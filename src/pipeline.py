"""One parameterised geolocation path, driven entirely by a PipelineConfig.

This replaces the six near-duplicate `geolocate_*` functions that used to live
in main.py. Which NER model runs, which linker links, whether a reranker is
used, which LLM answers and whether anything is routed are all read off the
config -- there is no per-configuration code path.

`Pipeline.run(text)` returns a `PipelineOutput` carrying every intermediate
stage, not just the winning candidate: the NER spans, the linked entities with
their confidences and ReFinED's own candidate set, the fully scored candidate
place list, and (for routed configurations) the branch that was taken. The
runner serialises those straight into the prediction record.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from config import PipelineConfig


@dataclass
class PipelineOutput:
    ner_spans: list = field(default_factory=list)
    linked_entities: list = field(default_factory=list)
    candidate_places: list = field(default_factory=list)
    routing_branch: str | None = None
    routing_rule: str | None = None
    # The gate's input features, structured. routing_rule is the human-readable
    # form of the same thing; this is what a threshold sweep reads, so that
    # refitting the threshold needs no re-run and no string parsing.
    routing_features: dict | None = None
    prediction: dict | None = None
    llm_raw_response: str | None = None
    error: str | None = None


def _candidate_record(cand):
    """Normalise an internal candidate dict into the stored shape."""
    return {
        "qid": cand.get("qid"),
        "lat": cand.get("lat"),
        "lon": cand.get("lon"),
        "country": cand.get("country"),
        "continent": cand.get("continent"),
        "score": cand.get("score"),
        "source_entity": cand.get("entity"),
        "source_entity_type": cand.get("entity_type"),
        # The description is what the reranker actually scored; without it the
        # score in this row cannot be interpreted after the fact.
        "description": cand.get("description"),
    }


def route_from_features(features, fallback="llm", ambiguity_threshold=2,
                        has_geoparser=False):
    """The routing rule, as a pure function of the gate's input features.

    features: {"toponym_present": bool, "max_candidates": int | None,
               "non_loc_spans": int}

    Returns (branch, reason). Hand-specified, not learned:

      no toponym, entity spans present             -> linking
      toponym present, unambiguous                 -> geoparser if configured,
                                                      else linking
      toponym present, ambiguous, entities present -> linking
      toponym present, ambiguous, no entity spans  -> llm
      nothing detected                             -> fallback

    Lives here, as one function, because scripts/fit_gate.py sweeps the
    threshold over stored features and must evaluate the same rule the
    pipeline executes rather than a copy of it.
    """
    non_loc = features.get("non_loc_spans") or 0
    if not features.get("toponym_present"):
        if non_loc:
            return "linking", "no toponym, entity spans present"
        return fallback, "nothing detected"

    max_candidates = features.get("max_candidates") or 0
    ambiguous = max_candidates >= ambiguity_threshold
    if not ambiguous:
        return ("geoparser" if has_geoparser else "linking",
                "toponym present, unambiguous")
    if non_loc:
        return "linking", "toponym ambiguous, entity spans present"
    return "llm", "toponym ambiguous, no entity spans"


class Pipeline:
    """Built once per run from a config; `run(text)` is called per item."""

    def __init__(self, cfg: PipelineConfig, dataset: str | None = None):
        self.cfg = cfg
        # Some configurations are constants fitted on the dev split of a
        # particular dataset (baseline_majority), so the pipeline needs to know
        # which dataset it is running over.
        self.dataset = dataset
        self._llm = None
        self._majority = _UNSET

    # ---------------------------------------------------------------- stages

    def _ner(self, text):
        from ner import extract_entities
        return extract_entities(text,
                                model=self.cfg.ner.model,
                                entity_types=self.cfg.ner.entity_types)

    def _link(self, text, spans):
        """Link mentions to QIDs. Returns the stored `linked_entities` shape."""
        li = self.cfg.linking
        if li.backend != "refined" or not spans:
            return [{"mention": e["text"], "mention_type": e["type"],
                     "qid": None, "confidence": None, "candidates": []}
                    for e in spans]
        from refined_linker import link_mentions
        linked = link_mentions(text, spans,
                               model_name=li.refined_model,
                               entity_set=li.refined_entity_set,
                               top_n=li.top_n)
        return [{"mention": ent["text"], "mention_type": ent["type"],
                 "qid": link["qid"], "confidence": link["confidence"],
                 "candidates": link["candidates"]}
                for ent, link in linked]

    def _places_from_links(self, spans, linked_entities):
        """Turn linked QIDs into geographic candidates via the property chain,
        with the optional per-mention wbsearchentities fallback."""
        from wikidata import resolve_qid_to_candidate, _search_entities
        li = self.cfg.linking
        out = []
        for ent, link in zip(spans, linked_entities):
            cand = None
            if link["qid"]:
                cand = resolve_qid_to_candidate(link["qid"], ent["text"], ent["type"])
            if cand is None and li.search_fallback:
                hits = _search_entities(ent["text"], limit=1)
                if hits:
                    cand = resolve_qid_to_candidate(hits[0], ent["text"], ent["type"])
            if cand:
                out.append(cand)
        return out

    def _places_from_search(self, spans):
        from wikidata import get_all_coords
        return get_all_coords(spans, top_n=self.cfg.linking.top_n)

    def _rank(self, text, candidates):
        if not candidates:
            return candidates
        if not self.cfg.rerank.enabled:
            return candidates
        from ranking import rank_candidates
        return rank_candidates(text, candidates, model=self.cfg.rerank.model)

    def _llm_pipeline(self):
        if self._llm is None:
            backend = self.cfg.llm.backend
            if backend == "gemma":
                from llm_gemma import build
            elif backend in ("groq_oss", "groq_llama"):
                from llm_groq import build
            else:
                raise ValueError(f"unknown llm.backend {backend!r}")
            self._llm = build(self.cfg.llm.model, self.cfg.llm.cache_name)
        return self._llm

    def save_llm_cache(self):
        if self._llm is not None:
            self._llm.save_cache()

    # ------------------------------------------------------------ baselines

    def _run_population_baseline(self, text, out):
        """Hard floor: NER toponyms -> gazetteer -> always the most populous
        candidate. No reranker, no linker, no context. In the Gritta et al.
        GeoWebNews comparison this heuristic on top of spaCy NER reached 95
        Accuracy@161km at 210 km mean error, beating several published
        geoparsers -- which is why it is the floor a real system must clear."""
        from gazetteer import lookup
        ba = self.cfg.baseline
        out.ner_spans = self._ner(text)

        candidates = []
        for span in out.ner_spans:
            for cand in lookup(span["text"], backend=ba.gazetteer,
                               top_n=self.cfg.linking.top_n,
                               min_population=ba.min_population):
                candidates.append({
                    "entity": span["text"],
                    "entity_type": span["type"],
                    "qid": cand["qid"],
                    "lat": cand["lat"], "lon": cand["lon"],
                    "country": cand["country"], "continent": cand["continent"],
                    "description": cand["description"],
                    # The population IS the score for this configuration; it is
                    # the only thing it ranks on.
                    "score": float(cand["population"] or 0),
                })
        candidates.sort(key=lambda c: c["score"], reverse=True)
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    def _majority_place(self):
        """The most frequent gold place in the DEV split of this dataset.

        Computed from the committed split file, dev side only -- a majority
        baseline fitted on test would be reading the answer sheet."""
        if self._majority is not _UNSET:
            return self._majority
        import csv as _csv
        from collections import Counter

        import pandas as pd

        from config import load_datasets, REPO_ROOT
        from wikidata import resolve_qid_to_candidate

        if not self.dataset:
            raise ValueError("baseline 'majority_dev_place' needs to know which "
                             "dataset it is running over")
        split_path = REPO_ROOT / "splits" / "place_split.csv"
        with open(split_path, encoding="utf-8") as f:
            side = {r["place_key"]: r["split"] for r in _csv.DictReader(f)}

        spec = load_datasets()[self.dataset]
        df = pd.read_csv(spec.resolved_path())
        counts = Counter()
        for _, row in df.iterrows():
            qid = row.get("qid")
            key = qid.strip() if isinstance(qid, str) and qid.strip() else None
            if key is None:
                name = str(row.get("location_name") or "").strip().casefold()
                key = f"name:{name}" if name else None
            if key and side.get(key) == "dev" and isinstance(qid, str):
                counts[qid.strip()] += 1
        if not counts:
            self._majority = None
            return None
        qid, n = counts.most_common(1)[0]
        cand = resolve_qid_to_candidate(qid, qid, "LOC")
        if cand is not None:
            cand = {**cand, "score": float(n)}
        self._majority = cand
        return cand

    def _run_majority_baseline(self, text, out):
        """Predict the same place for every item. Measures how much of a
        headline number is carried by the label distribution alone."""
        cand = self._majority_place()
        out.prediction = dict(cand) if cand else None

    # ----------------------------------------------------------- geoparser

    def _run_geoparser_path(self, text, out):
        from geoparser import geoparse
        gp = self.cfg.geoparser
        spans, candidates = geoparse(text, gp.backend, gp.spacy_model)
        out.ner_spans = spans
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    # ---------------------------------------------------------------- gates

    def _legacy_gate(self, text, out):
        """The original gate: did ReFinED link a Flair LOC to a QID that turns
        out to be a geographic-area type. It reads the FIRST SYSTEM'S OUTPUT,
        which is why it behaves as a fallback rather than a router."""
        from wikidata import is_geographic_qid
        out.ner_spans = self._ner(text)
        loc = [e for e in out.ner_spans if e["type"] == "LOC"]
        out.routing_features = {
            "toponym_present": bool(loc),
            "max_candidates": None,          # this gate never asks a gazetteer
            "non_loc_spans": len(out.ner_spans) - len(loc),
        }
        if not loc:
            out.routing_rule = "no LOC span -> fallback"
            return self.cfg.routing.fallback
        out.linked_entities = self._link(text, out.ner_spans)
        fired = any(is_geographic_qid(le["qid"])
                    for le in out.linked_entities
                    if le["mention_type"] == "LOC")
        out.routing_rule = ("LOC linked to a geographic QID -> linking"
                            if fired else "no LOC linked to a geographic QID -> fallback")
        return "linking" if fired else self.cfg.routing.fallback

    def _input_properties_gate(self, text, out):
        """Route on properties of the INPUT, computed before any system runs.

        Three features, all read off the text:
          * is a toponym present          -- a LOC span from NER
          * is that toponym ambiguous     -- candidate count from a gazetteer
          * how many non-LOC spans exist  -- PER / ORG / MISC mentions

        Rule (hand-specified, not learned):

          no toponym, entity spans present             -> entity-linking path
          toponym present, unambiguous                 -> geoparser, else linking
          toponym present, ambiguous, entities present -> linking (the entity
                                                          disambiguates)
          toponym present, ambiguous, no entities      -> LLM
          nothing detected                             -> LLM

        The old gate asked "did ReFinED link something that turned out to be a
        place", which reads the first system OUTPUT. A noisy toponym still
        links to a geographic QID, so that gate fired and sent a poisoned input
        down the ReFinED path. This gate never consults a system answer.
        """
        from gazetteer import count_candidates
        ro = self.cfg.routing

        out.ner_spans = self._ner(text)
        loc = [e for e in out.ner_spans if e["type"] == "LOC"]
        non_loc = [e for e in out.ner_spans if e["type"] != "LOC"]

        if loc:
            max_count = max(count_candidates(e["text"], backend=ro.gazetteer,
                                             top_n=ro.gazetteer_top_n)
                            for e in loc)
        else:
            max_count = None

        out.routing_features = {
            "toponym_present": bool(loc),
            "max_candidates": max_count,
            "non_loc_spans": len(non_loc),
        }
        branch, why = route_from_features(
            out.routing_features,
            fallback=ro.fallback,
            ambiguity_threshold=ro.ambiguity_threshold,
            has_geoparser=bool(self.cfg.geoparser.backend),
        )
        out.routing_rule = (
            f"toponym={int(bool(loc))} candidates={max_count} "
            f"non_loc={len(non_loc)} -> {branch} ({why})")
        return branch

    # ---------------------------------------------------------------- paths

    def _run_linking_path(self, text, out):
        """NER -> linking (or search retrieval) -> rerank -> top-1."""
        if not out.ner_spans:
            out.ner_spans = self._ner(text)
        if self.cfg.linking.backend == "wikidata_search":
            candidates = self._places_from_search(out.ner_spans)
        else:
            if not out.linked_entities:
                out.linked_entities = self._link(text, out.ner_spans)
            candidates = self._places_from_links(out.ner_spans, out.linked_entities)
        candidates = self._rank(text, candidates)
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    def _run_llm_path(self, text, out):
        """Whole-text LLM reading. candidate_places stays empty by
        construction: the LLM's answer is resolved with a limit=1 search, so
        there is no scored candidate list to record."""
        cand, raw = self._llm_pipeline().geolocate_verbose(text)
        out.llm_raw_response = raw
        out.prediction = cand

    # ---------------------------------------------------------------- entry

    def run(self, text) -> PipelineOutput:
        """Run one item. Never raises: a failure is attached to `out.error`
        with whatever stages completed before it, so the item is still
        recorded rather than dropped."""
        out = PipelineOutput()
        try:
            paradigm = self.cfg.paradigm
            if paradigm == "ner_dependent":
                self._run_linking_path(text, out)
            elif paradigm == "ner_free":
                self._run_llm_path(text, out)
            elif paradigm == "routed":
                branch = self._route(text, out)
                out.routing_branch = branch
                if branch == "linking":
                    self._run_linking_path(text, out)
                elif branch == "llm":
                    self._run_llm_path(text, out)
                elif branch == "geoparser":
                    self._run_geoparser_path(text, out)
                else:
                    raise ValueError(f"unhandled routing branch {branch!r}")
            elif paradigm == "geoparser":
                self._run_geoparser_path(text, out)
            elif paradigm == "baseline":
                strategy = self.cfg.baseline.strategy
                if strategy == "most_populous":
                    self._run_population_baseline(text, out)
                elif strategy == "majority_dev_place":
                    self._run_majority_baseline(text, out)
                else:
                    raise ValueError(f"unhandled baseline strategy {strategy!r}")
            else:
                raise ValueError(f"unhandled paradigm {paradigm!r}")
        except Exception as e:                      # noqa: BLE001 - recorded, not swallowed
            out.error = f"{type(e).__name__}: {e}"
            out.prediction = None
            if _DEBUG:
                traceback.print_exc()
        return out

    def _route(self, text, out):
        gate = self.cfg.routing.gate
        if gate == "legacy_is_geographic_qid":
            return self._legacy_gate(text, out)
        if gate == "input_properties":
            return self._input_properties_gate(text, out)
        raise ValueError(f"unknown routing gate {gate!r}")


_UNSET = object()
_DEBUG = False


def set_debug(flag: bool):
    """Print tracebacks for per-item failures instead of only recording them."""
    global _DEBUG
    _DEBUG = flag
