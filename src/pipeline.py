"""One parameterised geolocation path, driven entirely by a PipelineConfig.

Which NER model runs, which linker links, whether a reranker is used and which
LLM answers are all read off the config -- there is no per-configuration code
path.

`Pipeline.run(text)` returns a `PipelineOutput` carrying every intermediate
stage, not just the winning candidate: the NER spans, the linked entities with
their confidences and ReFinED's own candidate set, the fully scored candidate
place list and, for grounded configurations, the selection record. The
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
    prediction: dict | None = None
    llm_raw_response: str | None = None
    # Grounded LLM only: which candidate was selected, whether it abstained,
    # and whether it broke the selection contract. Stored so a wrong answer can
    # be attributed to bad retrieval versus bad selection.
    selection: dict | None = None
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


class Pipeline:
    """Built once per run from a config; `run(text)` is called per item."""

    def __init__(self, cfg: PipelineConfig, dataset: str | None = None):
        self.cfg = cfg
        # Some configurations are constants fitted on the dev split of a
        # particular dataset (baseline_majority), so the pipeline needs to know
        # which dataset it is running over.
        self.dataset = dataset
        self._llm = None
        self._grounded = None
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
        """Turn linked QIDs into geographic candidates via the property chain."""
        from wikidata import resolve_qid_to_candidate
        out = []
        for ent, link in zip(spans, linked_entities):
            if not link["qid"]:
                continue
            cand = resolve_qid_to_candidate(link["qid"], ent["text"], ent["type"])
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
            else:
                raise ValueError(f"unknown llm.backend {backend!r}")
            self._llm = build(self.cfg.llm.model, self.cfg.llm.cache_name)
        return self._llm

    def save_llm_cache(self):
        if self._llm is not None:
            self._llm.save_cache()
        if self._grounded is not None:
            self._grounded.save_cache()

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

    # ------------------------------------------------- tier 3: dense retrieval

    def _dense_candidates(self, text):
        """Top-k nearest places by cosine similarity. No mention detection, no
        linking, no LLM."""
        from dense_index import get_index
        index = get_index(self.cfg)
        return index.search(text, top_k=self.cfg.dense.top_k)

    def _run_dense_path(self, text, out):
        candidates = self._dense_candidates(text)
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    # -------------------------------------------------- tier 5: grounded LLM

    def _grounded_selector(self):
        if self._grounded is None:
            from llm_grounded import build
            self._grounded = build(self._llm_pipeline(), self._grounded_cache_name())
        return self._grounded

    def _grounded_cache_name(self):
        gr = self.cfg.grounded
        return f"{self.cfg.llm.cache_name}_grounded_{gr.candidate_source}"

    def _run_grounded_path(self, text, out):
        """Retrieve candidates, hand the list to the LLM, let it select one or
        decline. The retriever is an existing tier, never a new one, so the
        comparison against that tier is not confounded."""
        gr = self.cfg.grounded
        if gr.candidate_source == "dense":
            candidates = self._dense_candidates(text)[:gr.top_k]
        elif gr.candidate_source == "linking":
            out.ner_spans = self._ner(text)
            out.linked_entities = self._link(text, out.ner_spans)
            candidates = self._places_from_links(out.ner_spans,
                                                 out.linked_entities)[:gr.top_k]
        else:
            raise ValueError(f"unknown grounded.candidate_source "
                             f"{gr.candidate_source!r}")

        out.candidate_places = [_candidate_record(c) for c in candidates]

        if not candidates:
            # Nothing to select from. This is a retrieval failure, and the
            # selection record says so rather than blaming the model.
            out.selection = {"index": None, "qid": None, "abstained": True,
                             "violation": None, "no_candidates": True}
            return

        chosen, selection, raw = self._grounded_selector().select(text, candidates)
        out.llm_raw_response = raw
        out.selection = selection
        if selection.get("violation"):
            # Recorded, never silently scored as an ordinary wrong guess: a
            # grounded system ignoring its grounding has to be visible.
            out.error = f"selection violation: {selection['violation']}"
        out.prediction = chosen

    # ---------------------------------------------------------------- paths

    def _run_linking_path(self, text, out):
        """NER -> linking (or search retrieval) -> rerank -> top-1."""
        out.ner_spans = self._ner(text)
        if self.cfg.linking.backend == "wikidata_search":
            candidates = self._places_from_search(out.ner_spans)
        else:
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
            elif paradigm == "dense_retrieval":
                self._run_dense_path(text, out)
            elif paradigm == "grounded_llm":
                self._run_grounded_path(text, out)
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


_UNSET = object()
_DEBUG = False


def set_debug(flag: bool):
    """Print tracebacks for per-item failures instead of only recording them."""
    global _DEBUG
    _DEBUG = flag
