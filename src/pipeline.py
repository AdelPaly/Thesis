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

import csv
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TypedDict

from config import ConfigError, PipelineConfig
from keys import place_key

logger = logging.getLogger(__name__)

_UNSET = object()

# Errors that mean the run is misconfigured, not that one item is bad. They
# abort the run instead of being recorded as N identical per-item errors.
_FATAL_ERRORS = (ConfigError, ImportError, KeyError)

SPLIT_FILE = ("splits", "place_split.csv")


class Candidate(TypedDict, total=False):
    """Internal candidate place; the stored shape is `_candidate_record`."""
    qid: str
    lat: float
    lon: float
    country: str | None
    continent: str | None
    score: float
    entity: str
    entity_type: str
    description: str | None
    label: str | None


@dataclass
class PipelineOutput:
    """Every intermediate stage of one item, serialised by the runner as-is.

    `error` is set when a stage failed or a grounded selection broke its
    contract; stages completed before the failure stay populated."""
    ner_spans: list[dict] = field(default_factory=list)
    linked_entities: list[dict] = field(default_factory=list)
    candidate_places: list[dict] = field(default_factory=list)
    prediction: dict | None = None
    llm_raw_response: str | None = None
    # Grounded LLM only: which candidate was selected, whether it abstained,
    # and whether it broke the selection contract. Stored so a wrong answer can
    # be attributed to bad retrieval versus bad selection.
    selection: dict | None = None
    # LLM alone only: how the answer became a Wikidata item, or why it did not
    # (declined, unreadable, not_found, ambiguous). See src/resolver.py.
    resolution: dict | None = None
    error: str | None = None


def _str_or_none(value) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _float_or_none(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def _geonames_candidate(toponym: dict) -> Candidate:
    """A candidate straight from the GeoNames entry, for an entry with no
    Wikidata item: no QID and no continent (the entry carries neither), so it
    can be right at country level only."""
    data = toponym.get("data") or {}
    return {"qid": None,
            "lat": _float_or_none(data.get("latitude")),
            "lon": _float_or_none(data.get("longitude")),
            "country": _str_or_none(data.get("country_name")),
            "continent": None,
            "entity": toponym["text"], "entity_type": "LOC",
            "description": _str_or_none(data.get("name"))}


def _candidate_record(cand: Candidate) -> dict:
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
        # The place's own name, which the grounded list shows (linked
        # candidates only; None for the others). Stored from the development
        # run of 26-27 November on.
        "label": cand.get("label"),
        # The description is what the reranker actually scored; without it the
        # score in this row cannot be interpreted after the fact.
        "description": cand.get("description"),
    }


class Pipeline:
    """Built once per run from a config; `run(text)` is called per item."""

    def __init__(self, cfg: PipelineConfig, dataset: str | None = None):
        """The stage modules (ner, refined_linker, ranking, ...) are imported
        where a stage first runs, so a configuration loads only what it uses."""
        self.cfg = cfg
        # Some configurations are constants fitted on the dev split of a
        # particular dataset (majority_class), so the pipeline needs to know
        # which dataset it is running over.
        self.dataset = dataset
        self._llm = None
        self._grounded = None
        self._majority = _UNSET
        self._geoparser = None
        self._ner_memo: dict[str, list[dict]] = {}
        self._dense_index = None
        self._paradigms = {
            "ner_dependent": self._run_linking_path,
            "ner_free": self._run_llm_path,
            "dense_retrieval": self._run_dense_path,
            "grounded_llm": self._run_grounded_path,
            "baseline": self._run_baseline,
            "geoparser": self._run_geoparser_path,
        }

    # ---------------------------------------------------------------- stages

    def _ner(self, text: str) -> list[dict]:
        """Extract mention spans with the configured NER model. Memoised per
        run, because `prepare` and `run` both need the spans."""
        if text in self._ner_memo:
            return [dict(e) for e in self._ner_memo[text]]
        from ner import extract_entities
        # The fallback types are kept too (in text order with the others);
        # _link_with_fallback decides whether they are linked.
        types = self.cfg.ner.entity_types
        if types is not None and self.cfg.ner.fallback_types:
            types = [*types, *self.cfg.ner.fallback_types]
        spans = extract_entities(text, model=self.cfg.ner.model, entity_types=types)
        self._ner_memo[text] = [dict(e) for e in spans]
        return spans

    def _link(self, text: str, spans: list[dict]) -> list[dict]:
        """Link mentions to QIDs. Returns the stored `linked_entities` shape."""
        li = self.cfg.linking
        if li.backend == "blink" and spans:
            return self._link_blink(text, spans)
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

    def _link_blink(self, text: str, spans: list[dict]) -> list[dict]:
        """BLINK per mention: the cross-encoder's top entity, mapped to a QID
        by its Wikipedia title. The whole top-k ranking is kept as the
        linker's candidate set."""
        from blink_linker import get_linker, make_sample
        from wikidata import qids_for_enwiki_titles

        linker = get_linker(self.cfg.linking.top_n)
        ranked = [linker.link(make_sample(text, e["start"], e["end"])) for e in spans]
        qids = qids_for_enwiki_titles([c["title"] for r in ranked for c in r])
        out = []
        for ent, ranking in zip(spans, ranked):
            cands = [{"qid": qids.get(c["title"]), "title": c["title"],
                      "score": c["crossencoder_score"],
                      "biencoder_score": c["biencoder_score"]} for c in ranking]
            top = cands[0] if cands else {}
            out.append({"mention": ent["text"], "mention_type": ent["type"],
                        "qid": top.get("qid"), "confidence": top.get("score"),
                        "candidates": cands})
        return out

    def prepare(self, texts: list[str]) -> None:
        """Work a run can do once for all its items before the per-item loop.
        For BLINK that is the bi-encoder pass over the entity matrix: one pass
        for every mention of the run instead of one per item."""
        if self.cfg.linking.backend != "blink" or not texts:
            return
        from blink_linker import get_linker, make_sample

        samples = [make_sample(t, e["start"], e["end"])
                   for t in texts for e in self._ner(t)]
        get_linker(self.cfg.linking.top_n).link_all(samples)

    def _resolved_links(self, spans, linked_entities):
        """Yield (link, candidate) for each place a linked QID resolves to
        under `linking.place_rule`: at most one per link for chain_v1, every
        place for all_places (src/wikidata.py)."""
        from wikidata import resolve_qid_to_candidates
        rule = self.cfg.linking.place_rule
        for ent, link in zip(spans, linked_entities):
            if not link["qid"]:
                continue
            for cand in resolve_qid_to_candidates(link["qid"], ent["text"], ent["type"], rule):
                yield link, cand

    def _link_with_fallback(self, text: str) -> tuple[list[dict], list[dict], list[tuple]]:
        """NER, linking and resolution, with the fallback of `ner.fallback_types`.

        The mentions of the main types are linked first, in a linker call of
        their own, so their links are those of a run without the fallback.
        Only if none of them resolves to a place are the fallback mentions
        (MISC: demonyms, events) linked, in a second call, and resolved.

        Returns (spans, linked_entities, [(link, candidate), ...]). `spans` are
        all mentions in text order and `linked_entities` is aligned with it, one
        entry per span, as the analysis scripts expect: a fallback mention
        that was not needed gets an unlinked entry. Fallback entries carry
        `fallback`: "used" or "not_needed"."""
        spans = self._ner(text)
        fallback = set(self.cfg.ner.fallback_types or [])
        main_idx = [i for i, s in enumerate(spans) if s["type"] not in fallback]
        fb_idx = [i for i, s in enumerate(spans) if s["type"] in fallback]

        links: list[dict | None] = [None] * len(spans)
        main = [spans[i] for i in main_idx]
        for i, link in zip(main_idx, self._link(text, main)):
            links[i] = link
        pairs = list(self._resolved_links(main, [links[i] for i in main_idx]))

        if fb_idx:
            fb = [spans[i] for i in fb_idx]
            if pairs:
                fb_links = [{"mention": e["text"], "mention_type": e["type"], "qid": None,
                             "confidence": None, "candidates": [], "fallback": "not_needed"}
                            for e in fb]
            else:
                fb_links = [{**link, "fallback": "used"} for link in self._link(text, fb)]
                pairs = list(self._resolved_links(fb, fb_links))
            for i, link in zip(fb_idx, fb_links):
                links[i] = link
        return spans, links, pairs

    def _places_from_search(self, spans) -> list[Candidate]:
        from wikidata import get_all_coords
        return get_all_coords(spans, top_n=self.cfg.linking.top_n,
                              place_rule=self.cfg.linking.place_rule)

    def _rank(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """Rerank by text/description similarity; a no-op when disabled."""
        if not candidates or not self.cfg.rerank.enabled:
            return candidates
        from ranking import rank_candidates
        return rank_candidates(text, candidates, model=self.cfg.rerank.model)

    def _llm_pipeline(self):
        if self._llm is None:
            # Imported lazily so a config that never calls an LLM does not pull
            # in a provider's client library.
            from llm_geo import build
            self._llm = build(self.cfg.llm)
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
                               min_population=ba.min_population,
                               same_name=ba.strategy == "most_populous_named"):
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

    def _dev_place_counts(self) -> Counter:
        """Gold QID counts over the dev side of this dataset's split. A place
        is either dev or test, so nothing fitted here has seen a test place."""
        import pandas as pd

        from config import load_datasets, REPO_ROOT

        if not self.dataset:
            raise ConfigError(f"configuration {self.cfg.name!r} is fitted on dev and "
                              "needs to know which dataset it is running over")
        with open(REPO_ROOT.joinpath(*SPLIT_FILE), encoding="utf-8") as f:
            side = {r["place_key"]: r["split"] for r in csv.DictReader(f)}

        df = pd.read_csv(load_datasets()[self.dataset].resolved_path(),
                         keep_default_na=False, na_values=[""])   # "NA" is a continent
        counts: Counter = Counter()
        if "qid" not in df:
            return counts
        for row in df.to_dict("records"):
            qid = row.get("qid")
            if not (isinstance(qid, str) and qid.strip()):
                continue
            if side.get(place_key(qid, row.get("location_name"))) != "dev":
                continue
            counts[qid.strip()] += 1
        return counts

    def _majority_place(self) -> Candidate | None:
        """The most frequent gold place in the DEV split of this dataset.

        Computed from the committed split file, dev side only -- a majority
        baseline fitted on test would be reading the answer sheet."""
        if self._majority is not _UNSET:
            return self._majority
        counts = self._dev_place_counts()
        cand = None
        if counts:
            qid, n = counts.most_common(1)[0]
            from wikidata import resolve_qid_to_candidate
            cand = resolve_qid_to_candidate(qid, qid, "LOC")
            if cand is not None:
                cand = {**cand, "score": float(n)}
        self._majority = cand
        return cand

    def _run_majority_baseline(self, text, out):
        """Predict the same place for every item. Measures how much of a
        headline number is carried by the label distribution alone. `text` is
        unused (uniform stage signature) and `candidate_places` stays empty."""
        cand = self._majority_place()
        out.prediction = dict(cand) if cand else None

    def _run_baseline(self, text, out):
        strategy = self.cfg.baseline.strategy
        if strategy in ("most_populous", "most_populous_named"):
            self._run_population_baseline(text, out)
        elif strategy == "majority_dev_place":
            self._run_majority_baseline(text, out)
        else:
            raise ConfigError(f"unhandled baseline strategy {strategy!r}")

    # ------------------------------------------------- tier 3: dense retrieval

    def _dense_candidates(self, text: str, top_k: int | None = None) -> list[Candidate]:
        """Top-k nearest places by cosine similarity (`dense.top_k` unless
        given). No mention detection, no linking, no LLM."""
        if self._dense_index is None:
            from dense_index import get_index
            self._dense_index = get_index(self.cfg)
        return self._dense_index.search(text, top_k=top_k or self.cfg.dense.top_k)

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
        # The prompt version is in the name from revision 3 on, so a new prompt
        # starts its own cache and the replies of the first runs stay where
        # they are (llm_common.PickleCache would set them aside as .stale).
        # The candidate list version, per source, has replaced it there since
        # the development run of 26-27 November: linking v3, dense v2
        # (llm_grounded.LIST_VERSIONS).
        from llm_grounded import LIST_VERSIONS, PROMPT_VERSION
        gr = self.cfg.grounded
        name = f"{self.cfg.llm.cache_name}_grounded_{gr.candidate_source}"
        if PROMPT_VERSION == "v1":
            return name
        return f"{name}_{LIST_VERSIONS.get(gr.candidate_source, PROMPT_VERSION)}"

    def _top_linked(self, pairs, top_k: int) -> list[Candidate]:
        """Distinct place QIDs, most confident linker output first, then
        truncated. `pairs` is [(link, candidate), ...]; the places of one
        linked entity (all_places) share its confidence and keep their list
        order among themselves."""
        best: dict[str, tuple[float, Candidate]] = {}
        for link, cand in pairs:
            conf = link.get("confidence")
            conf = float(conf) if conf is not None else float("-inf")
            qid = cand.get("qid") or link["qid"]
            if qid not in best or conf > best[qid][0]:
                best[qid] = (conf, cand)
        ranked = sorted(best.values(), key=lambda t: t[0], reverse=True)
        return [cand for _, cand in ranked[:top_k]]

    def _run_grounded_path(self, text, out):
        """Retrieve candidates, hand the list to the LLM, let it select one or
        decline. The retriever is an existing tier, never a new one, so the
        comparison against that tier is not confounded."""
        gr = self.cfg.grounded
        if gr.candidate_source == "dense":
            candidates = self._dense_candidates(text, top_k=gr.top_k)
        elif gr.candidate_source == "linking":
            out.ner_spans, out.linked_entities, pairs = self._link_with_fallback(text)
            candidates = self._top_linked(pairs, gr.top_k)
        else:
            raise ConfigError(f"unknown grounded.candidate_source "
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

    # ------------------------------------------------- tier 1: geoparsers

    def _run_geoparser_path(self, text, out):
        """An established geoparser run as published: its own recogniser finds
        the toponyms, its own resolver picks a GeoNames entry for each. Each
        entry is mapped to Wikidata through P1566 so exact match is
        computable. The prediction is the first resolved toponym in reading
        order; the geoparser itself ranks nothing across toponyms."""
        if self._geoparser is None:
            from geoparser_irchel import GeoparserWorker
            self._geoparser = GeoparserWorker(self.cfg.geoparser)
        from wikidata import qid_for_geonames_id, resolve_qid_to_candidate

        toponyms = self._geoparser.parse(text)
        out.ner_spans = [{"text": t["text"], "start": t["start"], "end": t["end"],
                          "type": "TOPONYM"} for t in toponyms]
        candidates = []
        for t in toponyms:
            gid = t.get("identifier")
            qid = qid_for_geonames_id(gid) if gid else None
            out.linked_entities.append({"mention": t["text"], "mention_type": "TOPONYM",
                                        "qid": qid, "confidence": None,
                                        "candidates": [], "geonames_id": gid})
            if not gid:
                continue
            cand = resolve_qid_to_candidate(qid, t["text"], "LOC") if qid else None
            candidates.append(cand or _geonames_candidate(t))
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    # ---------------------------------------------------------------- paths

    def _run_linking_path(self, text, out):
        """NER -> linking (or search retrieval) -> rerank -> top-1."""
        if self.cfg.linking.backend == "wikidata_search":
            # The search path has no fallback (config.py refuses one): it has
            # no handler for MISC mentions.
            out.ner_spans = self._ner(text)
            candidates = self._places_from_search(out.ner_spans)
        else:
            out.ner_spans, out.linked_entities, pairs = self._link_with_fallback(text)
            candidates = [cand for _, cand in pairs]
        candidates = self._rank(text, candidates)
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    def _run_llm_path(self, text, out):
        """Whole-text LLM reading. candidate_places stays empty by
        construction: the LLM's answer is resolved by src/resolver.py (the rule
        that goes with the prompt template), which keeps no scored candidate
        list to record."""
        cand, raw, resolution = self._llm_pipeline().geolocate_verbose(text)
        out.llm_raw_response = raw
        out.resolution = resolution
        out.prediction = cand
        if (resolution or {}).get("status") == "error":
            # Rule v4: a Wikidata request failed while resolving the reply. The
            # record carries `error`, so the runner retries the post on resume;
            # the reply itself is in the LLM cache, so the retry calls no model.
            out.error = f"wikidata: {resolution.get('message', '')}"

    # ---------------------------------------------------------------- entry

    def run(self, text: str) -> PipelineOutput:
        """Run one item. A per-item failure (data, network) is attached to
        `out.error` with whatever stages completed before it, so the item is
        still recorded rather than dropped. Configuration and import errors
        propagate: they would otherwise repeat on every item."""
        out = PipelineOutput()
        try:
            handler = self._paradigms.get(self.cfg.paradigm)
            if handler is None:
                raise ConfigError(f"unhandled paradigm {self.cfg.paradigm!r}")
            handler(text, out)
        except _FATAL_ERRORS:
            raise
        except Exception as e:                      # noqa: BLE001 - recorded, not swallowed
            logger.debug("item failed", exc_info=True)
            out.error = f"{type(e).__name__}: {e}"
            out.prediction = None
        return out
