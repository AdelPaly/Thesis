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
from typing import Callable, TypedDict

from config import ConfigError, PipelineConfig
from keys import item_id, place_key

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
        # The description is what the reranker actually scored; without it the
        # score in this row cannot be interpreted after the fact.
        "description": cand.get("description"),
    }


class Pipeline:
    """Built once per run from a config; `run(text)` is called per item."""

    def __init__(self, cfg: PipelineConfig, dataset: str | None = None, *,
                 ner_fn: Callable | None = None,
                 link_fn: Callable | None = None,
                 rank_fn: Callable | None = None,
                 resolve_fn: Callable | None = None):
        """The `*_fn` arguments replace the real stage callables
        (extract_entities, link_mentions, rank_candidates,
        resolve_qid_to_candidate). Left as None they are imported lazily, so
        the pipeline can be exercised without Flair, ReFinED or the network."""
        self.cfg = cfg
        # Some configurations are constants fitted on the dev split of a
        # particular dataset (majority_class), so the pipeline needs to know
        # which dataset it is running over.
        self.dataset = dataset
        self._llm = None
        self._grounded = None
        self._majority = _UNSET
        self._lexical = None
        self._geoparser = None
        self._grid_lm = None
        self._ner_memo: dict[str, list[dict]] = {}
        self._dense_index = None
        self._ner_fn = ner_fn
        self._link_fn = link_fn
        self._rank_fn = rank_fn
        self._resolve_fn = resolve_fn
        self._paradigms = {
            "ner_dependent": self._run_linking_path,
            "ner_free": self._run_llm_path,
            "dense_retrieval": self._run_dense_path,
            "grounded_llm": self._run_grounded_path,
            "baseline": self._run_baseline,
            "geoparser": self._run_geoparser_path,
            "grid_lm": self._run_grid_lm_path,
            "lexical_retrieval": self._run_lexical_path,
        }

    # ---------------------------------------------------------------- stages

    def _ner(self, text: str) -> list[dict]:
        """Extract mention spans with the configured NER model. Memoised per
        run, because `prepare` and `run` both need the spans."""
        if text in self._ner_memo:
            return [dict(e) for e in self._ner_memo[text]]
        fn = self._ner_fn
        if fn is None:
            from ner import extract_entities as fn
        spans = fn(text, model=self.cfg.ner.model,
                   entity_types=self.cfg.ner.entity_types)
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
        fn = self._link_fn
        if fn is None:
            from refined_linker import link_mentions as fn
        linked = fn(text, spans,
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
        """Yield (link, candidate) for each linked QID that resolves."""
        resolve = self._resolve_fn
        if resolve is None:
            from wikidata import resolve_qid_to_candidate as resolve
        for ent, link in zip(spans, linked_entities):
            if not link["qid"]:
                continue
            cand = resolve(link["qid"], ent["text"], ent["type"])
            if cand:
                yield link, cand

    def _places_from_links(self, spans, linked_entities) -> list[Candidate]:
        """Turn linked QIDs into geographic candidates via the property chain."""
        return [cand for _, cand in self._resolved_links(spans, linked_entities)]

    def _places_from_search(self, spans) -> list[Candidate]:
        from wikidata import get_all_coords
        return get_all_coords(spans, top_n=self.cfg.linking.top_n)

    def _rank(self, text: str, candidates: list[Candidate]) -> list[Candidate]:
        """Rerank by text/description similarity; a no-op when disabled."""
        if not candidates or not self.cfg.rerank.enabled:
            return candidates
        fn = self._rank_fn
        if fn is None:
            from ranking import rank_candidates as fn
        return fn(text, candidates, model=self.cfg.rerank.model)

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

    def _dev_rows(self) -> list[dict]:
        """The dev side of this dataset's split, one dict per item with a gold
        QID. This is the training data of every fitted configuration: a place
        is either dev or test, so nothing fitted here has seen a test place."""
        import pandas as pd

        from config import load_datasets, REPO_ROOT

        if not self.dataset:
            raise ConfigError(f"configuration {self.cfg.name!r} is fitted on dev and "
                              "needs to know which dataset it is running over")
        with open(REPO_ROOT.joinpath(*SPLIT_FILE), encoding="utf-8") as f:
            side = {r["place_key"]: r["split"] for r in csv.DictReader(f)}

        df = pd.read_csv(load_datasets()[self.dataset].resolved_path())
        if "qid" not in df:
            return []
        rows = []
        for row in df.to_dict("records"):
            qid = row.get("qid")
            if not (isinstance(qid, str) and qid.strip()):
                continue
            if side.get(place_key(qid, row.get("location_name"))) != "dev":
                continue
            rows.append({
                "item_id": item_id(row["text"]),
                "text": row["text"],
                "qid": qid.strip(),
                "location_name": _str_or_none(row.get("location_name")),
                "country": _str_or_none(row.get("country")),
                "continent": _str_or_none(row.get("continent")),
                "lat": _float_or_none(row.get("latitude")),
                "lon": _float_or_none(row.get("longitude")),
            })
        return rows

    def _dev_place_counts(self) -> Counter:
        """Gold QID counts over the dev side of this dataset's split."""
        return Counter(r["qid"] for r in self._dev_rows())

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
            resolve = self._resolve_fn
            if resolve is None:
                from wikidata import resolve_qid_to_candidate as resolve
            cand = resolve(qid, qid, "LOC")
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
        if strategy == "most_populous":
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
        gr = self.cfg.grounded
        return f"{self.cfg.llm.cache_name}_grounded_{gr.candidate_source}"

    def _top_linked(self, spans, linked_entities, top_k: int) -> list[Candidate]:
        """Distinct QIDs, most confident linker output first, then truncated."""
        best: dict[str, tuple[float, Candidate]] = {}
        for link, cand in self._resolved_links(spans, linked_entities):
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
            out.ner_spans = self._ner(text)
            out.linked_entities = self._link(text, out.ner_spans)
            candidates = self._top_linked(out.ner_spans, out.linked_entities,
                                          gr.top_k)
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

    # ----------------------------------- tier 3: lexical and grid language model
    #
    # Both rank dense retrieval's own candidate places, described by the same
    # indexed text (dense_index.build_catalogue), so the three tier-3
    # mechanisms differ in the mechanism only.

    def _catalogue(self):
        from dense_index import get_catalogue
        return get_catalogue(self.cfg.dense)

    def _run_lexical_path(self, text, out):
        """Character n-gram overlap between the post and each place's indexed
        text. Nothing learned: no training texts, no encoder."""
        cat = self._catalogue()
        if self._lexical is None:
            from char_ngram import CharNgramIndex
            lx = self.cfg.lexical
            self._lexical = CharNgramIndex(cat.texts, lx.ngram_min, lx.ngram_max)
        candidates = [{**cat.places[cat.qids[i]], "score": score}
                      for i, score in self._lexical.rank(text, self.cfg.dense.top_k)]
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

    def _run_grid_lm_path(self, text, out):
        """Words in, cell out: the grid cells whose language models are closest
        to the post, each resolved to its candidate places. Cell models are
        learned from dev texts and the places' indexed texts; no gazetteer at
        query time, no mention detection, no LLM."""
        cat = self._catalogue()
        if self._grid_lm is None:
            from grid_lm import GridLM
            self._grid_lm = GridLM(self._dev_rows(), cat.places,
                                   dict(zip(cat.qids, cat.texts)),
                                   self.cfg.grid.cell_degrees)
        ranked = self._grid_lm.rank(text, item_id=item_id(text),
                                    top_k=self.cfg.dense.top_k)
        candidates = [{**cat.places[qid], "score": score} for qid, score in ranked]
        out.candidate_places = [_candidate_record(c) for c in candidates]
        out.prediction = candidates[0] if candidates else None

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


def set_debug(flag: bool) -> None:
    """Compatibility shim for older callers: per-item tracebacks are logged at
    DEBUG, so this just lowers this module's logger level."""
    if flag:
        logger.setLevel(logging.DEBUG)
