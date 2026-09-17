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


class Pipeline:
    """Built once per run from a config; `run(text)` is called per item."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        if cfg.paradigm in ("baseline", "geoparser"):
            raise NotImplementedError(
                f"paradigm {cfg.paradigm!r} is not implemented yet "
                f"(config {cfg.name!r})")
        self._llm = None

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

    # ---------------------------------------------------------------- gates

    def _legacy_gate(self, text, out):
        """The original gate: did ReFinED link a Flair LOC to a QID that turns
        out to be a geographic-area type. It reads the FIRST SYSTEM'S OUTPUT,
        which is why it behaves as a fallback rather than a router."""
        from wikidata import is_geographic_qid
        out.ner_spans = self._ner(text)
        if not any(e["type"] == "LOC" for e in out.ner_spans):
            out.routing_rule = "no LOC span -> fallback"
            return self.cfg.routing.fallback
        out.linked_entities = self._link(text, out.ner_spans)
        fired = any(is_geographic_qid(le["qid"])
                    for le in out.linked_entities
                    if le["mention_type"] == "LOC")
        out.routing_rule = ("LOC linked to a geographic QID -> linking"
                            if fired else "no LOC linked to a geographic QID -> fallback")
        return "linking" if fired else self.cfg.routing.fallback

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
                else:
                    raise ValueError(f"unhandled routing branch {branch!r}")
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
        raise ValueError(f"unknown routing gate {gate!r}")


_DEBUG = False


def set_debug(flag: bool):
    """Print tracebacks for per-item failures instead of only recording them."""
    global _DEBUG
    _DEBUG = flag
