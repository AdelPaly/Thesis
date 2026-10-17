"""ReFinED entity linker wrapper.

ReFinED (Amazon 2022) does joint NER + entity typing + disambiguation and links
directly to Wikidata QIDs, so there's no Wikipedia->QID hop. We feed it
Flair-detected mentions (via the `spans` override) so mention detection stays
identical across every non-LLM configuration.

Model identity (`refined_model`, `refined_entity_set`) comes from the
configuration; instances are cached per (model, entity_set) pair.
"""
import logging

from refined.inference.processor import Refined
from refined.data_types.base_types import Span

log = logging.getLogger(__name__)

_instances = {}


def clear_instances() -> None:
    """Drop cached ReFinED models (test / multi-config reset hook)."""
    _instances.clear()


def _load(model_name: str, entity_set: str) -> Refined:
    key = (model_name, entity_set)
    if key not in _instances:
        _instances[key] = Refined.from_pretrained(
            model_name=model_name,
            entity_set=entity_set,
        )
    return _instances[key]


def _candidates_of(span, top_n: int) -> list[tuple[str, float]]:
    """ReFinED's own ranked candidate set for a span, as [(qid, score)].

    These are already on the returned Span object -- `top_k_predicted_entities`
    when the model produced a ranking, `candidate_entities` (the generator's
    priors) otherwise -- so keeping them costs nothing beyond the copy.
    """
    out = []
    for entity, score in (getattr(span, "top_k_predicted_entities", None) or []):
        qid = getattr(entity, "wikidata_entity_id", None)
        if qid:
            out.append((qid, float(score)))
    if not out:
        for item in (getattr(span, "candidate_entities", None) or []):
            try:
                qid, score = item
            except (TypeError, ValueError):
                continue
            if isinstance(qid, str) and qid.startswith("Q"):
                out.append((qid, float(score)))
    return out[:top_n]


def link_mentions(text: str, entities: list[dict],
                  model_name: str = "wikipedia_model",
                  entity_set: str = "wikipedia",
                  top_n: int = 10) -> list[tuple[dict, dict]]:
    """Link each Flair mention to a Wikidata QID.

    Returns [(entity_dict, link_dict), ...] in input order, one entry per
    mention, where link_dict is::

        {"qid": str | None, "confidence": float | None,
         "candidates": [{"qid": str, "score": float}, ...]}

    Mentions carry Flair's own character offsets (ner.extract_entities), so the
    ReFinED spans address exactly the text Flair tagged. The previous version
    re-derived offsets with a case-insensitive `str.find` and a moving cursor,
    which silently dropped any mention it could not re-find and misaligned the
    rest. Every mention now produces a span, so the output is always the same
    length as the input. A mention whose offsets are not valid for `text`
    (0 <= start < end <= len(text)) is skipped with a warning and gets an
    empty link, so the output stays aligned with the input.
    """
    if not entities:
        return []

    spans = {}
    for i, ent in enumerate(entities):
        start, end = ent.get("start"), ent.get("end")
        if not (isinstance(start, int) and isinstance(end, int)
                and 0 <= start < end <= len(text)):
            log.warning("skipping mention %r with invalid offsets (%r, %r) for text of length %d",
                        ent.get("text"), start, end, len(text))
            continue
        spans[i] = Span(text=text[start:end], start=start, ln=end - start)

    linked = (_load(model_name, entity_set).process_text(text, spans=list(spans.values()))
              if spans else [])

    by_key = {}
    for s in linked:
        pred = getattr(s, "predicted_entity", None)
        qid = getattr(pred, "wikidata_entity_id", None) if pred else None
        conf = getattr(s, "entity_linking_model_confidence_score", None)
        by_key[(s.start, s.start + s.ln)] = {
            "qid": qid,
            "confidence": float(conf) if conf is not None else None,
            "candidates": [{"qid": q, "score": sc} for q, sc in _candidates_of(s, top_n)],
        }

    out = []
    for i, ent in enumerate(entities):
        sp = spans.get(i)
        link = by_key.get((sp.start, sp.start + sp.ln)) if sp else None
        out.append((ent, link or {"qid": None, "confidence": None, "candidates": []}))
    return out
