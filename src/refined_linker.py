"""ReFinED entity linker wrapper.

ReFinED (Amazon 2022) does joint NER + entity typing + disambiguation and links
directly to Wikidata QIDs, so there's no Wikipedia->QID hop. We feed it
Flair-detected mentions (via the `spans` override) so mention detection stays
identical across every non-LLM configuration.

Model identity (`refined_model`, `refined_entity_set`) comes from the
configuration; instances are cached per (model, entity_set) pair.
"""
from refined.inference.processor import Refined
from refined.data_types.base_types import Span

_instances = {}


def _load(model_name, entity_set):
    key = (model_name, entity_set)
    if key not in _instances:
        _instances[key] = Refined.from_pretrained(
            model_name=model_name,
            entity_set=entity_set,
        )
    return _instances[key]


def _candidates_of(span, top_n):
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


def link_mentions(text, entities, model_name="wikipedia_model",
                  entity_set="wikipedia", top_n=10):
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
    length as the input.
    """
    if not entities:
        return []

    spans = [
        Span(text=text[ent["start"]:ent["end"]],
             start=ent["start"],
             ln=ent["end"] - ent["start"])
        for ent in entities
    ]

    linked = _load(model_name, entity_set).process_text(text, spans=spans)

    by_start = {}
    for s in linked:
        pred = getattr(s, "predicted_entity", None)
        qid = getattr(pred, "wikidata_entity_id", None) if pred else None
        conf = getattr(s, "entity_linking_model_confidence_score", None)
        by_start[s.start] = {
            "qid": qid,
            "confidence": float(conf) if conf is not None else None,
            "candidates": [{"qid": q, "score": sc} for q, sc in _candidates_of(s, top_n)],
        }

    empty = {"qid": None, "confidence": None, "candidates": []}
    return [(ent, by_start.get(sp.start, dict(empty)))
            for ent, sp in zip(entities, spans)]
