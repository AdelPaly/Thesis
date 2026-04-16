"""ReFinED entity linker wrapper.

ReFinED (Amazon 2022) does joint NER + entity typing + disambiguation and links
directly to Wikidata QIDs, so there's no Wikipedia→QID hop. We still feed it
Flair-detected mentions (via the `spans` override) so NER stays consistent with
the original/hybrid/conditional modes.
"""
from refined.inference.processor import Refined
from refined.data_types.base_types import Span

_refined = None


def _load():
    global _refined
    if _refined is None:
        _refined = Refined.from_pretrained(
            model_name="wikipedia_model",
            entity_set="wikipedia",
        )
    return _refined


def link_mentions(text, entities):
    """For each Flair mention, return (entity_dict, qid). qid is None on failure."""
    if not entities:
        return []

    spans = []
    lower = text.lower()
    cursor = 0
    for ent in entities:
        m = ent["text"]
        idx = lower.find(m.lower(), cursor)
        if idx < 0:
            idx = lower.find(m.lower())
        if idx < 0:
            continue
        spans.append(Span(text=text[idx:idx + len(m)], start=idx, ln=len(m)))
        cursor = idx + len(m)

    if not spans:
        return [(ent, None) for ent in entities]

    linked = _load().process_text(text, spans=spans)

    qid_by_start = {}
    for s in linked:
        pred = getattr(s, "predicted_entity", None)
        qid = getattr(pred, "wikidata_entity_id", None) if pred else None
        qid_by_start[s.start] = qid

    out = []
    for ent, sp in zip(entities, spans):
        out.append((ent, qid_by_start.get(sp.start)))
    # Mentions dropped during span-building (not found in text) get None
    if len(out) < len(entities):
        seen = {id(e) for e, _ in out}
        for ent in entities:
            if id(ent) not in seen:
                out.append((ent, None))
    return out
