"""Flair NER wrapper.

Returns one dict per entity carrying the surface form, the tag, the tagger's
confidence and the *character offsets* Flair already computed. The offsets are
the contract that lets downstream linkers address a mention exactly instead of
searching for its string again.
"""
from flair.data import Sentence
from flair.nn import Classifier

_tagger = None
_tagger_name = None


def _load(model):
    """Load (once per process) the Flair tagger named in the config.

    Flair's own shorthand keys ("ner-large") and the HuggingFace-style names
    people write in configs ("flair/ner-large") are both accepted; the second
    is not a real ModelHub id, so fall back to the shorthand.
    """
    global _tagger, _tagger_name
    if _tagger is None or _tagger_name != model:
        try:
            _tagger = Classifier.load(model)
        except (ValueError, OSError):
            if "/" not in model:
                raise
            _tagger = Classifier.load(model.split("/", 1)[1])
        _tagger_name = model
    return _tagger


def extract_entities(text, model="flair/ner-large", entity_types=None):
    """Run NER over `text`.

    Returns [{text, type, score, start, end}, ...] where start/end are
    character offsets into `text` taken straight from Flair's data point, so
    text[start:end] is the mention. `entity_types` filters the tag set;
    None keeps every tag.
    """
    sentence = Sentence(text)
    _load(model).predict(sentence)

    entities = []
    for entity in sentence.get_labels('ner'):
        dp = entity.data_point
        entities.append({
            "text": dp.text,
            "type": entity.value,
            "score": entity.score,
            "start": dp.start_position,
            "end": dp.end_position,
        })
    if entity_types is not None:
        allowed = set(entity_types)
        entities = [e for e in entities if e["type"] in allowed]
    return entities
