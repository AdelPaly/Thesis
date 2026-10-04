"""Flair NER wrapper.

Returns one dict per entity carrying the surface form, the tag, the tagger's
confidence and the *character offsets* Flair already computed. The offsets are
the contract that lets downstream linkers address a mention exactly instead of
searching for its string again.
"""
from __future__ import annotations

from functools import lru_cache

from flair.data import Sentence
from flair.nn import Classifier

_HUB_PREFIX = "flair/"


def _hub_key(model: str) -> str:
    """Configs write `flair/ner-large`; Flair's ModelHub key is `ner-large`.
    Normalise here so the config string (and its hash) stays untouched."""
    return model[len(_HUB_PREFIX):] if model.startswith(_HUB_PREFIX) else model


@lru_cache(maxsize=None)
def load_tagger(model: str) -> Classifier:
    """Load (once per process and model name) the Flair tagger."""
    return Classifier.load(_hub_key(model))


def extract_entities(text: str, model: str,
                     entity_types: list[str] | None = None) -> list[dict]:
    """Run NER over `text`.

    Returns [{text, type, score, start, end}, ...] where start/end are
    character offsets into `text` taken straight from Flair's data point, so
    text[start:end] is the mention. `entity_types` filters the tag set;
    None keeps every tag.
    """
    sentence = Sentence(text)
    load_tagger(model).predict(sentence)

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
