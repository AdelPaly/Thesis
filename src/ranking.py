"""Semantic reranker: cosine similarity between the source text and each
candidate's Wikidata description.

The model identity comes from the configuration (`rerank.model`, normally the
shared encoder in configs/shared.yaml), not from a module constant. Loaded
encoders are cached per name, so a run pays the load cost once.

Input formatting: this reranker encodes the RAW source text and the RAW
description, with no prefixes. The dense-retrieval tier (src/dense_index.py)
uses the same encoder but wraps inputs in the encoder's query / passage
prefixes from configs/shared.yaml, when it has any. The two tiers therefore see the
same model with different input formatting and are not strictly like-for-like.
This is deliberate: adding prefixes here would change reranker results.

A candidate with no description: src/wikidata.py fills a missing English
description with the mention the candidate was found for. That string comes
from the source text, so its similarity to the text says nothing about the
place, and until 2026-11-23 such a candidate was ranked on it and often came
first. `has_description` tells the two apart, and candidates without a
description are now placed after those with one.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

MAX_ENCODE_BATCH_SIZE = 32


@lru_cache(maxsize=None)
def load_encoder(model_name: str) -> SentenceTransformer:
    """Load (once per process) the sentence encoder `model_name`, on the
    device sentence-transformers picks by default."""
    return SentenceTransformer(model_name)


def _plain(value) -> str:
    decomposed = unicodedata.normalize("NFKD", value if isinstance(value, str) else "")
    return "".join(c for c in decomposed if c.isalnum()).casefold()


def has_description(candidate: dict) -> bool:
    """False when the candidate's description is missing or is only the
    mention it was found for (the fill-in for an item with no description)."""
    description = _plain(candidate.get("description"))
    return bool(description) and description != _plain(candidate.get("entity"))


def rank_candidates(text: str, candidates: list[dict], model: str) -> list[dict]:
    """Score candidates by cosine similarity between source text and Wikidata
    descriptions.

    Returns a new list, each element a copy of the input candidate with a
    `score` added: the candidates that have a description first, highest score
    first, then those without one (see the module docstring), in score order
    too. The input list and dicts are not modified.
    """
    if not candidates:
        return []

    encoder = load_encoder(model)
    text_embedding = encoder.encode(text)
    descriptions = [c.get("description") or "" for c in candidates]
    candidate_embeddings = encoder.encode(
        descriptions, batch_size=min(len(descriptions), MAX_ENCODE_BATCH_SIZE))

    scores = util.cos_sim(text_embedding, candidate_embeddings)[0]
    scored = [{**c, "score": float(scores[i])} for i, c in enumerate(candidates)]
    scored.sort(key=lambda c: (has_description(c), c["score"]), reverse=True)
    return scored
