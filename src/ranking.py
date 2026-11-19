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
"""
from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer, util

MAX_ENCODE_BATCH_SIZE = 32


@lru_cache(maxsize=None)
def load_encoder(model_name: str) -> SentenceTransformer:
    """Load (once per process) the sentence encoder `model_name`, on the
    device sentence-transformers picks by default."""
    return SentenceTransformer(model_name)


def rank_candidates(text: str, candidates: list[dict], model: str) -> list[dict]:
    """Score candidates by cosine similarity between source text and Wikidata
    descriptions.

    Returns a new list sorted highest-first, each element a copy of the input
    candidate with a `score` added. The input list and dicts are not modified.
    """
    if not candidates:
        return []

    encoder = load_encoder(model)
    text_embedding = encoder.encode(text)
    descriptions = [c["description"] for c in candidates]
    candidate_embeddings = encoder.encode(
        descriptions, batch_size=min(len(descriptions), MAX_ENCODE_BATCH_SIZE))

    scores = util.cos_sim(text_embedding, candidate_embeddings)[0]
    scored = [{**c, "score": float(scores[i])} for i, c in enumerate(candidates)]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored
