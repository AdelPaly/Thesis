"""Semantic reranker: cosine similarity between the source text and each
candidate's Wikidata description.

The model identity comes from the configuration (`rerank.model`), not from a
module constant. Loaded models are cached per name so a run that sticks to one
configuration pays the load cost once.
"""
from sentence_transformers import SentenceTransformer, util

_models = {}


def _load(model_name):
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name)
    return _models[model_name]


def rank_candidates(text, candidates, model="intfloat/multilingual-e5-base"):
    """Score candidates by cosine similarity between source text and Wikidata
    descriptions; return the list sorted highest-first, each candidate carrying
    its `score`."""
    if not candidates:
        return candidates

    model = _load(model)
    text_embedding = model.encode(text)
    descriptions = [c["description"] for c in candidates]
    candidate_embeddings = model.encode(descriptions, batch_size=len(descriptions))

    scores = util.cos_sim(text_embedding, candidate_embeddings)[0]
    for i, candidate in enumerate(candidates):
        candidate["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
