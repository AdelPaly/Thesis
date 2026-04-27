from sentence_transformers import SentenceTransformer, util

# Semantic re-ranker: cosine similarity between source text and Wikidata descriptions
model = SentenceTransformer('intfloat/multilingual-e5-base')


def rank_candidates(text, candidates):
    """Score candidates by cosine similarity between source text and
    Wikidata descriptions, return sorted highest-first."""
    if not candidates:
        return candidates

    text_embedding = model.encode(text)
    descriptions = [c["description"] for c in candidates]
    candidate_embeddings = model.encode(descriptions, batch_size=len(descriptions))

    scores = util.cos_sim(text_embedding, candidate_embeddings)[0]
    for i, candidate in enumerate(candidates):
        candidate["score"] = float(scores[i])

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates
