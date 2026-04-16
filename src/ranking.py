from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('intfloat/multilingual-e5-base')

def rank_candidates(text, candidates):
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


def rank_per_entity(text, candidates):
    """Hybrid: score every candidate, then keep only the best per entity mention."""
    if not candidates:
        return candidates

    scored = rank_candidates(text, candidates)

    best_per_entity = {}
    for c in scored:
        key = c["entity"]
        if key not in best_per_entity or c["score"] > best_per_entity[key]["score"]:
            best_per_entity[key] = c

    winners = list(best_per_entity.values())
    winners.sort(key=lambda x: x["score"], reverse=True)
    return winners


#testing
if __name__ == "__main__":
    text = "The Eiffel Tower was evacuated after a bomb threat."

    candidates = [
        {"entity": "Paris", "entity_type": "LOC", "lat": 48.85, "lon": 2.35,
         "description": "capital and largest city of France"},
        {"entity": "Paris", "entity_type": "LOC", "lat": 33.66, "lon": -95.55,
         "description": "city in Texas, United States"},
    ]

    ranked = rank_candidates(text, candidates)
    for c in ranked:
        print(f"{c['description']} -> score: {c['score']:.4f}")