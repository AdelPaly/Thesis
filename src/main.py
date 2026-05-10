from ner import extract_entities
from ranking import rank_candidates
from wikidata import get_all_coords, resolve_qid_to_candidate


def geolocate(text):
    """two_stage: Flair NER -> Wikidata top-10 retrieval per entity ->
    E5 cross-candidate rerank -> top-1."""
    entities = extract_entities(text)
    candidates = get_all_coords(entities, top_n=10)
    if not candidates:
        return None
    candidates = rank_candidates(text, candidates)
    return candidates[0]


def geolocate_refined(text, _prelinked=None):
    """refined: Flair NER (LOC/PER/ORG only) -> ReFinED links each mention
    to a QID -> shared property chain -> E5 rerank -> top-1.

    _prelinked is an optional [(entity_dict, qid), ...] from a prior
    link_mentions call so callers (e.g. geolocate_refined_gemma) can reuse
    the linking work for routing-gate checks instead of running ReFinED
    twice on the same text."""
    if _prelinked is not None:
        linked = _prelinked
    else:
        from refined_linker import link_mentions
        entities = [e for e in extract_entities(text) if e["type"] in ("LOC", "PER", "ORG")]
        if not entities:
            return None
        linked = link_mentions(text, entities)

    candidates = []
    for ent, qid in linked:
        if not qid:
            continue
        cand = resolve_qid_to_candidate(qid, ent["text"], ent["type"])
        if cand:
            candidates.append(cand)

    if not candidates:
        return None

    ranked = rank_candidates(text, candidates)
    return ranked[0]


def geolocate_refined_search(text):
    """refined_search: per-mention blend. ReFinED links every Flair mention;
    for any mention ReFinED can't resolve to coordinates, fall back to a
    single wbsearchentities top-1 hit pushed through the same property chain.
    ReFinED owns disambiguation (where it's strongest); the search fallback
    only patches mentions ReFinED has no answer for."""
    from refined_linker import link_mentions
    from wikidata import _search_entities

    entities = [e for e in extract_entities(text) if e["type"] in ("LOC", "PER", "ORG")]
    if not entities:
        return None

    candidates = []
    for ent, qid in link_mentions(text, entities):
        if qid:
            cand = resolve_qid_to_candidate(qid, ent["text"], ent["type"])
            if cand:
                candidates.append(cand)
                continue
        hits = _search_entities(ent["text"], limit=1)
        if not hits:
            continue
        cand = resolve_qid_to_candidate(hits[0], ent["text"], ent["type"])
        if cand:
            candidates.append(cand)

    if not candidates:
        return None

    ranked = rank_candidates(text, candidates)
    return ranked[0]


def geolocate_refined_gemma(text):
    """refined_gemma: NER-routed combination, gated on ReFinED resolution.
    Flair LOC mentions are link-checked by ReFinED first; the entry takes
    the ReFinED branch only if at least one LOC resolves to a QID whose
    P31 (or one P279 hop above) is a real geographic-area type — city,
    town, country, admin region, etc. (see wikidata.GEO_AREA_P31). When
    Flair finds no LOC, when ReFinED can't link the LOC, or when the LOC
    resolves to a non-place (sports venue, generic 'old town', ocean,
    shopping center), the entry falls back to Gemma's whole-text reading
    where the org/person/contextual cues live."""
    from refined_linker import link_mentions
    from wikidata import is_geographic_qid

    entities = extract_entities(text)
    geo_entities = [e for e in entities if e["type"] in ("LOC", "PER", "ORG")]
    if geo_entities and any(e["type"] == "LOC" for e in geo_entities):
        linked = link_mentions(text, geo_entities)
        if any(is_geographic_qid(qid) for ent, qid in linked if ent["type"] == "LOC"):
            return geolocate_refined(text, _prelinked=linked)

    from llm_gemma import geolocate_gemma
    return geolocate_gemma(text)


def _refined_with_llm_fallback(text, llm_fn):
    """Same routing gate as geolocate_refined_gemma; LLM fn is parameterised
    so the same skeleton can host gemma, gpt-oss, or llama-4-scout."""
    from refined_linker import link_mentions
    from wikidata import is_geographic_qid

    entities = extract_entities(text)
    geo_entities = [e for e in entities if e["type"] in ("LOC", "PER", "ORG")]
    if geo_entities and any(e["type"] == "LOC" for e in geo_entities):
        linked = link_mentions(text, geo_entities)
        if any(is_geographic_qid(qid) for ent, qid in linked if ent["type"] == "LOC"):
            return geolocate_refined(text, _prelinked=linked)
    return llm_fn(text)


def geolocate_refined_groq_oss(text):
    """refined_groq_oss: refined_gemma's routing gate, but the LLM fallback
    is openai/gpt-oss-120b on Groq."""
    from llm_groq import geolocate_groq_oss
    return _refined_with_llm_fallback(text, geolocate_groq_oss)


def geolocate_refined_groq_llama(text):
    """refined_groq_llama: refined_gemma's routing gate, but the LLM fallback
    is meta-llama/llama-4-scout-17b-16e-instruct on Groq."""
    from llm_groq import geolocate_groq_llama
    return _refined_with_llm_fallback(text, geolocate_groq_llama)


if __name__ == "__main__":
    text = "The Eiffel Tower was evacuated after a bomb threat."
    print("two_stage          :", geolocate(text))
    print("refined            :", geolocate_refined(text))
    print("refined_search     :", geolocate_refined_search(text))
    print("refined_gemma      :", geolocate_refined_gemma(text))
    print("refined_groq_oss   :", geolocate_refined_groq_oss(text))
    print("refined_groq_llama :", geolocate_refined_groq_llama(text))
