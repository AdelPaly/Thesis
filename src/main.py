from ner import extract_entities
from ranking import rank_candidates, rank_per_entity
from wikidata import get_all_coords, get_labels, resolve_qid_to_candidate

def geolocate(text):
    entities = extract_entities(text)

    candidates = get_all_coords(entities, top_n=1)

    if not candidates:
        return None

    candidates = rank_candidates(text, candidates)

    return candidates[0]


def geolocate_hybrid(text):
    """Hybrid pipeline: rank Wikidata candidates per entity mention before
    cross-entity selection, so popularity bias in wbsearchentities doesn't
    silently lock in the wrong referent."""
    entities = extract_entities(text)

    candidates = get_all_coords(entities)
    if not candidates:
        return None

    winners = rank_per_entity(text, candidates)
    return winners[0] if winners else None


def geolocate_conditional(text):
    """Conditional pipeline: for each entity mention, if all top-10 candidates
    resolve to the same country, treat the entity as geographically unambiguous
    and use the top-1 candidate directly. Otherwise (candidates span 2+
    countries) apply the hybrid per-entity sentence-transformer rerank. Final
    cross-entity ranking is done as in hybrid mode."""
    entities = extract_entities(text)
    candidates = get_all_coords(entities)
    if not candidates:
        return None

    by_entity = {}
    for c in candidates:
        by_entity.setdefault(c["entity"], []).append(c)

    winners = []
    to_rerank = []
    for entity_text, cands in by_entity.items():
        countries = {c["country"] for c in cands if c["country"]}
        if len(countries) <= 1:
            winners.append(cands[0])
        else:
            to_rerank.extend(cands)

    if to_rerank:
        winners.extend(rank_per_entity(text, to_rerank))

    if not winners:
        return None

    winners = rank_candidates(text, winners)
    return winners[0]


def geolocate_refined(text):
    """ReFinED pipeline: Flair NER for mention detection → ReFinED for entity
    disambiguation (links straight to Wikidata QIDs, no Wikipedia hop) → shared
    geographic property chain → sentence-transformer cross-entity ranking."""
    from refined_linker import link_mentions

    entities = [e for e in extract_entities(text) if e["type"] in ("LOC", "PER", "ORG")]
    if not entities:
        return None

    candidates = []
    for ent, qid in link_mentions(text, entities):
        if not qid:
            continue
        cand = resolve_qid_to_candidate(qid, ent["text"], ent["type"])
        if cand:
            candidates.append(cand)

    if not candidates:
        return None

    ranked = rank_candidates(text, candidates)
    return ranked[0]


def geolocate_coverage(text):
    """Coverage-gated blend: ReFinED first, original as fallback.

    Run ReFinED on every mention; for each mention it fails to link, fall back
    to original's Wikidata top-1 search. ReFinED stays in charge of
    disambiguation (where it's strongest); original only patches coverage
    gaps, so its popularity bias can't silently overrule a correct ReFinED
    link."""
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
        # ReFinED failed (no QID) or the QID didn't resolve to coords ->
        # fall back to original's top-1 Wikidata search for this mention.
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


def geolocate_gated(text):
    """Ambiguity-gated blend of original and ReFinED.

    For each mention, fetch original's top-10 Wikidata candidates. If they all
    resolve to one country, the name is unambiguous -> keep original's top-1
    (preserves original's strength on non-toponyms and common names). If they
    span 2+ countries, hand that mention to ReFinED (which uses text context to
    disambiguate). Candidates from the two paths are never mixed in the same
    rerank pool, so ReFinED's context-picked descriptions can't dominate
    original's popularity-picked ones."""
    from refined_linker import link_mentions

    entities = extract_entities(text)
    candidates = get_all_coords(entities)
    if not candidates:
        return None

    by_entity = {}
    for c in candidates:
        by_entity.setdefault(c["entity"], []).append(c)

    winners = []
    ambiguous_entities = []
    ent_by_text = {e["text"]: e for e in entities}

    for entity_text, cands in by_entity.items():
        countries = {c["country"] for c in cands if c["country"]}
        if len(countries) <= 1:
            winners.append(cands[0])
        elif entity_text in ent_by_text:
            ambiguous_entities.append(ent_by_text[entity_text])

    if ambiguous_entities:
        for ent, qid in link_mentions(text, ambiguous_entities):
            if not qid:
                continue
            cand = resolve_qid_to_candidate(qid, ent["text"], ent["type"])
            if cand:
                winners.append(cand)

    if not winners:
        return None

    winners = rank_candidates(text, winners)
    return winners[0]


def geolocate_routed(text):
    """Routed pipeline: Flair NER → per-mention linker chosen by entity type.
    LOC mentions go through ReFinED (better geographic disambiguation); ORG /
    PER / MISC go through the original Wikidata wbsearchentities top-1."""
    from refined_linker import link_mentions as refined_link
    from wikidata import _search_entities

    entities = [e for e in extract_entities(text) if e["type"] in ("LOC", "PER", "ORG", "MISC")]
    if not entities:
        return None

    loc_entities = [e for e in entities if e["type"] == "LOC"]
    other_entities = [e for e in entities if e["type"] != "LOC"]

    candidates = []

    # ORG/PER/MISC entities → original Wikidata search
    for ent in other_entities:
        hits = _search_entities(ent["text"], limit=1)
        if not hits:
            continue
        cand = resolve_qid_to_candidate(hits[0], ent["text"], ent["type"])
        if cand:
            candidates.append(cand)

    # LOC entities → ReFinED, but only if no ORG/PER anchor was found.
    # When an organisation or person is present, treat it as the location
    # anchor and skip incidental place-name mentions that would otherwise
    # dominate the cross-entity rerank.
    has_org_per_anchor = any(c["type"] in ("ORG", "PER") for c in candidates)
    if loc_entities and not has_org_per_anchor:
        for ent, qid in refined_link(text, loc_entities):
            if not qid:
                continue
            cand = resolve_qid_to_candidate(qid, ent["text"], ent["type"])
            if cand:
                candidates.append(cand)

    if not candidates:
        return None

    ranked = rank_candidates(text, candidates)
    return ranked[0]


if __name__ == "__main__":
    text = "The Eiffel Tower was evacuated after a bomb threat."
    print("ORIGINAL:   ", geolocate(text))
    print("HYBRID:     ", geolocate_hybrid(text))
    print("CONDITIONAL:", geolocate_conditional(text))
    print("REFINED:    ", geolocate_refined(text))
    print("ROUTED:     ", geolocate_routed(text))
    print("GATED:      ", geolocate_gated(text))
    print("COVERAGE:   ", geolocate_coverage(text))
