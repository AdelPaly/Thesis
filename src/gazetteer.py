"""Gazetteer lookups: candidate places for a surface form, with population.

One backend, `wikidata`: the existing rate-limited wbsearchentities /
wbgetentities client and the shared src/.wikidata_cache. Candidates are
Wikidata QIDs, so a prediction is comparable to a gold QID, and population is
P1082.

A GeoNames backend previously sat behind a config switch and was never once
executed -- no GEONAMES_USERNAME is configured in this repo. It has been
removed rather than carried: untested code that has never run is worse than
absent code, and it returned GeoNames ids rather than QIDs, which would have
made exact-QID match uncomputable for anything predicted through it. If
GeoNames population data is wanted later, write it then and test it.

Candidates come back highest population first:

    {"qid", "name", "lat", "lon", "population", "country",
     "continent", "description"}
"""
from __future__ import annotations

import wikidata as wd


class GazetteerError(RuntimeError):
    pass


def _population(entity):
    """P1082 (population) as an int, or None. Prefers the highest-ranked
    non-deprecated claim; several claims means several census years."""
    best = None
    for claim in entity.get("claims", {}).get("P1082", []):
        if claim.get("rank") == "deprecated":
            continue
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not isinstance(val, dict):
            continue
        amount = val.get("amount")
        if amount is None:
            continue
        try:
            n = int(float(str(amount).lstrip("+")))
        except ValueError:
            continue
        if claim.get("rank") == "preferred":
            return n
        best = n if best is None else max(best, n)
    return best


def _wikidata_candidates(name, top_n):
    qids = wd._search_entities(name, limit=top_n)
    if not qids:
        return []
    entities = wd._fetch_entities(qids[:top_n])
    placed = {q: e for q, e in entities.items() if wd._get_coords(e)}
    if not placed:
        return []
    geo = wd._resolve_country_continent(placed)
    out = []
    for qid, ent in placed.items():
        lat, lon = wd._get_coords(ent)
        country, continent = geo.get(qid, ("", ""))
        out.append({
            "qid": qid,
            "name": ent.get("labels", {}).get("en", {}).get("value", name),
            "lat": lat, "lon": lon,
            "population": _population(ent),
            "country": country, "continent": continent,
            "description": ent.get("descriptions", {}).get("en", {}).get("value", name),
            "sitelinks": len(ent.get("sitelinks", {})),
        })
    return out


def lookup(name, backend="wikidata", top_n=10, min_population=0):
    """Candidate places for a surface form, most populous first.

    Candidates with no population at all sort last; a gazetteer entry with no
    population figure is not evidence of a large place. Ties (and the
    all-unpopulated case) break on sitelink count, which is the only
    prominence signal available for those rows.
    """
    if backend == "wikidata":
        cands = _wikidata_candidates(name, top_n)
    else:
        raise GazetteerError(f"unknown gazetteer backend {backend!r}")

    if min_population:
        cands = [c for c in cands
                 if (c["population"] or 0) >= min_population]
    cands.sort(key=lambda c: (c["population"] or -1, c["sitelinks"]), reverse=True)
    return cands


def count_candidates(name, backend="wikidata", top_n=10):
    """How many distinct places a surface form could denote -- the ambiguity
    signal the routing gate reads off the input. Counts gazetteer entries that
    have coordinates, so non-places in the search results do not inflate it."""
    return len(lookup(name, backend=backend, top_n=top_n))
