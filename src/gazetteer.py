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
     "continent", "description", "sitelinks"}
"""
from __future__ import annotations

import wikidata as wd


class GazetteerError(ValueError):
    pass


def _population(entity: dict) -> int | None:
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


def _wikidata_candidates(name: str, top_n: int) -> list[dict]:
    qids = wd.search_entities(name, limit=top_n)
    if not qids:
        return []
    entities = wd.fetch_entities(qids)
    placed = {q: e for q, e in entities.items() if wd.get_coords_of(e)}
    if not placed:
        return []
    geo = wd.resolve_country_continent(placed)
    out = []
    for qid, ent in placed.items():
        lat, lon = wd.get_coords_of(ent)
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


def lookup(name: str, backend: str = "wikidata", top_n: int = 10,
           min_population: int = 0) -> list[dict]:
    """Candidate places for a surface form, most populous first.

    Candidates with no population at all sort last; a gazetteer entry with no
    population figure is not evidence of a large place. Ties (and the
    all-unpopulated case) break on sitelink count, which is the only
    prominence signal available for those rows.
    """
    if backend != "wikidata":
        raise GazetteerError(f"unknown gazetteer backend {backend!r}")
    cands = _wikidata_candidates(name, top_n)

    if min_population:
        cands = [c for c in cands
                 if (c["population"] or 0) >= min_population]
    cands.sort(key=lambda c: (c["population"] or -1, c["sitelinks"]), reverse=True)
    return cands

