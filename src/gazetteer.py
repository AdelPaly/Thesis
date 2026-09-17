"""Gazetteer lookups: candidate places for a surface form, with population.

Two backends, named by configuration (`baseline.gazetteer`,
`routing.gazetteer_*`):

  wikidata -- goes through the existing rate-limited wbsearchentities /
              wbgetentities client and the shared src/.wikidata_cache. Returns
              Wikidata QIDs, so a prediction is comparable to a gold QID.
              Population is P1082.

  geonames -- the GeoNames search API. Requires GEONAMES_USERNAME in the
              environment (free registration; 1000 credits/hour). Returns
              GeoNames ids, NOT QIDs, so exact-QID match is not computable for
              predictions made through this backend -- they score on distance,
              country and continent only. Unexercised in this repo: no
              GEONAMES_USERNAME is configured, so it has never run here.

Both return the same candidate shape, highest population first:

    {"qid", "geonames_id", "name", "lat", "lon", "population",
     "country", "continent", "description"}
"""
from __future__ import annotations

import os

import requests

import wikidata as wd

GEONAMES_API = "http://api.geonames.org/searchJSON"
GEONAMES_UA = {"User-Agent": "GeoThesisBot/1.0 (student research)"}

# GeoNames reports a continent code; the datasets use names.
_CONTINENT_BY_CODE = {
    "AF": "Africa", "AS": "Asia", "EU": "Europe", "NA": "North America",
    "OC": "Oceania", "SA": "South America", "AN": "Antarctica",
}


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
            "geonames_id": None,
            "name": ent.get("labels", {}).get("en", {}).get("value", name),
            "lat": lat, "lon": lon,
            "population": _population(ent),
            "country": country, "continent": continent,
            "description": ent.get("descriptions", {}).get("en", {}).get("value", name),
            "sitelinks": len(ent.get("sitelinks", {})),
        })
    return out


def _geonames_candidates(name, top_n):
    username = os.environ.get("GEONAMES_USERNAME")
    if not username:
        raise GazetteerError(
            "baseline.gazetteer is 'geonames' but GEONAMES_USERNAME is not set. "
            "Register a free GeoNames account and put GEONAMES_USERNAME in "
            "src/.env, or set baseline.gazetteer to 'wikidata'.")
    resp = requests.get(GEONAMES_API, headers=GEONAMES_UA, timeout=30, params={
        "q": name, "maxRows": top_n, "featureClass": "P", "style": "FULL",
        "lang": "en", "username": username,
    })
    resp.raise_for_status()
    data = resp.json()
    if "status" in data:
        raise GazetteerError(f"GeoNames error: {data['status']}")
    out = []
    for hit in data.get("geonames", []):
        try:
            lat, lon = float(hit["lat"]), float(hit["lng"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "qid": None,
            "geonames_id": str(hit.get("geonameId")),
            "name": hit.get("name", name),
            "lat": lat, "lon": lon,
            "population": hit.get("population"),
            "country": hit.get("countryName", ""),
            "continent": _CONTINENT_BY_CODE.get(hit.get("continentCode", ""), ""),
            "description": hit.get("fcodeName", ""),
            "sitelinks": 0,
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
    elif backend == "geonames":
        cands = _geonames_candidates(name, top_n)
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
