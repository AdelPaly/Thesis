"""Turn an LLM's free-text answer into a Wikidata item, and say what happened.

An LLM-only configuration gets text back from the model:

    {"city": "Worcester", "country": "South Africa", "latitude": -33.6, "longitude": 19.4}

and has to find the Wikidata item for it. There are two rules, chosen by the
prompt template of the configuration:

  v1  `resolve_first_hit`. Search the city name, take the first result. The
      country is used only when the city finds nothing. This was the rule of the
      first runs of gemma4 and qwen3_vl (results/posthoc/original/) and it is
      kept so that they can be reproduced; it sends "Worcester, South Africa"
      to the city in Massachusetts.

  v2  `resolve_by_country_and_coordinates`. Search the city name (ten results),
      keep the results in the country the model named, and among several take
      the first within 25 km of the coordinates the model gave. See the
      function for the fallbacks. This was the rule of gemma4 and qwen3_vl
      until 2026-11-23 (results/posthoc/before_audit/).

  v3  v2 with a wider search. When the model named a country and none of the
      ten results lies in it, the name is searched again with fifty results
      before the rule goes on: "Cambridge, New Zealand" and "San Fernando,
      Trinidad and Tobago" are not among the first ten items of that name.
      Same prompt as v2. This is the rule of the reported gemma4 and qwen3_vl.

All return `(candidate or None, resolution)`. `resolution` is stored on the
prediction record, so a missing or coarse answer has a reason:

  status   resolved    the named place was matched to one Wikidata item
           declined    the model said the text has no location
           unreadable  the reply was not a JSON object
           not_found   the model named a place and the search found no item
                       for it (v2: none in the named country or near the
                       coordinates either)
           ambiguous   several items carry the name and the answer has nothing
                       that chooses between them (v2 only; v1 never knows)
  level    city | country | continent | None: what the prediction is. A
           not_found or ambiguous city still gives the country when the model
           named one, so `status` describes the named place and `level` the
           answer.
  method   how the item was chosen (first_hit, country, country+coordinates,
           country+search_order, coordinates, only_candidate).
  n_candidates, n_in_country, km: the numbers behind the choice; n_searched
           (v3) when the choice came from the wider search.

The Wikidata calls are parameters so the rule can be tested without a network
(tests/test_resolver.py).
"""
from __future__ import annotations

import math
import unicodedata
from typing import Callable

import wikidata as wd
from llm_common import parse_json_object

SEARCH_LIMIT = 10
# v3: results asked for when none of the first SEARCH_LIMIT is in the country
# the model named. 50 is the most one Wikidata search request returns.
WIDE_SEARCH_LIMIT = 50
# Farther than this from the model's coordinates, an item that is not in the
# named country is a namesake somewhere else, not the place the model meant.
# Fixed in advance (roughly the width of a mid-sized country), not tuned.
MAX_KM_WITHOUT_COUNTRY = 500.0

# Same-named items this close to the model's coordinates are not told apart by
# distance: a model's coordinates for a city are approximate, and the city, its
# district and its central suburb are often separate items a few km apart.
# The same limit decides whether the coordinates point at any candidate at all
# (see rule 4); that use was checked on the development split.
TIE_KM = 25.0

STATUSES = ("resolved", "declined", "unreadable", "not_found", "ambiguous")

# Models are asked for JSON null when a field is unknown, and mostly comply --
# but a few responses put the *string* "null" (or "none", "unknown") in the
# field instead. Searching Wikidata for the literal word "null" would produce a
# prediction that is never correct yet counts as one.
_PLACEHOLDER_NAMES = {"", "null", "none", "nil", "n/a", "na", "unknown",
                      "not specified", "unspecified", "no city", "no country"}


def clean_name(value) -> str | None:
    """A place name, or None for anything that is not one."""
    if not isinstance(value, str):
        return None
    name = value.strip()
    return None if name.lower() in _PLACEHOLDER_NAMES else name


def clean_coordinates(lat, lon) -> tuple[float, float] | None:
    """(lat, lon) in decimal degrees, or None unless both are valid numbers."""
    try:
        if isinstance(lat, bool) or isinstance(lon, bool) or lat is None or lon is None:
            return None
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        return None
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return None
    return lat, lon


def km_between(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance. Used only to choose between same-named items."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(h)))


def _resolution(status: str, level: str | None = None, method: str | None = None,
                **numbers) -> dict:
    assert status in STATUSES
    return {"status": status, "level": level, "method": method,
            **{k: v for k, v in numbers.items() if v is not None}}


def _candidate(name: str, qid: str, ent: dict, geo: Callable) -> dict:
    lat, lon = wd.get_coords_of(ent)
    country, continent = geo({qid: ent}).get(qid, ("", ""))
    return {"entity": name, "entity_type": "LOC", "lat": lat, "lon": lon,
            "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
            "qid": qid, "country": country, "continent": continent}


def _continent_only(parsed: dict) -> dict | None:
    """A continent the model named, with no item and no coordinate: 0.0/0.0
    would put the prediction on Null Island."""
    continent = clean_name(parsed.get("continent"))
    if not continent:
        return None
    return {"entity": continent, "entity_type": "LOC", "lat": None, "lon": None,
            "description": continent, "qid": None, "country": None, "continent": continent}


def _first_hit(name: str, search: Callable, fetch: Callable, geo: Callable) -> dict | None:
    qids = search(name, limit=1)
    if not qids:
        return None
    ent = fetch(qids).get(qids[0], {})
    return _candidate(name, qids[0], ent, geo) if wd.get_coords_of(ent) else None


def _fallback(parsed: dict, status: str, search: Callable, fetch: Callable, geo: Callable,
              **numbers) -> tuple[dict | None, dict]:
    """The city could not be pinned down: answer with the country if the model
    named one, else the continent, else nothing. `status` keeps the reason."""
    country = clean_name(parsed.get("country"))
    cand = _first_hit(country, search, fetch, geo) if country else None
    if cand:
        return cand, _resolution(status, "country", "first_hit", **numbers)
    cand = _continent_only(parsed)
    return cand, _resolution(status, "continent" if cand else None, None, **numbers)


def _read(raw_or_parsed) -> tuple[dict | None, dict | None]:
    """(parsed answer, None), or (None, the resolution that ends here)."""
    parsed = raw_or_parsed if isinstance(raw_or_parsed, dict) else parse_json_object(raw_or_parsed)
    if parsed is None:
        return None, _resolution("unreadable")
    if parsed.get("refused"):
        return None, _resolution("declined")
    return parsed, None


def resolve_first_hit(raw_or_parsed, *, search: Callable = wd.search_entities,
                      fetch: Callable = wd.fetch_entities,
                      geo: Callable = wd.resolve_country_continent) -> tuple[dict | None, dict]:
    """Rule v1: the first search result for the city, else for the country,
    else the continent as a string. The prediction is what the first runs of
    gemma4 and qwen3_vl produced; only the resolution is new."""
    parsed, ended = _read(raw_or_parsed)
    if ended:
        return None, ended
    city, country = clean_name(parsed.get("city")), clean_name(parsed.get("country"))
    if city:
        cand = _first_hit(city, search, fetch, geo)
        if cand:
            return cand, _resolution("resolved", "city", "first_hit")
        return _fallback(parsed, "not_found", search, fetch, geo)
    if country:
        cand = _first_hit(country, search, fetch, geo)
        if cand:
            return cand, _resolution("resolved", "country", "first_hit")
    cand = _continent_only(parsed)
    if cand:
        return cand, _resolution("not_found" if country else "resolved", "continent")
    # Not refused, yet no usable name in any field.
    return None, _resolution("not_found" if country else "declined")


def resolve_by_country_and_coordinates(
        raw_or_parsed, *, search: Callable = wd.search_entities,
        fetch: Callable = wd.fetch_entities,
        geo: Callable = wd.resolve_country_continent,
        wide_limit: int | None = None) -> tuple[dict | None, dict]:
    """Rule v2 (and v3 when `wide_limit` is given, see resolve_wide). For the
    city the model named:

      1. search Wikidata, ten results; keep those with coordinates;
      2. if the model named a country, keep the results in it (P17 is the item
         the country name resolves to, or the result is that country);
      3. of those, keep the ones whose label is exactly the name, if any (the
         search also returns "Worcester County" for "Worcester");
      4. one left: take it. Several left: take the first in search order of
         those within TIE_KM of the model's coordinates; if none is that near,
         the coordinates are not trusted and the first in search order is
         taken. `ambiguous` if the model gave no coordinates;
      5. none in the country, or no country named: take the result nearest the
         coordinates if it is within MAX_KM_WITHOUT_COUNTRY, which covers a
         country named in a way Wikidata does not use for P17 ("England").
         With neither a country nor coordinates, a single result is taken and
         several are `ambiguous`;
      6. otherwise `not_found`.

    A not_found or ambiguous city falls back to the country, then the
    continent, as in v1."""
    parsed, ended = _read(raw_or_parsed)
    if ended:
        return None, ended
    city, country = clean_name(parsed.get("city")), clean_name(parsed.get("country"))
    point = clean_coordinates(parsed.get("latitude"), parsed.get("longitude"))

    if not city:
        cand = _first_hit(country, search, fetch, geo) if country else None
        if cand:
            return cand, _resolution("resolved", "country", "first_hit")
        cand = _continent_only(parsed)
        if cand:
            return cand, _resolution("not_found" if country else "resolved", "continent")
        return None, _resolution("not_found" if country else "declined")

    country_qid = None
    if country:
        country_hits = search(country, limit=1)
        country_qid = country_hits[0] if country_hits else None

    def placed(limit):
        hits = search(city, limit=limit)
        ents = fetch(hits) if hits else {}
        found = [(q, ents[q]) for q in hits if ents.get(q) and wd.get_coords_of(ents[q])]
        return found, [(q, e) for q, e in found
                       if country_qid and (q == country_qid or wd._claim_id(e, "P17") == country_qid)]

    places, in_country = placed(SEARCH_LIMIT)
    searched = None
    if wide_limit and country_qid and not in_country:
        wide_places, wide_in_country = placed(wide_limit)
        if wide_in_country:
            places, in_country, searched = wide_places, wide_in_country, wide_limit
    if not places:
        return _fallback(parsed, "not_found", search, fetch, geo, n_candidates=0)
    numbers = {"n_candidates": len(places),
               "n_in_country": len(in_country) if country_qid else None,
               "n_searched": searched}

    def nearest(pool):
        """(km, qid, entity) of the item the coordinates point at. Items within
        TIE_KM of the point count as equally near and the search order decides,
        so a city is taken before the district or suburb of the same name
        whose coordinates lie inside it."""
        measured = [(km_between(point, wd.get_coords_of(e)), q, e) for q, e in pool]
        tied = [m for m in measured if m[0] <= TIE_KM]
        return tied[0] if tied else min(measured, key=lambda t: t[0])

    if in_country:
        pool = _same_name(city, in_country)
        if len(pool) == 1:
            q, e = pool[0]
            km = km_between(point, wd.get_coords_of(e)) if point else None
            return _candidate(city, q, e, geo), _resolution(
                "resolved", _level(q, country_qid), "country", km=_round(km), **numbers)
        if point:
            # Coordinates far from every candidate are a bad guess, not a
            # pointer to one of them: the search order then decides. On the
            # development split this was right more often than the nearest
            # item (Gemma 4: 64 of 96 such posts against 38; Qwen3-VL: 16 of
            # 40 against 13), at 25 km and at every wider limit tried.
            km, q, e = nearest(pool)
            method = "country+coordinates"
            if km > TIE_KM:
                q, e = pool[0]
                km, method = km_between(point, wd.get_coords_of(e)), "country+search_order"
            return _candidate(city, q, e, geo), _resolution(
                "resolved", _level(q, country_qid), method, km=_round(km), **numbers)
        return _fallback(parsed, "ambiguous", search, fetch, geo, **numbers)

    # Nothing in the named country, or no usable country.
    pool = _same_name(city, places)
    if point:
        km, q, e = nearest(pool)
        if km <= MAX_KM_WITHOUT_COUNTRY:
            return _candidate(city, q, e, geo), _resolution(
                "resolved", "city", "coordinates", km=_round(km), **numbers)
        return _fallback(parsed, "not_found", search, fetch, geo, km=_round(km), **numbers)
    if not country_qid:
        if len(pool) == 1:
            q, e = pool[0]
            return _candidate(city, q, e, geo), _resolution(
                "resolved", "city", "only_candidate", **numbers)
        return _fallback(parsed, "ambiguous", search, fetch, geo, **numbers)
    return _fallback(parsed, "not_found", search, fetch, geo, **numbers)


def resolve_wide(raw_or_parsed, *, search: Callable = wd.search_entities,
                 fetch: Callable = wd.fetch_entities,
                 geo: Callable = wd.resolve_country_continent) -> tuple[dict | None, dict]:
    """Rule v3: v2, and when none of the ten results is in the country the
    model named, the same rule over fifty results. The wider results are used
    only if one of them is in that country, so every answer v2 resolved in
    the named country is unchanged. `n_searched` on the resolution says when
    the wider search decided."""
    return resolve_by_country_and_coordinates(raw_or_parsed, search=search, fetch=fetch, geo=geo,
                                              wide_limit=WIDE_SEARCH_LIMIT)


def _level(qid: str, country_qid: str | None) -> str:
    return "country" if qid == country_qid else "city"


def _plain(name: str) -> str:
    """A name without case, accents or punctuation, for comparing labels."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if c.isalnum()).casefold()


def _same_name(name: str, pool: list) -> list:
    """The items whose English label is the name itself, if there are any.
    The search matches prefixes, so "Worcester" also returns Worcester County
    and Worcestershire; those are considered only when nothing is called
    exactly "Worcester"."""
    exact = [(q, e) for q, e in pool
             if _plain(e.get("labels", {}).get("en", {}).get("value", "")) == _plain(name)]
    return exact or pool


def _round(km: float | None) -> float | None:
    return None if km is None else round(km, 1)


RESOLVERS = {"v1": resolve_first_hit, "v2": resolve_by_country_and_coordinates, "v3": resolve_wide}
