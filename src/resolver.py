"""Turn an LLM's free-text answer into a Wikidata item, and say what happened.

An LLM-only configuration gets text back from the model:

    {"city": "Worcester", "country": "South Africa", "latitude": -33.6, "longitude": 19.4}

and has to find the Wikidata item for it. There are five rules, chosen by the
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
      Same prompt as v2. This is the rule of the reported gemma4 and qwen3_vl
      runs (commit 0389a2d).

  v4  `resolve_place_wide`. Rule v3 on the "place" field of prompt v4, which
      asks for the place itself rather than a city (revision 3, 2026-11-26;
      see results/REVISIONS.md). The search is that of v3, with four
      corrections (see the function): a far single namesake triggers the
      wider search, the country name is matched against the country items,
      and failed requests and empty replies get statuses of their own. A
      reply with a "city" field and no "place" field is read the same way.

  v5  `resolve_place_settlement`. Rule v4 on "place", then, when the place is
      not_found or ambiguous, the same search on the "settlement" field of
      prompt v5 (revision 3, after the development run of 26-27 November).
      `field` on the resolution says which of the two resolved.

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
           empty       (v4, v5) the provider returned no text or blocked the
                       reply, or every field of the reply is null and
                       "refused" false; v1-v3 call this declined
           error       (v4, v5) a Wikidata request failed; `message` says
                       which. v1-v3 cannot tell this from not_found
  level    city | country | continent | None: what the prediction is. A
           not_found or ambiguous city still gives the country when the model
           named one, so `status` describes the named place and `level` the
           answer. In v4 and v5 "city" means the named place, whatever its
           type (a street, an island or a park as well as a town), and in v5
           also the settlement when that is what resolved.
  field    (v5) "place" or "settlement": which field of the reply resolved;
           with "settlement", `place_status` is what happened to the place.
  method   how the item was chosen (first_hit, country, country+coordinates,
           country+search_order, coordinates, only_candidate; v4 also
           country_item, the country answered from the table of country items).
  n_candidates, n_in_country, km: the numbers behind the choice; n_searched
           (v3) when the choice came from the wider search.

The Wikidata calls are parameters so the rule can be tested without a network
(tests/test_resolver.py).
"""
from __future__ import annotations

import functools
import json
import math
import unicodedata
from pathlib import Path
from typing import Callable

import wikidata as wd
from llm_common import EMPTY_REPLY_SENTINEL, REFUSAL_SENTINEL, parse_json_object

COUNTRY_ITEMS = Path(__file__).resolve().parent / "country_items.json"

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

STATUSES = ("resolved", "declined", "unreadable", "not_found", "ambiguous",
            "empty", "error")      # the last two: rules v4 and v5 only

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


@functools.lru_cache(maxsize=1)
def _country_table() -> tuple[dict, dict]:
    """({plain label: {qid}}, {plain alias: {qid}}) of src/country_items.json
    (scripts/build_country_items.py)."""
    labels: dict[str, set] = {}
    aliases: dict[str, set] = {}
    items = json.loads(COUNTRY_ITEMS.read_text(encoding="utf-8"))["items"]
    for qid, it in items.items():
        labels.setdefault(_plain(it["label"]), set()).add(qid)
        for alias in it["aliases"]:
            aliases.setdefault(_plain(alias), set()).add(qid)
    return labels, aliases


def country_item(name: str) -> str | None:
    """Rule v4: the Wikidata country item a country name refers to, from the
    English labels, then the aliases, of the country items; None when no item
    or more than one has the name ("The Netherlands" is an alias of two)."""
    labels, aliases = _country_table()
    key = _plain(name)
    for table in (labels, aliases):
        found = table.get(key, set())
        if len(found) == 1:
            return next(iter(found))
        if found:
            return None
    return None


def _country_candidate(country: str, country_qid: str | None, search: Callable,
                       fetch: Callable, geo: Callable) -> dict | None:
    """The country as an answer: the country item when the table named one
    (rule v4), else the first search hit for the name."""
    if country_qid:
        ent = fetch([country_qid]).get(country_qid, {})
        if wd.get_coords_of(ent):
            return _candidate(country, country_qid, ent, geo)
    return _first_hit(country, search, fetch, geo)


def _fallback(parsed: dict, status: str, search: Callable, fetch: Callable, geo: Callable,
              country_qid: str | None = None, **numbers) -> tuple[dict | None, dict]:
    """The city could not be pinned down: answer with the country if the model
    named one, else the continent, else nothing. `status` keeps the reason.
    `country_qid` is the country item of rule v4's table; v1-v3 never pass it."""
    country = clean_name(parsed.get("country"))
    cand = _country_candidate(country, country_qid, search, fetch, geo) if country else None
    if cand:
        return cand, _resolution(status, "country", "country_item" if country_qid and
                                 cand["qid"] == country_qid else "first_hit", **numbers)
    cand = _continent_only(parsed)
    return cand, _resolution(status, "continent" if cand else None, None, **numbers)


_ANSWER_FIELDS = ("place", "city", "country", "continent")


def _read(raw_or_parsed, v4: bool = False) -> tuple[dict | None, dict | None]:
    """(parsed answer, None), or (None, the resolution that ends here).

    Rule v4 also ends with `empty`: the provider returned no text or blocked
    the reply (EMPTY_REPLY_SENTINEL; REFUSAL_SENTINEL in records written before
    revision 3), or the reply is a JSON object with every field null and
    "refused" false. v1-v3 call these `declined`, as they ran."""
    if v4 and isinstance(raw_or_parsed, str) and raw_or_parsed.strip() in (
            REFUSAL_SENTINEL, EMPTY_REPLY_SENTINEL):
        return None, _resolution("empty")
    parsed = raw_or_parsed if isinstance(raw_or_parsed, dict) else parse_json_object(raw_or_parsed)
    if parsed is None:
        return None, _resolution("unreadable")
    if parsed.get("refused"):
        return None, _resolution("empty" if v4 and parsed.get("empty_reply") else "declined")
    if v4 and not any(clean_name(parsed.get(f)) for f in _ANSWER_FIELDS) \
            and not clean_coordinates(parsed.get("latitude"), parsed.get("longitude")):
        return None, _resolution("empty")
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
        wide_limit: int | None = None,
        name_fields: tuple[str, ...] = ("city",),
        v4: bool = False,
        countries: Callable = country_item) -> tuple[dict | None, dict]:
    """Rule v2 (and v3 when `wide_limit` is given, see resolve_wide; v4 when
    `name_fields` also starts with "place", see resolve_place_wide). For the
    place the model named (the first of `name_fields` that holds a name;
    called `city` below):

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
    continent, as in v1.

    `v4` turns on the corrections of rule v4 (see resolve_place_wide); v2 and
    v3 run without them, as they ran."""
    parsed, ended = _read(raw_or_parsed, v4)
    if ended:
        return None, ended
    city = next((n for n in (clean_name(parsed.get(f)) for f in name_fields) if n), None)
    country = clean_name(parsed.get("country"))
    point = clean_coordinates(parsed.get("latitude"), parsed.get("longitude"))
    # v4: the country item the name refers to, from the table of country items.
    table_qid = countries(country) if v4 and country else None

    def fallback(status, **numbers):
        return _fallback(parsed, status, search, fetch, geo, country_qid=table_qid, **numbers)

    if not city:
        cand = _country_candidate(country, table_qid, search, fetch, geo) if country else None
        if cand:
            method = "country_item" if table_qid and cand["qid"] == table_qid else "first_hit"
            return cand, _resolution("resolved", "country", method)
        cand = _continent_only(parsed)
        if cand:
            return cand, _resolution("not_found" if country else "resolved", "continent")
        return None, _resolution("not_found" if country else "declined")

    country_qid = table_qid
    if country and not country_qid:
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
        return fallback("not_found", n_candidates=0)
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
        if v4 and point and wide_limit and searched is None and not any(
                km_between(point, wd.get_coords_of(e)) <= TIE_KM for _, e in pool):
            # v4: no namesake in the country lies within TIE_KM of the
            # coordinates, whether the ten results hold one or several. Search
            # wider; an in-country namesake within TIE_KM is taken (the first
            # in search order). Only if there is none does the rule below
            # decide on the ten results: one is kept, of several the first in
            # search order.
            wide_places, wide_in_country = placed(wide_limit)
            near = [m for m in ((km_between(point, wd.get_coords_of(we)), wq, we)
                                for wq, we in _same_name(city, wide_in_country))
                    if m[0] <= TIE_KM]
            if near:
                km, q, e = near[0]
                wide_numbers = {"n_candidates": len(wide_places),
                                "n_in_country": len(wide_in_country),
                                "n_searched": wide_limit}
                return _candidate(city, q, e, geo), _resolution(
                    "resolved", _level(q, country_qid), "country+coordinates",
                    km=_round(km), **wide_numbers)
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
        return fallback("ambiguous", **numbers)

    # Nothing in the named country, or no usable country.
    pool = _same_name(city, places)
    if point:
        km, q, e = nearest(pool)
        if km <= MAX_KM_WITHOUT_COUNTRY:
            return _candidate(city, q, e, geo), _resolution(
                "resolved", "city", "coordinates", km=_round(km), **numbers)
        return fallback("not_found", km=_round(km), **numbers)
    if not country_qid:
        if len(pool) == 1:
            q, e = pool[0]
            return _candidate(city, q, e, geo), _resolution(
                "resolved", "city", "only_candidate", **numbers)
        return fallback("ambiguous", **numbers)
    return fallback("not_found", **numbers)


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


def resolve_place_wide(raw_or_parsed, *, search: Callable = wd.search_entities_checked,
                       fetch: Callable = wd.fetch_entities_checked,
                       geo: Callable = wd.resolve_country_continent_checked,
                       countries: Callable = country_item) -> tuple[dict | None, dict]:
    """Rule v4 (revision 3, 2026-11-26): rule v3 on the "place" field of
    prompt v4, falling back to "city", with four corrections.

    Prompts v1-v3 asked for a city, so for an island, a dam or a park the
    models named the nearest town (Visby for Gotland). Prompt v4 asks for the
    place itself, which need not be a settlement. The search is v3's (ten
    results, fifty when none of the ten is in the named country).

    The corrections (v1-v3 stay as they ran):

      1. No namesake in the named country within TIE_KM of the model's
         coordinates, whether the ten results hold one or several: the fifty
         results are searched too, and an in-country namesake within TIE_KM
         is taken (the first in search order). Only if there is none does v3's
         rule decide on the ten: one is kept, of several the first in search
         order (the choice made on the development split stays the final
         fallback). v3 took a single namesake whatever the distance ("Richmond,
         Canada" with coordinates in British Columbia gave Richmond, Quebec).
      2. The country name is matched against the Wikidata country items
         (labels, then aliases; src/country_items.json) before the first
         search hit is used: "Georgia" is the country, not the US state that
         the search returns first. The country fallback answers with that
         item too.
      3. A failed Wikidata request ends in status `error` (with `message`), not
         in `not_found`: the default lookups raise instead of returning
         nothing. Nothing about the failure is cached, and the pipeline puts
         the message on the record's `error`, so the runner retries the post.
      4. An empty or blocked reply ends in status `empty`, not `declined`, and
         so does a reply with every field null and "refused" false.
    """
    try:
        return resolve_by_country_and_coordinates(
            raw_or_parsed, search=search, fetch=fetch, geo=geo, wide_limit=WIDE_SEARCH_LIMIT,
            name_fields=("place", "city"), v4=True, countries=countries)
    except wd.WikidataError as e:
        return None, _resolution("error", message=str(e))


def resolve_place_settlement(raw_or_parsed, *, search: Callable = wd.search_entities_checked,
                             fetch: Callable = wd.fetch_entities_checked,
                             geo: Callable = wd.resolve_country_continent_checked,
                             countries: Callable = country_item) -> tuple[dict | None, dict]:
    """Rule v5 (revision 3, after the development run of 26-27 November): rule
    v4 on the "place" field, then on the "settlement" field of prompt v5 when
    the place is not pinned down.

    Prompt v4 asks for the location itself. On posts with clues only the
    models then named a street, a venue, a club or a person that often has no
    Wikidata item, and rule v4 fell straight back to the country. Prompt v5
    also asks for the settlement the location is in.

      1. Rule v4 on "place" (falling back to "city", as v4 does). When it
         resolves, the settlement is never searched; the resolution gets
         `field` "place".
      2. When the place is not_found or ambiguous and "settlement" holds a
         name that is not the place's (compared without case, accents or
         punctuation), the same search and choice run on the settlement, with
         the same country and coordinates. If it resolves, that is the answer:
         `field` "settlement", and `place_status` keeps what happened to the
         place.
      3. Otherwise the answer is that of step 1: the country, then the
         continent.

    A failed Wikidata request in either search ends the whole resolution in
    status `error`, as in v4."""
    parsed, ended = _read(raw_or_parsed, v4=True)
    if ended:
        return None, ended

    def v4_rule(answer):
        return resolve_by_country_and_coordinates(
            answer, search=search, fetch=fetch, geo=geo, wide_limit=WIDE_SEARCH_LIMIT,
            name_fields=("place", "city"), v4=True, countries=countries)

    place = next((n for n in (clean_name(parsed.get(f)) for f in ("place", "city")) if n), None)
    settlement = clean_name(parsed.get("settlement"))
    try:
        cand, res = v4_rule(parsed)
        if res["status"] == "resolved":
            return cand, ({**res, "field": "place"} if place else res)
        if res["status"] not in ("not_found", "ambiguous") or not settlement \
                or (place and _plain(settlement) == _plain(place)):
            return cand, res
        by_settlement = {k: v for k, v in parsed.items() if k not in ("place", "city")}
        s_cand, s_res = v4_rule({**by_settlement, "place": settlement})
    except wd.WikidataError as e:
        return None, _resolution("error", message=str(e))
    if s_res["status"] == "resolved":
        return s_cand, {**s_res, "field": "settlement", "place_status": res["status"]}
    return cand, res


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


RESOLVERS = {"v1": resolve_first_hit, "v2": resolve_by_country_and_coordinates, "v3": resolve_wide,
             "v4": resolve_place_wide, "v5": resolve_place_settlement}
