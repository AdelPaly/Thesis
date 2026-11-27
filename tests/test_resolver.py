"""Unit tests for src/resolver.py: hand-made answers resolved against a small
copy of Wikidata.

    python -m pytest tests/test_resolver.py

No network and no dataset. The search results below were copied from Wikidata
on 2026-11-21 (item, label, coordinates, country, in the order the search
returned them), so each case is a real ambiguity. The answers are written by
hand, not taken from model output or from the evaluation posts. Springfield and
Singapore do not occur in the dataset as place names; Worcester, Malé, San
José, Perth and Tbilisi do.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import resolver  # noqa: E402

US, UK, ZA, MV, IT, CR, PH, GE, SG, FR = ("Q30", "Q145", "Q258", "Q826", "Q38", "Q800",
                                          "Q928", "Q230", "Q334", "Q142")

# qid: (English label, latitude, longitude, country item); None = no coordinates
ITEMS = {
    # countries and the like
    US: ("United States", 39.83, -98.58, US), UK: ("United Kingdom", 54.6, -2.0, UK),
    ZA: ("South Africa", -29.0, 24.0, ZA), MV: ("Maldives", 4.18, 73.51, MV),
    CR: ("Costa Rica", 10.0, -84.0, CR), GE: ("Georgia", 42.0, 44.0, GE),
    SG: ("Singapore", 1.3, 103.8, SG), FR: ("France", 46.0, 2.0, FR),
    IT: ("Italy", 42.5, 12.5, IT), PH: ("Philippines", 12.0, 123.0, PH),
    "Q21": ("England", 53.0, -1.0, UK),
    "Q1428": ("Georgia", 33.0, -83.5, US),               # the US state
    # Worcester
    "Q49179": ("Worcester", 42.27, -71.80, US),          # Massachusetts
    "Q1646181": ("Worcester", 52.19, -2.22, UK),         # England
    "Q16916691": ("Worcester", None, None, None),        # family name
    "Q67575442": ("Worcestershire", 52.24, -2.16, UK),
    "Q847508": ("Worcester Art Museum", 42.27, -71.80, US),
    "Q1025649": ("Worcester", -33.65, 19.44, ZA),        # Western Cape
    # Malé
    "Q6581097": ("male", None, None, None),
    "Q9347": ("Malé", 4.17, 73.51, MV),
    "Q295126": ("Malé", 46.35, 10.91, IT),
    "Q70222": ("Malestroit", 47.81, -2.38, FR),
    # Springfield
    "Q2369961": ("Springfield", 51.75, 0.51, UK),
    "Q28515": ("Springfield", 39.80, -89.64, US),        # Illinois
    "Q49158": ("Springfield", 42.11, -72.55, US),        # Massachusetts
    "Q135615": ("Springfield", 37.20, -93.29, US),       # Missouri
    # San José
    "Q16553": ("San Jose", 37.30, -121.87, US),
    "Q3070": ("San José", 9.93, -84.08, CR),
    "Q55599": ("San Jose", 15.79, 120.99, PH),
    # Perth
    "Q408": ("Australia", -25.0, 133.0, "Q408"),
    "Q3183": ("Perth", -31.96, 115.86, "Q408"),          # Western Australia
    "Q203000": ("Perth", 56.40, -3.43, UK),              # Scotland
    "Q1009433": ("Perth", -41.57, 147.17, "Q408"),       # Tasmania
    "Q68816359": ("Perth", 56.40, -3.43, UK),            # parish around the Scottish city
    "Q14229145": ("Perth", -31.95, 115.85, "Q408"),      # central suburb of the city
    # Tbilisi, Singapore
    "Q994": ("Tbilisi", 41.72, 44.79, GE),
    "Q1341835": ("Tbilisi Central Station", 41.72, 44.80, GE),
    "Q7522983": ("Singapore General Hospital", 1.28, 103.83, SG),
    "Q1054746": ("Singapore Island", 1.38, 103.80, SG),
}

SEARCHES = {
    "Worcester": ["Q49179", "Q1646181", "Q16916691", "Q67575442", "Q847508", "Q1025649"],
    "Malé": ["Q6581097", "Q9347", "Q295126", "Q70222"],
    "Springfield": ["Q2369961", "Q28515", "Q49158", "Q135615"],
    "San José": ["Q16553", "Q3070", "Q55599"],
    "Perth": ["Q3183", "Q203000", "Q1009433", "Q68816359", "Q14229145"],
    "Australia": ["Q408"],
    "Tbilisi": ["Q994", "Q1341835"],
    "Singapore": ["Q334", "Q7522983", "Q1054746"],
    "United States": [US], "United Kingdom": [UK], "South Africa": [ZA], "Maldives": [MV],
    "Costa Rica": [CR], "France": [FR], "Italy": [IT], "England": ["Q21"],
    "Georgia": ["Q1428", "Q230"],                         # the state comes first
}


def _entity(qid: str) -> dict:
    """The item in the shape the Wikidata API returns it."""
    label, lat, lon, country = ITEMS[qid]
    claims = {}
    if lat is not None:
        claims["P625"] = [{"rank": "normal", "mainsnak": {"datavalue": {
            "value": {"latitude": lat, "longitude": lon}}}}]
    if country:
        claims["P17"] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": country}}}}]
    return {"labels": {"en": {"value": label}}, "claims": claims}


def search(name, limit=10):
    return SEARCHES.get(name, [])[:limit]


def fetch(qids):
    return {q: _entity(q) for q in qids}


def geo(entities):
    return {q: (ITEMS[ITEMS[q][3]][0] if ITEMS[q][3] else "", "") for q in entities}


def v1(answer):
    return resolver.resolve_first_hit(answer, search=search, fetch=fetch, geo=geo)


def v2(answer):
    return resolver.resolve_by_country_and_coordinates(answer, search=search, fetch=fetch, geo=geo)


def answer(city=None, country=None, lat=None, lon=None, continent=None, refused=False):
    return json.dumps({"city": city, "country": country, "continent": continent,
                       "latitude": lat, "longitude": lon, "refused": refused})


def qid(result):
    return (result[0] or {}).get("qid")


# ------------------------------------------------- the error that prompted it

def test_v1_sends_worcester_south_africa_to_massachusetts():
    """The reported rule ignores the country. Kept as it was, on purpose."""
    cand, res = v1(answer("Worcester", "South Africa"))
    assert cand["qid"] == "Q49179" and cand["country"] == "United States"
    assert res == {"status": "resolved", "level": "city", "method": "first_hit"}


@pytest.mark.parametrize("country, lat, lon, expected", [
    ("South Africa", -33.6, 19.4, "Q1025649"),
    ("United Kingdom", 52.2, -2.2, "Q1646181"),
    ("United States", 42.3, -71.8, "Q49179"),
])
def test_v2_worcester_follows_the_country(country, lat, lon, expected):
    cand, res = v2(answer("Worcester", country, lat, lon))
    assert cand["qid"] == expected
    assert res["status"] == "resolved" and res["level"] == "city"


def test_v2_country_alone_is_enough_when_one_place_has_the_name():
    cand, res = v2(answer("Worcester", "South Africa"))
    assert cand["qid"] == "Q1025649" and res["method"] == "country"


def test_v2_prefers_the_exact_name_to_a_longer_one_nearby():
    """Coordinates on top of the Worcester Art Museum still give the city."""
    cand, _ = v2(answer("Worcester", "United States", 42.2731, -71.8019))
    assert cand["qid"] == "Q49179"


# ------------------------------------------------------------- other namesakes

def test_v2_male_is_the_capital_not_the_word():
    cand, res = v2(answer("Malé", "Maldives", 4.2, 73.5))
    assert cand["qid"] == "Q9347" and res["n_candidates"] == 3


def test_v2_male_in_italy():
    assert qid(v2(answer("Malé", "Italy", 46.35, 10.9))) == "Q295126"


def test_v1_male_finds_nothing_for_the_city_and_gives_the_country():
    """First result is the word "male", which has no coordinates."""
    cand, res = v1(answer("Malé", "Maldives"))
    assert cand["qid"] == MV
    assert res["status"] == "not_found" and res["level"] == "country"


@pytest.mark.parametrize("lat, lon, expected", [
    (39.8, -89.6, "Q28515"),      # Illinois
    (42.1, -72.6, "Q49158"),      # Massachusetts
    (37.2, -93.3, "Q135615"),     # Missouri
])
def test_v2_springfield_needs_the_coordinates(lat, lon, expected):
    cand, res = v2(answer("Springfield", "United States", lat, lon))
    assert cand["qid"] == expected
    assert res["method"] == "country+coordinates" and res["n_in_country"] == 3


def test_v2_coordinates_far_from_every_candidate_are_not_trusted():
    """A point in Texas is 900 km from the nearest Springfield: the search
    order decides (Illinois), not the least distant one (Missouri)."""
    cand, res = v2(answer("Springfield", "United States", 31.0, -100.0))
    assert cand["qid"] == "Q28515" and res["method"] == "country+search_order"


def test_v2_springfield_without_coordinates_is_ambiguous_and_gives_the_country():
    cand, res = v2(answer("Springfield", "United States"))
    assert res["status"] == "ambiguous" and res["level"] == "country"
    assert cand["qid"] == US


def test_v2_san_jose_costa_rica_and_california():
    assert qid(v2(answer("San José", "Costa Rica", 9.9, -84.1))) == "Q3070"
    assert qid(v2(answer("San José", "United States", 37.3, -121.9))) == "Q16553"


@pytest.mark.parametrize("country, lat, lon, expected", [
    ("Australia", -31.95, 115.85, "Q3183"),     # exactly on the suburb: still the city
    ("Australia", -41.6, 147.2, "Q1009433"),    # Tasmania
    ("United Kingdom", 56.4, -3.4, "Q203000"),  # the city, not the parish around it
])
def test_v2_perth_city_before_the_suburb_inside_it(country, lat, lon, expected):
    assert qid(v2(answer("Perth", country, lat, lon))) == expected


# ------------------------------------------------- when the country is no help

def test_v2_country_wikidata_does_not_use_falls_back_on_coordinates():
    """Worcester's country is the United Kingdom in Wikidata, not England."""
    cand, res = v2(answer("Worcester", "England", 52.19, -2.22))
    assert cand["qid"] == "Q1646181" and res["method"] == "coordinates"


def test_v2_country_name_resolving_to_the_wrong_item():
    """"Georgia" finds the US state first; the coordinates still find Tbilisi."""
    cand, res = v2(answer("Tbilisi", "Georgia", 41.7, 44.8))
    assert cand["qid"] == "Q994" and res["method"] == "coordinates"


def test_v2_no_country_uses_coordinates():
    assert qid(v2(answer("Worcester", None, -33.65, 19.44))) == "Q1025649"


def test_v2_city_state_gives_the_country_item():
    cand, res = v2(answer("Singapore", "Singapore", 1.35, 103.82))
    assert cand["qid"] == SG and res["level"] == "country"


# ------------------------------------------------------------ failure reasons

@pytest.mark.parametrize("resolve", [v1, v2])
def test_declined(resolve):
    cand, res = resolve(answer(refused=True))
    assert cand is None and res["status"] == "declined" and res["level"] is None


@pytest.mark.parametrize("resolve", [v1, v2])
@pytest.mark.parametrize("reply", ["I think this is about Paris.", "", '["Paris"]', None,
                                   '{"city": "Paris", "country": '])
def test_unreadable(resolve, reply):
    cand, res = resolve(reply)
    assert cand is None and res["status"] == "unreadable"


@pytest.mark.parametrize("resolve", [v1, v2])
def test_not_found_with_nothing_to_fall_back_on(resolve):
    cand, res = resolve(answer("Xyzzyville"))
    assert cand is None and res["status"] == "not_found" and res["level"] is None


@pytest.mark.parametrize("resolve", [v1, v2])
def test_not_found_falls_back_to_the_country(resolve):
    cand, res = resolve(answer("Xyzzyville", "France", 48.0, 2.0))
    assert cand["qid"] == FR and res["status"] == "not_found" and res["level"] == "country"


def test_v2_namesake_far_from_the_coordinates_is_not_found():
    """No Worcester in France: the answer is France, not Massachusetts."""
    cand, res = v2(answer("Worcester", "France", 48.0, 2.0))
    assert cand["qid"] == FR
    assert res["status"] == "not_found" and res["level"] == "country" and res["km"] > 500


def test_v2_several_namesakes_and_nothing_to_choose_with():
    cand, res = v2(answer("Springfield"))
    assert cand is None and res["status"] == "ambiguous" and res["n_candidates"] == 4


def test_v2_one_candidate_and_nothing_else():
    cand, res = v2(answer("Malestroit"))            # not in SEARCHES: nothing found
    assert cand is None and res["status"] == "not_found"


@pytest.mark.parametrize("resolve", [v1, v2])
def test_continent_only_has_no_item_and_no_coordinates(resolve):
    cand, res = resolve(answer(continent="Africa"))
    assert cand["qid"] is None and cand["lat"] is None and cand["continent"] == "Africa"
    assert res["level"] == "continent"


@pytest.mark.parametrize("resolve", [v1, v2])
@pytest.mark.parametrize("placeholder", ["null", "None", "unknown", " N/A ", ""])
def test_placeholder_strings_are_not_names(resolve, placeholder):
    cand, res = resolve(answer(placeholder, placeholder, continent=placeholder))
    assert cand is None and res["status"] == "declined"


# ----------------------------------------------------------------- coordinates

@pytest.mark.parametrize("lat, lon", [(200, 19), (-33, 400), ("south", "east"), (None, 19.4),
                                      (True, False), (float("nan"), 1.0)])
def test_invalid_coordinates_are_ignored(lat, lon):
    assert resolver.clean_coordinates(lat, lon) is None


def test_coordinates_given_as_strings_are_read():
    assert resolver.clean_coordinates("-33.65", "19.44") == (-33.65, 19.44)
    assert qid(v2(json.dumps({"city": "Springfield", "country": "United States",
                              "latitude": "42.1", "longitude": "-72.6"}))) == "Q49158"


def test_bad_coordinates_leave_the_country_rule():
    cand, res = v2(answer("Worcester", "South Africa", 999, 999))
    assert cand["qid"] == "Q1025649" and "km" not in res


def test_distance():
    london, paris = (51.5074, -0.1278), (48.8566, 2.3522)
    assert 340 < resolver.km_between(london, paris) < 347
    assert resolver.km_between(london, london) == 0


def test_reply_wrapped_in_a_code_fence_is_read():
    reply = "```json\n" + answer("Worcester", "South Africa", -33.6, 19.4) + "\n```"
    assert qid(v2(reply)) == "Q1025649"


# ------------------------------------------------------- v3: the wider search

NZ, TT = "Q664", "Q754"
ITEMS.update({
    NZ: ("New Zealand", -41.0, 174.0, NZ), TT: ("Trinidad and Tobago", 10.7, -61.2, TT),
    "Q350": ("Cambridge", 52.21, 0.12, UK), "Q49111": ("Cambridge", 42.37, -71.11, US),
    "Q1028261": ("Cambridge", -37.88, 175.47, NZ),          # Waikato
    "Q99999001": ("Cambridge", -41.20, 175.10, NZ),         # a second one, made up for the tie
})
# ten namesakes outside New Zealand come first, as on Wikidata
for _i in range(8):
    ITEMS[f"Q9000{_i}"] = ("Cambridge", 40.0 + _i, -80.0 - _i, US)
SEARCHES.update({
    "Cambridge": ["Q350", "Q49111", *[f"Q9000{i}" for i in range(8)], "Q1028261", "Q99999001"],
    "New Zealand": [NZ], "Trinidad and Tobago": [TT],
})


def v3(answer):
    return resolver.resolve_wide(answer, search=search, fetch=fetch, geo=geo)


def test_v2_does_not_reach_the_eleventh_result():
    cand, res = v2(answer("Cambridge", "New Zealand", -37.9, 175.5))
    assert qid((cand, res)) == NZ and res["status"] == "not_found" and res["level"] == "country"


def test_v3_finds_the_place_beyond_the_first_ten():
    cand, res = v3(answer("Cambridge", "New Zealand", -37.9, 175.5))
    assert cand["qid"] == "Q1028261"
    assert res["status"] == "resolved" and res["level"] == "city" and res["n_searched"] == 50


def test_v3_wider_results_follow_the_same_rule():
    """Two in the country: the coordinates choose, as in v2."""
    assert qid(v3(answer("Cambridge", "New Zealand", -41.2, 175.1))) == "Q99999001"
    cand, res = v3(answer("Cambridge", "New Zealand"))
    assert res["status"] == "ambiguous" and cand["qid"] == NZ


def test_v3_does_not_search_again_when_the_first_ten_have_the_country():
    cand, res = v3(answer("Cambridge", "United Kingdom", 52.2, 0.1))
    assert cand["qid"] == "Q350" and "n_searched" not in res


def test_v3_wider_search_with_nothing_in_the_country_changes_nothing():
    """Not in the country even among fifty: the ten results decide, as in v2."""
    a = answer("Cambridge", "Trinidad and Tobago", 10.6, -61.4)
    assert v3(a) == v2(a)
    a = answer("Cambridge", "Trinidad and Tobago", 52.2, 0.1)      # coordinates point at England
    assert v3(a) == v2(a) and qid(v3(a)) == "Q350"


def test_v3_without_a_country_is_v2():
    a = answer("Cambridge", None, 42.4, -71.1)
    assert v3(a) == v2(a)


@pytest.mark.parametrize("a", [
    answer("Worcester", "South Africa", -33.6, 19.4), answer("Malé", "Maldives", 4.2, 73.5),
    answer("Springfield", "United States", 39.8, -89.6), answer("Perth", "Australia", -31.95, 115.86),
    answer("Tbilisi", "Georgia", 41.7, 44.8), answer("Singapore", "Singapore", 1.3, 103.8),
    answer("Nowhere", "France"), answer(None, "France"), answer(refused=True)])
def test_v3_agrees_with_v2_on_the_earlier_cases(a):
    assert v3(a) == v2(a)


# ---------------------- rule v4 (revision 3, 2026-11-26): v3 on the "place" field

def v4(answer, search=search, **kw):
    return resolver.resolve_place_wide(answer, search=search, fetch=fetch, geo=geo, **kw)


def place_answer(place=None, country=None, lat=None, lon=None, continent=None, refused=False):
    return json.dumps({"place": place, "country": country, "continent": continent,
                       "latitude": lat, "longitude": lon, "refused": refused})


def test_v4_reads_the_place_field():
    assert qid(v4(place_answer("Worcester", "South Africa", -33.6, 19.4))) == "Q1025649"


def test_v4_reads_a_city_field_when_there_is_no_place():
    assert qid(v4(answer("Worcester", "South Africa", -33.6, 19.4))) == "Q1025649"


def test_v4_prefers_place_to_city():
    reply = json.dumps({"place": "Tbilisi", "city": "Worcester", "country": "Georgia",
                        "latitude": 41.7, "longitude": 44.8})
    assert qid(v4(reply)) == "Q994"


def test_v2_and_v3_ignore_the_place_field():
    """v2 and v3 are kept as they ran: a v4 reply has no city, so they give the country."""
    for rule in (v2, v3):
        cand, res = rule(place_answer("Worcester", "South Africa", -33.6, 19.4))
        assert cand["qid"] == ZA and res["level"] == "country"


def test_v4_searches_past_the_tenth_result_as_v3_does():
    """Worcester, South Africa as the eleventh search result: v2 never sees it.
    None of the first ten is in South Africa, so v3 and v4 search again with
    fifty results and find it there."""
    padded = SEARCHES["Worcester"][:-1] + ["Q49179"] * 5 + ["Q1025649"]

    def long_search(name, limit=10):
        return (padded if name == "Worcester" else SEARCHES.get(name, []))[:limit]

    assert len(padded) > resolver.SEARCH_LIMIT
    assert qid(resolver.resolve_by_country_and_coordinates(
        answer("Worcester", "South Africa", -33.6, 19.4),
        search=long_search, fetch=fetch, geo=geo)) == ZA
    cand, res = v4(place_answer("Worcester", "South Africa", -33.6, 19.4), search=long_search)
    assert cand["qid"] == "Q1025649" and res["n_searched"] == resolver.WIDE_SEARCH_LIMIT


@pytest.mark.parametrize("a", [
    answer("Worcester", "South Africa", -33.6, 19.4), answer("Cambridge", "United Kingdom", 52.2, 0.1),
    answer("Cambridge", "Trinidad and Tobago", 10.6, -61.4), answer("Cambridge", None, 42.4, -71.1),
    answer("Tbilisi", "Georgia", 41.7, 44.8), answer("Nowhere", "France"), answer(None, "France")])
def test_v4_on_a_city_reply_is_v3(a):
    """With the table of country items left out (correction 2), v4 reads a
    city reply as v3 does; corrections 1, 3 and 4 do not apply to these
    replies. tests/test_resolver_outside.py tests the corrections."""
    assert v4(a, countries=lambda name: None) == v3(a)


@pytest.mark.parametrize("resolve", ["v1", "v2", "v3", "v4"])
def test_v4_declined_like_the_others(resolve):
    cand, res = resolver.RESOLVERS[resolve](place_answer(refused=True), search=search,
                                            fetch=fetch, geo=geo)
    assert cand is None and res["status"] == "declined"


# ------------ rule v5 (after the development run of 26-27 November): the settlement

class Searches:
    """`search` that records every name it is asked for, and fails for one."""

    def __init__(self, fail_on=None):
        self.names, self.fail_on = [], fail_on

    def __call__(self, name, limit=10):
        self.names.append(name)
        if name == self.fail_on:
            raise resolver.wd.WikidataError("HTTP 429 from Wikidata")
        return search(name, limit)


def v5(answer, search=search, **kw):
    return resolver.resolve_place_settlement(answer, search=search, fetch=fetch, geo=geo, **kw)


def settled_answer(place=None, settlement=None, country=None, lat=None, lon=None, continent=None):
    return json.dumps({"place": place, "settlement": settlement, "country": country,
                       "continent": continent, "latitude": lat, "longitude": lon,
                       "refused": False})


def test_v5_place_not_found_falls_back_to_the_settlement():
    a = settled_answer("Pervez Musharraf Street", "Worcester", "South Africa", -33.6, 19.4)
    assert v4(a)[1]["status"] == "not_found" and qid(v4(a)) == ZA
    cand, res = v5(a)
    assert cand["qid"] == "Q1025649" and cand["entity"] == "Worcester"
    assert res["status"] == "resolved" and res["field"] == "settlement"
    assert res["place_status"] == "not_found" and res["level"] == "city"


def test_v5_ambiguous_place_falls_back_to_the_settlement():
    a = settled_answer("Springfield", "Worcester", "United States")
    assert v4(a)[1]["status"] == "ambiguous"
    cand, res = v5(a)
    assert cand["qid"] == "Q49179"
    assert res["field"] == "settlement" and res["place_status"] == "ambiguous"


def test_v5_place_that_resolves_never_searches_the_settlement():
    searches = Searches()
    cand, res = v5(settled_answer("Tbilisi", "Worcester", "Georgia", 41.7, 44.8), search=searches)
    assert cand["qid"] == "Q994" and res["status"] == "resolved" and res["field"] == "place"
    assert "Worcester" not in searches.names


def test_v5_settlement_equal_to_the_place_is_not_searched_again():
    a = settled_answer("Xyzzyville", "xyzzyville!", "France", 48.0, 2.0)
    by_v4, by_v5 = Searches(), Searches()
    expected = v4(a, search=by_v4)
    assert v5(a, search=by_v5) == expected and by_v5.names == by_v4.names
    assert "xyzzyville!" not in by_v5.names


def test_v5_settlement_not_found_either_gives_what_the_place_gave():
    a = settled_answer("Pervez Musharraf Street", "Xyzzyville", "France", 48.0, 2.0)
    searches = Searches()
    assert v5(a, search=searches) == v4(a)
    assert "Xyzzyville" in searches.names
    assert qid(v5(a)) == FR and v5(a)[1]["status"] == "not_found"


def test_v5_without_a_settlement_is_v4():
    a = settled_answer("Pervez Musharraf Street", None, "France", 48.0, 2.0)
    assert v5(a) == v4(a)


@pytest.mark.parametrize("fail_on", ["Pervez Musharraf Street", "Worcester"])
def test_v5_error_in_either_search_ends_in_error(fail_on):
    a = settled_answer("Pervez Musharraf Street", "Worcester", "South Africa", -33.6, 19.4)
    cand, res = v5(a, search=Searches(fail_on=fail_on))
    assert cand is None and res["status"] == "error" and "429" in res["message"]


@pytest.mark.parametrize("a", [
    place_answer("Worcester", "South Africa", -33.6, 19.4), place_answer("Cambridge", "New Zealand", -37.9, 175.5),
    place_answer("Springfield", "United States"), place_answer("Nowhere", "France"),
    place_answer(None, "France"), place_answer(continent="Africa"), place_answer(refused=True),
    answer("Tbilisi", "Georgia", 41.7, 44.8), "not json"])
def test_v5_on_a_v4_reply_is_v4(a):
    """A v4 reply has no settlement: v5 gives v4's answer and only adds
    `field` "place" when a named place resolved."""
    cand4, res4 = v4(a)
    cand5, res5 = v5(a)
    assert cand5 == cand4
    assert {k: v for k, v in res5.items() if k != "field"} == res4
    parsed = resolver.parse_json_object(a) or {}
    named = any(resolver.clean_name(parsed.get(f)) for f in ("place", "city"))
    assert res5.get("field") == ("place" if res4["status"] == "resolved" and named else None)