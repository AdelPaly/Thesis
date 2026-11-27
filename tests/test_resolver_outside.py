"""Rule v4 (src/resolver.py) on ambiguous place names that are not in the dataset.

    python -m pytest tests/test_resolver_outside.py
    GEO_LIVE_WIKIDATA=1 python -m pytest tests/test_resolver_outside.py -k live   # optional, network

tests/test_resolver.py uses places that are gold places or rivals in the
synthetic dataset (Worcester, Malé, Perth, ...). The cases here use three names
none of whose items occurs in it: Richmond, Toledo and Aurangabad.
`test_no_fixture_item_is_in_the_dataset` checks that and fails if that
changes. The replies are written by hand in the format of prompt v4 ("place",
"country", "continent", "latitude", "longitude", "refused"); none comes from a
model.

The search results and items in tests/fixtures/namesakes.json were copied from
Wikidata (date and method in the file), fifty results per place name, in the
order the search returned them. So each case is a real ambiguity, and the
eleventh-to-fiftieth results are the real ones. No network is used unless
GEO_LIVE_WIKIDATA is set.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import resolver  # noqa: E402

FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "namesakes.json").read_text(encoding="utf-8"))
ITEMS, SEARCHES = FIXTURE["items"], FIXTURE["searches"]
PLACE_NAMES = ("Richmond", "Toledo", "Aurangabad")
DATASET = ROOT / "generation" / "synthetic" / "output" / "synthetic_dataset.csv"


def _entity(qid: str) -> dict:
    """The item in the shape the Wikidata API returns it."""
    it = ITEMS[qid]
    claims = {}
    if it["lat"] is not None:
        claims["P625"] = [{"rank": "normal", "mainsnak": {"datavalue": {
            "value": {"latitude": it["lat"], "longitude": it["lon"]}}}}]
    if it["country"]:
        claims["P17"] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": it["country"]}}}}]
    return {"labels": {"en": {"value": it["label"]}},
            "descriptions": {"en": {"value": it["description"]}}, "claims": claims}


def search(name, limit=10):
    return SEARCHES.get(name, [])[:limit]


def fetch(qids):
    return {q: _entity(q) for q in qids if q in ITEMS}


def geo(entities):
    return {q: (ITEMS.get(ITEMS[q]["country"] or "", {}).get("label", ""), "") for q in entities}


def v4(reply, **kw):
    return resolver.resolve_place_wide(reply, **{"search": search, "fetch": fetch, "geo": geo, **kw})


def reply(place=None, country=None, lat=None, lon=None, continent=None, refused=False):
    """A reply in the JSON format prompt v4 asks for."""
    return json.dumps({"place": place, "country": country, "continent": continent,
                       "latitude": lat, "longitude": lon, "refused": refused})


# --------------------------------------------------------- outside the dataset

def _dataset_qids() -> set[str]:
    df = pd.read_csv(DATASET, keep_default_na=False, na_values=[""], dtype=str)
    out = set()
    for column in ("qid", "rival_qid", "noise_donor_qid"):
        out |= {q.strip() for q in df[column].dropna() if q.strip()}
    return out


@pytest.mark.skipif(not DATASET.exists(), reason="dataset not present")
def test_no_fixture_item_is_in_the_dataset():
    """No item of the fixture is a gold place, a rival or a noise donor of the
    dataset, and no place name of the cases is a gold location name. (The
    country items are in the fixture too; none is a gold or rival QID.)"""
    df = pd.read_csv(DATASET, keep_default_na=False, na_values=[""], dtype=str)
    in_dataset = sorted(set(ITEMS) & _dataset_qids())
    assert not in_dataset, f"fixture items that are in the dataset: {in_dataset}"
    names = {n.strip().casefold() for n in df["location_name"].dropna()}
    assert not [n for n in PLACE_NAMES if n.casefold() in names]


def test_fixture_is_the_real_ambiguity():
    """Guards the fixture: several same-named places per name, and the results
    the wider-search cases rely on really are past the tenth."""
    for name in PLACE_NAMES:
        same = [q for q in SEARCHES[name] if ITEMS[q]["label"] == name and ITEMS[q]["lat"] is not None]
        assert len(same) >= 5, name
    assert SEARCHES["Richmond"].index("Q985819") >= resolver.SEARCH_LIMIT      # South Africa
    assert SEARCHES["Richmond"].index("Q1208462") >= resolver.SEARCH_LIMIT     # New Zealand
    assert SEARCHES["Toledo"].index("Q2096039") >= resolver.SEARCH_LIMIT       # Uruguay


# ------------------------------------------------- the right namesake by country

@pytest.mark.parametrize("country, lat, lon, expected", [
    ("Spain", 39.86, -4.02, "Q5836"),            # Toledo, Castile-La Mancha
    ("Philippines", 10.38, 123.64, "Q316527"),   # Toledo, Cebu
    ("Brazil", None, None, "Q1776774"),          # Toledo, Paraná: the country alone is enough
    ("United States", 41.65, -83.54, "Q49239"),  # Toledo, Ohio
])
def test_namesake_by_country(country, lat, lon, expected):
    cand, res = v4(reply("Toledo", country, lat, lon))
    assert cand["qid"] == expected
    assert res["status"] == "resolved" and res["level"] == "city" and res["method"] == "country"
    assert "n_searched" not in res


# --------------------------------------------- the right namesake by coordinates

@pytest.mark.parametrize("name, country, lat, lon, expected", [
    ("Toledo", "Colombia", 7.31, -72.48, "Q1525656"),      # Norte de Santander
    ("Toledo", "Colombia", 7.01, -75.70, "Q2078498"),      # Antioquia
    ("Aurangabad", "India", 19.88, 75.32, "Q200713"),      # Maharashtra
    ("Aurangabad", "India", 24.75, 84.37, "Q590550"),      # Bihar
    ("Richmond", "United Kingdom", 54.40, -1.74, "Q1009324"),   # North Yorkshire
    ("Richmond", "United Kingdom", 51.46, -0.30, "Q2500974"),   # London
])
def test_namesake_by_coordinates_within_the_country(name, country, lat, lon, expected):
    cand, res = v4(reply(name, country, lat, lon))
    assert cand["qid"] == expected
    assert res["status"] == "resolved" and res["method"] == "country+coordinates"
    assert res["n_in_country"] >= 2 and res["km"] <= resolver.TIE_KM


def test_coordinates_alone_without_a_country():
    cand, res = v4(reply("Toledo", None, 41.65, -83.54))
    assert cand["qid"] == "Q49239" and res["method"] == "coordinates"


def test_country_wikidata_does_not_use_falls_back_on_coordinates():
    """P17 of Richmond, London is the United Kingdom, not England."""
    cand, res = v4(reply("Richmond", "England", 51.46, -0.30))
    assert cand["qid"] == "Q2500974" and res["method"] == "coordinates" and res["n_in_country"] == 0


# ------------------------------------------- found only by the wider search

@pytest.mark.parametrize("name, country, lat, lon, expected", [
    ("Richmond", "South Africa", -29.87, 30.27, "Q985819"),    # 47th result
    ("Richmond", "South Africa", None, None, "Q985819"),
    ("Toledo", "Uruguay", -34.74, -56.09, "Q2096039"),         # 25th result
])
def test_namesake_found_only_by_the_wider_search(name, country, lat, lon, expected):
    cand, res = v4(reply(name, country, lat, lon))
    assert cand["qid"] == expected
    assert res["status"] == "resolved" and res["n_searched"] == resolver.WIDE_SEARCH_LIMIT


def test_rule_v2_does_not_reach_it():
    """The same reply under v2 (ten results, "city" field): no Richmond in
    South Africa, and the coordinates are far from every one of the ten."""
    a = json.dumps({"city": "Richmond", "country": "South Africa", "latitude": -29.87,
                    "longitude": 30.27})
    cand, res = resolver.resolve_by_country_and_coordinates(a, search=search, fetch=fetch, geo=geo)
    assert cand["qid"] == "Q258" and res["status"] == "not_found" and res["level"] == "country"


@pytest.mark.parametrize("lat, lon, expected", [
    (-41.33, 173.18, "Q1208462"),     # Tasman District
    (-43.52, 172.66, "Q7330726"),     # Christchurch suburb, 49th result
])
def test_wider_results_are_chosen_by_the_coordinates(lat, lon, expected):
    cand, res = v4(reply("Richmond", "New Zealand", lat, lon))
    assert cand["qid"] == expected
    assert res["method"] == "country+coordinates" and res["n_searched"] == resolver.WIDE_SEARCH_LIMIT


# ------------- correction 1: one namesake in the country, far from the coordinates

def v3(a):
    return resolver.resolve_wide(a, search=search, fetch=fetch, geo=geo)


def city(place, country=None, lat=None, lon=None):
    """The same answer in the reply format of prompts v1-v3."""
    return json.dumps({"city": place, "country": country, "latitude": lat, "longitude": lon})


def test_far_single_namesake_sends_v4_to_the_wider_search():
    """The first ten results hold one Richmond in Canada (Quebec); the
    coordinates point at Richmond, British Columbia (16th result), 3,800 km
    from it. v4 searches fifty results and takes the one within TIE_KM."""
    cand, res = v4(reply("Richmond", "Canada", 49.17, -123.13))
    assert cand["qid"] == "Q236837"
    assert res["method"] == "country+coordinates" and res["km"] <= resolver.TIE_KM
    assert res["n_searched"] == resolver.WIDE_SEARCH_LIMIT


def test_v3_still_takes_the_far_single_namesake():
    cand, res = v3(city("Richmond", "Canada", 49.17, -123.13))
    assert cand["qid"] == "Q139694" and res["method"] == "country" and res["km"] > 3000


def test_far_single_namesake_is_kept_when_the_wider_search_has_none_near():
    """Coordinates on Madrid, 70 km from Toledo: no Toledo in Spain within
    TIE_KM among fifty results either, so the one in the first ten stays."""
    cand, res = v4(reply("Toledo", "Spain", 40.42, -3.70))
    assert cand["qid"] == "Q5836" and res["method"] == "country" and res["km"] > resolver.TIE_KM
    assert "n_searched" not in res


def test_several_far_namesakes_send_v4_to_the_wider_search_too():
    """The same correction when the ten results hold several namesakes in the
    country and none is near: five Aurangabads in India in the first ten, the
    nearest 180 km from the coordinates; the eleventh result, a village in
    Uttar Pradesh, is at the coordinates. v3 takes the first in search order."""
    a = reply("Aurangabad", "India", 27.085, 78.841)
    cand, res = v4(a)
    assert cand["qid"] == "Q120487800" and res["method"] == "country+coordinates"
    assert res["n_searched"] == resolver.WIDE_SEARCH_LIMIT
    cand, res = v3(city("Aurangabad", "India", 27.085, 78.841))
    assert cand["qid"] == "Q200713" and res["method"] == "country+search_order"


def test_several_far_namesakes_fall_back_to_search_order():
    """Coordinates on Bogotá: no Toledo in Colombia within TIE_KM among fifty
    results either, so the first of the ten in search order is taken, as v3
    does (the choice made on the development split)."""
    a = reply("Toledo", "Colombia", 4.71, -74.07)
    cand, res = v4(a)
    assert cand["qid"] == "Q1525656" and res["method"] == "country+search_order"
    assert "n_searched" not in res
    assert v3(city("Toledo", "Colombia", 4.71, -74.07))[0]["qid"] == "Q1525656"


def test_near_single_namesake_searches_no_further():
    calls = []

    def counting(name, limit=10):
        calls.append((name, limit))
        return search(name, limit)

    cand, _ = v4(reply("Toledo", "Philippines", 10.38, 123.64), search=counting)
    assert cand["qid"] == "Q316527"
    assert ("Toledo", resolver.WIDE_SEARCH_LIMIT) not in calls


# --------------------------------------------- country and continent fallback

def test_not_found_falls_back_to_the_country():
    cand, res = v4(reply("Qwxzvillebury", "Uruguay", -34.7, -56.1))
    assert cand["qid"] == "Q77"
    assert res["status"] == "not_found" and res["level"] == "country" and res["n_candidates"] == 0


def test_not_found_without_a_country_falls_back_to_the_continent():
    cand, res = v4(reply("Qwxzvillebury", None, continent="South America"))
    assert cand["qid"] is None and cand["lat"] is None and cand["continent"] == "South America"
    assert res["status"] == "not_found" and res["level"] == "continent"


def test_country_alone_is_the_answer():
    cand, res = v4(reply(None, "India"))
    assert cand["qid"] == "Q668" and res == {"status": "resolved", "level": "country",
                                            "method": "country_item"}


def test_continent_alone_is_the_answer():
    cand, res = v4(reply(None, None, continent="Asia"))
    assert cand["qid"] is None and res["status"] == "resolved" and res["level"] == "continent"


# ------------------------------------------------------------ failure reasons

def test_declined():
    cand, res = v4(reply(refused=True))
    assert cand is None and res == {"status": "declined", "level": None, "method": None}


def test_declined_with_a_place_still_declines():
    """refused=true wins over any field: the reply is not searched."""
    cand, res = v4(reply("Toledo", "Spain", 39.86, -4.02, refused=True))
    assert cand is None and res["status"] == "declined"


@pytest.mark.parametrize("raw", [
    '{"place": "Toledo", "country": ',            # cut off
    "It is probably Toledo in Spain.",             # prose
    '["Toledo", "Spain"]',                         # JSON, not an object
    "",
    None,
])
def test_unreadable(raw):
    cand, res = v4(raw)
    assert cand is None and res["status"] == "unreadable"


def test_fenced_reply_is_read():
    raw = "```json\n" + reply("Toledo", "Philippines", 10.38, 123.64) + "\n```"
    assert v4(raw)[0]["qid"] == "Q316527"


def test_not_found_with_nothing_to_fall_back_on():
    cand, res = v4(reply("Qwxzvillebury"))
    assert cand is None and res["status"] == "not_found" and res["level"] is None


def test_namesake_far_from_the_coordinates_is_not_found():
    """A Toledo in a country that has none, with coordinates there: the
    nearest Toledo is more than MAX_KM_WITHOUT_COUNTRY away, so the answer is
    the country, not a Toledo elsewhere."""
    cand, res = v4(reply("Toledo", "New Zealand", -41.0, 174.0))
    assert cand["qid"] == "Q664"
    assert res["status"] == "not_found" and res["km"] > resolver.MAX_KM_WITHOUT_COUNTRY


@pytest.mark.parametrize("name, country, n_in_country", [
    ("Toledo", "Colombia", 2), ("Aurangabad", "India", 9)])
def test_ambiguous_in_the_country_gives_the_country(name, country, n_in_country):
    cand, res = v4(reply(name, country))
    assert res["status"] == "ambiguous" and res["level"] == "country"
    assert res["n_in_country"] == n_in_country
    assert cand["qid"] == SEARCHES[country][0]


def test_ambiguous_with_nothing_to_choose_with():
    cand, res = v4(reply("Richmond"))
    assert cand is None and res["status"] == "ambiguous" and res["level"] is None


def test_ambiguous_falls_back_to_the_continent():
    cand, res = v4(reply("Richmond", continent="Oceania"))
    assert res["status"] == "ambiguous" and res["level"] == "continent"
    assert cand["continent"] == "Oceania" and cand["qid"] is None


def test_older_city_reply_is_read_as_a_place():
    a = json.dumps({"city": "Toledo", "country": "Philippines", "latitude": 10.38,
                    "longitude": 123.64})
    assert v4(a)[0]["qid"] == "Q316527"


# ----------------------- correction 2: the country name against the country items

@pytest.mark.parametrize("name, expected", [
    ("Georgia", "Q230"), ("UK", "Q145"), ("United Kingdom", "Q145"), ("USA", "Q30"),
    ("united states", "Q30"), ("Czechia", "Q213"), ("Jordan", "Q810"), ("Chad", "Q657"),
    ("The Netherlands", None),     # an alias of two items: no match, the search decides
    ("Holland", None), ("Atlantis", None)])
def test_country_item(name, expected):
    assert resolver.country_item(name) == expected


def test_georgia_is_the_country_not_the_state():
    """The first search hit for "Georgia" is the US state. v3 answers with it;
    v4 answers with the country."""
    a = reply("Qwxzvillebury", "Georgia", 41.7, 44.8)
    assert v3(a)[0]["qid"] == "Q1428"                    # v3 reads "city": none here
    cand, res = v4(a)
    assert cand["qid"] == "Q230" and res["status"] == "not_found"
    assert res["level"] == "country" and res["method"] == "country_item"


def test_uk_is_the_united_kingdom_not_ukrainian():
    """The first search hit for "UK" is the Ukrainian language, which has
    coordinates in Ukraine. v3 finds no Richmond in "UK" and, with no
    coordinates, answers with that item; v4 finds the two Richmonds in the
    United Kingdom and, with nothing to choose between them, answers with
    the United Kingdom."""
    cand, res = v3(city("Richmond", "UK"))
    assert cand["qid"] == "Q8798" and res["status"] == "not_found"
    cand, res = v4(reply("Richmond", "UK"))
    assert cand["qid"] == "Q145" and res["status"] == "ambiguous" and res["n_in_country"] == 2
    cand, res = v4(reply("Richmond", "UK", 54.40, -1.74))
    assert cand["qid"] == "Q1009324" and res["method"] == "country+coordinates"


def test_unmatched_country_name_still_uses_the_search():
    cand, res = v4(reply("Richmond", "England", 51.46, -0.30), countries=lambda n: None)
    assert cand["qid"] == "Q2500974" and res["method"] == "coordinates"


# ----------------------------------- correction 3: a failed Wikidata request

def failing(name, limit=10):
    raise resolver.wd.WikidataError("HTTP 503")


@pytest.mark.parametrize("a", [reply("Toledo", "Spain", 39.86, -4.02),
                               reply(None, "Holland"),        # not in the table: searched
                               reply("Qwxzvillebury", "Uruguay")])
def test_failed_search_is_an_error_not_not_found(a):
    cand, res = v4(a, search=failing)
    assert cand is None and res["status"] == "error" and "503" in res["message"]


def test_failed_fetch_is_an_error():
    def broken_fetch(qids):
        raise resolver.wd.WikidataError("entity fetch failed")
    cand, res = v4(reply("Toledo", "Spain", 39.86, -4.02), fetch=broken_fetch)
    assert cand is None and res["status"] == "error"


def test_v3_still_reads_a_failed_search_as_nothing_found():
    """v3's default search returns [] on failure; with that, v3 says not_found."""
    cand, res = v3(city("Toledo", "Spain", 39.86, -4.02).replace("Toledo", "Nowhere"))
    assert res["status"] == "not_found"


def test_checked_lookups_raise_and_cache_nothing(monkeypatch):
    wd = resolver.wd

    def down(params, **kw):
        raise wd.WikidataError("HTTP 503")
    monkeypatch.setattr(wd, "_api_get", down)
    key = ("Somewhere-not-cached-xyz", 10)
    with pytest.raises(wd.WikidataError):
        wd.search_entities_checked(*key)
    assert key not in wd._search_cache
    with pytest.raises(wd.WikidataError):
        wd.fetch_entities_checked(["Q999999999991"])
    assert "Q999999999991" not in wd._entity_cache
    assert wd.search_entities(*key) == []                 # v1-v3: unchanged


def test_pipeline_puts_the_error_on_the_record(monkeypatch):
    """The runner retries records whose `error` is set (src/runner.py)."""
    import config
    import pipeline

    class LLM:
        def geolocate_verbose(self, text):
            return None, "{}", {"status": "error", "level": None, "method": None,
                                "message": "HTTP 503"}
    cfg = config.load_config(ROOT / "configs" / "gemma4.yaml")
    pipe = pipeline.Pipeline(cfg)
    monkeypatch.setattr(pipe, "_llm_pipeline", lambda: LLM())
    out = pipe.run("Back in Toledo")
    assert out.error == "wikidata: HTTP 503" and out.prediction is None
    assert out.resolution["status"] == "error"


# ------------------------------------- correction 4: empty and blocked replies

from llm_common import EMPTY_REPLY_SENTINEL, REFUSAL_SENTINEL  # noqa: E402


@pytest.mark.parametrize("raw", [
    EMPTY_REPLY_SENTINEL,                    # empty or blocked reply (revision 3 on)
    REFUSAL_SENTINEL,                        # the same, in records written before
    reply(),                                 # every field null, refused false
    json.dumps({"place": "null", "country": "unknown", "continent": None,
                "latitude": None, "longitude": None, "refused": False}),
])
def test_empty_reply_is_its_own_status(raw):
    cand, res = v4(raw)
    assert cand is None and res["status"] == "empty"


def test_a_refusal_is_still_declined():
    assert v4(reply(refused=True))[1]["status"] == "declined"
    full = json.dumps({"place": None, "country": None, "continent": None,
                       "latitude": None, "longitude": None, "refused": True})
    assert v4(full)[1]["status"] == "declined"


@pytest.mark.parametrize("raw", [EMPTY_REPLY_SENTINEL, REFUSAL_SENTINEL, city(None)])
def test_v2_and_v3_still_call_an_empty_reply_declined(raw):
    for rule in (v3, lambda a: resolver.resolve_by_country_and_coordinates(
            a, search=search, fetch=fetch, geo=geo)):
        assert rule(raw)[1]["status"] == "declined"


def test_coordinates_alone_are_not_empty():
    """Not every field is null, so not `empty` (the reply names no place:
    `declined`, as before)."""
    assert v4(reply(None, None, 10.38, 123.64))[1]["status"] == "declined"


def test_empty_replies_are_not_cached(monkeypatch):
    import llm_geo

    class Client:
        model_id, generation_params = "stub", {}

        def call(self, prompt):
            return EMPTY_REPLY_SENTINEL
    stored = {}

    class Cache:
        def __init__(self, *a):
            pass

        def get(self, key):
            return stored.get(key)

        def put(self, key, value):
            stored[key] = value

        def save(self):
            pass
    monkeypatch.setattr(llm_geo, "PickleCache", Cache)
    pipe = llm_geo.LLMGeoPipeline(Client(), "unused", "v4")
    monkeypatch.setattr(pipe, "resolve", lambda raw: (None, resolver._read(raw, v4=True)[1]))
    cand, raw, res = pipe.geolocate_verbose("Back in Toledo")
    assert raw == EMPTY_REPLY_SENTINEL and res["status"] == "empty" and not stored


# ------------------------------------------------------ optional: live Wikidata

LIVE = os.environ.get("GEO_LIVE_WIKIDATA") == "1"


@pytest.mark.skipif(not LIVE, reason="live Wikidata check; set GEO_LIVE_WIKIDATA=1 to run")
@pytest.mark.parametrize("a, expected", [
    (reply("Toledo", "Philippines", 10.38, 123.64), "Q316527"),
    (reply("Toledo", "Colombia", 7.01, -75.70), "Q2078498"),
    (reply("Aurangabad", "India", 24.75, 84.37), "Q590550"),
    (reply("Richmond", "South Africa", -29.87, 30.27), "Q985819"),
])
def test_live_wikidata(a, expected):
    """The same rule with the real search and items. Wikidata changes, so a
    failure here means the fixture may be out of date, not necessarily that
    the rule is wrong."""
    cand, res = resolver.resolve_place_wide(a)
    assert cand is not None and cand["qid"] == expected, res
