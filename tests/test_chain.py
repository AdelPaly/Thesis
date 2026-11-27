"""Unit tests for the place rules of src/wikidata.py (`linking.place_rule`):

  chain_v1    the reported runs: the first place along GEO_PROP_CHAIN_V1 on the
              linking path; place of birth (persons) and headquarters
              (organisations) on the search path;
  all_places  revision 3: every place a linked entity without coordinates
              links to, deduplicated, with no order of preference.

    python -m pytest tests/test_chain.py

No network: the Wikidata calls are replaced with hand-made items.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import wikidata as wd  # noqa: E402


def _value(qid, rank="normal", ended=False):
    claim = {"rank": rank, "mainsnak": {"datavalue": {"value": {"id": qid}}}}
    if ended:
        claim["qualifiers"] = {"P582": [{}]}
    return claim


def _claims(**props):
    """_claims(P19="Qborn", P551=["Qa", ("Qb", "deprecated")]) -> claims."""
    out = {}
    for prop, values in props.items():
        values = values if isinstance(values, list) else [values]
        out[prop] = [_value(*v) if isinstance(v, tuple) else _value(v) for v in values]
    return out


def _place(lat, lon, **extra):
    return {"claims": {"P625": [{"rank": "normal", "mainsnak": {"datavalue": {
        "value": {"latitude": lat, "longitude": lon}}}}], **_claims(**extra)}}


HUMAN = "Q5"
ITEMS = {
    "Qborn": _place(1, 1), "Qhome": _place(2, 2), "Qwork": _place(3, 3),
    "Qhq": _place(4, 4), "Qdied": _place(6, 6), "Qtown": _place(7, 7),
    "Qformed": _place(8, 8), "Qnowhere": {"claims": {}},
    "Qdistrict": {"claims": _claims(P131="Qtown")},            # no coordinates: hop to P131
    # people
    "Qall": {"claims": _claims(P31=HUMAN, P19="Qborn", P551="Qhome", P937="Qwork")},
    "Qwork_born": {"claims": _claims(P31=HUMAN, P19="Qborn", P937="Qwork")},
    "Qborn_only": {"claims": _claims(P31=HUMAN, P19="Qborn")},
    "Qhome_no_coords": {"claims": _claims(P31=HUMAN, P551="Qnowhere", P19="Qborn")},
    "Qsame_place": {"claims": _claims(P31=HUMAN, P19="Qhome", P551="Qhome", P937="Qhome")},
    "Qvia_district": {"claims": _claims(P31=HUMAN, P551="Qdistrict")},
    "Qranks": {"claims": _claims(P31=HUMAN, P19="Qborn",
                                 P551=[("Qhome", "deprecated"), ("Qwork", "normal", True)])},
    "Qtwo_homes": {"claims": _claims(P31=HUMAN, P551=["Qhome", "Qwork"], P20="Qdied")},
    # organisations
    "Qorg": {"claims": _claims(P159="Qhq", P551="Qhome", P740="Qformed")},
    "Qorg_no_hq": {"claims": _claims(P740="Qformed")},
    "Qorg_hq_district": {"claims": _claims(P159="Qdistrict")},
    "Qorg_moved": {"claims": _claims(P159=[("Qhq", "normal", True), "Qwork"])},  # old HQ ended
    # an item with coordinates of its own is returned as itself, under both rules
    "Qstation": _place(5, 5, P131="Qhome", P276="Qwork"),
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    fetch = lambda qids: {q: ITEMS.get(q, {}) for q in qids}  # noqa: E731
    monkeypatch.setattr(wd, "fetch_entities", fetch)
    monkeypatch.setattr(wd, "_fetch_entities_status", lambda qids: (fetch(qids), True))
    monkeypatch.setattr(wd, "resolve_country_continent", lambda ents: {q: ("", "") for q in ents})
    monkeypatch.setattr(wd, "_resolve_cc", lambda ents: ({q: ("", "") for q in ents}, True))


def all_places(qid, kind="PER"):
    return [c["qid"] for c in wd.resolve_qid_to_candidates(qid, "x", kind, "all_places")]


# ------------------------------------------------ chain_v1: the reported runs

@pytest.mark.parametrize("person, expected", [
    ("Qall", "Qborn"),             # P19 comes before P937 in GEO_PROP_CHAIN_V1
    ("Qwork_born", "Qborn"),
    ("Qborn_only", "Qborn"),
    ("Qhome_no_coords", "Qborn"),  # residence is not in the chain at all
    ("Qvia_district", None),       # nor is it reached through P131
])
def test_chain_v1_linked_person(person, expected):
    cand = wd.resolve_qid_to_candidate(person, "x", "PER")
    assert (cand and cand["qid"]) == expected
    assert [c["qid"] for c in wd.resolve_qid_to_candidates(person, "x", "PER")] == (
        [expected] if expected else [])


def test_chain_v1_is_the_default_and_the_reported_chain():
    assert wd.GEO_PROP_CHAIN_V1 == ["P159", "P131", "P19", "P937", "P276"]
    assert wd.resolve_qid_to_candidates("Qall", "x", "PER") == [
        wd.resolve_qid_to_candidate("Qall", "x", "PER")]


def test_chain_v1_organisation_and_hop():
    assert wd.resolve_qid_to_candidate("Qorg", "x", "ORG")["qid"] == "Qhq"
    assert wd.resolve_qid_to_candidate("Qorg_hq_district", "x", "ORG")["qid"] == "Qtown"


def test_chain_v1_skips_an_end_dated_value_when_there_is_another():
    """_claim_id, as in the reported runs: the current headquarters, not the old one."""
    assert wd.resolve_qid_to_candidate("Qorg_moved", "x", "ORG")["qid"] == "Qwork"


def test_all_places_keeps_the_end_dated_headquarters_too():
    assert sorted(all_places("Qorg_moved", "ORG")) == ["Qhq", "Qwork"]


def test_residence_first_chain_is_only_documented():
    assert wd.GEO_PROP_CHAIN == ["P159", "P131", "P551", "P937", "P19", "P276"]
    assert not hasattr(wd, "PERSON_PROP_CHAIN")


@pytest.mark.parametrize("person, expected", [
    ("Qall", ["Qborn"]), ("Qwork_born", ["Qborn"]), ("Qborn_only", ["Qborn"]),
    ("Qhome_no_coords", ["Qborn"]), ("Qtwo_homes", []),
])
def test_chain_v1_search_path_person_is_the_place_of_birth(person, expected):
    out, complete = wd._per_candidates("x", [person], cap=5)
    assert complete and [c["qid"] for c in out] == expected


def test_chain_v1_search_path_organisation_is_the_headquarters():
    out, _ = wd._org_candidates("x", ["Qorg", "Qorg_no_hq"], cap=10)
    assert [c["qid"] for c in out] == ["Qhq"]


# ------------------------------------------------ all_places: revision 3

def test_all_places_gives_every_place_of_a_person():
    assert sorted(all_places("Qall")) == ["Qborn", "Qhome", "Qwork"]
    assert sorted(all_places("Qtwo_homes")) == ["Qdied", "Qhome", "Qwork"]


def test_all_places_follows_the_property_list_without_preference():
    assert wd.ALL_PLACE_PROPS == ["P159", "P740", "P131", "P276", "P551", "P937", "P19", "P20"]
    assert sorted(all_places("Qorg", "ORG")) == ["Qformed", "Qhome", "Qhq"]


def test_own_coordinates_give_only_the_item_itself():
    assert all_places("Qstation", "LOC") == ["Qstation"]
    assert wd.resolve_qid_to_candidate("Qstation", "x", "LOC")["qid"] == "Qstation"


def test_all_places_deduplicates():
    assert all_places("Qsame_place") == ["Qhome"]


def test_all_places_hops_to_p131_when_a_target_has_no_coordinates():
    assert all_places("Qvia_district") == ["Qtown"]
    assert all_places("Qorg_hq_district", "ORG") == ["Qtown"]
    assert all_places("Qhome_no_coords") == ["Qborn"]     # Qnowhere has no P131 either


def test_all_places_skips_deprecated_and_keeps_end_dated_values():
    assert sorted(all_places("Qranks")) == ["Qborn", "Qwork"]   # Qhome is deprecated


def test_all_places_candidate_carries_the_place_name_and_the_mention():
    ITEMS["Qhome"]["labels"] = {"en": {"value": "Hometown"}}
    try:
        cands = wd.resolve_qid_to_candidates("Qsame_place", "Anna Example", "PER", "all_places")
    finally:
        del ITEMS["Qhome"]["labels"]
    assert [(c["label"], c["entity"], c["entity_type"]) for c in cands] == [
        ("Hometown", "Anna Example", "PER")]


def test_unknown_rule_is_refused():
    with pytest.raises(ValueError):
        wd.resolve_qid_to_candidates("Qall", "x", "PER", "residence_first")


# ------------------------------------------------ all_places on the search path

def test_search_path_person_gives_all_places_and_the_cap_counts_persons():
    out, complete = wd._per_candidates("x", ["Qall", "Qborn_only", "Qtwo_homes"], cap=2,
                                       place_rule="all_places")
    # Qall and Qborn_only are the first two persons; Qborn is listed once.
    assert complete and sorted(c["qid"] for c in out) == ["Qborn", "Qhome", "Qwork"]


def test_search_path_skips_non_humans_for_a_person():
    out, _ = wd._per_candidates("x", ["Qorg"], cap=5, place_rule="all_places")
    assert out == []


def test_search_path_organisation_gives_all_places_and_skips_humans():
    out, _ = wd._org_candidates("x", ["Qall", "Qorg", "Qnowhere", "Qorg_no_hq"], cap=10,
                                place_rule="all_places")
    assert sorted(c["qid"] for c in out) == ["Qformed", "Qhome", "Qhq"]


def test_search_path_organisation_cap_counts_organisations():
    out, _ = wd._org_candidates("x", ["Qorg_no_hq", "Qorg"], cap=1, place_rule="all_places")
    assert [c["qid"] for c in out] == ["Qformed"]


def test_cache_key_keeps_chain_v1_entries_and_separates_all_places(monkeypatch):
    """chain_v1 keeps the key of the reported runs (text, type, top_n), so a
    cached answer is reused; all_places gets its own entry. A dict stands in
    for the on-disk cache, which the test must not touch."""
    cache = {("x", "PER", 10): ["cached chain_v1 answer"]}
    monkeypatch.setattr(wd, "_coords_cache", cache)
    monkeypatch.setattr(wd, "_search_with_fallback", lambda text, limit: ["Qall"])
    assert wd.get_coords("x", "PER", top_n=10) == ["cached chain_v1 answer"]
    out = wd.get_coords("x", "PER", top_n=10, place_rule="all_places")
    assert sorted(c["qid"] for c in out) == ["Qborn", "Qhome", "Qwork"]
    assert ("x", "PER", 10, "all_places") in cache
    assert cache[("x", "PER", 10)] == ["cached chain_v1 answer"]
