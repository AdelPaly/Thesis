"""Unit tests for src/gazetteer.py: the population floor's candidates.

    python -m pytest tests/test_gazetteer.py

No network and no dataset. The search results are hand-made after real cases:
the Wikidata search matches the start of a label and any alias, so a province
or a namesake found by an alias comes back for a city's name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import gazetteer  # noqa: E402

# qid: (English label, population or None, has coordinates)
ITEMS = {
    "Q8717": ("Seville", 690_000, True),
    "Q95088": ("Province of Seville", 1_950_000, True),
    "Q1": ("Seville", 5_000, True),                 # a small namesake
    "Q2": ("Seville", None, True),                  # a namesake with no population
    "Q3": ("Seville Cathedral", None, True),
    "Q4": ("Seville", 9_000_000, False),            # a family name: no coordinates
    "Q181557": ("Kaohsiung", 2_730_000, True),      # found for "Gao" by an alias
    "Q188904": ("Gao", 86_000, True),
    "Q1156": ("Mumbai", 12_400_000, True),          # found for "Bombay" by an alias
    "Q5": ("Bombay Beach", 300, True),
    "Q6": ("Sevilla", 40_000, True),                # same letters, another spelling
    "Q7": ("São Tomé", 70_000, True),
}
SEARCH = {
    "Seville": ["Q4", "Q8717", "Q95088", "Q1", "Q2", "Q3"],
    "Gao": ["Q181557", "Q188904"],
    "Bombay": ["Q1156", "Q5"],
    "Sao Tome": ["Q7"],
    "Atlantis": [],
}


def _entity(qid):
    label, population, placed = ITEMS[qid]
    claims = {}
    if placed:
        claims["P625"] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": {
            "latitude": 1.0, "longitude": 2.0}}}}]
    if population is not None:
        claims["P1082"] = [{"rank": "normal", "mainsnak": {"datavalue": {"value": {
            "amount": f"+{population}"}}}}]
    return {"labels": {"en": {"value": label}}, "claims": claims, "sitelinks": {}}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(gazetteer.wd, "search_entities", lambda name, limit=10: SEARCH[name][:limit])
    monkeypatch.setattr(gazetteer.wd, "fetch_entities", lambda qids: {q: _entity(q) for q in qids})
    monkeypatch.setattr(gazetteer.wd, "resolve_country_continent",
                        lambda ents: {q: ("", "") for q in ents})


def qids(name, **kw):
    return [c["qid"] for c in gazetteer.lookup(name, **kw)]


def test_all_results_most_populous_first_is_the_old_rule():
    assert qids("Seville") == ["Q95088", "Q8717", "Q1", "Q2", "Q3"]


def test_same_name_drops_the_province_and_the_cathedral():
    assert qids("Seville", same_name=True) == ["Q8717", "Q1", "Q2"]


def test_same_name_still_takes_the_most_populous_namesake():
    assert qids("Seville", same_name=True)[0] == "Q8717"


def test_item_without_coordinates_is_never_a_candidate():
    assert "Q4" not in qids("Seville") and "Q4" not in qids("Seville", same_name=True)


def test_alias_match_loses_to_the_item_of_that_name():
    assert qids("Gao")[0] == "Q181557"
    assert qids("Gao", same_name=True) == ["Q188904"]


def test_place_found_only_by_an_alias_is_kept():
    assert qids("Bombay", same_name=True) == ["Q1156", "Q5"]


def test_name_is_compared_without_accents_or_case():
    assert qids("Sao Tome", same_name=True) == ["Q7"]
    assert gazetteer.named_as("SEVILLE", [{"qid": "a", "label": "Seville"},
                                          {"qid": "b", "label": "Sevilla"}]) == [
        {"qid": "a", "label": "Seville"}]


def test_no_result():
    assert qids("Atlantis", same_name=True) == []


def test_min_population_applies_after_the_name_filter():
    assert qids("Seville", same_name=True, min_population=10_000) == ["Q8717"]


def test_candidate_without_a_label_never_counts_as_same_name():
    assert gazetteer.named_as("Gao", [{"qid": "a", "label": ""}, {"qid": "b", "label": "Gao"}]) == [
        {"qid": "b", "label": "Gao"}]
