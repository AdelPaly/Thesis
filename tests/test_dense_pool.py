"""Unit tests for the pool filter of src/dense_index.py
(gold_plus_answers_described).

    python -m pytest tests/test_dense_pool.py

No network, no model, no dataset: hand-made Wikidata items.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

dense_index = pytest.importorskip("dense_index")

EARTH = "http://www.wikidata.org/entity/Q2"
MARS = "http://www.wikidata.org/entity/Q111"


def coords(globe=EARTH, rank="normal"):
    value = {"latitude": 1.0, "longitude": 2.0}
    if globe is not None:
        value["globe"] = globe
    return {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}


def item(description="city in Tunisia", p625=None, p31=()):
    ent = {"claims": {"P625": [coords()] if p625 is None else p625,
                      "P31": [{"mainsnak": {"datavalue": {"value": {"id": q}}}} for q in p31]}}
    if description is not None:
        ent["descriptions"] = {"en": {"value": description}}
    return ent


def test_a_described_place_with_a_country_is_kept():
    assert dense_index.is_described_place(item(), "Tunisia")


def test_no_description_is_left_out():
    assert not dense_index.is_described_place(item(description=None), "Kosovo")
    assert not dense_index.is_described_place(item(description="  "), "Kosovo")


def test_no_country_is_left_out():
    assert not dense_index.is_described_place(item(), "")
    assert not dense_index.is_described_place(item(), None)


def test_crater_on_mars_is_left_out():
    crater = item(description="Mars crater", p625=[coords(MARS)])
    assert not dense_index.on_earth(crater)
    assert not dense_index.is_described_place(crater, "Tunisia")


def test_disambiguation_page_is_left_out():
    assert not dense_index.is_described_place(
        item(description="Wikimedia disambiguation page", p31=["Q4167410"]), "Japan")


def test_claim_without_a_globe_counts_as_earth():
    assert dense_index.on_earth(item(p625=[coords(None)]))


def test_preferred_claim_decides():
    assert dense_index.on_earth(item(p625=[coords(MARS), coords(EARTH, "preferred")]))
    assert not dense_index.on_earth(item(p625=[coords(EARTH), coords(MARS, "preferred")]))


def test_deprecated_claim_is_ignored():
    assert dense_index.on_earth(item(p625=[coords(MARS, "deprecated"), coords(EARTH)]))


def test_item_without_coordinates_is_not_called_off_earth():
    assert dense_index.on_earth({"claims": {}})


def test_gold_place_removed_by_the_filter_stops_the_build():
    with pytest.raises(RuntimeError):
        dense_index._check_gold({"Q1"}, {"Q1", "Q2"})
