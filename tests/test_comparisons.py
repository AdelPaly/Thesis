"""The planned comparisons and the exploratory ones (configs/comparisons.yaml,
scripts/analyse.py), after the development run of 26-27 November.

    python -m pytest tests/test_comparisons.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import analyse  # noqa: E402

PLANNED = {
    ("geoparser_irchel", "population_wikidata"), ("refined", "population_wikidata"),
    ("gemma4", "population_wikidata"), ("refined", "retrieve_rerank"),
    ("blink", "retrieve_rerank"), ("refined", "geoparser_irchel"),
    ("dense_retrieval", "refined"), ("refined", "gemma4"), ("dense_retrieval", "gemma4"),
    ("qwen3_vl", "gemma4"), ("grounded_gemma", "gemma4"), ("grounded_gemma", "dense_retrieval"),
    ("grounded_gemma_linking", "grounded_gemma"), ("grounded_gemma_linking", "refined"),
}


def test_planned_family_is_the_fourteen_pairs_chosen_in_advance():
    planned = [(a, b) for a, b, _ in analyse.load_comparisons()]
    assert len(planned) == 14 and set(planned) == PLANNED


def test_size_pair_is_exploratory_and_outside_the_family():
    exploratory = [(a, b) for a, b, _ in analyse.load_comparisons(section="exploratory")]
    assert exploratory == [("qwen3_vl", "qwen3_vl_32b")]
    assert not set(exploratory) & PLANNED
