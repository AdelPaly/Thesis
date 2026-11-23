"""Unit tests for src/ranking.py: where a candidate with no description goes.

    python -m pytest tests/test_ranking.py

No model is loaded: the encoder is replaced by one that returns hand-made
vectors, so the scores are known in advance.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

ranking = pytest.importorskip("ranking")

TEXT = "the bins by the old lot in Sitra are overflowing again"
# what the fake encoder returns for each string: similarity to TEXT is the first number
VECTORS = {
    TEXT: [1.0, 0.0],
    "Sitra": [0.99, 0.14],                      # the mention itself: very similar to the text
    "island of Bahrain": [0.6, 0.8],
    "depression in Egypt": [0.3, 0.95],
    "": [0.0, 1.0],
}


class FakeEncoder:
    def encode(self, texts, **_):
        if isinstance(texts, str):
            return np.array(VECTORS[texts], dtype="float32")
        return np.array([VECTORS[t] for t in texts], dtype="float32")


@pytest.fixture(autouse=True)
def fake_encoder(monkeypatch):
    monkeypatch.setattr(ranking, "load_encoder", lambda name: FakeEncoder())


def cand(qid, description, entity="Sitra"):
    return {"qid": qid, "entity": entity, "description": description}


def order(candidates):
    return [c["qid"] for c in ranking.rank_candidates(TEXT, candidates, model="fake")]


def test_described_candidates_are_ordered_by_similarity():
    assert order([cand("egypt", "depression in Egypt"), cand("bahrain", "island of Bahrain")]) == [
        "bahrain", "egypt"]


def test_mention_text_as_description_does_not_win():
    """The item with no description used to be scored on "Sitra" and come first."""
    assert order([cand("nodesc", "Sitra"), cand("bahrain", "island of Bahrain"),
                  cand("egypt", "depression in Egypt")]) == ["bahrain", "egypt", "nodesc"]


def test_the_score_is_still_recorded():
    ranked = ranking.rank_candidates(TEXT, [cand("nodesc", "Sitra"), cand("bahrain", "island of Bahrain")],
                                     model="fake")
    assert ranked[1]["qid"] == "nodesc" and ranked[1]["score"] > ranked[0]["score"]


def test_only_undescribed_candidates_keep_their_score_order():
    assert order([cand("a", None), cand("b", "Sitra")]) == ["b", "a"]


def test_missing_and_empty_descriptions_count_as_none():
    assert not ranking.has_description(cand("a", None))
    assert not ranking.has_description(cand("a", "  "))
    assert not ranking.has_description(cand("a", "sitra"))
    assert ranking.has_description(cand("a", "island of Bahrain"))


def test_comparison_ignores_case_and_accents():
    assert not ranking.has_description(cand("a", "Malé", entity="male"))


def test_another_mentions_text_is_a_description():
    """Only the candidate's own mention is the fill-in."""
    assert ranking.has_description(cand("a", "Sitra", entity="Manama"))


def test_input_is_not_modified_and_empty_list():
    candidates = [cand("bahrain", "island of Bahrain")]
    ranking.rank_candidates(TEXT, candidates, model="fake")
    assert "score" not in candidates[0]
    assert ranking.rank_candidates(TEXT, [], model="fake") == []
