"""Unit tests for the grounded selection prompt (src/llm_grounded.py), revision 3.

    python -m pytest tests/test_grounded.py

No network and no model: only the text the model would be shown is checked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import llm_grounded as g  # noqa: E402

# A place reached through a person (linking): `entity` is the mention.
LINKED = {"qid": "Q1015681", "entity": "Nick Willis", "entity_type": "PER",
          "label": "Lower Hutt", "country": "New Zealand",
          "description": "city in the Wellington Region of New Zealand"}
# A dense candidate: `entity` is already the place name, and there is no label.
DENSE = {"qid": "Q1015681", "entity": "Lower Hutt", "entity_type": "PLACE",
         "country": "New Zealand",
         "description": "city in the Wellington Region of New Zealand"}


def format_candidates_v2(candidates):
    """List v2 as it ran in the development run of 26-27 November, copied
    here so that the dense list can be checked against it byte for byte."""
    lines = []
    for i, cand in enumerate(candidates, start=1):
        label = cand.get("label") or cand.get("entity") or cand.get("qid") or "?"
        qid = cand.get("qid") or "-"
        country = cand.get("country") or "unknown country"
        desc = (cand.get("description") or "").strip()
        line = f"{i}. {qid} — {label} — {country}"
        if desc and desc.lower() != label.lower():
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no candidates were retrieved)"


def test_linked_candidate_shows_the_place_name_then_the_mention():
    line = g.format_candidates([LINKED])
    assert line == ('1. Q1015681 — Lower Hutt — New Zealand — city in the Wellington '
                    'Region of New Zealand — from "Nick Willis"')


def test_the_mention_is_the_only_change_from_list_v2():
    assert g.format_candidates([LINKED]) == (format_candidates_v2([LINKED])
                                             + ' — from "Nick Willis"')


def test_mention_that_is_the_place_name_is_not_repeated():
    for mention in ("Lower Hutt", "lower hutt", "Lower-Hutt"):
        cand = {**LINKED, "entity": mention, "entity_type": "LOC"}
        assert g.format_candidates([cand]) == format_candidates_v2([cand])


def test_candidate_without_label_falls_back_to_the_mention():
    cand = {k: v for k, v in LINKED.items() if k != "label"}
    line = g.format_candidates([cand])
    assert "Nick Willis" in line and "from" not in line


def test_dense_candidates_render_exactly_as_in_list_v2():
    cands = [DENSE, {**DENSE, "qid": "Q2", "entity": "Earth", "description": "Earth"},
             {"qid": "Q3", "entity": "Somewhere", "country": None, "description": None}]
    assert g.format_candidates(cands) == format_candidates_v2(cands)
    assert g.format_candidates([]) == format_candidates_v2([])


def test_dense_candidates_of_the_pinned_index_render_exactly_as_in_list_v2():
    """Every place of the revision 3 index, as dense retrieval hands it to the
    model: list v3 is byte-identical to v2, so the dense cache stays v2."""
    import pickle
    path = Path(__file__).resolve().parent.parent / "src" / ".dense_index" / "index_3530d31fa83aacb7.pkl"
    if not path.exists():
        import pytest
        pytest.skip("pinned dense index not on this machine")
    with path.open("rb") as f:
        places = list(pickle.load(f)["places"].values())
    assert len(places) == 3599 and not any("label" in p for p in places)
    for start in range(0, len(places), 10):
        block = places[start:start + 10]
        assert g.format_candidates(block) == format_candidates_v2(block)


def test_prompt_no_longer_asks_to_decline_when_unsure():
    assert "do not believe in" in g.PROMPT_V1
    assert "do not believe in" not in g.PROMPT
    assert "none of the candidates is the location" in g.PROMPT


def _cache_name(source):
    import pipeline
    cfg = SimpleNamespace(llm=SimpleNamespace(cache_name="gemma4_12b"),
                          grounded=SimpleNamespace(candidate_source=source))
    return pipeline.Pipeline._grounded_cache_name(SimpleNamespace(cfg=cfg))


def test_new_prompt_has_its_own_cache():
    assert g.PROMPT_VERSION != "v1"
    assert _cache_name("dense") == "gemma4_12b_grounded_dense_v2"


def test_linking_list_v3_has_its_own_cache_and_dense_keeps_v2():
    assert g.LIST_VERSIONS == {"dense": "v2", "linking": "v3"}
    assert _cache_name("linking") == "gemma4_12b_grounded_linking_v3"


def test_stored_candidate_keeps_the_label():
    import pipeline
    rec = pipeline._candidate_record(LINKED)
    assert rec["label"] == "Lower Hutt" and rec["source_entity"] == "Nick Willis"
    assert pipeline._candidate_record(DENSE)["label"] is None
