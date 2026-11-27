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


def test_linked_candidate_shows_the_place_name_not_the_mention():
    line = g.format_candidates([LINKED])
    assert "Lower Hutt" in line and "Nick Willis" not in line


def test_linked_and_dense_candidates_are_shown_the_same_way():
    assert g.format_candidates([LINKED]) == g.format_candidates([DENSE])


def test_candidate_without_label_falls_back_to_the_mention():
    cand = {k: v for k, v in LINKED.items() if k != "label"}
    assert "Nick Willis" in g.format_candidates([cand])


def test_prompt_no_longer_asks_to_decline_when_unsure():
    assert "do not believe in" in g.PROMPT_V1
    assert "do not believe in" not in g.PROMPT
    assert "none of the candidates is the location" in g.PROMPT


def test_new_prompt_has_its_own_cache():
    import pipeline
    cfg = SimpleNamespace(llm=SimpleNamespace(cache_name="gemma4_12b"),
                          grounded=SimpleNamespace(candidate_source="linking"))
    name = pipeline.Pipeline._grounded_cache_name(SimpleNamespace(cfg=cfg))
    assert name == f"gemma4_12b_grounded_linking_{g.PROMPT_VERSION}"
    assert g.PROMPT_VERSION != "v1"
