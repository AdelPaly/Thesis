"""Unit tests for the predict-only mode (scripts/predict.py).

    python -m pytest tests/test_predict.py

No model and no network: the pipeline is replaced by a stand-in that returns
hand-made PipelineOutputs, and labels are not looked up on Wikidata.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
import predict  # noqa: E402
from pipeline import PipelineOutput  # noqa: E402

ALL_CONFIGS = sorted(p.stem for p in (ROOT / "configs").glob("*.yaml")
                     if p.stem not in ("datasets", "shared", "comparisons"))


def cfg(name):
    return config.load_config(ROOT / "configs" / f"{name}.yaml")


# ----------------------------------------------- configurations that need gold

@pytest.mark.parametrize("name", ["majority_class", "dense_retrieval", "grounded_gemma"])
def test_gold_dependent_configurations_are_refused(name, tmp_path):
    (tmp_path / "in.csv").write_text("text\nhello\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot run"):
        predict.predict(name, tmp_path / "in.csv", tmp_path / "out",
                        pipeline_factory=lambda *a, **k: pytest.fail("pipeline built"))
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", [n for n in ALL_CONFIGS
                                  if n not in ("majority_class", "dense_retrieval", "grounded_gemma")])
def test_other_configurations_need_no_gold(name):
    assert predict.needs_gold(cfg(name)) is None


# ---------------------------------------------------------------- statuses

def out(**kw):
    o = PipelineOutput()
    for k, v in kw.items():
        setattr(o, k, v)
    return o


SPAN = {"text": "Toledo", "type": "LOC", "start": 0, "end": 6}
PLACE = {"qid": "Q5836", "lat": 39.86, "lon": -4.03, "country": "Spain",
         "continent": "Europe", "description": "municipality in central Spain"}


@pytest.mark.parametrize("o, expected", [
    (out(), "no_mention"),
    (out(ner_spans=[SPAN], linked_entities=[{"qid": None}]), "no_link"),
    (out(ner_spans=[SPAN], linked_entities=[{"qid": "Q1"}]), "no_place"),
    (out(ner_spans=[SPAN], linked_entities=[{"qid": "Q5836"}], candidate_places=[PLACE],
         prediction=PLACE), "predicted"),
    (out(ner_spans=[SPAN], error="HTTPError: 500"), "error"),
])
def test_linker_statuses(o, expected):
    assert predict.status_of(cfg("refined"), o) == (expected, "")


def test_search_path_has_no_link_stage():
    """population_wikidata and retrieve_rerank search by name: a mention with
    no hit is no_place, never no_link."""
    assert predict.status_of(cfg("population_wikidata"), out(ner_spans=[SPAN]))[0] == "no_place"


@pytest.mark.parametrize("resolution", [
    {"status": "resolved", "level": "city"}, {"status": "declined", "level": None},
    {"status": "unreadable", "level": None}, {"status": "not_found", "level": "country"},
    {"status": "ambiguous", "level": "continent"}])
def test_llm_status_is_the_resolution(resolution):
    status, level = predict.status_of(cfg("gemma4"), out(resolution=resolution))
    assert status == resolution["status"] and level == (resolution["level"] or "")


@pytest.mark.parametrize("o, expected", [
    (out(ner_spans=[], selection={"no_candidates": True, "abstained": True}), "no_mention"),
    (out(ner_spans=[SPAN], linked_entities=[{"qid": None}],
         selection={"no_candidates": True, "abstained": True}), "no_link"),
    (out(ner_spans=[SPAN], candidate_places=[PLACE], selection={"abstained": True}), "abstained"),
    (out(ner_spans=[SPAN], candidate_places=[PLACE], selection={"violation": "index 12"},
         error="selection violation: index 12"), "violation"),
    (out(ner_spans=[SPAN], candidate_places=[PLACE], selection={"index": 1}, prediction=PLACE),
     "predicted"),
])
def test_grounded_statuses(o, expected):
    assert predict.status_of(cfg("grounded_gemma_linking"), o)[0] == expected


@pytest.mark.parametrize("links, expected", [
    ([{"qid": "Q1"}], ""), ([{"qid": None, "fallback": "not_needed"}, {"qid": "Q1"}], "not_needed"),
    ([{"qid": "Q20", "fallback": "used"}], "used")])
def test_fallback_marker(links, expected):
    assert predict.fallback_of(out(linked_entities=links)) == expected


# ------------------------------------------------------------- end to end

class StandIn:
    """Pipeline stand-in: Toledo is found, the second post has no mention."""

    def __init__(self, cfg, dataset=None):
        assert dataset is None
        self.seen = []

    def prepare(self, texts):
        pass

    def run(self, text):
        self.seen.append(text)
        if "Toledo" in text:
            return out(ner_spans=[SPAN], linked_entities=[{"qid": "Q5836", "mention": "Toledo"}],
                       candidate_places=[PLACE], prediction={**PLACE, "entity": "Toledo"})
        return out()

    def save_llm_cache(self):
        pass


def test_text_only_csv_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(predict, "labels_for", lambda qids: {"Q5836": "Toledo"})
    monkeypatch.setattr("wikidata.save_caches", lambda: None)
    src = tmp_path / "posts.csv"
    src.write_text('text\n"Back in Toledo, Spain"\n"Monday again"\n', encoding="utf-8")
    csv_path, jsonl_path = predict.predict("refined", src, tmp_path / "out",
                                           pipeline_factory=StandIn)
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert list(rows[0]) == predict.COLUMNS
    assert [r["status"] for r in rows] == ["predicted", "no_mention"]
    assert rows[0]["predicted_qid"] == "Q5836" and rows[0]["predicted_label"] == "Toledo"
    assert float(rows[0]["lat"]) == 39.86 and rows[0]["country"] == "Spain"
    records = [json.loads(l) for l in jsonl_path.open(encoding="utf-8")]
    gold_keys = {"place_qid", "gold_latlon", "gold_country", "gold_continent", "condition"}
    assert not any(gold_keys & set(r) for r in records)


def test_only_the_text_column_is_read(tmp_path, monkeypatch):
    """A gold column in the input never reaches the pipeline."""
    monkeypatch.setattr(predict, "labels_for", lambda qids: {})
    monkeypatch.setattr("wikidata.save_caches", lambda: None)
    src = tmp_path / "labelled.csv"
    src.write_text("qid,text,country\nQ5836,Back in Toledo,Spain\n", encoding="utf-8")
    seen = []

    class Recording(StandIn):
        def run(self, text):
            seen.append(text)
            return super().run(text)

    predict.predict("refined", src, tmp_path / "out", pipeline_factory=Recording)
    assert seen == ["Back in Toledo"]
    assert predict.read_texts(src, "text") == ["Back in Toledo"]


def test_missing_text_column_stops(tmp_path):
    src = tmp_path / "bad.csv"
    src.write_text("body\nhello\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no column 'text'"):
        predict.read_texts(src, "text")
