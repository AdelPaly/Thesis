"""Unit tests for the demonym fallback (`ner.fallback_types`, revision 3) in
src/pipeline.py, and for the two config fields revision 3 adds.

    python -m pytest tests/test_fallback.py

No model and no network: NER, the linker and the resolver are replaced with
stand-ins, so only the pipeline's own logic runs.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import pipeline  # noqa: E402
import wikidata  # noqa: E402

TEXT = "French tourists met Anna in Paris"


def span(text, kind):
    start = TEXT.find(text)
    return {"text": text, "type": kind, "score": 0.99, "start": start, "end": start + len(text)}


# mention -> linked QID, and QID -> the places it resolves to
LINKS = {"French": "Qfrance", "Anna": "Qanna", "Paris": "Qparis", "Nobody": "Qnobody"}
PLACES = {"Qfrance": ["Qfrance"], "Qparis": ["Qparis"], "Qanna": ["Qhome", "Qwork"],
          "Qnobody": []}


@pytest.fixture
def run(monkeypatch):
    """run(spans, cfg) -> (output, linker calls). The linker records the
    mentions of each call."""
    calls = []

    def fake_link(self, text, spans):
        calls.append([s["text"] for s in spans])
        return [{"mention": s["text"], "mention_type": s["type"], "qid": LINKS.get(s["text"]),
                 "confidence": 0.9, "candidates": []} for s in spans]

    def fake_resolve(qid, text, kind, rule="chain_v1"):
        return [{"qid": p, "entity": text, "entity_type": kind, "lat": 0.0, "lon": 0.0,
                 "description": p, "country": "", "continent": ""} for p in PLACES.get(qid, [])]

    monkeypatch.setattr(pipeline.Pipeline, "_link", fake_link)
    monkeypatch.setattr(pipeline.Pipeline, "_rank", lambda self, text, cands: cands)
    monkeypatch.setattr(wikidata, "resolve_qid_to_candidates", fake_resolve)

    def _run(spans, cfg):
        calls.clear()
        monkeypatch.setattr(pipeline.Pipeline, "_ner", lambda self, text: [dict(s) for s in spans])
        out = pipeline.PipelineOutput()
        pipeline.Pipeline(cfg)._run_linking_path(TEXT, out)
        return out, list(calls)
    return _run


@pytest.fixture(scope="module")
def refined():
    return config.load_config(ROOT / "configs" / "refined.yaml")


def without_fallback(cfg):
    return replace(cfg, ner=replace(cfg.ner, fallback_types=None))


def test_refined_is_configured_with_the_fallback(refined):
    assert refined.ner.entity_types == ["LOC", "PER", "ORG"]
    assert refined.ner.fallback_types == ["MISC"]
    assert refined.linking.place_rule == "all_places"


def test_fallback_mention_is_not_linked_when_another_gives_a_place(run, refined):
    spans = [span("French", "MISC"), span("Paris", "LOC")]
    out, calls = run(spans, refined)
    assert calls == [["Paris"]]                       # one linker call, main mentions only
    assert out.prediction["qid"] == "Qparis"
    assert out.linked_entities[0]["fallback"] == "not_needed"
    assert out.linked_entities[0]["qid"] is None


def test_fallback_is_used_when_nothing_else_gives_a_place(run, refined):
    spans = [span("French", "MISC"), span("Nobody", "PER")]
    out, calls = run(spans, refined)
    assert calls == [["Nobody"], ["French"]]          # second call only for the fallback
    assert out.prediction["qid"] == "Qfrance"
    assert out.linked_entities[0]["fallback"] == "used"
    assert out.linked_entities[0]["qid"] == "Qfrance"
    assert "fallback" not in out.linked_entities[1]


def test_fallback_without_main_mentions(run, refined):
    out, calls = run([span("French", "MISC")], refined)
    assert calls == [[], ["French"]] or calls == [["French"]]
    assert out.prediction["qid"] == "Qfrance"


@pytest.mark.parametrize("spans", [
    [span("French", "MISC"), span("Paris", "LOC")],
    [span("French", "MISC"), span("Nobody", "PER")],
    [span("Anna", "PER"), span("French", "MISC"), span("Paris", "LOC")],
])
def test_main_results_are_those_of_a_run_without_the_fallback(run, refined, spans):
    with_fb, _ = run(spans, refined)
    main = [s for s in spans if s["type"] != "MISC"]
    plain, _ = run(main, without_fallback(refined))
    main_links = [l for l in with_fb.linked_entities if "fallback" not in l]
    assert main_links == plain.linked_entities
    if plain.prediction:          # the fallback changes nothing when the main mentions answer
        assert with_fb.candidate_places == plain.candidate_places
        assert with_fb.prediction == plain.prediction


def test_linked_entities_stay_aligned_with_ner_spans(run, refined):
    spans = [span("French", "MISC"), span("Anna", "PER"), span("Paris", "LOC")]
    out, _ = run(spans, refined)
    assert [s["text"] for s in out.ner_spans] == ["French", "Anna", "Paris"]
    assert [l["mention"] for l in out.linked_entities] == ["French", "Anna", "Paris"]
    assert len(out.linked_entities) == len(out.ner_spans)


def test_all_places_of_one_person_all_reach_the_reranker(run, refined):
    out, _ = run([span("Anna", "PER")], refined)
    assert [c["qid"] for c in out.candidate_places] == ["Qhome", "Qwork"]


def test_top_linked_deduplicates_and_shares_the_entity_confidence(refined):
    link_a = {"qid": "Qanna", "confidence": 0.9}
    link_b = {"qid": "Qparis", "confidence": 0.5}
    pairs = [(link_a, {"qid": "Qhome"}), (link_a, {"qid": "Qwork"}),
             (link_b, {"qid": "Qparis"}), (link_b, {"qid": "Qhome"})]
    top = pipeline.Pipeline(refined)._top_linked(pairs, top_k=10)
    assert [c["qid"] for c in top] == ["Qhome", "Qwork", "Qparis"]


# ------------------------------------------------------------ config fields

def raw(changes=None):
    import yaml
    d = yaml.safe_load((ROOT / "configs" / "refined.yaml").read_text(encoding="utf-8"))
    for (section, key), value in (changes or {}).items():
        d[section][key] = value
    return d


def build(d):
    return config.from_dict(d, source_path="configs/refined.yaml")


def test_defaults_leave_the_hash_unchanged():
    """A field that holds its default is left out of the hash (_LATE_FIELDS),
    so writing it out explicitly changes nothing."""
    base = raw()
    del base["ner"]["fallback_types"]
    del base["linking"]["place_rule"]
    explicit = raw({("ner", "fallback_types"): None, ("linking", "place_rule"): "chain_v1"})
    assert build(base).config_hash == build(explicit).config_hash


def test_reported_refined_hash_is_kept_by_chain_v1_without_fallback():
    """With both fields at their defaults, refined hashes as its reported run
    (results/predictions/refined__synthetic.jsonl)."""
    import json
    with (ROOT / "results" / "predictions" / "refined__synthetic.jsonl").open(encoding="utf-8") as f:
        reported = json.loads(f.readline())["config_hash"]
    base = raw()
    del base["ner"]["fallback_types"]
    del base["linking"]["place_rule"]
    for k in ("design", "description"):        # documentation, not hashed
        base[k] = ""
    assert build(base).config_hash == reported


def test_setting_a_field_moves_the_hash():
    base = raw()
    assert build(base).config_hash != build(raw({("linking", "place_rule"): "chain_v1"})).config_hash


@pytest.mark.parametrize("changes, message", [
    ({("linking", "place_rule"): "residence_first"}, "place_rule"),
    ({("ner", "fallback_types"): []}, "fallback_types"),
    ({("ner", "entity_types"): None}, "needs ner.entity_types"),
    ({("ner", "fallback_types"): ["MISC", "LOC"]}, "overlap"),
    ({("linking", "backend"): "wikidata_search"}, "refined or blink"),
])
def test_invalid_fallback_and_rule_are_refused(changes, message):
    with pytest.raises(config.ConfigError, match=message):
        build(raw(changes))
