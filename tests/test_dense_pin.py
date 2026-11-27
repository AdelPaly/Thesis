"""The pinned dense index (`dense.index_signature`, src/dense_index.py), added
after the development run of 26-27 November.

    python -m pytest tests/test_dense_pin.py

No network and no encoder: a small index file is written to a temporary
directory and the build path is replaced with one that fails the test.
"""
from __future__ import annotations

import hashlib
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import dense_index  # noqa: E402

SIG = "0123456789abcdef"


def _cfg(sig=SIG, **dense):
    d = {"index_signature": sig, "encoder": "enc", "pool": "gold_plus_answers_described",
         "index_fields": ["label"], "passage_prefix": "passage: ", "query_prefix": "query: ",
         **dense}
    return SimpleNamespace(dense=SimpleNamespace(**d))


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(dense_index, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(dense_index, "_dataset_gold_qids", lambda: {"Q1"})

    def no_build(*a, **k):
        raise AssertionError("a pinned index must never be built")

    monkeypatch.setattr(dense_index, "build_catalogue", no_build)
    monkeypatch.setattr(dense_index, "build_pool", no_build)
    payload = {"qids": ["Q1", "Q2"], "matrix": np.eye(2, dtype="float32"),
               "places": {"Q1": {"qid": "Q1"}, "Q2": {"qid": "Q2"}},
               "meta": {"signature": SIG, "encoder": "enc", "pool": "gold_plus_answers_described",
                        "index_fields": ["label"], "passage_prefix": "passage: ", "n_places": 2}}
    path = tmp_path / f"index_{SIG}.pkl"
    path.write_bytes(pickle.dumps(payload))
    path.with_suffix(".sha256").write_text(hashlib.sha256(path.read_bytes()).hexdigest() + "\n")
    return path


def test_pinned_index_is_loaded_as_it_is(index_dir):
    idx = dense_index.load_or_build(_cfg())
    assert idx.qids == ["Q1", "Q2"] and idx.meta["signature"] == SIG


def test_missing_pinned_index_is_a_config_error_and_nothing_is_built(index_dir):
    index_dir.unlink()
    with pytest.raises(config.ConfigError, match="never rebuilt"):
        dense_index.load_or_build(_cfg())


def test_changed_file_is_a_config_error(index_dir):
    index_dir.write_bytes(index_dir.read_bytes() + b"x")
    with pytest.raises(config.ConfigError, match="not the pinned index"):
        dense_index.load_or_build(_cfg())


def test_missing_recorded_sha256_is_a_config_error(index_dir):
    index_dir.with_suffix(".sha256").unlink()
    with pytest.raises(config.ConfigError, match="no recorded sha256"):
        dense_index.load_or_build(_cfg())


def test_settings_that_differ_from_the_index_are_a_config_error(index_dir):
    with pytest.raises(config.ConfigError, match="other settings"):
        dense_index.load_or_build(_cfg(index_fields=["label", "description"]))


def test_pinned_index_is_cached_by_its_signature_without_reading_the_pool(index_dir, monkeypatch):
    monkeypatch.setattr(dense_index, "_indexes", {})
    a = dense_index.get_index(_cfg())
    assert dense_index.get_index(_cfg()) is a


def test_index_signature_is_a_late_field():
    """Unset, it leaves every configuration's hash where it was; set, the
    configuration hashes differently."""
    raw = config.load_config(ROOT / "configs" / "dense_retrieval.yaml")
    assert raw.dense.index_signature == "3530d31fa83aacb7"
    unpinned = config.from_dict({**_raw("dense_retrieval"),
                                 "dense": {**_raw("dense_retrieval")["dense"], "index_signature": None}},
                                source_path="x")
    bare = _raw("dense_retrieval")
    bare["dense"] = {k: v for k, v in bare["dense"].items() if k != "index_signature"}
    assert config.from_dict(bare).config_hash == unpinned.config_hash != raw.config_hash


def test_bad_signature_is_refused():
    raw = _raw("dense_retrieval")
    raw["dense"]["index_signature"] = "not-hex"
    with pytest.raises(config.ConfigError):
        config.from_dict(raw)


def test_pool_configurations_are_pinned_to_the_same_index():
    for name in ("dense_retrieval", "grounded_gemma"):
        assert config.load_config(ROOT / "configs" / f"{name}.yaml").dense.index_signature \
            == "3530d31fa83aacb7"


def test_record_carries_the_index_signature():
    import runner
    cfg = config.load_config(ROOT / "configs" / "grounded_gemma.yaml")
    assert runner.models_of(cfg)["dense_index"] == "3530d31fa83aacb7"
    assert "dense_index" not in runner.models_of(config.load_config(ROOT / "configs" / "gemma4.yaml"))


def _raw(name):
    import yaml
    return yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))
