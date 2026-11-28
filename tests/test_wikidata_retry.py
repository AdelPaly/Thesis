"""Transient Wikidata API errors are retried inside wikidata._api_get; other API
errors are raised at once. No network is used.

    python -m pytest tests/test_wikidata_retry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import wikidata as wd  # noqa: E402


class FakeResponse:
    status_code = 200
    headers: dict = {}

    def __init__(self, body):
        self.body = body

    def json(self):
        return self.body


@pytest.fixture
def replies(monkeypatch):
    queue = []
    monkeypatch.setattr(wd.requests, "get", lambda *a, **k: FakeResponse(queue.pop(0)))
    monkeypatch.setattr(wd.time, "sleep", lambda s: None)
    monkeypatch.setattr(wd._rate_limiter, "wait", lambda: None)
    return queue


@pytest.mark.parametrize("code", ["cirrussearch-too-busy-error", "maxlag", "ratelimited"])
def test_transient_error_is_retried(replies, code):
    replies += [{"error": {"code": code}}, {"error": {"code": code}}, {"search": ["ok"]}]
    assert wd._api_get({"action": "wbsearchentities"}) == {"search": ["ok"]}
    assert replies == []


def test_other_api_error_is_not_retried(replies):
    replies += [{"error": {"code": "no-such-entity"}}, {"search": ["ok"]}]
    with pytest.raises(wd.WikidataError, match="no-such-entity"):
        wd._api_get({"action": "wbgetentities"})
    assert len(replies) == 1


def test_gives_up_after_max_attempts(replies):
    replies += [{"error": {"code": "cirrussearch-too-busy-error"}}] * wd.MAX_ATTEMPTS
    with pytest.raises(wd.WikidataError, match="after"):
        wd._api_get({"action": "wbsearchentities"})
