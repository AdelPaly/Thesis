"""Geoparser adapter (paradigm: geoparser).

A geoparser is a *toponym resolver*: it detects place names in text and
resolves them against a gazetteer. It is not an entity-linking system and it
has no way to reach a location that is only implied. On conditions where the
text contains no toponym it cannot score above zero by construction. That is
the expected result and it is the point of including one, not a bug.

Backend: mordecai3 (Halterman 2023), spaCy-based, reported at 184 km mean
error and 94 Accuracy@161km on GeoWebNews.

Runtime requirement, verified against the published wheel
(mordecai3-3.0.0a0): mordecai3 talks to a local **Elasticsearch 7.x instance
holding the GeoNames gazetteer** at localhost:9200 --
`mordecai3/elastic_utilities.py` builds `Elasticsearch(hosts=['localhost'],
port=9200)` and searches `index="geonames"`, and `geoparse.py` raises
`ConnectionError("Could not locate Elasticsearch. Are you sure it's
running?")` without one. It is pip-installable but not self-contained.

It also cannot be installed into this environment as it stands. Its pins are
torch<2.0, pandas<2.0, numpy<2.0 and elasticsearch<8.0, against the installed
torch 2.10 / pandas 2.2 / numpy 2.4 that flair, refined and
sentence-transformers are built on; and torch<2.0 has no cp313 wheel, so the
pin is unsatisfiable on Python 3.13 at any cost. Running it means a separate
Python <=3.11 environment plus an Elasticsearch container with the GeoNames
index loaded.

This adapter is therefore written and wired to the config, but unexercised:
`load()` raises a GeoparserUnavailable naming the missing piece rather than
guessing. When the environment exists, the configuration runs unchanged.
"""
from __future__ import annotations

_parsers = {}


class GeoparserUnavailable(RuntimeError):
    """The configured geoparser cannot run in this environment."""


def _load_mordecai3(spacy_model):
    try:
        from mordecai3 import Geoparser
    except ImportError as e:
        raise GeoparserUnavailable(
            "mordecai3 is not installed. It pins torch<2.0 / pandas<2.0 / "
            "numpy<2.0, which conflicts with the installed torch 2.10 / "
            "pandas 2.2 / numpy 2.4, and torch<2.0 has no cp313 wheel, so the "
            "pin cannot be satisfied on this Python 3.13. Install it in a "
            "separate Python <=3.11 environment."
        ) from e
    try:
        return Geoparser(nlp=spacy_model) if spacy_model else Geoparser()
    except Exception as e:                                # noqa: BLE001
        raise GeoparserUnavailable(
            f"mordecai3 failed to start ({type(e).__name__}: {e}). It needs a "
            f"local Elasticsearch 7.x at localhost:9200 with the GeoNames "
            f"gazetteer loaded into an index named 'geonames'."
        ) from e


def load(backend, spacy_model=None):
    key = (backend, spacy_model)
    if key not in _parsers:
        if backend == "mordecai3":
            _parsers[key] = _load_mordecai3(spacy_model)
        else:
            raise GeoparserUnavailable(f"unknown geoparser backend {backend!r}")
    return _parsers[key]


def _pick(results):
    """Mordecai3 returns one resolved entry per detected toponym. Take the
    highest-scoring one as the item-level prediction, keeping the rest as the
    candidate list so the choice is auditable."""
    scored = []
    for ent in results.get("geolocated_ents", []):
        lat, lon = ent.get("lat"), ent.get("lon")
        if lat is None or lon is None:
            continue
        scored.append({
            "entity": ent.get("search_name") or ent.get("name"),
            "entity_type": "LOC",
            "qid": None,                       # GeoNames ids, not QIDs
            "geonames_id": str(ent.get("geonameid")) if ent.get("geonameid") else None,
            "lat": float(lat), "lon": float(lon),
            "country": ent.get("country_code3") or "",
            "continent": "",
            "description": ent.get("feature_code") or "",
            "score": float(ent.get("score", 0.0) or 0.0),
        })
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored


def geoparse(text, backend, spacy_model=None):
    """Return (toponym_spans, candidates). The winner is candidates[0]."""
    parser = load(backend, spacy_model)
    results = parser.geoparse_doc(text)
    spans = [{
        "text": ent.get("search_name") or ent.get("name") or "",
        "type": "LOC",
        "score": float(ent.get("score", 0.0) or 0.0),
        "start": ent.get("start_char"),
        "end": ent.get("end_char"),
    } for ent in results.get("geolocated_ents", [])]
    return spans, _pick(results)
