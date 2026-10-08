"""Dense retrieval index over candidate places (tier 3).

No mention detection, no entity linking, no LLM: embed the post, embed a text
description of every candidate place, take nearest neighbours.

Why the tier exists: without it, "NER-free" and "LLM" are the same category in
the results, because every system lacking a mention-detection stage happens to
be an LLM. This is the missing cell -- it separates "no mention detection" from
"parametric knowledge".

The encoder is the one already used by the reranker (`rerank.model`), so a
comparison against tier 2 is not confounded by a different encoder.

WHAT GOES INTO THE INDEX is a configuration parameter (`dense.index_fields`)
and is recorded in the index metadata, because it defines what this tier can
possibly retrieve. A place whose description says nothing about football cannot
be retrieved from a post about football.

THE CANDIDATE POOL defines the difficulty. Retrieving from 150 places is not
the same task as retrieving from 100,000, so the pool size is recorded in the
metadata and reported. The `gold_plus_predicted` pool is:

    every gold place across all configured datasets
  + every QID any configuration ever proposed as a candidate place
  + every QID in a linker candidate set
  filtered to entities that actually have coordinates

Two properties of that pool must be stated when reporting this tier:

* The gold place is in the pool BY CONSTRUCTION. So "gold not in the top-k" is
  a ranking failure, never a coverage failure, and 100% recall is reachable in
  principle. This is a closed-set retrieval evaluation.
* The distractors are endogenous -- they are the places other systems actually
  confused for the gold, which makes them hard negatives, but the pool lacks
  the long tail of world geography. Scores here are therefore optimistic
  relative to retrieval over a full gazetteer.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
from pathlib import Path

import numpy as np

import wikidata as wd

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".dense_index"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"


def _dataset_gold_qids() -> set[str]:
    from config import load_datasets
    out = set()
    for spec in load_datasets().values():
        path = spec.resolved_path()
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                qid = (row.get("qid") or "").strip()
                if qid.startswith("Q"):
                    out.add(qid)
    return out


def _predicted_qids() -> set[str]:
    """Every place any configuration has proposed."""
    out = set()
    if not PREDICTIONS_DIR.exists():
        return out
    for path in PREDICTIONS_DIR.glob("*.jsonl"):
        with path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    log.warning("skipping corrupt JSON line %d in %s", lineno, path.name)
                    continue
                for cand in rec.get("candidate_places") or []:
                    if cand.get("qid"):
                        out.add(cand["qid"])
                for link in rec.get("linked_entities") or []:
                    if link.get("qid"):
                        out.add(link["qid"])
                    for cand in link.get("candidates") or []:
                        if cand.get("qid"):
                            out.add(cand["qid"])
    return out


def build_pool(pool_spec: str) -> list[str]:
    """QIDs in the candidate pool, before the coordinate filter."""
    if pool_spec == "gold_only":
        return sorted(_dataset_gold_qids())
    if pool_spec == "gold_plus_predicted":
        return sorted(_dataset_gold_qids() | _predicted_qids())
    raise ValueError(f"unknown dense.pool {pool_spec!r}")


def index_text(qid: str, entity: dict, fields, geo: dict, aliases: dict,
               passage_prefix: str = "") -> str:
    """The text that represents a place in the index."""
    label = entity.get("labels", {}).get("en", {}).get("value", "") or qid
    description = entity.get("descriptions", {}).get("en", {}).get("value", "")
    country, continent = geo.get(qid, ("", ""))

    parts = []
    if "label" in fields:
        parts.append(label)
    if "description" in fields and description:
        parts.append(description)
    if "aliases" in fields:
        alt = aliases.get(qid) or []
        if alt:
            parts.append("also known as " + ", ".join(alt[:8]))
    if "country" in fields and country:
        parts.append(f"in {country}")
    if "continent" in fields and continent:
        parts.append(continent)
    return passage_prefix + ". ".join(p for p in parts if p)


def _signature(cfg, pool=None) -> str:
    """Cache key: anything that changes what the index contains or means,
    including the pool contents (they depend on prediction files on disk)."""
    d = cfg.dense
    if pool is None:
        pool = build_pool(d.pool)
    payload = {
        "pool_hash": hashlib.sha256(",".join(pool).encode()).hexdigest(),
        "encoder": d.encoder,
        "pool": d.pool,
        "index_fields": list(d.index_fields or []),
        "passage_prefix": d.passage_prefix,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class DenseIndex:
    def __init__(self, qids, matrix, places, meta, encoder_name, query_prefix):
        self.qids = qids
        self.matrix = matrix              # (n, d) float32, L2-normalised
        self.places = places              # qid -> candidate dict
        self.meta = meta
        self.encoder_name = encoder_name
        self.query_prefix = query_prefix
        self._model = None

    def _encoder(self):
        if self._model is None:
            from ranking import load_encoder
            self._model = load_encoder(self.encoder_name)
        return self._model

    def search(self, text: str, top_k: int = 10) -> list[dict]:
        """Top-k nearest places by cosine similarity."""
        if len(self.qids) == 0 or top_k <= 0:
            return []
        vec = self._encoder().encode(self.query_prefix + text,
                                     normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        scores = self.matrix @ vec
        if top_k >= len(scores):
            order = np.argsort(-scores)
        else:
            top = np.argpartition(-scores, top_k)[:top_k]
            order = top[np.argsort(-scores[top])]
        out = []
        for i in order[:top_k]:
            qid = self.qids[i]
            place = dict(self.places[qid])
            place["score"] = float(scores[i])
            out.append(place)
        return out


def _cache_path(signature: str) -> Path:
    return CACHE_DIR / f"index_{signature}.pkl"


def _check_gold(qids, gold):
    """The gold places must all be in the index. If one is missing, every item
    about that place is unretrievable and the tier's numbers are wrong in a
    way nothing downstream would reveal -- so this fails loudly rather than
    producing a quietly degraded index."""
    missing_gold = sorted(gold - set(qids))
    if missing_gold:
        raise RuntimeError(
            f"{len(missing_gold)} gold place(s) missing from the dense index, "
            f"e.g. {missing_gold[:5]}. Items about them could never be "
            f"retrieved. Check for fetch failures and rebuild.")


class Catalogue:
    """The candidate places of a pool: their QIDs in index order, their place
    records, and the text that represents each one (no encoder prefix)."""

    def __init__(self, qids, places, texts, pool_size, unfetched):
        self.qids = qids
        self.places = places
        self.texts = texts
        self.pool_size = pool_size
        self.unfetched = unfetched


def build_catalogue(d, pool=None, gold=None) -> Catalogue:
    """Fetch the pool's places and build their index texts.

    Shared by every configuration that ranks the same candidate places from
    the same descriptions -- dense retrieval, the character n-gram floor and
    the grid language model -- so that none of them sees a different pool."""
    if pool is None:
        pool = build_pool(d.pool)
    if gold is None:
        gold = _dataset_gold_qids()
    log.info("pool before coordinate filter: %d QIDs", len(pool))

    entities = wd.fetch_entities(pool)
    unfetched = [q for q in pool if not entities.get(q)]
    placed = {q: e for q, e in entities.items() if e and wd.get_coords_of(e)}
    log.info("with coordinates: %d", len(placed))
    if unfetched:
        # A transient 429 makes an entity look exactly like an entity with no
        # coordinates, which is how the retired results lost data quietly.
        # Retry once before accepting the loss, and count what is still missing.
        log.info("%d QID(s) returned nothing; retrying once", len(unfetched))
        entities.update(wd.fetch_entities(unfetched))
        unfetched = [q for q in pool if not entities.get(q)]
        placed = {q: e for q, e in entities.items() if e and wd.get_coords_of(e)}
        log.info("after retry: %d placed, %d unfetched", len(placed), len(unfetched))

    _check_gold(placed, gold)

    geo = wd.resolve_country_continent(placed)
    aliases = wd.fetch_aliases(list(placed)) if "aliases" in (d.index_fields or []) else {}

    qids, texts, places = [], [], {}
    for qid, ent in placed.items():
        lat, lon = wd.get_coords_of(ent)
        country, continent = geo.get(qid, ("", ""))
        label = ent.get("labels", {}).get("en", {}).get("value", "") or qid
        places[qid] = {
            "qid": qid,
            "entity": label,
            "entity_type": "PLACE",
            "lat": lat, "lon": lon,
            "country": country, "continent": continent,
            "description": ent.get("descriptions", {}).get("en", {}).get("value", label),
        }
        qids.append(qid)
        texts.append(index_text(qid, ent, d.index_fields or [], geo, aliases))
    wd.save_caches()
    return Catalogue(qids, places, texts, len(pool), unfetched)


_catalogues: dict = {}


def get_catalogue(d) -> Catalogue:
    """The catalogue for a dense section, built once per process and pool."""
    pool = build_pool(d.pool)
    key = (hashlib.sha256(",".join(pool).encode()).hexdigest(),
           d.pool, tuple(d.index_fields or []))
    if key not in _catalogues:
        _catalogues[key] = build_catalogue(d, pool)
    return _catalogues[key]


def load_or_build(cfg) -> DenseIndex:
    """Load the cached index, or build it once and cache it."""
    d = cfg.dense
    pool = build_pool(d.pool)
    signature = _signature(cfg, pool)
    path = _cache_path(signature)
    gold = _dataset_gold_qids()
    if path.exists():
        with path.open("rb") as f:
            payload = pickle.load(f)
        _check_gold(payload["qids"], gold)
        m = payload["meta"]
        log.info("dense index loaded: %d places, encoder=%s, pool=%s",
                 m["n_places"], m["encoder"], m["pool"])
        return DenseIndex(payload["qids"], payload["matrix"], payload["places"],
                          payload["meta"], d.encoder, d.query_prefix)

    log.info("building dense index (pool=%s, encoder=%s)...", d.pool, d.encoder)
    cat = build_catalogue(d, pool, gold)
    qids, places, unfetched = cat.qids, cat.places, cat.unfetched
    texts = [d.passage_prefix + t for t in cat.texts]

    from ranking import load_encoder
    model = load_encoder(d.encoder)
    log.info("embedding %d passages...", len(texts))
    matrix = model.encode(texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False)
    matrix = np.asarray(matrix, dtype="float32")

    meta = {
        "encoder": d.encoder,
        "pool": d.pool,
        "pool_size_before_coord_filter": len(pool),
        "n_pool_unfetched": len(unfetched),
        "n_gold": len(gold),
        "n_gold_in_index": len(gold & set(qids)),
        "n_places": len(qids),
        "index_fields": list(d.index_fields or []),
        "passage_prefix": d.passage_prefix,
        "query_prefix": d.query_prefix,
        "example_passage": texts[0] if texts else "",
        "signature": signature,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"qids": qids, "matrix": matrix, "places": places,
                     "meta": meta}, f)
    os.replace(tmp, path)
    wd.save_caches()
    log.info("wrote %s (%d places)", path.name, len(qids))
    return DenseIndex(qids, matrix, places, meta, d.encoder, d.query_prefix)


_indexes = {}


def clear_indexes() -> None:
    """Drop in-process cached indexes (test / multi-config reset hook)."""
    _indexes.clear()


def get_index(cfg) -> DenseIndex:
    key = _signature(cfg)
    if key not in _indexes:
        _indexes[key] = load_or_build(cfg)
    return _indexes[key]
