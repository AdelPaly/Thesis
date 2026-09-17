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
import os
import pickle
from pathlib import Path

import numpy as np

import wikidata as wd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(__file__).resolve().parent / ".dense_index"
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"


def _dataset_gold_qids():
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


def _predicted_qids():
    """Every place any configuration has proposed, including retired runs."""
    out = set()
    if not PREDICTIONS_DIR.exists():
        return out
    paths = list(PREDICTIONS_DIR.glob("*.jsonl"))
    paths += list((PREDICTIONS_DIR / "retired").glob("*.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
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


def build_pool(pool_spec):
    """QIDs in the candidate pool, before the coordinate filter."""
    if pool_spec == "gold_only":
        return sorted(_dataset_gold_qids())
    if pool_spec == "gold_plus_predicted":
        return sorted(_dataset_gold_qids() | _predicted_qids())
    raise ValueError(f"unknown dense.pool {pool_spec!r}")


def index_text(qid, entity, fields, geo, aliases, passage_prefix=""):
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


def _signature(cfg):
    """Cache key: anything that changes what the index contains or means."""
    d = cfg.dense
    payload = {
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
            from ranking import _load
            self._model = _load(self.encoder_name)
        return self._model

    def search(self, text, top_k=10):
        """Top-k nearest places by cosine similarity."""
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


def _cache_path(cfg):
    return CACHE_DIR / f"index_{_signature(cfg)}.pkl"


def load_or_build(cfg, verbose=True):
    """Load the cached index, or build it once and cache it."""
    path = _cache_path(cfg)
    if path.exists():
        with path.open("rb") as f:
            payload = pickle.load(f)
        if verbose:
            m = payload["meta"]
            print(f"  dense index loaded: {m['n_places']} places, "
                  f"encoder={m['encoder']}, pool={m['pool']}")
        return DenseIndex(payload["qids"], payload["matrix"], payload["places"],
                          payload["meta"], cfg.dense.encoder, cfg.dense.query_prefix)

    d = cfg.dense
    if verbose:
        print(f"  building dense index (pool={d.pool}, encoder={d.encoder})...")
    pool = build_pool(d.pool)
    if verbose:
        print(f"    pool before coordinate filter: {len(pool)} QIDs")

    entities = wd._fetch_entities(pool)
    unfetched = [q for q in pool if not entities.get(q)]
    placed = {q: e for q, e in entities.items() if e and wd._get_coords(e)}
    if verbose:
        print(f"    with coordinates: {len(placed)}")
    if unfetched:
        # A transient 429 makes an entity look exactly like an entity with no
        # coordinates, which is how the retired results lost data quietly.
        # Retry once before accepting the loss, and count what is still missing.
        if verbose:
            print(f"    {len(unfetched)} QID(s) returned nothing; retrying once")
        entities.update(wd._fetch_entities(unfetched))
        unfetched = [q for q in pool if not entities.get(q)]
        placed = {q: e for q, e in entities.items() if e and wd._get_coords(e)}
        if verbose:
            print(f"    after retry: {len(placed)} placed, {len(unfetched)} unfetched")

    geo = wd._resolve_country_continent(placed)
    aliases = wd.fetch_aliases(list(placed)) if "aliases" in (d.index_fields or []) else {}

    qids, texts, places = [], [], {}
    for qid, ent in placed.items():
        lat, lon = wd._get_coords(ent)
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
        texts.append(index_text(qid, ent, d.index_fields or [], geo, aliases,
                                d.passage_prefix))

    from ranking import _load
    model = _load(d.encoder)
    if verbose:
        print(f"    embedding {len(texts)} passages...")
    matrix = model.encode(texts, batch_size=64, normalize_embeddings=True,
                          show_progress_bar=False)
    matrix = np.asarray(matrix, dtype="float32")

    # The gold places must all be in the index. If one is missing, every item
    # about that place is unretrievable and the tier's numbers are wrong in a
    # way nothing downstream would reveal -- so this fails loudly rather than
    # producing a quietly degraded index.
    gold = _dataset_gold_qids()
    missing_gold = sorted(gold - set(qids))
    if missing_gold:
        raise RuntimeError(
            f"{len(missing_gold)} gold place(s) missing from the dense index, "
            f"e.g. {missing_gold[:5]}. Items about them could never be "
            f"retrieved. Check for fetch failures and rebuild.")

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
        "signature": _signature(cfg),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({"qids": qids, "matrix": matrix, "places": places,
                     "meta": meta}, f)
    os.replace(tmp, path)
    wd.save_caches()
    if verbose:
        print(f"    wrote {path.name} ({len(qids)} places)")
    return DenseIndex(qids, matrix, places, meta, d.encoder, d.query_prefix)


_indexes = {}


def get_index(cfg):
    key = _signature(cfg)
    if key not in _indexes:
        _indexes[key] = load_or_build(cfg)
    return _indexes[key]
