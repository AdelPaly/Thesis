#!/usr/bin/env python3
"""Uncapped clue totals for every place, alongside the capped sample from 02_clues.py.

02_clues.py samples at most --per-property clues per place per query. That cap is
what generation wants -- a bounded, notability-neutral sample -- but it destroys
the column's meaning as a measure of abundance: a place with 30 inbound P131s and
one with 3000 both report 25. Ordinal analysis over clue_count therefore saturates.

This script produces clue_total: the same 14 queries with the LIMIT removed,
counted rather than materialised.

WHAT ACTUALLY GETS QUERIED. A branch whose cached sample came back under the cap
was never limited -- the LIMIT did not bind, so the cached length IS the exact
uncapped count. Only branches sitting AT the cap need a query. That turns
147 places x 14 queries into one query per capped (place, query) pair.

Each branch is counted by its own query, so a timeout costs one branch rather
than the whole place, and a retry re-runs only what failed.

EVENTS IS DIFFERENT. It is not a single SPARQL shape: 02_clues.py samples up to
--event-pool dated things located in the place, then intersects their P31 classes
locally against the ~55k subclasses of Q1656682. That two-step does not collapse
into a COUNT, and the server-side form it replaces (P31/P279* Q1656682) is
precisely the one that times out on large places.

It also means EVENTS carries a SECOND ceiling: the 400-item pool. A cached EVENTS
value below 25 is therefore NOT evidence of exactness the way it is for the other
thirteen branches -- the pool may have saturated upstream of the cap. So this
script counts the dated pool for every place, and wherever that pool saturated,
or the EVENTS branch hit the cap, it recomputes EVENTS uncapped: every dated
thing in the place, chunked P31 lookups, local intersection. Where even that
cannot be completed the row is marked rather than silently under-reported.

Output columns written back into the places CSV:

    clue_total   uncapped count over the same 14 queries
    capped       true if any branch hit the cap (i.e. clue_count understates)

Usage:
    py -3 scripts/03_clue_totals.py [--input CSV] [--per-property N] [--seed S]
                                    [--event-pool N] [--dry-run] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_clues = import_module("02_clues")

QueryError = _clues.QueryError
sparql = _clues.sparql
value_of = _clues.value_of
qid_from_uri = _clues.qid_from_uri
flat_where = _clues.flat_where
sports_where = _clues.sports_where
works_where = _clues.works_where
event_pool_where = _clues.event_pool_where
fetch_event_classes = _clues.fetch_event_classes
fetch_instance_classes = _clues.fetch_instance_classes
PROPERTIES = _clues.PROPERTIES
OPTIONAL_PROPERTIES = _clues.OPTIONAL_PROPERTIES
PATTERN_LABELS = _clues.PATTERN_LABELS
SLEEP_BETWEEN = _clues.SLEEP_BETWEEN

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "01_places_final.csv"
CLUE_CACHE = ROOT / "data" / "02_clues_cache.json"
TOTAL_CACHE = ROOT / "data" / "03_clue_totals_cache.json"

# The query set 02_clues.py was run with, in its report order.
QUERY_KEYS = list(PROPERTIES) + list(OPTIONAL_PROPERTIES) + list(PATTERN_LABELS)

CACHE_VERSION = 1
SPOT_CHECKS = 20          # uncapped counts run against branches known to be exact
POOL_CHUNK_SLEEP = 0.25


# --------------------------------------------------------------------------
# Counting
# --------------------------------------------------------------------------

def count_query(where: str, place_qid: str) -> str:
    """The 02_clues.py subquery with the LIMIT dropped and COUNT in its place.

    The `wikibase:sitelinks` triple and the self-exclusion FILTER are kept so
    this counts exactly the population the sampler drew from, not a near-miss.
    """
    return f"""
SELECT (COUNT(DISTINCT ?clue) AS ?n) WHERE {{
  {where}
  ?clue wikibase:sitelinks ?sitelinks .
  FILTER(?clue != wd:{place_qid})
}}
"""


def where_for(place_qid: str, key: str) -> str:
    if key == "SPORTS":
        return sports_where(place_qid)
    if key == "WORKS":
        return works_where(place_qid)
    if key == "EVENTS":
        return event_pool_where(place_qid)
    return flat_where(place_qid, key)


def branch_total(place_qid: str, key: str) -> int:
    rows = sparql(count_query(where_for(place_qid, key), place_qid))
    if not rows:
        return 0
    return int(value_of(rows[0], "n") or 0)


def event_pool_qids(place_qid: str) -> list[str]:
    """Every dated thing in the place -- the uncapped form of the EVENTS pool."""
    rows = sparql(f"""
SELECT DISTINCT ?clue WHERE {{
  {event_pool_where(place_qid)}
  ?clue wikibase:sitelinks ?sitelinks .
  FILTER(?clue != wd:{place_qid})
}}
""", timeout=300)
    out = []
    for row in rows:
        clue = value_of(row, "clue")
        if clue:
            out.append(qid_from_uri(clue))
    return out


def events_total(place_qid: str, event_classes: set[str]) -> int:
    """EVENTS with no pool limit and no result limit, same two-step as 02_clues.py."""
    pool = event_pool_qids(place_qid)
    if not pool:
        return 0
    classes = fetch_instance_classes(pool)
    return sum(1 for q in pool if classes.get(q, set()) & event_classes)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

def load_cache(use_cache: bool) -> dict:
    if not use_cache or not TOTAL_CACHE.exists():
        return {}
    try:
        payload = json.loads(TOTAL_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("version") != CACHE_VERSION:
        return {}
    return payload.get("entries", {})


def save_cache(entries: dict) -> None:
    TOTAL_CACHE.write_text(
        json.dumps({"version": CACHE_VERSION, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def clue_cache_key(qid: str, key: str, per_property: int, seed: str, pool: int) -> str:
    ck = f"{qid}|{key}|{per_property}|{seed}"
    if key == "EVENTS":
        ck += f"|{pool}"
    return ck


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--per-property", type=int, default=25)
    parser.add_argument("--seed", default=_clues.DEFAULT_SEED)
    parser.add_argument("--event-pool", type=int, default=_clues.EVENT_POOL)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the query plan and exit without querying")
    args = parser.parse_args(argv)

    cap = args.per_property

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    if not CLUE_CACHE.exists():
        print(f"clue cache not found: {CLUE_CACHE}; run 02_clues.py first", file=sys.stderr)
        return 2

    clue_cache = json.loads(CLUE_CACHE.read_text(encoding="utf-8"))["entries"]

    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    # ---- plan: which branches are exact already, which need a query ----
    exact: dict[tuple[str, str], int] = {}
    to_query: list[tuple[str, str]] = []
    missing: list[tuple[str, str]] = []
    for row in rows:
        qid = row["wikidata_id"]
        for key in QUERY_KEYS:
            ck = clue_cache_key(qid, key, cap, args.seed, args.event_pool)
            if ck not in clue_cache:
                missing.append((qid, key))
                continue
            n = len(clue_cache[ck])
            if key == "EVENTS":
                continue          # decided by pool saturation, below
            if n >= cap:
                to_query.append((qid, key))
            else:
                exact[(qid, key)] = n

    if missing:
        print(f"!! {len(missing)} (place, query) pairs absent from the clue cache; "
              f"run 02_clues.py to completion first", file=sys.stderr)
        for qid, key in missing[:10]:
            print(f"   {qid} {key}", file=sys.stderr)
        return 2

    print(f"places                      : {len(rows)}")
    print(f"branches exact from cache   : {len(exact)}")
    print(f"branches needing a count    : {len(to_query)}")
    print(f"EVENTS branches             : {len(rows)} (pool size checked for each)")
    if args.dry_run:
        return 0

    use_cache = not args.no_cache
    cache = load_cache(use_cache)
    failures: list[tuple[str, str, str]] = []
    done = 0

    def cached_count(qid: str, key: str, kind: str, fn) -> int | None:
        nonlocal done
        ck = f"{kind}|{qid}|{key}"
        if use_cache and ck in cache:
            return cache[ck]
        try:
            value = fn()
        except QueryError as exc:
            failures.append((qid, key, str(exc)))
            print(f"      ! {key} failed for {qid}: {exc}", flush=True)
            return None
        cache[ck] = value
        done += 1
        if use_cache and done % 10 == 0:
            save_cache(cache)
        time.sleep(SLEEP_BETWEEN)
        return value

    # ---- pass 1: uncapped counts for capped non-EVENTS branches ----
    print(f"\npass 1: {len(to_query)} uncapped branch counts", flush=True)
    totals: dict[tuple[str, str], int] = dict(exact)
    for index, (qid, key) in enumerate(to_query, start=1):
        if index % 20 == 1:
            print(f"  [{index}/{len(to_query)}]", flush=True)
        value = cached_count(qid, key, "count", lambda q=qid, k=key: branch_total(q, k))
        if value is not None:
            totals[(qid, key)] = value

    # ---- pass 2: EVENTS pool sizes, then uncapped EVENTS where needed ----
    print(f"\npass 2: EVENTS pool sizes for {len(rows)} places", flush=True)
    pool_sizes: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        qid = row["wikidata_id"]
        if index % 25 == 1:
            print(f"  [{index}/{len(rows)}]", flush=True)
        value = cached_count(qid, "POOL", "pool",
                             lambda q=qid: branch_total(q, "EVENTS"))
        if value is not None:
            pool_sizes[qid] = value

    needs_events: list[str] = []
    for row in rows:
        qid = row["wikidata_id"]
        ck = clue_cache_key(qid, "EVENTS", cap, args.seed, args.event_pool)
        sampled = len(clue_cache[ck])
        pool = pool_sizes.get(qid)
        if pool is None:
            needs_events.append(qid)            # unknown pool: recompute to be safe
        elif sampled >= cap or pool > args.event_pool:
            needs_events.append(qid)
        else:
            totals[(qid, "EVENTS")] = sampled   # pool never saturated, cap never bound

    print(f"\npass 3: uncapped EVENTS for {len(needs_events)} places "
          f"(pool saturated or branch at cap)", flush=True)
    event_classes: set[str] = set()
    if needs_events:
        print("  fetching event subclass tree ...", flush=True)
        try:
            event_classes = fetch_event_classes()
            print(f"  {len(event_classes)} event classes", flush=True)
        except QueryError as exc:
            print(f"  ! could not fetch event classes ({exc})", file=sys.stderr)

    for index, qid in enumerate(needs_events, start=1):
        print(f"  [{index}/{len(needs_events)}] {qid} (pool {pool_sizes.get(qid, '?')})",
              flush=True)
        if not event_classes:
            failures.append((qid, "EVENTS", "event subclass tree unavailable"))
            continue
        value = cached_count(qid, "EVENTS", "events",
                             lambda q=qid: events_total(q, event_classes))
        if value is not None:
            totals[(qid, "EVENTS")] = value

    # ---- pass 4: validate the query shape against branches already known exact ----
    print(f"\npass 4: {SPOT_CHECKS} spot checks on branches known to be exact", flush=True)
    spot = [(q, k) for (q, k), n in sorted(exact.items()) if n > 0][:SPOT_CHECKS]
    bad = 0
    for qid, key in spot:
        value = cached_count(qid, key, "spot", lambda q=qid, k=key: branch_total(q, k))
        if value is None:
            continue
        if value != exact[(qid, key)]:
            bad += 1
            print(f"  MISMATCH {qid} {key}: cache {exact[(qid, key)]} vs count {value}")
    print(f"  {len(spot) - bad}/{len(spot)} agree with the cached sample length")

    if use_cache:
        save_cache(cache)

    # ---- assemble ----
    incomplete = {qid for qid, key, _ in failures}
    for row in rows:
        qid = row["wikidata_id"]
        branch = {k: totals.get((qid, k)) for k in QUERY_KEYS}
        row["clue_total"] = ("" if any(v is None for v in branch.values())
                             else str(sum(branch.values())))
        row["capped"] = str(any(
            len(clue_cache[clue_cache_key(qid, k, cap, args.seed, args.event_pool)]) >= cap
            for k in QUERY_KEYS
        )).lower()

    for offset, column in enumerate(("clue_total", "capped"), start=1):
        if column not in fields:
            fields.insert(fields.index("clue_count") + offset, column)

    with args.input.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    # ---- report ----
    print(f"\nwrote {args.input.name}: {len(rows)} rows")
    if bad:
        print(f"!! {bad} spot checks disagreed -- clue_total semantics are suspect")
    if failures:
        print(f"!! {len(failures)} branch queries failed across {len(incomplete)} places; "
              f"their clue_total is blank. Re-run to retry just those.")
        for qid, key, err in failures[:20]:
            print(f"   {qid} {key}: {err[:120]}")
    return 1 if (bad or failures) else 0


if __name__ == "__main__":
    raise SystemExit(main())
