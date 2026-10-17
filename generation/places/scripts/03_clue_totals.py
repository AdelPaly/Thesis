#!/usr/bin/env python3
"""Uncapped clue totals for every place, alongside the capped sample from 02_clues.py.

02_clues.py samples at most --per-property clues per place per query. That cap is
what generation wants -- a bounded, notability-neutral sample -- but it destroys
the column's meaning as a measure of abundance: a place with 30 inbound P131s and
one with 3000 both report 25. Ordinal analysis over clue_count therefore saturates.

This script produces clue_total: the same queries as 02_clues.py (13 by default,
14 with --include-p2848) with the LIMIT removed, counted rather than materialised.

WHAT ACTUALLY GETS QUERIED. A branch whose cached sample came back under the cap
was never limited -- the LIMIT did not bind, so the cached length IS the exact
uncapped count. Only branches sitting AT the cap need a query. That turns
147 places x 13 queries into one query per capped (place, query) pair.

Each branch is counted by its own query, so a timeout costs one branch rather
than the whole place, and a retry re-runs only what failed.

EVENTS IS DIFFERENT. It is not a single SPARQL shape: 02_clues.py samples up to
--event-pool dated things located in the place, then intersects their P31 classes
locally against the ~55k subclasses of Q1656682. That two-step does not collapse
into a COUNT, and the server-side form it replaces (P31/P279* Q1656682) is
precisely the one that times out on large places.

It also means EVENTS carries a SECOND ceiling: the 400-item pool. A cached EVENTS
value below 25 is therefore NOT evidence of exactness the way it is for the other
branches -- the pool may have saturated upstream of the cap. So this
script counts the dated pool for every place, and wherever that pool saturated,
or the EVENTS branch hit the cap, it recomputes EVENTS uncapped: every dated
thing in the place, chunked P31 lookups, local intersection. Where even that
cannot be completed the row is marked rather than silently under-reported.

Output columns written back into the places CSV:

    clue_total   uncapped count over the same query set as 02_clues.py
    capped       true if any branch hit the cap (i.e. clue_count understates)

--include-p2848 must match the flag 02_clues.py was run with. The query set is
built by the same clue_queries.query_keys(), so a (place, query) pair that 02
never ran is reported as missing (exit 2) rather than silently counted as zero.
--no-cache ignores the totals cache and does not write it.

The places CSV is rewritten atomically (tmp + os.replace) after the previous
copy is saved to data/archive/01_places_final.bak-totals.csv.

Usage:
    py -3 scripts/03_clue_totals.py [--input CSV] [--per-property N] [--seed S]
                                    [--event-pool N] [--include-p2848]
                                    [--dry-run] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

from clue_queries import (
    DEFAULT_SEED,
    EVENT_POOL,
    PER_PROPERTY,
    SLEEP_BETWEEN,
    event_pool_where,
    fetch_event_classes,
    fetch_instance_classes,
    flat_where,
    query_keys,
    sports_where,
    works_where,
)
from io_utils import utf8_stdout, write_text_atomic
from jsoncache import JsonCache
from wdqs import QueryError, qid_from_uri, sparql, value_of

# The cache-key builder lives in 02_clues.py (a numeric-prefix name, so it can
# only be imported this way).
clue_cache_key = import_module("02_clues").clue_cache_key

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "01_places_final.csv"
CLUE_CACHE = ROOT / "data" / "02_clues_cache.json"
TOTAL_CACHE = ROOT / "data" / "03_clue_totals_cache.json"
ARCHIVE_DIR_NAME = "archive"
BACKUP_TAG = "bak-totals"

CACHE_VERSION = 1
SPOT_CHECKS = 20          # uncapped counts run against branches known to be exact
POOL_QUERY_TIMEOUT = 300  # seconds: listing every dated thing in a big place
CACHE_SAVE_EVERY = 10     # fresh counts between saves of the totals cache
PASS1_PROGRESS_EVERY = 20 # progress line cadence in pass 1 (1 in N)
PASS2_PROGRESS_EVERY = 25 # progress line cadence in pass 2 (1 in N)
MAX_FAILURES_SHOWN = 20
MAX_MISSING_SHOWN = 10
ERROR_SNIPPET = 120
EXIT_OK = 0
EXIT_PROBLEM = 1
EXIT_BAD_INPUT = 2


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
    """The WHERE fragment 02_clues.py uses for `key` (P-number, SPORTS, WORKS or EVENTS)."""
    if key == "SPORTS":
        return sports_where(place_qid)
    if key == "WORKS":
        return works_where(place_qid)
    if key == "EVENTS":
        return event_pool_where(place_qid)
    return flat_where(place_qid, key)


def branch_total(place_qid: str, key: str) -> int:
    """Uncapped number of distinct clues for one (place, query)."""
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
""", timeout=POOL_QUERY_TIMEOUT)
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
# Cached counting
# --------------------------------------------------------------------------

@dataclass
class CountRunner:
    """Runs count functions through the totals cache, recording failures."""

    cache: JsonCache
    failures: list[tuple[str, str, str]] = field(default_factory=list)
    done: int = 0

    def count(self, qid: str, key: str, kind: str, fn) -> int | None:
        """Cached value for (kind, qid, key), else fn(); None if the query failed."""
        ck = f"{kind}|{qid}|{key}"
        if ck in self.cache:
            return self.cache.get(ck)
        try:
            value = fn()
        except QueryError as exc:
            self.failures.append((qid, key, str(exc)))
            print(f"      ! {key} failed for {qid}: {exc}", flush=True)
            return None
        self.cache.set(ck, value)
        self.done += 1
        if self.done % CACHE_SAVE_EVERY == 0:
            self.cache.save()
        time.sleep(SLEEP_BETWEEN)
        return value


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_places(path: Path) -> tuple[list[str], list[dict]]:
    """The places CSV as (field names, rows)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def load_clue_cache(path: Path) -> dict:
    """The 02_clues cache entries (key -> list of clue records)."""
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------

@dataclass
class Plan:
    """Which branches are exact already and which need a query."""

    exact: dict[tuple[str, str], int] = field(default_factory=dict)
    to_query: list[tuple[str, str]] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)


def make_plan(rows: list[dict], keys: list[str], clue_cache: dict, cap: int,
              seed: str, event_pool: int) -> Plan:
    """Split every (place, query) pair into exact / needs-a-count / missing."""
    plan = Plan()
    for row in rows:
        qid = row["wikidata_id"]
        for key in keys:
            ck = clue_cache_key(qid, key, cap, seed, event_pool)
            if ck not in clue_cache:
                plan.missing.append((qid, key))
                continue
            n = len(clue_cache[ck])
            if key == "EVENTS":
                continue          # decided by pool saturation, in pass 2
            if n >= cap:
                plan.to_query.append((qid, key))
            else:
                plan.exact[(qid, key)] = n
    return plan


def count_capped_branches(counter: CountRunner, plan: Plan) -> dict[tuple[str, str], int]:
    """Pass 1: uncapped counts for the capped non-EVENTS branches, plus the exact ones."""
    print(f"\npass 1: {len(plan.to_query)} uncapped branch counts", flush=True)
    totals: dict[tuple[str, str], int] = dict(plan.exact)
    for index, (qid, key) in enumerate(plan.to_query, start=1):
        if index % PASS1_PROGRESS_EVERY == 1:
            print(f"  [{index}/{len(plan.to_query)}]", flush=True)
        value = counter.count(qid, key, "count", lambda q=qid, k=key: branch_total(q, k))
        if value is not None:
            totals[(qid, key)] = value
    return totals


def count_events(counter: CountRunner, rows: list[dict], clue_cache: dict, cap: int,
                 seed: str, event_pool: int, totals: dict[tuple[str, str], int]) -> None:
    """Passes 2 and 3: EVENTS pool sizes, then uncapped EVENTS where the pool or cap bound."""
    print(f"\npass 2: EVENTS pool sizes for {len(rows)} places", flush=True)
    pool_sizes: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        qid = row["wikidata_id"]
        if index % PASS2_PROGRESS_EVERY == 1:
            print(f"  [{index}/{len(rows)}]", flush=True)
        value = counter.count(qid, "POOL", "pool", lambda q=qid: branch_total(q, "EVENTS"))
        if value is not None:
            pool_sizes[qid] = value

    needs_events: list[str] = []
    for row in rows:
        qid = row["wikidata_id"]
        ck = clue_cache_key(qid, "EVENTS", cap, seed, event_pool)
        sampled = len(clue_cache[ck])
        pool = pool_sizes.get(qid)
        if pool is None:
            needs_events.append(qid)            # unknown pool: recompute to be safe
        elif sampled >= cap or pool > event_pool:
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
            counter.failures.append((qid, "EVENTS", "event subclass tree unavailable"))
            continue
        value = counter.count(qid, "EVENTS", "events",
                              lambda q=qid: events_total(q, event_classes))
        if value is not None:
            totals[(qid, "EVENTS")] = value


def spot_check(counter: CountRunner, exact: dict[tuple[str, str], int]) -> int:
    """Pass 4: re-count branches known exact; returns how many disagreed."""
    print(f"\npass 4: {SPOT_CHECKS} spot checks on branches known to be exact", flush=True)
    spot = [(q, k) for (q, k), n in sorted(exact.items()) if n > 0][:SPOT_CHECKS]
    bad = 0
    for qid, key in spot:
        value = counter.count(qid, key, "spot", lambda q=qid, k=key: branch_total(q, k))
        if value is None:
            continue
        if value != exact[(qid, key)]:
            bad += 1
            print(f"  MISMATCH {qid} {key}: cache {exact[(qid, key)]} vs count {value}")
    print(f"  {len(spot) - bad}/{len(spot)} agree with the cached sample length")
    return bad


def assemble(rows: list[dict], keys: list[str], totals: dict[tuple[str, str], int],
             clue_cache: dict, cap: int, seed: str, event_pool: int) -> None:
    """Fill clue_total and capped on every row (in place)."""
    for row in rows:
        qid = row["wikidata_id"]
        branch = {k: totals.get((qid, k)) for k in keys}
        row["clue_total"] = ("" if any(v is None for v in branch.values())
                             else str(sum(branch.values())))
        row["capped"] = str(any(
            len(clue_cache[clue_cache_key(qid, k, cap, seed, event_pool)]) >= cap
            for k in keys
        )).lower()


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def backup_to_archive(path: Path) -> Path:
    """Copy `path` to <its dir>/archive/<stem>.bak-totals<suffix>; returns the copy."""
    archive = path.parent / ARCHIVE_DIR_NAME
    archive.mkdir(parents=True, exist_ok=True)
    backup = archive / f"{path.stem}.{BACKUP_TAG}{path.suffix}"
    shutil.copy2(path, backup)
    return backup


def write_places(path: Path, fields: list[str], rows: list[dict]) -> Path:
    """Back the CSV up to archive/, then rewrite it atomically with clue_total/capped."""
    fields = list(fields)
    for offset, column in enumerate(("clue_total", "capped"), start=1):
        if column not in fields:
            fields.insert(fields.index("clue_count") + offset, column)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    backup = backup_to_archive(path)
    write_text_atomic(path, buffer.getvalue())
    return backup


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_plan(rows: list[dict], plan: Plan) -> None:
    """The query plan summary."""
    print(f"places                      : {len(rows)}")
    print(f"branches exact from cache   : {len(plan.exact)}")
    print(f"branches needing a count    : {len(plan.to_query)}")
    print(f"EVENTS branches             : {len(rows)} (pool size checked for each)")


def print_report(path: Path, rows: list[dict], bad: int,
                 failures: list[tuple[str, str, str]]) -> None:
    """Final summary: rows written, spot-check mismatches, failed branches."""
    incomplete = {qid for qid, _, _ in failures}
    print(f"\nwrote {path.name}: {len(rows)} rows")
    if bad:
        print(f"!! {bad} spot checks disagreed -- clue_total semantics are suspect")
    if failures:
        print(f"!! {len(failures)} branch queries failed across {len(incomplete)} places; "
              f"their clue_total is blank. Re-run to retry just those.")
        for qid, key, err in failures[:MAX_FAILURES_SHOWN]:
            print(f"   {qid} {key}: {err[:ERROR_SNIPPET]}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--per-property", type=int, default=PER_PROPERTY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--event-pool", type=int, default=EVENT_POOL)
    parser.add_argument("--include-p2848", action="store_true",
                        help="count P2848 too; must match the flag 02_clues.py ran with")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the totals cache and do not write it")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the query plan and exit without querying")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load -> plan -> count -> assemble -> write -> report."""
    args = parse_args(argv)
    cap = args.per_property

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return EXIT_BAD_INPUT
    if not CLUE_CACHE.exists():
        print(f"clue cache not found: {CLUE_CACHE}; run 02_clues.py first", file=sys.stderr)
        return EXIT_BAD_INPUT

    clue_cache = load_clue_cache(CLUE_CACHE)
    fields, rows = load_places(args.input)
    keys = list(query_keys(args.include_p2848))

    plan = make_plan(rows, keys, clue_cache, cap, args.seed, args.event_pool)
    if plan.missing:
        print(f"!! {len(plan.missing)} (place, query) pairs absent from the clue cache; "
              f"run 02_clues.py to completion first (same --include-p2848 setting)",
              file=sys.stderr)
        for qid, key in plan.missing[:MAX_MISSING_SHOWN]:
            print(f"   {qid} {key}", file=sys.stderr)
        return EXIT_BAD_INPUT

    print_plan(rows, plan)
    if args.dry_run:
        return EXIT_OK

    cache = JsonCache(TOTAL_CACHE, CACHE_VERSION, enabled=not args.no_cache)
    counter = CountRunner(cache)
    totals = count_capped_branches(counter, plan)
    count_events(counter, rows, clue_cache, cap, args.seed, args.event_pool, totals)
    bad = spot_check(counter, plan.exact)
    cache.save()

    assemble(rows, keys, totals, clue_cache, cap, args.seed, args.event_pool)
    write_places(args.input, fields, rows)
    print_report(args.input, rows, bad, counter.failures)
    return EXIT_PROBLEM if (bad or counter.failures) else EXIT_OK


if __name__ == "__main__":
    utf8_stdout()
    raise SystemExit(main())
