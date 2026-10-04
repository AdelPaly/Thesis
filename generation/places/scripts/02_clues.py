#!/usr/bin/env python3
"""Collect Wikidata "clue" entities for every place in a place CSV (default data/01_places_final.csv).

A clue is any entity that points AT the place. Two kinds of query find them.

FLAT PROPERTIES -- one hop, `?clue wdt:PROP wd:PLACE`:

    P19    place of birth          people born there
    P20    place of death          people who died there
    P276   location                things located there
    P159   headquarters location   organisations based there
    P937   work location           people who worked there
    P138   named after             things named after it
    P1071  location of creation    works/objects made there
    P495   country of origin       works originating there
    P1435  heritage designation    listed/protected things there
    P131   located in admin entity  things inside the place

    P2848 (Wi-Fi access) is NOT queried by default. It is an outbound facility
    property; probed inbound over a 15-place sample and over all 150 places it
    returned zero rows every time. --include-p2848 turns it back on.

PATTERNS -- multi-hop shapes that catch associations no single property does:

    SPORTS  clubs/teams whose home venue (P115) is the place or sits in it,
            plus sports organisations (P279* of Q4438121) sited there
    EVENTS  festivals, competitions and other instances of an event class
            (P279* of Q1656682) whose location is the place
    WORKS   books, films and songs set in (P840) or filmed in (P915) the place
            (works *named after* the place already arrive via P138)

EVENTS runs in two steps. A server-side `P31/P279* Q1656682` walk times out on
large places, so instead a cheap query samples dated things located in the
place, and their P31 classes are intersected locally against the event subclass
set -- fetched once per run in a single ~10s query.

SAMPLING. Each query returns a RANDOM sample of up to --per-property items,
ordered by MD5(CONCAT(STR(?clue), seed)) with a seed fixed by --seed. The old
ORDER BY DESC(?sitelinks) biased every clue set toward the most famous
associated entities, which contaminated difficulty control: easy places got the
easiest clues. Sitelink counts are still recorded, just no longer used to rank.

Output: data/02_clues.jsonl, one JSON object per clue:
    place_qid, place_name, clue_qid, clue_label, property, property_label,
    sitelinks

`property` is a P-number for a flat property, or SPORTS / EVENTS / WORKS for a
pattern. A place whose queries all fail is reported as failed, not as "0 clues".

The query specs and fetchers live in clue_queries.py; the WDQS client in
wdqs.py. This file is the CLI and re-exports the names the later numbered
scripts still import from it.

Exit code: 0 on success, 1 if any place lost one or more queries (the output is
still written, but thinner than a complete run), 2 on bad input.
--allow-partial keeps exit 0 in that case. --no-cache ignores the cache and
does not write it.

Usage:
    py -3 scripts/02_clues.py [--limit N] [--per-property N] [--seed S]
                              [--include-p2848] [--no-cache] [--allow-partial]
                              [--input CSV] [--out JSONL]
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from clue_queries import (  # noqa: F401  (re-exported for 03/04/04b/05/05b)
    DEFAULT_SEED,
    EVENT_CLASS,
    EVENT_POOL,
    OPTIONAL_PROPERTIES,
    PATTERN_LABELS,
    PER_PROPERTY,
    PROPERTIES,
    SLEEP_BETWEEN,
    SPORTS_ORG_CLASS,
    VALUES_CHUNK,
    event_pool_where,
    fetch_event_classes,
    fetch_events,
    fetch_instance_classes,
    fetch_query,
    fetch_sampled,
    flat_where,
    query_keys as build_query_keys,
    sample_key,
    sampled_query,
    sports_where,
    works_where,
)
from io_utils import utf8_stdout, write_jsonl_atomic
from jsoncache import JsonCache, cache_key
from wdqs import (  # noqa: F401  (re-exported for 03/04/04b/05/05b)
    ATTEMPTS,
    ENDPOINT,
    HTTP_TIMEOUT,
    USER_AGENT,
    QueryError,
    qid_from_uri,
    sparql,
    value_of,
)

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "01_places_final.csv"
OUTPUT_JSONL = ROOT / "data" / "02_clues.jsonl"
CACHE_PATH = ROOT / "data" / "02_clues_cache.json"

CACHE_VERSION = 2     # bumped: sampling changed, v1 entries are ordered by fame
CACHE_SAVE_EVERY = 5  # places between cache saves, so an interrupt loses little
BUCKET_FEW_MAX = 2    # "few clues" means 1..2; 3+ is "many"
EXIT_OK = 0
EXIT_PARTIAL = 1
EXIT_BAD_INPUT = 2


# --------------------------------------------------------------------------
# Input / cache
# --------------------------------------------------------------------------

def read_places(path: Path, limit: int | None) -> list[dict]:
    """Places from the CSV as [{'qid', 'name'}], skipping rows with no wikidata_id."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    places = []
    for row in rows:
        qid = (row.get("wikidata_id") or "").strip()
        if not qid:
            continue
        places.append({"qid": qid, "name": (row.get("place_name") or "").strip()})
    if limit is not None:
        places = places[:limit]
    return places


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_table(places: list[dict], counts: dict[str, int], failures: dict[str, str]) -> None:
    """Per-place clue counts, flagging places whose every query failed."""
    name_width = max([len(p["name"]) for p in places] + [len("place")])
    qid_width = max([len(p["qid"]) for p in places] + [len("qid")])
    header = f"{'place':<{name_width}}  {'qid':<{qid_width}}  {'clues':>5}"
    print()
    print(header)
    print("-" * len(header))
    for place in places:
        note = ""
        if place["qid"] in failures:
            note = f"  ! {failures[place['qid']]}"
        print(
            f"{place['name']:<{name_width}}  {place['qid']:<{qid_width}}  "
            f"{counts.get(place['qid'], 0):>5}{note}"
        )


def print_buckets(places: list[dict], counts: dict[str, int]) -> None:
    """How many places have zero, few or many clues."""
    zero = sum(1 for p in places if counts.get(p["qid"], 0) == 0)
    few = sum(1 for p in places if 1 <= counts.get(p["qid"], 0) <= BUCKET_FEW_MAX)
    many = sum(1 for p in places if counts.get(p["qid"], 0) > BUCKET_FEW_MAX)
    print()
    print(f"places with 0 clues   : {zero}")
    print(f"places with 1-2 clues : {few}")
    print(f"places with 3+ clues  : {many}")
    print(f"total places          : {len(places)}")


def print_per_property(records: list[dict], keys: dict[str, str]) -> None:
    """Records and distinct places touched, per query. Zero rows are the point."""
    print()
    print(f"{'query':<8} {'meaning':<34} {'records':>8} {'places':>7}")
    print("-" * 60)
    for key, label in keys.items():
        hits = [r for r in records if r["property"] == key]
        places = {r["place_qid"] for r in hits}
        print(f"{key:<8} {label:<34} {len(hits):>8} {len(places):>7}")


def report(places: list[dict], counts: dict[str, int], failures: dict[str, str],
           lost: dict[str, list[str]], records: list[dict],
           keys: dict[str, str], out: Path) -> None:
    """Console summary after the output file is written."""
    print_table(places, counts, failures)
    print_buckets(places, counts)
    print_per_property(records, keys)
    unlabelled = sum(1 for r in records if not r["clue_label"])
    print(f"\nwrote {len(records)} clue records to {out}")
    if unlabelled:
        print(f"  ({unlabelled} have no English label)")
    if failures:
        print(f"  ({len(failures)} places returned nothing because every query failed)")
    if lost:
        print(f"  ({len(lost)} places lost one or more queries; their clue sets are incomplete)")


# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------

def clue_cache_key(qid: str, key: str, per_property: int, seed: str, event_pool: int) -> str:
    """Cache key for one (place, query); EVENTS also depends on the pool size."""
    parts = [qid, key, per_property, seed]
    if key == "EVENTS":
        parts.append(event_pool)
    return cache_key(*parts)


def collect_place(place: dict, keys: dict[str, str], properties: dict[str, str],
                  cache: JsonCache, event_classes: set[str],
                  args: argparse.Namespace) -> tuple[list[dict], list[str]]:
    """Run (or recall) every query for one place. Returns (records, error messages)."""
    qid = place["qid"]
    place_records: list[dict] = []
    errors: list[str] = []

    for key in keys:
        ck = clue_cache_key(qid, key, args.per_property, args.seed, args.event_pool)
        cached = cache.get(ck)
        if cached is not None:
            clues = cached
        elif key == "EVENTS" and not event_classes:
            errors.append("EVENTS: event classes unavailable")
            continue
        else:
            try:
                clues = fetch_query(key, qid, properties, args.per_property, args.seed,
                                    args.event_pool, event_classes)
            except QueryError as exc:
                errors.append(f"{key}: {exc}")
                print(f"      ! {key} failed: {exc}", flush=True)
                continue
            cache.set(ck, clues)
            time.sleep(SLEEP_BETWEEN)

        for clue in clues:
            place_records.append(
                {
                    "place_qid": qid,
                    "place_name": place["name"],
                    "clue_qid": clue["clue_qid"],
                    "clue_label": clue["clue_label"],
                    "property": clue["property"],
                    "property_label": clue["property_label"],
                    "sitelinks": clue["sitelinks"],
                }
            )

    # Deterministic but notability-neutral order, matching the sampling.
    place_records.sort(key=lambda r: (sample_key(r["clue_qid"], args.seed), r["property"]))
    return place_records, errors


def load_event_classes() -> set[str]:
    """Fetch the event subclass set once; an empty set disables EVENTS."""
    print("fetching event subclass tree ...", flush=True)
    try:
        event_classes = fetch_event_classes()
    except QueryError as exc:
        print(f"  ! could not fetch event classes ({exc}); EVENTS disabled", file=sys.stderr)
        return set()
    print(f"  {len(event_classes)} event classes", flush=True)
    return event_classes


def collect_all(places: list[dict], keys: dict[str, str], properties: dict[str, str],
                cache: JsonCache, args: argparse.Namespace
                ) -> tuple[list[dict], dict[str, int], dict[str, str], dict[str, list[str]]]:
    """Run every place. Returns (records, counts, all-failed places, places with lost queries)."""
    print(f"{len(places)} places x {len(keys)} queries "
          f"(<= {args.per_property} clues each, seed {args.seed!r})")
    event_classes = load_event_classes()

    records: list[dict] = []
    counts: dict[str, int] = {}
    failures: dict[str, str] = {}
    lost: dict[str, list[str]] = {}

    for index, place in enumerate(places, start=1):
        qid = place["qid"]
        print(f"[{index}/{len(places)}] {place['name']} ({qid})", flush=True)
        place_records, errors = collect_place(place, keys, properties, cache, event_classes, args)
        records.extend(place_records)
        counts[qid] = len(place_records)
        if errors:
            lost[qid] = errors
            if not place_records:
                failures[qid] = f"{len(errors)}/{len(keys)} queries failed"
        if index % CACHE_SAVE_EVERY == 0:
            cache.save()
    cache.save()
    return records, counts, failures, lost


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process the first N places")
    parser.add_argument(
        "--per-property",
        type=int,
        default=PER_PROPERTY,
        help=f"max clues sampled per place per query (default {PER_PROPERTY})",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"sampling seed mixed into the MD5 ordering (default {DEFAULT_SEED!r})",
    )
    parser.add_argument(
        "--event-pool",
        type=int,
        default=EVENT_POOL,
        help=f"candidates drawn before the event-class filter (default {EVENT_POOL})",
    )
    parser.add_argument(
        "--include-p2848",
        action="store_true",
        help="also query P2848 (Wi-Fi access); returns nothing, off by default",
    )
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the cache and do not write it")
    parser.add_argument("--allow-partial", action="store_true",
                        help="exit 0 even if some places lost queries (default: exit 1)")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out", type=Path, default=OUTPUT_JSONL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load places -> collect clues -> write -> report."""
    args = parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return EXIT_BAD_INPUT

    places = read_places(args.input, args.limit)
    if not places:
        print("no places with a wikidata_id in the input", file=sys.stderr)
        return EXIT_BAD_INPUT

    keys = build_query_keys(args.include_p2848)
    properties = {k: v for k, v in keys.items() if k not in PATTERN_LABELS}
    cache = JsonCache(CACHE_PATH, CACHE_VERSION, enabled=not args.no_cache)

    records, counts, failures, lost = collect_all(places, keys, properties, cache, args)

    write_jsonl_atomic(args.out, records)
    report(places, counts, failures, lost, records, keys, args.out)

    if lost and not args.allow_partial:
        print("exit 1: some places lost queries (use --allow-partial to accept)", file=sys.stderr)
        return EXIT_PARTIAL
    return EXIT_OK


if __name__ == "__main__":
    utf8_stdout()
    raise SystemExit(main())
