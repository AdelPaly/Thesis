#!/usr/bin/env python3
"""Re-derive area_km2 for every place from Wikidata, rank-aware.

THE BUG THIS FIXES. An item can carry several P2046 statements -- a municipality
boundary and an urban core, a historic extent and a current one, or simply two
sourced measurements. The earlier extraction took whichever one came back first,
so Las Cruces landed on 0.3 km2 when its best-rank value is ~199.5, and Borobudur
took 0.6 against a best-rank 0.6257. An arbitrary pick is not a small error: the
values differ by three orders of magnitude on the worst row.

SELECTION RULE, in order:
  1. statements marked `a wikibase:BestRank` -- Wikidata's own answer to "which
     of these is current", i.e. preferred rank if any exists, else normal rank
  2. among those, the LARGEST, on the reasoning that where a place has both an
     administrative and a built-up-area figure, the administrative boundary is
     the one the rest of this dataset means by "the place"
  3. if nothing is marked best rank, the largest non-deprecated value

Deprecated statements are excluded throughout. Values come from psn: (normalised)
so they arrive in square metres regardless of the unit each statement uses;
km2 = m2 / 1e6.

This script WRITES to the places CSV -- unlike verify_usable.py. It copies the
file to data/archive/<name>.bak-areas.csv first, replaces it atomically (tmp +
os.replace), and reports every change. Rows whose item has no P2046 are left
blank, which is what a blank in this column is defined to mean.

It also runs the size pass over the corrected values and reports what crosses
--size-limit. It does NOT remove anything: in the shipped file rows above
1000 km2 were kept and merely noted, so removal is a dataset decision with
knock-on effects on clue_count and clue_total, not a mechanical cleanup.

Usage:
    py -3 scripts/04_fix_areas.py [--csv PATH] [--size-limit 1000] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

from io_utils import utf8_stdout, write_text_atomic
from verify_places import BATCH_SIZE, SLEEP_BETWEEN, chunks, format_number, parse_float
from wdqs import CONTACT, QueryError, qid_from_uri, sparql, value_of

# Identifies this tool to WDQS; passed per call, never assigned onto another module.
USER_AGENT = (
    f"places-areas/1.0 (area re-derivation for a place dataset; contact: {CONTACT}) "
    f"python-urllib"
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "data" / "01_places_final.csv"
ARCHIVE_DIR_NAME = "archive"
DEFAULT_SIZE_LIMIT_KM2 = 1000.0
SQ_M_PER_SQ_KM = 1_000_000
LARGE_AREA_NOTE = "large area"
CLEARED_NAME_WIDTH = 26
NOTE_WIDTH = 44
EXIT_OK = 0
EXIT_BAD_INPUT = 2

Statements = list[tuple[float, bool]]
Change = tuple[dict, str, str, str]   # (row, before, after, how)


def query_areas(qids: list[str]) -> dict[str, Statements]:
    """Every non-deprecated normalised P2046 for a batch, with its best-rank flag.

    FILTER NOT EXISTS is anchored on ?statement, which is already bound from
    ?item through the VALUES clause, so the pattern stays bounded.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?area ?best WHERE {{
  VALUES ?item {{ {values} }}
  ?item p:P2046 ?statement .
  ?statement psn:P2046/wikibase:quantityAmount ?area .
  OPTIONAL {{ ?statement a wikibase:BestRank . BIND(true AS ?best) }}
  FILTER NOT EXISTS {{ ?statement wikibase:rank wikibase:DeprecatedRank }}
}}
"""
    out: dict[str, Statements] = defaultdict(list)
    for row in sparql(query, user_agent=USER_AGENT):
        item = value_of(row, "item")
        area = value_of(row, "area")
        if not item or area is None:
            continue
        out[qid_from_uri(item)].append((float(area), value_of(row, "best") == "true"))
    return out


def choose_area(statements: Statements) -> tuple[float | None, str]:
    """Apply the selection rule. Returns (km2 or None, how it was chosen)."""
    if not statements:
        return None, "no P2046"
    best = [value for value, is_best in statements if is_best]
    pool, how = (best, "best rank") if best else (
        [value for value, _ in statements], "no best rank; largest non-deprecated")
    chosen = max(pool)
    if len(pool) > 1:
        how += f"; largest of {len(pool)}"
    return chosen / SQ_M_PER_SQ_KM, how


# --------------------------------------------------------------------------
# Load / fetch
# --------------------------------------------------------------------------

def load_places(path: Path) -> tuple[list[str], list[dict]]:
    """The places CSV as (field names, rows)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    return fields, rows


def fetch_areas(qids: list[str]) -> tuple[dict[str, Statements], set[str]]:
    """Query every place in batches. Returns (areas by qid, qids whose batch failed)."""
    areas: dict[str, Statements] = {}
    failed: set[str] = set()
    total = (len(qids) + BATCH_SIZE - 1) // BATCH_SIZE
    for index, chunk in enumerate(chunks(qids, BATCH_SIZE), start=1):
        print(f"  batch {index}/{total} ({len(chunk)})", flush=True)
        try:
            areas.update(query_areas(chunk))
        except QueryError as exc:
            print(f"    ! batch {index} failed: {exc}", flush=True)
            failed.update(chunk)
        time.sleep(SLEEP_BETWEEN)
    return areas, failed


# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------

def apply_areas(rows: list[dict], areas: dict[str, Statements], failed: set[str]
                ) -> tuple[list[Change], list[tuple[dict, str]], list[tuple[dict, str]], int]:
    """Update area_km2 in place. Returns (changed, backfilled, cleared, multi-statement count)."""
    changed: list[Change] = []
    backfilled: list[tuple[dict, str]] = []
    cleared: list[tuple[dict, str]] = []
    multi = 0

    for row in rows:
        qid = (row.get("wikidata_id") or "").strip()
        if qid in failed:
            continue
        statements = areas.get(qid, [])
        if len(statements) > 1:
            multi += 1
        km2, how = choose_area(statements)
        before = (row.get("area_km2") or "").strip()
        after = "" if km2 is None else format_number(km2)
        if before == after:
            continue
        row["area_km2"] = after
        if not before and after:
            backfilled.append((row, how))
        elif before and not after:
            cleared.append((row, how))
        else:
            changed.append((row, before, after, how))
    return changed, backfilled, cleared, multi


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_changes(changed: list[Change], backfilled: list[tuple[dict, str]],
                  cleared: list[tuple[dict, str]], multi: int) -> None:
    """Counts and per-row detail for every change."""
    print(f"\nitems with more than one non-deprecated P2046: {multi}")
    print(f"values corrected : {len(changed)}")
    print(f"blanks backfilled: {len(backfilled)}")
    print(f"values cleared   : {len(cleared)}")

    if changed:
        width = max(len(r["place_name"]) for r, _, _, _ in changed)
        print(f"\n{'place':<{width}}  {'origin':<12} {'before':>10} {'after':>12}  how")
        print("-" * (width + 60))
        for row, before, after, how in sorted(
            changed, key=lambda c: -abs((parse_float(c[2]) or 0) - (parse_float(c[1]) or 0))
        ):
            print(f"{row['place_name']:<{width}}  {row.get('origin', ''):<12} {before:>10} {after:>12}  {how}")

    if backfilled:
        width = max(len(r["place_name"]) for r, _ in backfilled)
        print("\nbackfilled blanks:")
        for row, how in backfilled:
            print(f"  {row['place_name']:<{width}}  {row.get('origin', ''):<12} "
                  f"{row['area_km2']:>12} km2  ({how})")

    if cleared:
        print("\ncleared (file had a value, Wikidata has no P2046):")
        for row, how in cleared:
            print(f"  {row['place_name']:<{CLEARED_NAME_WIDTH}} {row.get('origin', '')}")


def print_size_pass(rows: list[dict], size_limit: float) -> None:
    """Report places whose corrected area exceeds the limit. Removes nothing."""
    over = [r for r in rows
            if (parse_float(r.get("area_km2", "")) or 0) > size_limit]
    print(f"\n{'=' * 68}\nsize pass: {len(over)} places exceed {format_number(size_limit)} km2")
    if over:
        width = max(len(r["place_name"]) for r in over)
        print(f"\n{'place':<{width}}  {'origin':<12} {'area_km2':>12}  {'flagged?':<9} note")
        print("-" * (width + 60))
        for row in sorted(over, key=lambda r: -(parse_float(r.get("area_km2", "")) or 0)):
            flagged = "yes" if LARGE_AREA_NOTE in (row.get("note") or "") else "NO"
            print(f"{row['place_name']:<{width}}  {row.get('origin', ''):<12} "
                  f"{row['area_km2']:>12}  {flagged:<9} {(row.get('note') or '')[:NOTE_WIDTH]}")
        newly = [r for r in over if LARGE_AREA_NOTE not in (r.get("note") or "")]
        print(f"\n{len(newly)} of these carry no 'large area' note.")
    print("\nNothing removed. Rows above the limit were kept and merely noted in the\n"
          "shipped file, so exclusion is a dataset decision, not a mechanical fix.")


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def backup_to_archive(path: Path) -> Path:
    """Copy `path` to <its dir>/archive/<stem>.bak-areas<suffix>; returns the copy."""
    archive = path.parent / ARCHIVE_DIR_NAME
    archive.mkdir(parents=True, exist_ok=True)
    backup = archive / f"{path.stem}.bak-areas{path.suffix}"
    shutil.copy2(path, backup)
    return backup


def write_places(path: Path, fields: list[str], rows: list[dict]) -> Path:
    """Back the CSV up to archive/, then rewrite it atomically. Returns the backup path."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    backup = backup_to_archive(path)
    write_text_atomic(path, buffer.getvalue())
    return backup


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Re-derive area_km2 from Wikidata.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--size-limit", type=float, default=DEFAULT_SIZE_LIMIT_KM2,
                        help="report places whose corrected area exceeds this (km2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="query and report, but do not write the CSV")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load -> query -> compute -> report -> write."""
    args = parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"input not found: {csv_path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    fields, rows = load_places(csv_path)
    if "area_km2" not in fields:
        print("input has no area_km2 column", file=sys.stderr)
        return EXIT_BAD_INPUT

    qids = [(r.get("wikidata_id") or "").strip() for r in rows]
    print(f"re-deriving area for {len(rows)} places")

    areas, failed = fetch_areas(qids)
    if failed:
        print(f"\n!! {len(failed)} items could not be queried; "
              f"their area_km2 is left untouched", file=sys.stderr)

    changed, backfilled, cleared, multi = apply_areas(rows, areas, failed)
    print_changes(changed, backfilled, cleared, multi)
    print_size_pass(rows, args.size_limit)

    if args.dry_run:
        print("\n--dry-run: CSV not written")
        return EXIT_OK

    if not (changed or backfilled or cleared):
        print("\nno changes; CSV not rewritten")
        return EXIT_OK

    backup = write_places(csv_path, fields, rows)
    print(f"\nbacked up to {backup.parent.name}/{backup.name}")
    print(f"wrote {csv_path.name}: {len(rows)} rows")
    return EXIT_OK


if __name__ == "__main__":
    utf8_stdout()
    raise SystemExit(main())
