#!/usr/bin/env python3
"""Apply the stage-7 clue filter: drop name leaks and unlabelled clues.

05_name_leakage.py measured; this script acts. Two rules, both from that
measurement:

  name_leakage  The clue label contains the place name as a case-insensitive
                substring. This is the STRICT test, deliberately -- the relaxed
                token-boundary variant was measured alongside it and the
                difference in the other direction is negligible, while strict
                catches within-word forms ("Lagosian", "Lagosport") that token
                matching lets through.

  no_label      clue_label is null. A clue with no English label cannot be
                realised as a surface form in an English post, so it is not a
                usable clue regardless of what it denotes.

Note that the two rules cannot both fire on one record: the substring test
needs a label to test. The counts are reported jointly anyway so the overlap is
visible rather than assumed.

The input is opened read-only and never rewritten. Survivors go to
02_clues_filtered.jsonl; every dropped record is written verbatim to
03_removed_clues.jsonl with a removed_reason field, so the exclusions stay
auditable instead of vanishing.

Continent for the emptied-place report is joined from the places CSV on
wikidata_id; places absent from it report "?" rather than failing the run.

Usage:
    py -3 scripts/03_filter_clues.py [--clues data/02_clues.jsonl]
                                     [--out data/02_clues_filtered.jsonl]
                                     [--removed data/03_removed_clues.jsonl]
                                     [--places data/01_places_final.csv] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "02_clues.jsonl"
OUT = ROOT / "data" / "02_clues_filtered.jsonl"
REMOVED = ROOT / "data" / "03_removed_clues.jsonl"
PLACES = ROOT / "data" / "01_places_final.csv"

BUCKETS = (
    ("0", lambda n: n == 0),
    ("1-2", lambda n: 1 <= n <= 2),
    ("3-9", lambda n: 3 <= n <= 9),
    ("10+", lambda n: n >= 10),
)


def leaks_strict(place: str, label: str) -> bool:
    """The specified test, as written in 05_name_leakage.py: substring, folded."""
    return place.casefold() in label.casefold()


def load_places(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["wikidata_id"]: row for row in csv.DictReader(handle)}


def distribution(counts: dict[str, int], places: list[str]) -> list[tuple[str, int]]:
    return [(name, sum(1 for q in places if test(counts.get(q, 0)))) for name, test in BUCKETS]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description="Filter leaking and unlabelled clues.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--removed", type=Path, default=REMOVED)
    parser.add_argument("--places", type=Path, default=PLACES)
    parser.add_argument("--dry-run", action="store_true",
                        help="report only; write neither output file")
    args = parser.parse_args(argv)

    if not args.clues.exists():
        print(f"clue file not found: {args.clues}", file=sys.stderr)
        return 2
    if args.out.resolve() == args.clues.resolve() or args.removed.resolve() == args.clues.resolve():
        print("refusing to overwrite the input clue file", file=sys.stderr)
        return 2

    records = [json.loads(line) for line in args.clues.open(encoding="utf-8") if line.strip()]
    if not records:
        print("no clue records", file=sys.stderr)
        return 2

    # ---- classify ----
    kept: list[dict] = []
    dropped: list[dict] = []
    for record in records:
        label = record.get("clue_label")
        reasons = []
        if label is None:
            reasons.append("no_label")
        elif leaks_strict(record["place_name"], label):
            reasons.append("name_leakage")
        if reasons:
            dropped.append({**record, "removed_reason": "+".join(reasons)})
        else:
            kept.append(record)

    by_reason = Counter(r["removed_reason"] for r in dropped)
    leak_n = sum(n for reason, n in by_reason.items() if "name_leakage" in reason)
    nolabel_n = sum(n for reason, n in by_reason.items() if "no_label" in reason)
    both_n = sum(n for reason, n in by_reason.items() if "+" in reason)

    # ---- write ----
    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        for path, rows in ((args.out, kept), (args.removed, dropped)):
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmp.replace(path)

    # ---- report ----
    places = sorted({r["place_qid"] for r in records})
    name_of = {r["place_qid"]: r["place_name"] for r in records}
    before = Counter(r["place_qid"] for r in records)
    after = Counter(r["place_qid"] for r in kept)
    place_meta = load_places(args.places)

    print("=" * 70)
    print("RECORDS")
    print(f"  before                    : {len(records)}")
    print(f"  removed                   : {len(dropped)}"
          f"   {len(dropped)/len(records):.1%}")
    print(f"    name leakage (strict)   : {leak_n}")
    print(f"    no English label        : {nolabel_n}")
    print(f"    both rules              : {both_n}"
          + ("   (impossible by construction: a null label cannot be substring-tested)"
             if both_n == 0 else ""))
    print(f"  after                     : {len(kept)}"
          f"   {len(kept)/len(records):.1%} of input")
    print(f"  places                    : {len(places)}")

    print("\n" + "=" * 70)
    print("PER-PLACE CLUE COUNT, AFTER FILTERING")
    print(f"\n{'bucket':<8} {'places after':>13} {'places before':>14}")
    print("-" * 38)
    dist_after = dict(distribution(after, places))
    dist_before = dict(distribution(before, places))
    for name, _ in BUCKETS:
        print(f"{name:<8} {dist_after[name]:>13} {dist_before[name]:>14}")
    print("-" * 38)
    print(f"{'total':<8} {len(places):>13} {len(places):>14}")

    emptied = [q for q in places if after.get(q, 0) == 0]
    print("\n" + "=" * 70)
    print(f"PLACES LEFT AT ZERO: {len(emptied)}")
    if emptied:
        print(f"\n{'qid':<11} {'name':<26} {'continent':<15} {'had':>5} "
              f"{'leak':>5} {'nolab':>6}")
        print("-" * 72)
        for qid in sorted(emptied, key=lambda q: -before[q]):
            meta = place_meta.get(qid, {})
            reasons = Counter(r["removed_reason"] for r in dropped if r["place_qid"] == qid)
            print(f"{qid:<11} {(meta.get('place_name') or name_of[qid])[:26]:<26} "
                  f"{(meta.get('continent') or '?')[:15]:<15} {before[qid]:>5} "
                  f"{reasons.get('name_leakage', 0):>5} {reasons.get('no_label', 0):>6}")
        if not place_meta:
            print(f"\n  (continent unavailable: {args.places} not found)")

    print("\n" + "=" * 70)
    print("BY PROPERTY")
    props = sorted({r["property"] for r in records}, key=lambda p: -sum(
        1 for r in records if r["property"] == p))
    prop_label = {r["property"]: r["property_label"] for r in records}
    print(f"\n{'property':<9} {'before':>7} {'leak':>6} {'nolab':>6} {'after':>7} "
          f"{'kept':>7}  {'label':<40}")
    print("-" * 90)
    for prop in props:
        group = [r for r in records if r["property"] == prop]
        gone = Counter(r["removed_reason"] for r in dropped if r["property"] == prop)
        survived = sum(1 for r in kept if r["property"] == prop)
        rate = f"{survived/len(group):.1%}" if group else "-"
        print(f"{prop:<9} {len(group):>7} {gone.get('name_leakage', 0):>6} "
              f"{gone.get('no_label', 0):>6} {survived:>7} {rate:>7}  "
              f"{prop_label.get(prop, '')[:40]:<40}")
    print("-" * 90)
    print(f"{'ALL':<9} {len(records):>7} {leak_n:>6} {nolabel_n:>6} {len(kept):>7} "
          f"{len(kept)/len(records):>6.1%}")

    print("\nplaces still holding at least one clue of each property")
    print(f"\n{'property':<9} {'before':>7} {'after':>7} {'lost':>6}")
    print("-" * 32)
    holders_before = defaultdict(set)
    holders_after = defaultdict(set)
    for record in records:
        holders_before[record["property"]].add(record["place_qid"])
    for record in kept:
        holders_after[record["property"]].add(record["place_qid"])
    for prop in props:
        b, a = len(holders_before[prop]), len(holders_after[prop])
        print(f"{prop:<9} {b:>7} {a:>7} {b - a:>6}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nwrote {args.out}      {len(kept)} records")
        print(f"wrote {args.removed}  {len(dropped)} records (+ removed_reason)")
        print(f"input {args.clues} untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
