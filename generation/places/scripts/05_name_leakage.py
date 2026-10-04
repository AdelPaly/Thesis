#!/usr/bin/env python3
"""Measure name leakage in the clue set. Measurement only -- deletes nothing.

An inbound P131 clue is something located inside the place, and those things are
very often named after it: "Lagos State Library", "Sangardo Airport". A clue
whose label contains the place's own name gives the answer away, so it is not an
indirect clue at all -- it burns generation budget and is thrown out later at the
stage 7 filter anyway.

THE TEST, as specified: case-insensitive substring, place_name in clue_label.

Records whose clue_label is null are counted as UNTESTABLE, never as clean. They
cannot leak by this test because there is nothing to test, and folding them into
the denominator would understate the leak rate.

The strict test is a LOWER BOUND. It misses a leak whenever the file's
place_name carries a qualifier the clue label does not ("Cala, Eastern Cape" vs
"Cala Airport") or differs by diacritics. A relaxed variant -- diacritic-folded,
matched on the place name's leading comma-segment at token boundaries -- is
reported alongside as the upper bound. The strict number is the answer to the
question asked; the pair is what the keep/drop decision should be made on.

Usage:
    py -3 scripts/05_name_leakage.py [--clues data/02_clues.jsonl] [--out CSV]
                                     [--examples N]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import leaks_strict, read_jsonl, utf8_stdout  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "02_clues.jsonl"
OUT_CSV = ROOT / "data" / "05_name_leakage.csv"

HEAVY_P131_SHARE = 0.5   # a place is "P131 heavy" when P131 is at least this share of its clues


def fold(text: str) -> str:
    """Lowercase, strip diacritics, reduce punctuation to single spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9a-z]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def leaks_relaxed(place: str, label: str) -> bool:
    """Diacritic-folded, leading comma-segment, matched at token boundaries."""
    stem = fold(place.split(",")[0])
    if not stem:
        return False
    return re.search(rf"(?:^| ){re.escape(stem)}(?:$| )", fold(label)) is not None


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()

    parser = argparse.ArgumentParser(description="Measure name leakage in the clue set.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--examples", type=int, default=12)
    args = parser.parse_args(argv)

    if not args.clues.exists():
        print(f"clue file not found: {args.clues}", file=sys.stderr)
        return 2

    records = read_jsonl(args.clues)
    if not records:
        print("no clue records", file=sys.stderr)
        return 2

    for record in records:
        label = record.get("clue_label")
        record["_testable"] = bool(label)
        record["_strict"] = bool(label) and leaks_strict(record["place_name"], label)
        record["_relaxed"] = bool(label) and leaks_relaxed(record["place_name"], label)

    places = sorted({r["place_qid"] for r in records})
    per_place = Counter(r["place_qid"] for r in records)
    name_of = {r["place_qid"]: r["place_name"] for r in records}

    print(f"clue records : {len(records)}")
    print(f"places       : {len(places)}")
    print(f"untestable   : {sum(1 for r in records if not r['_testable'])} "
          f"(no English label -- excluded from every rate below)")

    # ---- headline: P131 ----
    p131 = [r for r in records if r["property"] == "P131"]
    testable = [r for r in p131 if r["_testable"]]
    strict = [r for r in p131 if r["_strict"]]
    relaxed = [r for r in p131 if r["_relaxed"]]
    print("\n" + "=" * 70)
    print("P131 (located in administrative entity)")
    print(f"  records            : {len(p131)}")
    print(f"  testable           : {len(testable)}")
    print(f"  LEAK (strict)      : {len(strict)}"
          + (f"   {len(strict)/len(testable):.1%} of testable" if testable else ""))
    print(f"  leak (relaxed)     : {len(relaxed)}"
          + (f"   {len(relaxed)/len(testable):.1%} of testable" if testable else ""))

    # ---- every property, for comparison ----
    print("\n" + "=" * 70)
    print("all properties, same test")
    print(f"\n{'property':<9} {'records':>8} {'testable':>9} {'strict':>7} {'rate':>7} "
          f"{'relaxed':>8} {'rate':>7}")
    print("-" * 62)
    rows_out = []
    for prop in sorted({r["property"] for r in records},
                       key=lambda p: -sum(1 for r in records if r["property"] == p)):
        group = [r for r in records if r["property"] == prop]
        can = [r for r in group if r["_testable"]]
        s = sum(1 for r in group if r["_strict"])
        x = sum(1 for r in group if r["_relaxed"])
        srate = f"{s/len(can):.1%}" if can else "-"
        xrate = f"{x/len(can):.1%}" if can else "-"
        print(f"{prop:<9} {len(group):>8} {len(can):>9} {s:>7} {srate:>7} {x:>8} {xrate:>7}")
        rows_out.append((prop, len(group), len(can), s, x))

    total_can = sum(1 for r in records if r["_testable"])
    total_s = sum(1 for r in records if r["_strict"])
    total_x = sum(1 for r in records if r["_relaxed"])
    print("-" * 62)
    all_srate = f"{total_s/total_can:.1%}" if total_can else "-"
    all_xrate = f"{total_x/total_can:.1%}" if total_can else "-"
    print(f"{'ALL':<9} {len(records):>8} {total_can:>9} {total_s:>7} "
          f"{all_srate:>6} {total_x:>8} {all_xrate:>6}")

    # ---- place-level survival ----
    def survivors(drop) -> dict[str, int]:
        out = Counter()
        for record in records:
            if not drop(record):
                out[record["place_qid"]] += 1
        return out

    scenarios = [
        ("remove leaking P131 only (strict)",
         lambda r: r["property"] == "P131" and r["_strict"]),
        ("remove leaking P131 only (relaxed)",
         lambda r: r["property"] == "P131" and r["_relaxed"]),
        ("remove P131 entirely",
         lambda r: r["property"] == "P131"),
        ("remove leaking clues, every property (strict)",
         lambda r: r["_strict"]),
        ("remove leaking clues, every property (relaxed)",
         lambda r: r["_relaxed"]),
        ("remove leaking (strict) AND unlabelled, every property",
         lambda r: r["_strict"] or not r["_testable"]),
    ]

    print("\n" + "=" * 70)
    print("places losing their ENTIRE clue set")
    print(f"\n{'scenario':<52} {'emptied':>8} {'records lost':>13}")
    print("-" * 76)
    emptied_detail: dict[str, list[str]] = {}
    for title, drop in scenarios:
        left = survivors(drop)
        gone = [q for q in places if left.get(q, 0) == 0]
        lost = sum(1 for r in records if drop(r))
        emptied_detail[title] = gone
        print(f"{title:<52} {len(gone):>8} {lost:>13}")

    for title in ("remove leaking P131 only (strict)", "remove P131 entirely"):
        gone = emptied_detail[title]
        if gone:
            print(f"\n  emptied by '{title}':")
            for qid in gone:
                print(f"    {name_of[qid]} ({qid}) -- had {per_place[qid]}")

    # ---- how much of each place's set is P131, and how much of that leaks ----
    p131_by_place: dict[str, list[dict]] = {}
    for record in p131:
        p131_by_place.setdefault(record["place_qid"], []).append(record)
    p131_share = []
    for qid in places:
        total = per_place[qid]
        group = p131_by_place.get(qid, [])
        mine = len(group)
        leak = sum(1 for r in group if r["_strict"])
        if mine:
            p131_share.append((qid, total, mine, leak, mine / total))
    heavy = [p for p in p131_share if p[4] >= HEAVY_P131_SHARE]
    print(f"\nplaces with any P131 clue          : {len(p131_share)} / {len(places)}")
    print(f"places where P131 is >={HEAVY_P131_SHARE:.0%} of set  : {len(heavy)}")
    fully = [p for p in p131_share if p[2] and p[3] == p[2]]
    print(f"places where every P131 clue leaks : {len(fully)}")

    # ---- examples ----
    print("\n" + "=" * 70)
    print(f"example P131 leaks (strict), first {args.examples}:")
    for record in strict[: args.examples]:
        print(f"  {record['place_name']:<24} <- {record['clue_label']}")
    caught = [r for r in p131 if r["_relaxed"] and not r["_strict"]]
    if caught:
        print(f"\ncaught only by the relaxed test, first {args.examples} "
              f"(these are leaks the strict test misses):")
        for record in caught[: args.examples]:
            print(f"  {record['place_name']:<24} <- {record['clue_label']}")

    # ---- detail CSV ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["place_qid", "place_name", "property", "clue_qid",
                         "clue_label", "leak_strict", "leak_relaxed"])
        for record in records:
            if record["_strict"] or record["_relaxed"]:
                writer.writerow([
                    record["place_qid"], record["place_name"], record["property"],
                    record["clue_qid"], record["clue_label"] or "",
                    str(record["_strict"]).lower(), str(record["_relaxed"]).lower(),
                ])
    print(f"\ndetail: {args.out}  (every flagged record; nothing deleted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
