#!/usr/bin/env python3
"""Join every stage's output back onto the place list, so how far each place
got is one row rather than four files.

The pipeline narrows at every stage and each stage reports only its own
survivors: 02 emits clues for every place it was given, 02-filtered keeps fewer,
05 scores sets for those but only some of them resolve. A place that vanished
between two stages is invisible in both. This script puts the whole descent in
one table and names the stage each place stopped at.

STAGE_PASSED is the summary column.

  sets        at least one 3-to-5 clue set intersects to exactly this place.
              The place is generatable and solvable.
  clues_only  usable clues survive the filters, but no set pins the place
              down. Either too few clues to form a set, or sets that stay
              ambiguous.
  dropped     nothing survived 03's leakage and label filtering.

  The distinction the third table exists to draw is WHY a clues_only place
  failed. clues_raw against clues_usable separates a place that never had
  clues from one whose clues were all thrown out -- different problems, and
  only the second is fixable by loosening a filter.

COUNTS COME FROM THE FILES, NOT THE SPREADSHEET. `clue_count` in
data/01_places_final.csv is capped at 25 per query branch and understates many
rows; `clue_total` is the uncapped ordinal measure. Neither says what is
actually generatable after filtering, so `clues_raw` and `clues_usable` are
recounted from 02_clues.jsonl and 02_clues_filtered.jsonl directly. All three
existing columns are carried through unchanged for comparison.

RESOLVING_SETS is counted over the sets 05 enumerated, which is its
family-balanced pool of <= 12 clues per place, not every subset of every clue.
It answers "does a good set exist", not "how many exist in principle".

Input files are read, never written. data/01_places_final.csv is not modified.

The clue files (02, 02-filtered, 05) can cover MORE places than --places: they
were collected for a larger candidate list, and the shipped final list is the
102-place subset of it. The 147-row candidate CSV that list was built from is
NOT shipped, so such extra places are reported as a WARNING and do not fail the
run. Only a broken join, an impossible count, or a missing input fails.

Exit codes: 0 ok (warnings allowed), 1 a sanity check failed, 2 bad input.

Output:
    data/01_places_status.csv   every input column, plus clues_raw,
                                clues_usable, resolving_sets,
                                best_set_ambiguity, families_available,
                                stage_passed

Usage:
    py -3 scripts/06_status.py [--places data/01_places_final.csv]
                               [--target 17] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import read_jsonl, utf8_stdout, write_text_atomic  # noqa: E402

# The family mapping is 04's (the module name starts with a digit, so it cannot
# be a plain import). Restating it here would let the two drift.
FAMILIES = import_module("04_score_clues").FAMILIES

ROOT = Path(__file__).resolve().parent.parent
PLACES = ROOT / "data" / "01_places_final.csv"
CLUES_RAW = ROOT / "data" / "02_clues.jsonl"
CLUES_FILTERED = ROOT / "data" / "02_clues_filtered.jsonl"
CLUE_SETS = ROOT / "data" / "05_clue_sets.jsonl"
OUT_CSV = ROOT / "data" / "01_places_status.csv"

NEW_COLUMNS = ["clues_raw", "clues_usable", "resolving_sets",
               "best_set_ambiguity", "families_available", "stage_passed"]

STAGES = ["sets", "clues_only", "dropped"]

DEFAULT_TARGET = 17     # places at stage 'sets' wanted per continent
RULE_WIDTH = 70
EXIT_OK, EXIT_CHECK_FAILED, EXIT_BAD_INPUT = 0, 1, 2


# --------------------------------------------------------------------------
# Load, join, write
# --------------------------------------------------------------------------

def load_places(path: Path) -> tuple[list[dict], list[str]]:
    """The place CSV as (rows, column names)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = list(reader.fieldnames or [])
    return rows, columns


def join_status(places: list[dict], raw: list[dict], filtered: list[dict],
                clue_sets: dict[str, dict]) -> tuple[list[dict], Counter, Counter]:
    """One row per place: its columns plus counts from the stage files and its stage.

    Returns (rows, raw_counts, usable_counts).
    """
    raw_counts = Counter(r["place_qid"] for r in raw)
    usable_counts = Counter(r["place_qid"] for r in filtered)
    families_by_place: dict[str, set[str]] = defaultdict(set)
    for record in filtered:
        families_by_place[record["place_qid"]].add(
            FAMILIES.get(record["property"], "other"))

    rows = []
    for place in places:
        qid = place["wikidata_id"]
        usable = usable_counts.get(qid, 0)
        scored = clue_sets.get(qid)
        resolving = (scored or {}).get("sets_resolving_to_place") or 0
        best = (scored or {}).get("intersection_min")

        if resolving > 0:
            stage = "sets"
        elif usable > 0:
            stage = "clues_only"
        else:
            stage = "dropped"

        rows.append({
            **place,
            "clues_raw": raw_counts.get(qid, 0),
            "clues_usable": usable,
            "resolving_sets": resolving,
            "best_set_ambiguity": "" if best is None else best,
            "families_available": len(families_by_place.get(qid, ())),
            "stage_passed": stage,
        })
    return rows, raw_counts, usable_counts


def write_status(path: Path, columns: list[str], rows: list[dict]) -> None:
    """Write the status CSV atomically (UTF-8 with BOM, LF line endings)."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns + NEW_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue(), encoding="utf-8-sig")


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def report_descent(rows: list[dict]) -> list[str]:
    """Table 1: stage reached, by continent. Returns the sorted continents."""
    continents = sorted({r["continent"] for r in rows})
    print("\n" + "=" * RULE_WIDTH)
    print("STAGE REACHED, BY CONTINENT")
    print(f"\n{'continent':<18} " + " ".join(f"{s:>11}" for s in STAGES)
          + f" {'total':>7}")
    print("-" * 66)
    for continent in continents:
        group = [r for r in rows if r["continent"] == continent]
        counts = Counter(r["stage_passed"] for r in group)
        print(f"{continent:<18} " + " ".join(f"{counts.get(s, 0):>11}" for s in STAGES)
              + f" {len(group):>7}")
    print("-" * 66)
    overall = Counter(r["stage_passed"] for r in rows)
    print(f"{'ALL':<18} " + " ".join(f"{overall.get(s, 0):>11}" for s in STAGES)
          + f" {len(rows):>7}")
    return continents


def report_deficit(rows: list[dict], continents: list[str], target: int) -> None:
    """Table 2: continents short of `target` places at stage 'sets'."""
    print("\n" + "=" * RULE_WIDTH)
    print(f"DEFICIT AGAINST {target} PLACES AT STAGE 'sets'")
    short = []
    for continent in continents:
        group = [r for r in rows if r["continent"] == continent]
        have = sum(1 for r in group if r["stage_passed"] == "sets")
        if have < target:
            short.append((continent, have, len(group)))
    if not short:
        print(f"\nnone -- every continent has at least {target}.")
        return
    print(f"\n{'continent':<18} {'at sets':>8} {'deficit':>8} {'in list':>8} "
          f"{'recoverable':>12}")
    print("-" * 60)
    for continent, have, total in short:
        group = [r for r in rows if r["continent"] == continent]
        # A clues_only place can in principle be recovered by better clues
        # or a wider pool. A continent short of `target` places ENTIRELY is
        # short of places, and no amount of clue work fixes that.
        recoverable = sum(1 for r in group if r["stage_passed"] != "sets")
        print(f"{continent:<18} {have:>8} {target - have:>8} {total:>8} "
              f"{recoverable:>12}")
    deficit = sum(target - have for _, have, _ in short)
    print(f"\ntotal deficit: {deficit} places across {len(short)} continent(s)")
    for continent, have, total in short:
        if total < target:
            print(f"  !! {continent} holds only {total} places in the list, "
                  f"below the target of {target}.")
            print(f"     {target - total} of its deficit cannot be closed by "
                  f"clue work at all -- the list itself is short.")


def report_stalled(rows: list[dict], clue_sets: dict[str, dict]) -> None:
    """Table 3: who failed to reach 'sets', and whether it was clues or distinctness."""
    stalled = [r for r in rows if r["stage_passed"] != "sets"]
    print("\n" + "=" * RULE_WIDTH)
    print(f"PLACES THAT DID NOT REACH 'sets': {len(stalled)}")
    print(f"\n{'place':<26} {'continent':<15} {'stage':<11} {'raw':>5} {'usable':>7} "
          f"{'fam':>4} {'best':>5}")
    print("-" * 78)
    for record in sorted(stalled, key=lambda r: (r["stage_passed"], r["continent"],
                                                 r["place_name"])):
        print(f"{record['place_name'][:26]:<26} {record['continent'][:15]:<15} "
              f"{record['stage_passed']:<11} {record['clues_raw']:>5} "
              f"{record['clues_usable']:>7} {record['families_available']:>4} "
              f"{str(record['best_set_ambiguity'] or '-'):>5}")

    no_clues = [r for r in stalled if r["clues_raw"] == 0]
    all_filtered = [r for r in stalled if r["clues_raw"] > 0 and r["clues_usable"] == 0]
    too_few = [r for r in stalled if r["clues_usable"] > 0 and not r["best_set_ambiguity"]]
    ambiguous = [r for r in stalled if r["best_set_ambiguity"] not in ("", None)
                 and r["best_set_ambiguity"] > 1]

    print("\nwhy they stalled")
    print("-" * RULE_WIDTH)
    print(f"  no clues found at all (02)          {len(no_clues):>4}")
    print(f"  every clue filtered out (03)        {len(all_filtered):>4}")
    print(f"  clues survive but form no set (05)  {len(too_few):>4}")
    print(f"  sets exist but stay ambiguous       {len(ambiguous):>4}")
    print("\n  The first two are clue-supply problems and the fix is upstream. The")
    print("  third is a set-search problem only if the clue count is near the")
    print("  minimum; below it, it is also clue supply. Only the fourth is a")
    print("  distinctness problem.")

    # A place can have usable clues from four families and still lose all but
    # one to 05's surface cut. Worth seeing, because it points at the cut
    # rather than at the place.
    narrowed = [r for r in rows
                if r["stage_passed"] != "dropped"
                and clue_sets.get(r["wikidata_id"])
                and len(clue_sets[r["wikidata_id"]]["families_available"])
                < r["families_available"]]
    if narrowed:
        print(f"\n{len(narrowed)} places have usable clues from more families than "
              f"survived into 05's")
        print("  eligible pool -- the surface-ambiguity cut, not the place, "
              "narrowed them.")


def sanity_checks(places: list[dict], rows: list[dict], places_name: str,
                  clue_file_places: list[set[str]]) -> int:
    """Print the sanity checks; return how many FAILED (warnings do not count)."""
    print("\n" + "=" * RULE_WIDTH)
    print("SANITY CHECKS")
    failed = 0

    csv_qids = [r["wikidata_id"] for r in places]
    joined = {r["wikidata_id"] for r in rows}
    dupes = [q for q, n in Counter(csv_qids).items() if n > 1]
    unjoined = [q for q in csv_qids if q not in joined]
    extra = sorted(set().union(*clue_file_places) - set(csv_qids))

    if len(rows) != len(places) or unjoined or dupes:
        failed += 1
        print(f"  FAIL  join is not one row per place: {len(places)} in, {len(rows)} out"
              + (f", {len(dupes)} duplicate qid(s): {', '.join(dupes[:5])}" if dupes else "")
              + (f", {len(unjoined)} unjoined" if unjoined else ""))
    else:
        print(f"  ok    all {len(places)} places appear in the join, one row each")

    if extra:
        # Not a failure: the clue files were collected for a larger candidate
        # list than the shipped place list (see the module docstring).
        print(f"  WARN  {len(extra)} place_qid(s) in the clue files are absent from "
              f"{places_name} (it is a subset of the clue files' places; ignored):")
        for qid in extra[:10]:
            print(f"          {qid}")
    else:
        print(f"  ok    no place appears in the clue files that is not in "
              f"{places_name}")

    over = [r for r in rows if r["clues_usable"] > r["clues_raw"]]
    if over:
        failed += 1
        print(f"  FAIL  {len(over)} places have clues_usable > clues_raw:")
        for record in over[:10]:
            print(f"          {record['place_name']:<24} raw {record['clues_raw']}, "
                  f"usable {record['clues_usable']}")
    else:
        print(f"  ok    clues_usable <= clues_raw for all {len(rows)} places")

    impossible = [r for r in rows if r["resolving_sets"] > 0 and r["clues_usable"] == 0]
    if impossible:
        failed += 1
        print(f"  FAIL  {len(impossible)} places resolve sets with no usable clues:")
        for record in impossible[:10]:
            print(f"          {record['place_name']:<24} "
                  f"{record['resolving_sets']} sets")
    else:
        print("  ok    no place has resolving sets without usable clues")
    return failed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(description="Join stage outputs onto the place list.")
    parser.add_argument("--places", type=Path, default=PLACES)
    parser.add_argument("--raw", type=Path, default=CLUES_RAW)
    parser.add_argument("--filtered", type=Path, default=CLUES_FILTERED)
    parser.add_argument("--sets", type=Path, default=CLUE_SETS)
    parser.add_argument("--out", type=Path, default=OUT_CSV)
    parser.add_argument("--target", type=int, default=DEFAULT_TARGET,
                        help=f"places at stage 'sets' wanted per continent "
                             f"(default {DEFAULT_TARGET})")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Join, write and report. Exit 0 ok (warnings allowed), 1 a check failed, 2 bad input."""
    utf8_stdout()
    args = build_parser().parse_args(argv)

    missing = [p for p in (args.places, args.raw, args.filtered, args.sets)
               if not p.exists()]
    if missing:
        for path in missing:
            print(f"input not found: {path}", file=sys.stderr)
        return EXIT_BAD_INPUT

    places, columns = load_places(args.places)
    raw = read_jsonl(args.raw)
    filtered = read_jsonl(args.filtered)
    clue_sets = {r["place_qid"]: r for r in read_jsonl(args.sets)}

    clash = [c for c in NEW_COLUMNS if c in columns]
    if clash:
        print(f"input already has column(s) {', '.join(clash)}; refusing to "
              f"overwrite them", file=sys.stderr)
        return EXIT_BAD_INPUT

    rows, raw_counts, usable_counts = join_status(places, raw, filtered, clue_sets)

    print(f"places      : {len(places)} from {args.places.name}")
    print(f"raw clues   : {len(raw)} over {len(raw_counts)} places")
    print(f"usable clues: {len(filtered)} over {len(usable_counts)} places")
    print(f"scored sets : {len(clue_sets)} places in {args.sets.name}")

    if not args.dry_run:
        write_status(args.out, columns, rows)

    continents = report_descent(rows)
    report_deficit(rows, continents, args.target)
    report_stalled(rows, clue_sets)
    failed = sanity_checks(places, rows, args.places.name,
                           [set(raw_counts), set(usable_counts), set(clue_sets)])

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nwrote {args.out}   {len(rows)} rows, "
              f"{len(columns) + len(NEW_COLUMNS)} columns "
              f"({len(columns)} carried through, {len(NEW_COLUMNS)} added)")
    return EXIT_CHECK_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
