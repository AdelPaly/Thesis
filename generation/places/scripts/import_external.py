#!/usr/bin/env python3
"""Turn the external clue sets into the planner's 05_clue_sets.jsonl.

The external export (data/external/) has the same clue_sets.jsonl shape as
stage 5, except that best_sets[].clues[] carry no property_label, which the
planner requires and generate.py uses to word each clue's relation. This adds
it back from the external clues.jsonl, matched on (place_qid, clue_qid,
property). Nothing else is changed.

A clue with no exact match takes the label of any clue with the same property;
one with neither is reported and left without, so plan.py stops on it rather
than writing a prompt with a raw P-number.

Usage:
    py -3 scripts/import_external.py [--clues data/external/clues.jsonl]
                                     [--sets data/external/clue_sets.jsonl]
                                     [--out data/05_clue_sets.jsonl]
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import read_jsonl, utf8_stdout, write_jsonl_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "external" / "clues.jsonl"
SETS = ROOT / "data" / "external" / "clue_sets.jsonl"
OUT_JSONL = ROOT / "data" / "05_clue_sets.jsonl"


def main(argv: list[str] | None = None) -> int:
    utf8_stdout()

    parser = argparse.ArgumentParser(description="Add property_label to the external clue sets.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--sets", type=Path, default=SETS)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    args = parser.parse_args(argv)

    for path in (args.clues, args.sets):
        if not path.exists():
            print(f"file not found: {path}", file=sys.stderr)
            return 2

    exact: dict[tuple[str, str, str], str] = {}
    by_property: dict[str, str] = {}
    for clue in read_jsonl(args.clues):
        label = clue.get("property_label")
        if label:
            exact[(clue["place_qid"], clue["clue_qid"], clue["property"])] = label
            by_property.setdefault(clue["property"], label)

    records = read_jsonl(args.sets)
    stats: Counter[str] = Counter()
    missing: list[str] = []
    for record in records:
        for best in record.get("best_sets") or []:
            for clue in best["clues"]:
                if clue.get("property_label"):
                    stats["already set"] += 1
                    continue
                key = (record["place_qid"], clue["clue_qid"], clue["property"])
                if key in exact:
                    clue["property_label"] = exact[key]
                    stats["exact match"] += 1
                elif clue["property"] in by_property:
                    clue["property_label"] = by_property[clue["property"]]
                    stats["by property"] += 1
                else:
                    stats["missing"] += 1
                    missing.append(f"{record['place_name']}: {clue['clue_label']} ({clue['property']})")

    count = write_jsonl_atomic(args.out, records)
    print(f"clue entries: {dict(stats)}")
    for line in missing[:20]:
        print(f"  no property_label: {line}")
    print(f"wrote {args.out}   {count} places")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
