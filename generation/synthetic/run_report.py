#!/usr/bin/env python3
"""
Report on a generation run (Task 6, the pilot).

    python run_report.py --run output/pilot

Reads the run's raw_generations.jsonl and swaps.jsonl (generate.py --out) and
the plan, and writes report.md in the run folder: how many items passed on
try 1, 2 and 3 or failed, the failure reasons per condition, clue and place
swaps, one example post per condition with its scale and angle, and the
lengths against their targets.
"""
from __future__ import annotations

import argparse
import statistics
import sys
from collections import Counter
from pathlib import Path

import generate as gen
import plan

HERE = Path(__file__).resolve().parent
CONDITIONS = list(plan.CONDITION_SPECS)


def report(rows: list[dict], items: dict[str, dict], swaps: list[dict]) -> str:
    out = ["# Generation run report", "",
           f"{len(rows)} items run, {sum(1 for r in rows if r.get('swapped_from'))} of them "
           f"reserves swapped in. Writer {rows[0].get('model') if rows else '-'}, judge "
           f"{rows[0].get('judge_model') if rows else '-'}.", ""]

    out += ["## Outcome per condition", "",
            "| condition | items | pass try 1 | pass try 2 | pass try 3 | failed |", "|---|---|---|---|---|---|"]
    for c in CONDITIONS + ["all"]:
        mine = [r for r in rows if c == "all" or r["condition"] == c]
        if not mine:
            continue
        n = Counter(f"try_{r['tries']}" if r["accepted"] else "failed" for r in mine)
        cell = lambda k: f"{n[k]} ({100 * n[k] / len(mine):.0f}%)"  # noqa: E731
        out.append(f"| {c} | {len(mine)} | {cell('try_1')} | {cell('try_2')} | {cell('try_3')} | {cell('failed')} |")

    out += ["", "## Why tries failed", "",
            "Every failed try, also those followed by a passing one.", "",
            "| condition | reason | tries |", "|---|---|---|"]
    for c in CONDITIONS:
        reasons = Counter(t["status"] for r in rows if r["condition"] == c
                          for t in r["try_log"] if t["status"] != "kept")
        for status, k in reasons.most_common():
            out.append(f"| {c} | {status} | {k} |")

    clue_swaps = [(r, s) for r in rows for s in r.get("clue_swaps") or []]
    out += ["", "## Swaps", "",
            f"- clue swaps (a decorative clue replaced by a spare): {len(clue_swaps)}, in "
            f"{len({r['item_id'] for r, _ in clue_swaps})} items"]
    for r, s in clue_swaps:
        out.append(f"  - {r['condition']} {r['place_name']}: {s['from_label']} -> {s['to_label']}")
    out.append(f"- place swaps (all tries failed): {len(swaps)}")
    for s in swaps:
        out.append(f"  - {s['set']} set: {s['place_name']} ({s['failed_condition']}, {s['failed_status']}) -> "
                   f"{s['replacement_qid'] or 'no matching reserve left'}")

    out += ["", "## One post per condition", ""]
    for c in CONDITIONS:
        kept = [r for r in rows if r["condition"] == c and r["accepted"]]
        if kept:
            r = kept[0]
            out += [f"**{c}**, {r['place_name']} (scale {r.get('scale')}, angle \"{r.get('angle')}\", "
                    f"{len(r['post_text'].split())} words, target {r.get('target_len')}, try {r['tries']})", "",
                    f"> {r['post_text']}", ""]

    out += ["## Length against target", "",
            "| condition | posts | median target | median length | median off by | within 25% of target |",
            "|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        mine = [r for r in rows if r["condition"] == c and r["accepted"] and r.get("target_len")]
        if not mine:
            continue
        lens = [len(r["post_text"].split()) for r in mine]
        targets = [r["target_len"] for r in mine]
        off = [abs(a - t) for a, t in zip(lens, targets)]
        near = sum(abs(a - t) <= 0.25 * t for a, t in zip(lens, targets))
        out.append(f"| {c} | {len(mine)} | {statistics.median(targets)} | {statistics.median(lens)} | "
                   f"{statistics.median(off)} | {near} ({100 * near / len(mine):.0f}%) |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, default=HERE / "output" / "pilot")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = ap.parse_args(argv)
    config = gen.load_config(args.config)
    items = {i["item_id"]: i for i in gen.read_jsonl(Path(config["paths"]["item_plan"]))}
    rows = [r for r in gen.latest_rows(gen.read_jsonl(args.run / "raw_generations.jsonl")).values()
            if "accepted" in r]
    swaps_path = args.run / "swaps.jsonl"
    swaps = gen.read_jsonl(swaps_path) if swaps_path.exists() else []
    text = report(rows, items, swaps)
    (args.run / "report.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
