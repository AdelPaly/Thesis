#!/usr/bin/env python3
"""
The dataset's quality criteria as numbers, from the kept files.

    python quality_report.py                      # tables, and output/quality_report.csv
    python quality_report.py --audit output/audit/vse_Qwen3.8_27B.jsonl

Reads the plan, the generations, the swaps, finalise_report.csv, the dataset
and an audit log (audit.py). Calls no model and changes none of them. A second
dataset made from the same plan can be put through this script and the two
tables compared; that is what "of comparable quality" is measured on.

Sections (the `section` column of the CSV):

    yield        planned, generated and kept posts per condition; why the rest
                 were dropped
    tries        how many tries a kept post needed; places swapped for a reserve
    balance      the kept posts against the plan (pool "plan"), as shares, by
                 condition, continent, place type, prominence, register, scale,
                 valence, noise kind and split
    rules        finalise.py's checks run again on the dataset's own text; every
                 count should be 0
    style        post length in words beside the real posts it was modelled on
                 (sampler/length_dist.json), and the longest post in characters
    repetition   posts that share a stock scene (regenerate.stock_scene), posts
                 per writer configuration, and how each kept post was written
    second_judge the audit model's verdicts on the kept posts. It read only
                 posts the generation-time judge accepted, so this is how often
                 it accepts those, not agreement over all posts written.
    difficulty   exact-match accuracy of the two floor configurations, when
                 their predictions exist (results/predictions/)

No thresholds are applied: none were measured (there is no second generation
of this plan to take the spread from), so the report gives the numbers only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import audit
import finalise
import generate as gen
import plan
import regenerate

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
COLUMNS = ["section", "table", "group", "value", "n", "of", "share", "reference", "note"]
FLOORS = ("majority_class", "population_wikidata")
# what a balance table is called, and where the value is on a plan item
BALANCE = [("condition", "condition"), ("continent", "place_continent"), ("place type", "place_type"),
           ("place prominence", "pbin"), ("register", "register"), ("scale", "scale"),
           ("valence", "valence"), ("noise kind", "_noise_kind"), ("split", "place_split")]


class Report:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(self, section: str, table: str, group: Any, value: Any, n: Any = "", of: Any = "",
            reference: Any = "", note: str = "") -> None:
        share = f"{100 * n / of:.1f}" if isinstance(n, (int, float)) and isinstance(of, int) and of else ""
        self.rows.append({"section": section, "table": table, "group": group, "value": value, "n": n,
                          "of": of, "share": share, "reference": reference, "note": note})

    def show(self) -> None:
        last = None
        for r in self.rows:
            head = (r["section"], r["table"])
            if head != last:
                print(f'\n{r["section"]}: {r["table"]}')
                last = head
            share = f'{r["share"]:>6}%' if r["share"] != "" else " " * 7
            of = f'/ {r["of"]:<6}' if r["of"] != "" else " " * 8
            tail = "  ".join(str(x) for x in (r["reference"], r["note"]) if x != "")
            print(f'  {str(r["group"]):<16} {str(r["value"]):<46} {str(r["n"]):>7} {of} {share}  {tail}'.rstrip())


def noise_kind(item: dict[str, Any]) -> str:
    noise = item.get("noise_entity") or {}
    return noise.get("noise_kind") or noise.get("kind") or ("none" if not noise else "entity")


def words(text: str) -> int:
    return len(text.split())


def quantile(sorted_values: list[int], q: float) -> float:
    return statistics.quantiles(sorted_values, n=100, method="inclusive")[round(q * 100) - 1]


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--audit", type=Path, default=None,
                        help="an audit.py log (default: the only .jsonl in output/audit/)")
    parser.add_argument("--out", type=Path, default=None, help="default output/quality_report.csv")
    args = parser.parse_args(argv)

    config = gen.load_config(args.config)
    paths = config["paths"]
    dataset_path = Path(paths["dataset_csv"])
    items = gen.read_jsonl(Path(paths["item_plan"]))
    by_id = {i["item_id"]: i for i in items}
    for i in items:
        i["_noise_kind"] = noise_kind(i)
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))
    with Path(paths["finalise_report"]).open(encoding="utf-8", newline="") as f:
        report = list(csv.DictReader(f))
    with dataset_path.open(encoding="utf-8", newline="") as f:
        dataset = list(csv.DictReader(f))
    conditions = sorted({i["condition"] for i in items})
    out = Report()

    # ---- yield
    for c in conditions + ["all"]:
        mine = [r for r in report if c in ("all", r["condition"])]
        planned = sum(1 for i in items if c in ("all", i["condition"]) and i.get("pool", "plan") == "plan")
        generated = [r for r in mine if r["item_id"] in latest]
        kept = sum(1 for r in mine if r["status"] == "kept")
        out.add("yield", "posts per condition", c, "planned (pool plan)", planned)
        out.add("yield", "posts per condition", c, "in the plan with reserve", len(mine))
        out.add("yield", "posts per condition", c, "generated", len(generated), len(mine))
        out.add("yield", "posts per condition", c, "kept of generated", kept, len(generated))
    for c in conditions:
        mine = [r for r in report if r["condition"] == c]
        for status, n in Counter(r["status"] for r in mine).most_common():
            out.add("yield", "outcome of every planned post", c, status, n, len(mine))

    # ---- tries
    for c in conditions:
        mine = [d for d in dataset if d["signal_type"] == c]
        for tries, n in sorted(Counter(d["tries"] for d in mine).items(), key=lambda kv: int(kv[0] or 0)):
            out.add("tries", "tries a kept post needed", c, f"{tries} tries", n, len(mine))
    swaps_path = Path(paths["raw_generations"]).parent / "swaps.jsonl"
    swaps = gen.read_jsonl(swaps_path) if swaps_path.exists() else []
    for cset in sorted({s["set"] for s in swaps}):
        mine = [s for s in swaps if s["set"] == cset]
        out.add("tries", "places swapped out of a set", cset, "swapped out", len(mine))
        out.add("tries", "places swapped out of a set", cset, "a reserve came in",
                sum(1 for s in mine if s.get("replacement_qid")), len(mine))

    # ---- balance: shares in the plan (pool "plan") against shares in the dataset
    planned_items = [i for i in items if i.get("pool", "plan") == "plan"]
    kept_items = [by_id[d["item_id"]] for d in dataset]
    for name, key in BALANCE:
        was, now = Counter(str(i.get(key) or "-") for i in planned_items), Counter(str(i.get(key) or "-") for i in kept_items)
        for value in sorted(set(was) | set(now)):
            p = 100 * was[value] / len(planned_items)
            k = 100 * now[value] / len(kept_items)
            out.add("balance", name, name, value, now[value], len(kept_items),
                    reference=f"planned {p:.1f}% ({was[value]})", note=f"{k - p:+.1f} pp")

    # ---- rules: the checks again, on the text in the dataset
    aliases = finalise.load_aliases(Path(paths["place_inputs"]))
    limit = finalise.char_limit(config.get("finalise"))
    failures: Counter[str] = Counter()
    for d in dataset:
        item = finalise.used_item(by_id[d["item_id"]], latest.get(d["item_id"]))
        names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
        verdict = finalise.check(item, d["text"].strip(), names, None, limit)
        if verdict:
            failures[verdict[0]] += 1
    out.add("rules", "string checks on the dataset", "all", "posts failing any check",
            sum(failures.values()), len(dataset), note="place name where forbidden, clue or distractor "
            "missing, unplanned name, length")
    for status, n in failures.most_common():
        out.add("rules", "string checks on the dataset", "all", status, n, len(dataset))
    for label, bad in (
            ("post without its plan item's clues", sum(
                1 for d in dataset
                if "|".join(c["clue_qid"] for c in finalise.used_item(by_id[d["item_id"]], latest.get(d["item_id"]))["clues"])
                != (d.get("clue_qids") or ""))),
            ("place on both sides of the split", len(
                {d["qid"] for d in dataset if d["split"] == "dev"} & {d["qid"] for d in dataset if d["split"] == "test"})),
            ("paired post whose place has no EXPLICIT post", sum(
                1 for d in dataset if d["signal_type"] != "EXPLICIT"
                and d["qid"] not in {e["qid"] for e in dataset if e["signal_type"] == "EXPLICIT"})),
            ("duplicate text", len(dataset) - len({d["text"] for d in dataset}))):
        out.add("rules", "structure of the dataset", "all", label, bad, len(dataset))

    # ---- style: length in words against the real posts
    reference = json.loads(Path(paths["length_dist"]).read_text(encoding="utf-8"))
    ref = {"p5": reference["p5"], "q1": reference["q1"], "median": reference["median"],
           "q3": reference["q3"], "p95": reference["p95"]}
    for c in conditions + ["all"]:
        lengths = sorted(words(d["text"]) for d in dataset if c in ("all", d["signal_type"]))
        for label, q in (("p5", .05), ("q1", .25), ("median", .5), ("q3", .75), ("p95", .95)):
            out.add("style", "words per post", c, label, round(quantile(lengths, q), 1),
                    reference=f"real posts {ref[label]}")
        inside = sum(1 for n in lengths if ref["q1"] <= n <= ref["q3"])
        out.add("style", "words per post", c, "within the real posts' q1-q3", inside, len(lengths))
        out.add("style", "words per post", c, "above the real posts' p95",
                sum(1 for n in lengths if n > ref["p95"]), len(lengths))
    out.add("style", "characters per post", "all", "longest post", max(len(d["text"]) for d in dataset),
            reference=f"limit {limit}")

    # ---- repeated scenes: posts that share one of the two stock scenes, and
    # how many came from the prompt before and after it lost its examples
    for c in conditions + ["all"]:
        mine = [d for d in dataset if c in ("all", d["signal_type"])]
        scenes = Counter(regenerate.stock_scene(d["text"]) or "none" for d in mine)
        out.add("repetition", "posts in a stock scene", c, "any stock scene", len(mine) - scenes["none"], len(mine))
        if c == "all":
            for scene, n in scenes.most_common():
                out.add("repetition", "posts in a stock scene", c, scene, n, len(mine))
    hashes = Counter((latest.get(d["item_id"]) or {}).get("config_hash") or "?" for d in dataset)
    for config_hash, n in hashes.most_common():
        out.add("repetition", "posts per writer configuration", "all", config_hash, n, len(dataset))
    # rows rewritten on 2026-11-17 carry no "regenerated" field: over 3 tries, or a manual removal in the try log
    how = Counter(r.get("regenerated") or ("manual or failed (2026-11-17)" if (r.get("tries") or 0) > 3
                                           or any(t.get("manual_removed") for t in r.get("try_log") or [])
                                           else "first generation")
                  for r in ((latest.get(d["item_id"]) or {}) for d in dataset))
    for mode, n in how.most_common():
        out.add("repetition", "how the kept post was written", "all", mode, n, len(dataset))

    # ---- second judge
    audit_path = args.audit
    if audit_path is None:
        logs = sorted((dataset_path.parent / "audit").glob("*.jsonl"))
        audit_path = logs[0] if len(logs) == 1 else None
    if audit_path is None or not audit_path.exists():
        out.add("second_judge", "audit", "all", "no audit log", note="run audit.py, or give --audit")
    else:
        text_now = {d["item_id"]: d["text"].strip() for d in dataset}
        rows = {r["item_id"]: r for r in gen.read_jsonl(audit_path)
                if r.get("answer") is not None and text_now.get(r["item_id"]) == r["text"]}
        model = next(iter(rows.values()), {}).get("model", "")
        first = sorted({(latest.get(i) or {}).get("judge_model") or "?" for i in rows})
        out.add("second_judge", "coverage", "all", "kept posts the audit read", len(rows), len(dataset),
                reference=f"audit model {model}", note=f'generation judge {", ".join(first)}')
        for name, strict in (("accepted (audit.py's default scoring)", frozenset()),
                             ("accepted (off-theme and unquoted links counted too)", frozenset({"theme", "quotes"}))):
            verdicts = {i: audit.judge_verdict(by_id[i], r["answer"], r["text"], strict) for i, r in rows.items()}
            for c in conditions + ["all"]:
                mine = [i for i in rows if c in ("all", by_id[i]["condition"])]
                out.add("second_judge", name, c, "accepted by the audit model",
                        sum(1 for i in mine if verdicts[i] is None), len(mine))
            for status, n in Counter(v[0] for v in verdicts.values() if v).most_common():
                out.add("second_judge", name + ": reasons", "all", status, n, len(rows))

    # ---- difficulty: the floor configurations, if they have been run
    for name in FLOORS:
        path = REPO / "results" / "predictions" / f"{name}__synthetic.jsonl"
        if not path.exists():
            out.add("difficulty", "exact match of the floor configurations", name, "not run",
                    note=f"no {path.relative_to(REPO).as_posix()}")
            continue
        texts = {d["text"]: d for d in dataset}
        preds = [p for p in gen.read_jsonl(path) if p.get("text") in texts]
        for c in conditions + ["all"]:
            mine = [p for p in preds if c in ("all", texts[p["text"]]["signal_type"])]
            out.add("difficulty", "exact match of the floor configurations", name, c,
                    sum(1 for p in mine if p.get("predicted_qid") and p["predicted_qid"] == p.get("place_qid")),
                    len(mine), note="" if len(preds) == len(dataset) else f"{len(preds)} of {len(dataset)} posts predicted")

    out.show()
    out_path = args.out or dataset_path.with_name("quality_report.csv")
    finalise.write_csv(out_path, COLUMNS, out.rows)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
