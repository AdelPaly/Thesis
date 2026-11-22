"""List every post that was taken out of the dataset by hand.

    python generation/synthetic/manual_edits.py

A post is removed by hand by adding a line to output/manual_removed.csv (item,
date, status, reason). `regenerate.py` then sends the item back to the writer
with that reason, and on success takes the line out of the list again, so the
list itself is empty once the work is done. What stays is the generation log:
every such removal is a try in the item's `try_log` in
output/raw_generations.jsonl, marked `manual_removed` with the date, next to
the removed post, the reason given and the tries that followed.

This script reads that log and writes output/manual_edits.csv, one line per
removal:

  item_id, condition, place_name
  removed_on     the date entered in manual_removed.csv
  status, reason the fault and the instruction given to the writer
  removed_text   the post that was taken out
  new_tries      tries made after the removal, in the run that replaced it
  outcome        replaced | in list again (a later line removed the new post too)
  rewritten_utc  when the replacement was written
  in_dataset     whether the item is in synthetic_dataset.csv
  dataset_text   the post the dataset has for the item now

It calls no model and changes nothing else. generation/MANUAL_EDITS.md says
how the posts were chosen.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "output" / "raw_generations.jsonl"
DATASET = HERE / "output" / "synthetic_dataset.csv"
OUT = HERE / "output" / "manual_edits.csv"
FIELDS = ["item_id", "condition", "place_name", "removed_on", "status", "reason",
          "removed_text", "new_tries", "outcome", "rewritten_utc", "in_dataset", "dataset_text"]


def main() -> None:
    with DATASET.open(encoding="utf-8", newline="") as f:
        dataset = {row["item_id"]: row["text"] for row in csv.DictReader(f)}

    seen, edits = set(), []
    with RAW.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            log = row.get("try_log") or []
            marks = [i for i, t in enumerate(log) if t.get("manual_removed")]
            for n, i in enumerate(marks):
                t = log[i]
                # A later row of the same item repeats its earlier tries.
                key = (row["item_id"], t.get("try"), t.get("reason"), t.get("text"))
                if key in seen:
                    continue
                seen.add(key)
                end = marks[n + 1] if n + 1 < len(marks) else len(log)
                edits.append({
                    "item_id": row["item_id"],
                    "condition": row["condition"],
                    "place_name": row["place_name"],
                    "removed_on": t["manual_removed"],
                    "status": t.get("status"),
                    "reason": t.get("reason"),
                    "removed_text": t.get("text"),
                    "new_tries": end - i - 1,
                    "outcome": "replaced" if n + 1 == len(marks) else "in list again",
                    "rewritten_utc": row.get("generated_utc"),
                    "in_dataset": row["item_id"] in dataset,
                    "dataset_text": dataset.get(row["item_id"], ""),
                })

    edits.sort(key=lambda e: (e["rewritten_utc"], e["item_id"]))
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(edits)

    items = {e["item_id"] for e in edits}
    print(f"{len(edits)} removals by hand, {len(items)} items, "
          f"{sum(i in dataset for i in items)} of them in the dataset")
    for status, n in Counter(e["status"] for e in edits).most_common():
        print(f"  {status:24} {n:3} removals, {len({e['item_id'] for e in edits if e['status'] == status})} items")
    unchanged = [e for e in edits if e["in_dataset"] and e["removed_text"] == e["dataset_text"]]
    print(f"removed posts still in the dataset unchanged: {len(unchanged)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
