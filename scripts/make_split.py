"""Build the dev/test split and write it to splits/place_split.csv.

    python scripts/make_split.py            # rebuild (refuses to clobber silently)
    python scripts/make_split.py --force

The split unit is a PLACE, not an item. Every item about the same place lands
on the same side of the boundary, so a configuration cannot be tuned on one
post about Bath and then reported on another post about Bath.

The assignment is:

* global -- a place keeps one side across every dataset it appears in, so a
  place seen in both the social-media and the GeoCorpora sets cannot be dev in
  one and test in the other;
* stratified by (dataset, condition) of the place's first occurrence, so both
  sides cover every signal condition and every corpus;
* deterministic -- a place's rank inside its stratum is sha256(place_key +
  SALT), so the assignment does not depend on row order, and row order in
  these CSVs has been in doubt before;
* committed -- the output file is the assignment. Rebuilding is only needed
  when a dataset gains rows, and then only the new places move.

Places with no gold QID are keyed by their normalised location name instead,
so corpora with coordinates but no QIDs still split by place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_PATH = REPO_ROOT / "splits" / "place_split.csv"
ITEM_SPLIT_PATH = REPO_ROOT / "splits" / "item_split.csv"
DATASET_REGISTRY = REPO_ROOT / "configs" / "datasets.yaml"

# Changing either of these changes the assignment, which invalidates every
# threshold fitted on dev. They are constants on purpose.
SALT = "geo-dev-test-split-v1"
DEV_FRACTION = 0.30


def place_key(qid, location_name):
    """The split unit. A gold QID when there is one, else the place's name."""
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    name = str(location_name or "").strip().casefold()
    return f"name:{name}" if name else ""


def item_id(text):
    """Must match runner.item_id -- it is the join key between the split and
    the prediction records."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _rank(key):
    return hashlib.sha256((SALT + key).encode("utf-8")).hexdigest()


def build():
    registry = yaml.safe_load(DATASET_REGISTRY.read_text(encoding="utf-8"))

    places = {}          # key -> {qid, name, stratum, items, datasets, conditions}
    items = []           # (item_id, dataset, place_key, condition)
    for ds_name in registry:                      # registry order is the tie-break
        df = pd.read_csv(REPO_ROOT / registry[ds_name]["path"])
        for _, row in df.iterrows():
            key = place_key(row.get("qid"), row.get("location_name"))
            if not key:
                continue
            # NaN is truthy, so `or "none"` would yield the string "nan".
            raw_condition = row.get("signal_type")
            condition = "none" if pd.isna(raw_condition) else str(raw_condition)
            entry = places.get(key)
            if entry is None:
                entry = places[key] = {
                    "place_key": key,
                    "place_qid": row.get("qid") if isinstance(row.get("qid"), str) else "",
                    "location_name": str(row.get("location_name") or ""),
                    "stratum": f"{ds_name}|{condition}",
                    "n_items": 0,
                    "datasets": set(),
                    "conditions": set(),
                }
            entry["n_items"] += 1
            entry["datasets"].add(ds_name)
            entry["conditions"].add(condition)
            items.append((item_id(row["text"]), ds_name, key, condition))

    by_stratum = defaultdict(list)
    for entry in places.values():
        by_stratum[entry["stratum"]].append(entry)

    for stratum, entries in by_stratum.items():
        entries.sort(key=lambda e: _rank(e["place_key"]))
        n_dev = round(len(entries) * DEV_FRACTION)
        # Never let a stratum be all-dev or all-test when it has >1 place.
        if len(entries) > 1:
            n_dev = max(1, min(len(entries) - 1, n_dev))
        for i, entry in enumerate(entries):
            entry["split"] = "dev" if i < n_dev else "test"

    rows = sorted(places.values(), key=lambda e: (e["stratum"], e["place_key"]))
    return rows, items


def write_items(rows, items, path=ITEM_SPLIT_PATH):
    """The item-level view of the same assignment.

    Derived from the place assignment, never decided independently. It exists
    so the scorer can attach a split to a prediction record by item_id without
    reading the datasets -- the scorer is not allowed to reach into src/ or
    reconstruct gold data, and a record does not carry location_name, so a
    place key cannot always be rebuilt from the record alone.
    """
    side = {r["place_key"]: r["split"] for r in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "dataset", "place_key", "condition", "split"])
        for iid, ds, key, condition in items:
            if (iid, ds) in seen:
                continue
            seen.add((iid, ds))
            w.writerow([iid, ds, key, condition, side[key]])
    return path


def write(rows, path=SPLIT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["place_key", "place_qid", "location_name", "split", "stratum",
              "n_items", "datasets", "conditions"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "place_key": r["place_key"],
                "place_qid": r["place_qid"],
                "location_name": r["location_name"],
                "split": r["split"],
                "stratum": r["stratum"],
                "n_items": r["n_items"],
                "datasets": ";".join(sorted(r["datasets"])),
                "conditions": ";".join(sorted(r["conditions"])),
            })
    return path


def summarise(rows):
    print(f"places: {len(rows)}   items: {sum(r['n_items'] for r in rows)}")
    print(f"dev fraction requested: {DEV_FRACTION}  salt: {SALT}\n")

    per_ds = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # ds -> split -> [places, items]
    for r in rows:
        for ds in r["datasets"]:
            cell = per_ds[ds][r["split"]]
            cell[0] += 1
            cell[1] += r["n_items"] if len(r["datasets"]) == 1 else 0

    print(f"{'dataset':22} {'dev places':>10} {'test places':>11}")
    print("-" * 46)
    for ds in sorted(per_ds):
        print(f"{ds:22} {per_ds[ds]['dev'][0]:>10} {per_ds[ds]['test'][0]:>11}")

    print(f"\n{'condition':22} {'dev places':>10} {'test places':>11}")
    print("-" * 46)
    per_cond = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for c in r["conditions"]:
            per_cond[c][r["split"]] += 1
    for c in sorted(per_cond):
        print(f"{c:22} {per_cond[c]['dev']:>10} {per_cond[c]['test']:>11}")

    n_dev = sum(1 for r in rows if r["split"] == "dev")
    dev_items = sum(r["n_items"] for r in rows if r["split"] == "dev")
    all_items = sum(r["n_items"] for r in rows)
    print(f"\ntotal: {n_dev} dev places / {len(rows) - n_dev} test places")
    print(f"       {dev_items} dev items / {all_items - dev_items} test items "
          f"({dev_items / all_items:.1%} dev)")
    multi = sum(1 for r in rows if len(r["datasets"]) > 1)
    print(f"       {multi} place(s) appear in more than one dataset "
          f"(single global assignment each)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="overwrite an existing split file")
    p.add_argument("--out", default=str(SPLIT_PATH))
    args = p.parse_args()

    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(
            f"{out} already exists. The split is meant to be stable across runs; "
            f"pass --force only if you intend to move places between dev and test "
            f"(which invalidates every threshold fitted on dev).")

    rows, items = build()
    summarise(rows)
    print(f"\nwrote {write(rows, out)}")
    print(f"wrote {write_items(rows, items)}")


if __name__ == "__main__":
    main()
