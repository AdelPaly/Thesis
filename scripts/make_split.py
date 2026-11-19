"""Build the dev/test split and write it to splits/place_split.csv.

    python scripts/make_split.py            # refuses to overwrite an existing split
    python scripts/make_split.py --force    # write the files again

The split unit is a PLACE, not an item. Every item about the same place lands
on the same side of the boundary, so a configuration cannot be tuned on one
post about Bath and then reported on another post about Bath.

This script decides nothing. Each dataset in configs/datasets.yaml carries its
own `split` column (the synthetic set was split by place in
generation/reform_places.py before any post was written), and the script
copies that assignment into the two files the runner and the scorer read:

* splits/place_split.csv -- one row per place: its side, the (dataset,
  condition) of its first occurrence (the `stratum` column, for reporting),
  the number of items, and the datasets and conditions it appears in;
* splits/item_split.csv -- one row per item with the side of its place.

A row whose `split` is neither dev nor test stops the run, and so does a place
that two rows put on different sides. The output depends on the datasets
alone, so a rebuild from the same data writes the same files.

A place is keyed by its gold QID, or by its normalised location name when it
has none (src/keys.py).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from keys import item_id, place_key   # noqa: E402  (shared with runner/pipeline)

SPLIT_PATH = REPO_ROOT / "splits" / "place_split.csv"
ITEM_SPLIT_PATH = REPO_ROOT / "splits" / "item_split.csv"
DATASET_REGISTRY = REPO_ROOT / "configs" / "datasets.yaml"


def build() -> tuple[list[dict], list[tuple]]:
    """Return (rows, items), each place on the side its dataset gives it.
    Pure: reads the datasets, writes nothing."""
    registry = yaml.safe_load(DATASET_REGISTRY.read_text(encoding="utf-8"))

    places = {}          # key -> {qid, name, stratum, items, datasets, conditions}
    items = []           # (item_id, dataset, place_key, condition)
    for ds_name in registry:                      # registry order is the tie-break
        df = pd.read_csv(REPO_ROOT / registry[ds_name]["path"])
        dup = df["text"].duplicated()
        assert not dup.any(), (f"{ds_name}: {int(dup.sum())} duplicate text(s); "
                               f"item_id would collapse them")
        for row in df.itertuples(index=False):
            qid = row.qid if isinstance(row.qid, str) else ""
            name = "" if pd.isna(row.location_name) else str(row.location_name)
            key = place_key(qid, name)
            if not key:
                continue
            # NaN is truthy, so `or "none"` would yield the string "nan".
            condition = "none" if pd.isna(row.signal_type) else str(row.signal_type)
            # The dataset's own split (assigned by place before generation) is
            # used as is.
            own = getattr(row, "split", None)
            if own not in ("dev", "test"):
                raise SystemExit(f"{ds_name}: {key} has split {own!r}; every row "
                                 f"needs dev or test in its `split` column")
            entry = places.get(key)
            if entry is not None and entry["split"] != own:
                raise SystemExit(f"{key}: {ds_name} puts it in {own}, but it is already "
                                 f"{entry['split']} from {entry['stratum']}")
            if entry is None:
                entry = places[key] = {
                    "place_key": key,
                    "place_qid": qid,
                    "location_name": name,
                    "stratum": f"{ds_name}|{condition}",
                    "n_items": 0,
                    "datasets": set(),
                    "conditions": set(),
                    "split": own,
                }
            entry["n_items"] += 1
            entry["datasets"].add(ds_name)
            entry["conditions"].add(condition)
            items.append((item_id(row.text), ds_name, key, condition))

    rows = sorted(places.values(), key=lambda e: (e["stratum"], e["place_key"]))
    return rows, items


def write_items(rows: list[dict], items: list[tuple], path: Path = ITEM_SPLIT_PATH) -> Path:
    """The item-level view of the same assignment.

    Derived from the place assignment, never decided independently. It exists
    so the scorer can attach a split to a prediction record by item_id without
    reading the datasets -- the scorer is not allowed to reach into src/ or
    reconstruct gold data, and a record does not carry location_name, so a
    place key cannot always be rebuilt from the record alone.
    """
    side = {r["place_key"]: r["split"] for r in rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "dataset", "place_key", "condition", "split"])
        for iid, ds, key, condition in items:
            w.writerow([iid, ds, key, condition, side[key]])
    return path


def write(rows: list[dict], path: Path = SPLIT_PATH) -> Path:
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


def summarise(rows: list[dict]) -> None:
    if not rows:
        print("no places found")
        return
    print(f"places: {len(rows)}   items: {sum(r['n_items'] for r in rows)}")
    print()

    per_ds = defaultdict(lambda: defaultdict(int))   # ds -> split -> places
    for r in rows:
        for ds in r["datasets"]:
            per_ds[ds][r["split"]] += 1

    print(f"{'dataset':22} {'dev places':>10} {'test places':>11}")
    print("-" * 46)
    for ds in sorted(per_ds):
        print(f"{ds:22} {per_ds[ds]['dev']:>10} {per_ds[ds]['test']:>11}")

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
          f"({dev_items / max(all_items, 1):.1%} dev)")
    multi = sum(1 for r in rows if len(r["datasets"]) > 1)
    print(f"       {multi} place(s) appear in more than one dataset "
          f"(single global assignment each)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true",
                   help="overwrite existing split files")
    p.add_argument("--out", default=str(SPLIT_PATH), help="place split output path")
    p.add_argument("--items-out", default=None,
                   help="item split output path (default: splits/item_split.csv "
                        "when --out is the default, else item_split.csv beside --out)")
    args = p.parse_args()

    out = Path(args.out)
    if args.items_out:
        items_out = Path(args.items_out)
    elif out.resolve() == SPLIT_PATH.resolve():
        items_out = ITEM_SPLIT_PATH
    else:
        items_out = out.with_name("item_split.csv")

    # Both or neither: check every target before writing anything.
    clash = [str(t) for t in (out, items_out) if t.exists()]
    if clash and not args.force:
        raise SystemExit(
            f"{', '.join(clash)} already exists. The split is meant to be stable; "
            f"--force writes it again from the datasets' own split column.")

    rows, items = build()
    summarise(rows)
    print(f"\nwrote {write(rows, out)}")
    print(f"wrote {write_items(rows, items, items_out)}")


if __name__ == "__main__":
    main()
