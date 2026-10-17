"""Build the dev/test split and write it to splits/place_split.csv.

    python scripts/make_split.py            # refuses to clobber an existing split
    python scripts/make_split.py --force    # rebuild; existing places keep their side

The split unit is a PLACE, not an item. Every item about the same place lands
on the same side of the boundary, so a configuration cannot be tuned on one
post about Bath and then reported on another post about Bath.

The assignment is:

* global -- a place keeps one side across every dataset it appears in, so a
  place seen in both the social-media and the GeoCorpora sets cannot be dev in
  one and test in the other;
* labelled with the (dataset, condition) of the place's first occurrence (the
  `stratum` column), for reporting;
* deterministic -- a place not yet in the committed split goes to dev when
  sha256(SALT + place_key), read as a fraction of 2**256, is below
  DEV_FRACTION. That depends on the place alone, never on how many other
  places exist or on row order (which has been in doubt before);
* committed -- the existing place_split.csv IS the assignment. On a rebuild
  every place already in it keeps its side; only unseen places are assigned.
  Nothing ever moves, so thresholds fitted on dev stay valid.

Places with no gold QID are keyed by their normalised location name instead,
so corpora with coordinates but no QIDs still split by place.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

# Changing either of these changes the assignment of unseen places. They are
# constants on purpose.
SALT = "geo-dev-test-split-v1"
DEV_FRACTION = 0.30


def _rank(key: str) -> str:
    return hashlib.sha256((SALT + key).encode("utf-8")).hexdigest()


def hash_side(key: str) -> str:
    """Side for an unseen place, from its hash alone (no rounding, no
    dependence on stratum size)."""
    return "dev" if int(_rank(key), 16) / 16 ** 64 < DEV_FRACTION else "test"


def load_existing(path: Path = SPLIT_PATH) -> dict[str, str]:
    """place_key -> side from a committed split, or {} when there is none."""
    if not path.exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False)
    return dict(zip(df["place_key"], df["split"]))


def build(existing: dict[str, str] | None = None) -> tuple[list[dict], list[tuple]]:
    """Return (rows, items). Places in `existing` keep their side; the rest are
    assigned by hash_side. Pure: reads the datasets, writes nothing."""
    existing = existing or {}
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
            entry = places.get(key)
            if entry is None:
                entry = places[key] = {
                    "place_key": key,
                    "place_qid": qid,
                    "location_name": name,
                    "stratum": f"{ds_name}|{condition}",
                    "n_items": 0,
                    "datasets": set(),
                    "conditions": set(),
                    "split": existing.get(key) or hash_side(key),
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
    print(f"dev fraction for unseen places: {DEV_FRACTION}  salt: {SALT}\n")

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
                   help="overwrite existing split files (existing places keep their side)")
    p.add_argument("--out", default=str(SPLIT_PATH), help="place split output path")
    p.add_argument("--items-out", default=None,
                   help="item split output path (default: splits/item_split.csv "
                        "when --out is the default, else item_split.csv beside --out)")
    p.add_argument("--existing", default=str(SPLIT_PATH),
                   help="committed place split whose assignments are kept")
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
            f"--force rebuilds it, and every place already in {args.existing} "
            f"keeps its side, so only unseen places are assigned.")

    rows, items = build(load_existing(Path(args.existing)))
    summarise(rows)
    print(f"\nwrote {write(rows, out)}")
    print(f"wrote {write_items(rows, items, items_out)}")


if __name__ == "__main__":
    main()
