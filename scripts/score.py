"""Scorer. Reads prediction records, computes every metric, loads no model.

    python scripts/score.py                                   # everything
    python scripts/score.py --config refined --dataset geocorpora
    python scripts/score.py --by condition

This module deliberately imports nothing from src/. It touches no model, makes
no network call, and needs no API key: it only reads
results/predictions/*.jsonl. Changing a metric definition is a re-run of this
file, not of the pipeline.

It is also the single home of the haversine implementation.

Metric definitions
------------------
has_prediction : the configuration returned something -- a QID, a coordinate,
                 or at minimum a continent.
no_result      : the complement of has_prediction. Reported per item, as a
                 column, not as a separate denominator.
exact          : predicted_qid == place_qid, over items that HAVE a gold QID.
                 place_qid is allowed to be null: some corpora have gold
                 coordinates but no gold Wikidata id, and reporting those as
                 0% exact would be wrong. The n_gold_qid column is the
                 denominator, so a dataset without gold QIDs reports exact as
                 "-" rather than zero.
country        : exact, or predicted_country == gold_country. The hierarchy is
                 derived HERE, at report time, not baked into stored booleans.
continent      : country, or predicted_continent == gold_continent.
distance_km    : haversine, only when both gold and predicted coordinates
                 exist. A prediction with no coordinate has no distance -- it
                 is never scored as a distance to 0,0.
acc@161km      : share of items WITH GOLD COORDINATES whose distance exists and
                 is <= 161 km. Items with a gold coordinate but no predicted
                 one count as failures, which is the field convention.
mean/median    : over items that have a distance, per the same convention.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
ITEM_SPLIT_PATH = REPO_ROOT / "splits" / "item_split.csv"

EARTH_RADIUS_KM = 6371.0
# 161 km is the standard ~100 mile threshold used in the geoparsing literature
# (Gritta et al.); we report this as acc@161km.
ACC_RADIUS_KM = 161.0


def haversine_km(a, b):
    """Great-circle distance in km between two [lat, lon] pairs.
    Returns None if either point is missing or incomplete -- a missing
    coordinate is not a coordinate at 0,0."""
    if not a or not b:
        return None
    lat1, lon1 = a
    lat2, lon2 = b
    for v in (lat1, lon1, lat2, lon2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(h)))


def load_item_split(path=ITEM_SPLIT_PATH):
    """{(item_id, dataset): {"place_key", "split", "condition"}} from the
    committed split file. Derived from the place-level assignment, so items
    about the same place are always on the same side."""
    import csv
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return {(r["item_id"], r["dataset"]): r for r in csv.DictReader(f)}


def attach_split(records, split_index=None):
    """Add `split` and `place_key` to each record. `place_key` is the pairing
    unit for McNemar, Wilcoxon and the bootstrap: items within a place are not
    independent, so nothing may resample or pair by item."""
    split_index = load_item_split() if split_index is None else split_index
    out = []
    for rec in records:
        info = split_index.get((rec.get("item_id"), rec.get("dataset")), {})
        out.append({
            **rec,
            "split": info.get("split"),
            "place_key": info.get("place_key") or rec.get("place_qid"),
        })
    return out


def _norm(value):
    return value.strip().casefold() if isinstance(value, str) and value.strip() else None


def score_record(rec: dict) -> dict:
    """Per-item scoring. Returns the record plus derived fields; the stored
    record itself is never mutated with metric booleans."""
    gold_qid = rec.get("place_qid")
    pred_qid = rec.get("predicted_qid")
    gold_latlon = rec.get("gold_latlon")
    pred_latlon = rec.get("predicted_latlon")

    has_prediction = bool(pred_qid or pred_latlon or rec.get("predicted_continent"))
    exact = bool(gold_qid and pred_qid and gold_qid == pred_qid)

    gold_country, pred_country = _norm(rec.get("gold_country")), _norm(rec.get("predicted_country"))
    country = exact or bool(gold_country and pred_country and gold_country == pred_country)

    gold_cont, pred_cont = _norm(rec.get("gold_continent")), _norm(rec.get("predicted_continent"))
    continent = country or bool(gold_cont and pred_cont and gold_cont == pred_cont)

    distance = haversine_km(gold_latlon, pred_latlon)

    return {
        **rec,
        "has_prediction": has_prediction,
        "no_result": not has_prediction,
        "is_error": rec.get("error") is not None,
        "exact": exact,
        "country_correct": country,
        "continent_correct": continent,
        "distance_km": distance,
        "scorable_distance": gold_latlon is not None,
        "scorable_exact": gold_qid is not None,
        "within_161km": distance is not None and distance <= ACC_RADIUS_KM,
    }


def aggregate(scored: list[dict]) -> dict:
    """Metric block for a set of scored items."""
    n = len(scored)
    if n == 0:
        return {"n": 0}
    dists = [s["distance_km"] for s in scored if s["distance_km"] is not None]
    n_scorable = sum(1 for s in scored if s["scorable_distance"])
    n_gold_qid = sum(1 for s in scored if s["scorable_exact"])
    within = sum(1 for s in scored if s["within_161km"])
    return {
        "n": n,
        "n_gold_qid": n_gold_qid,
        "exact": (sum(s["exact"] for s in scored) / n_gold_qid) if n_gold_qid else None,
        "country": sum(s["country_correct"] for s in scored) / n,
        "continent": sum(s["continent_correct"] for s in scored) / n,
        "acc@161km": (within / n_scorable) if n_scorable else None,
        "n_with_distance": len(dists),
        "median_km": statistics.median(dists) if dists else None,
        "mean_km": (sum(dists) / len(dists)) if dists else None,
        "no_result_rate": sum(s["no_result"] for s in scored) / n,
        "error_rate": sum(s["is_error"] for s in scored) / n,
    }


def load_predictions(paths) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def find_prediction_files(config=None, dataset=None, directory=PREDICTIONS_DIR):
    pattern = f"{config or '*'}__{dataset or '*'}.jsonl"
    return sorted(Path(p) for p in glob.glob(str(Path(directory) / pattern)))


def _fmt(value, pct=False, width=7):
    if value is None:
        return "-".rjust(width)
    if pct:
        return f"{value * 100:{width}.1f}"
    return f"{value:{width}.1f}"


def print_table(rows, group_label="config"):
    header = (f"{group_label:34} {'n':>4} {'nQID':>5} {'exact':>7} {'country':>7} "
              f"{'cont':>7} {'a@161':>7} {'med km':>8} {'mean km':>8} {'no-res':>7} {'err':>6}")
    print(header)
    print("-" * len(header))
    for label, m in rows:
        if m["n"] == 0:
            print(f"{label:34} {0:>4}")
            continue
        print(f"{label:34} {m['n']:>4} {m['n_gold_qid']:>5} "
              f"{_fmt(m['exact'], True)} {_fmt(m['country'], True)} "
              f"{_fmt(m['continent'], True)} {_fmt(m['acc@161km'], True)} "
              f"{_fmt(m['median_km'], width=8)} {_fmt(m['mean_km'], width=8)} "
              f"{_fmt(m['no_result_rate'], True)} {_fmt(m['error_rate'], True, 6)}")


def group_by(scored, key):
    groups = {}
    for s in scored:
        groups.setdefault(s.get(key), []).append(s)
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--predictions", nargs="*", default=None,
                   help="explicit jsonl paths; defaults to results/predictions/")
    p.add_argument("--config", default=None, help="filter by configuration name")
    p.add_argument("--dataset", default=None, help="filter by dataset name")
    p.add_argument("--split", default="all", choices=["all", "dev", "test"],
                   help="which side of the place-level split to report "
                        "(default all; configurations and thresholds are chosen "
                        "on dev, test is reported once)")
    p.add_argument("--by", default=None,
                   choices=["condition", "paradigm", "dataset"],
                   help="extra breakdown within each config/dataset pair")
    p.add_argument("--out", default=None, help="write the table to a CSV file")
    args = p.parse_args()

    paths = ([Path(x) for x in args.predictions] if args.predictions
             else find_prediction_files(args.config, args.dataset))
    if not paths:
        raise SystemExit("no prediction files found -- run src/runner.py first")

    records = attach_split(load_predictions(paths))
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
        if not records:
            raise SystemExit(f"no records on the {args.split} side")
    scored = [score_record(r) for r in records]
    print(f"split: {args.split}  records: {len(scored)}\n")
    rows = []
    for (cfg_name, ds), items in sorted(
            group_by_pairs(scored).items(), key=lambda kv: kv[0]):
        rows.append((f"{cfg_name} / {ds}", aggregate(items)))
        if args.by:
            for sub, sub_items in sorted(group_by(items, args.by).items(),
                                         key=lambda kv: str(kv[0])):
                rows.append((f"    {args.by}={sub}", aggregate(sub_items)))

    print_table(rows)

    if args.out:
        import csv
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fields = ["group", "n", "n_gold_qid", "exact", "country", "continent", "acc@161km",
                  "n_with_distance", "median_km", "mean_km", "no_result_rate", "error_rate"]
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for label, m in rows:
                row = {"group": label.strip()}
                for k in fields[1:]:
                    v = m.get(k)
                    row[k] = round(v, 4) if isinstance(v, float) else v
                w.writerow(row)
        print(f"\nwrote {out}")


def group_by_pairs(scored):
    groups = {}
    for s in scored:
        groups.setdefault((s.get("config_name"), s.get("dataset")), []).append(s)
    return groups


if __name__ == "__main__":
    main()
