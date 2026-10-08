"""Scorer. Reads prediction records, computes every metric, loads no model.

    python scripts/score.py                                   # everything
    python scripts/score.py --config refined --dataset geocorpora
    python scripts/score.py --by condition

It touches no model, makes no network call, and needs no API key: it only reads
results/predictions/*.jsonl and splits/item_split.csv (the one src/ import is
the stdlib-only key helper in src/keys.py). Changing a metric definition is a
re-run of this file, not of the pipeline.

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
                 Denominator: items that have a gold country.
continent      : country, or predicted_continent == gold_continent.
                 Denominator: items that have a gold continent.
no_result_rate, error_rate : over all items.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from keys import current_config_name, place_key  # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
CONFIG_DIR = REPO_ROOT / "configs"
ITEM_SPLIT_PATH = REPO_ROOT / "splits" / "item_split.csv"

LABEL_WIDTH = 34
CSV_FIELDS = ["group", "n", "n_gold_qid", "exact", "country", "continent",
              "no_result_rate", "error_rate"]


def config_order(config_dir: Path | str = CONFIG_DIR) -> dict[str, tuple]:
    """{config name: (tier, variant)} read from configs/*.yaml. Reading YAML is
    not loading a model: it only orders the rows."""
    import yaml

    out = {}
    for path in sorted(Path(config_dir).glob("*.yaml")):
        if path.name == "datasets.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            print(f"warning: skipping unreadable config {path.name}: {exc}", file=sys.stderr)
            continue
        if isinstance(raw, dict) and raw.get("name"):
            out[raw["name"]] = (raw.get("tier"), raw.get("variant") or "")
    return out


def row_order(name: str, order: dict) -> tuple:
    """Sort key for a report row: tier, then variant, then name. A name with no
    config file left sorts after every tier."""
    tier, variant = order.get(name, (None, ""))
    return (tier if isinstance(tier, int) else 99, variant, str(name))


def pct(num: int, den: int) -> float | None:
    """num / den, or None when the denominator is empty."""
    return num / den if den else None


def group_by(records: Iterable[dict], key: str | Callable[[dict], object]) -> dict:
    """Group records by a field name or by a key function."""
    def key_fn(r: dict):
        return r.get(key) if isinstance(key, str) else key(r)

    groups: dict = {}
    for r in records:
        groups.setdefault(key_fn(r), []).append(r)
    return groups


def group_by_pairs(scored: Iterable[dict]) -> dict:
    return group_by(scored, lambda r: (r.get("config_name"), r.get("dataset")))


def load_item_split(path: Path | str = ITEM_SPLIT_PATH) -> dict:
    """{(item_id, dataset): {"place_key", "split", "condition"}} from the
    committed split file. Derived from the place-level assignment, so items
    about the same place are always on the same side."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing -- run scripts/make_split.py first; without it "
            "no record can be assigned to dev/test or to a place")
    with path.open(encoding="utf-8") as f:
        return {(r["item_id"], r["dataset"]): r for r in csv.DictReader(f)}


def attach_split(records: list[dict], split_index: dict | None = None) -> list[dict]:
    """Add `split` and `place_key` to each record. `place_key` is the pairing
    unit for McNemar and the bootstrap: items within a place are not
    independent, so nothing may resample or pair by item. Records absent from
    the split index (stale split) get split=None and are counted in a warning."""
    split_index = load_item_split() if split_index is None else split_index
    out = []
    missing = 0
    for rec in records:
        info = split_index.get((rec.get("item_id"), rec.get("dataset")))
        if info is None:
            missing += 1
            info = {}
        out.append({
            **rec,
            "split": info.get("split"),
            "place_key": info.get("place_key")
                         or place_key(rec.get("place_qid"), rec.get("location_name")),
        })
    if missing:
        print(f"warning: {missing} of {len(out)} record(s) not found in the item "
              "split (split=None); re-run scripts/make_split.py", file=sys.stderr)
    return out


def _norm(value) -> str | None:
    return value.strip().casefold() if isinstance(value, str) and value.strip() else None


def score_record(rec: dict) -> dict:
    """Per-item scoring. Returns the record plus derived fields; the stored
    record itself is never mutated with metric booleans."""
    gold_qid = rec.get("place_qid")
    pred_qid = rec.get("predicted_qid")

    has_prediction = bool(pred_qid or rec.get("predicted_latlon")
                          or rec.get("predicted_continent"))
    exact = bool(gold_qid and pred_qid and gold_qid == pred_qid)

    gold_country, pred_country = _norm(rec.get("gold_country")), _norm(rec.get("predicted_country"))
    country = exact or bool(gold_country and pred_country and gold_country == pred_country)

    gold_cont, pred_cont = _norm(rec.get("gold_continent")), _norm(rec.get("predicted_continent"))
    continent = country or bool(gold_cont and pred_cont and gold_cont == pred_cont)

    return {
        **rec,
        "has_prediction": has_prediction,
        "no_result": not has_prediction,
        "is_error": rec.get("error") is not None,
        "exact": exact,
        "country_correct": country,
        "continent_correct": continent,
        "scorable_exact": gold_qid is not None,
        "scorable_country": gold_country is not None,
        "scorable_continent": gold_cont is not None,
    }


def summarise_metrics(scored: list[dict]) -> dict:
    """Metric block for a set of scored items."""
    n = len(scored)
    if n == 0:
        return {"n": 0}
    n_gold_qid = sum(s["scorable_exact"] for s in scored)
    n_gold_country = sum(s["scorable_country"] for s in scored)
    n_gold_continent = sum(s["scorable_continent"] for s in scored)
    return {
        "n": n,
        "n_gold_qid": n_gold_qid,
        "n_gold_country": n_gold_country,
        "n_gold_continent": n_gold_continent,
        "exact": pct(sum(s["exact"] for s in scored), n_gold_qid),
        "country": pct(sum(s["country_correct"] for s in scored if s["scorable_country"]),
                       n_gold_country),
        "continent": pct(sum(s["continent_correct"] for s in scored if s["scorable_continent"]),
                         n_gold_continent),
        "no_result_rate": sum(s["no_result"] for s in scored) / n,
        "error_rate": sum(s["is_error"] for s in scored) / n,
    }


aggregate = summarise_metrics  # name analyse.py still imports


def load_predictions(paths: Iterable[Path | str]) -> list[dict]:
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    rec["config_name"] = current_config_name(rec.get("config_name"))
                    records.append(rec)
    keys = Counter((r.get("config_name"), r.get("dataset"), r.get("item_id")) for r in records)
    dupes = [k for k, c in keys.items() if c > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate (config_name, dataset, item_id), "
                         f"e.g. {dupes[0]}")
    hashes: dict = {}
    for r in records:
        hashes.setdefault(r.get("config_name"), set()).add(r.get("config_hash"))
    for cfg, hs in sorted(hashes.items(), key=lambda kv: str(kv[0])):
        if len(hs) > 1:
            print(f"warning: config {cfg} has {len(hs)} different config_hash "
                  "values across its predictions", file=sys.stderr)
    return records


def prediction_file_key(path: Path) -> tuple[str, str]:
    """(current config name, dataset) for a predictions file name."""
    cfg, dataset = Path(path).stem.split("__", 1)
    return current_config_name(cfg), dataset


def find_prediction_files(config: str | None = None, dataset: str | None = None,
                          directory: Path | str = PREDICTIONS_DIR) -> list[Path]:
    """One file per (config, dataset). A file under a renamed configuration's
    old name is used only when no file under the current name exists."""
    config = current_config_name(config)
    chosen: dict[tuple[str, str], Path] = {}
    for path in sorted(Path(directory).glob("*__*.jsonl")):
        key = prediction_file_key(path)
        if (config and key[0] != config) or (dataset and key[1] != dataset):
            continue
        is_legacy = path.stem.split("__", 1)[0] != key[0]
        if key in chosen and is_legacy:
            continue
        chosen[key] = path
    return sorted(chosen.values())


def _fmt(value: float | None, as_pct: bool = False, width: int = 7) -> str:
    if value is None:
        return "-".rjust(width)
    return f"{value * 100 if as_pct else value:{width}.1f}"


def print_table(rows: list[tuple[str, dict]], group_label: str = "config") -> None:
    header = (f"{group_label:{LABEL_WIDTH}} {'n':>4} {'nQID':>5} {'exact':>7} "
              f"{'country':>7} {'cont':>7} {'no-res':>7} {'err':>6}")
    print(header)
    print("-" * len(header))
    for label, m in rows:
        if m["n"] == 0:
            print(f"{label:{LABEL_WIDTH}} {0:>4}")
            continue
        print(f"{label:{LABEL_WIDTH}} {m['n']:>4} {m['n_gold_qid']:>5} "
              f"{_fmt(m['exact'], True)} {_fmt(m['country'], True)} "
              f"{_fmt(m['continent'], True)} "
              f"{_fmt(m['no_result_rate'], True)} {_fmt(m['error_rate'], True, 6)}")


def write_csv(rows: list[tuple[str, dict]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for label, m in rows:
            row = {"group": label.strip()}
            for k in CSV_FIELDS[1:]:
                v = m.get(k)
                row[k] = round(v, 4) if isinstance(v, float) else v
            w.writerow(row)
    print(f"\nwrote {out}")


def main() -> None:
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
    order = config_order()
    for (cfg_name, ds), items in sorted(group_by_pairs(scored).items(),
                                        key=lambda kv: (row_order(kv[0][0], order), kv[0][1])):
        rows.append((f"{cfg_name} / {ds}", summarise_metrics(items)))
        if args.by:
            for sub, sub_items in sorted(group_by(items, args.by).items(),
                                         key=lambda kv: str(kv[0])):
                rows.append((f"    {args.by}={sub}", summarise_metrics(sub_items)))

    print_table(rows)
    if args.out:
        write_csv(rows, Path(args.out))


if __name__ == "__main__":
    main()
