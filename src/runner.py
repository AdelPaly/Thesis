"""Inference runner. Computes no metrics.

    python src/runner.py --config refined --dataset social_media_clean

Runs one configuration over one dataset and writes one JSON object per item to
results/predictions/<config_name>__<dataset>.jsonl. Every item produces a
record, including failures: a failed item is recorded with `error` set, never
skipped.

The split from scoring is the point. Changing a metric definition means
re-running scripts/score.py over these files -- no GPU, no API calls, no
network. Re-running the runner itself is close to free too: response caches
(src/.llm_cache, src/.wikidata_cache) are preserved, and by default items
already present in the output file are not recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

SRC = Path(__file__).resolve().parent
REPO_ROOT = SRC.parent
sys.path.insert(0, str(SRC))
load_dotenv(SRC / ".env")
load_dotenv(REPO_ROOT / ".env")

from config import load_config, load_datasets, PipelineConfig   # noqa: E402
from pipeline import Pipeline, set_debug                        # noqa: E402

PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"


def item_id(text: str) -> str:
    """sha256 of the text, first 16 chars. No dataset carries an id column and
    row order has been in doubt, so a positional index is not a safe key. The
    text hash is free and matches what the LLM response caches key on."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def _clean(value):
    """CSV blanks arrive as NaN; store them as null."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _latlon(row):
    lat, lon = _clean(row.get("latitude")), _clean(row.get("longitude"))
    if lat is None or lon is None:
        return None
    return [float(lat), float(lon)]


def build_record(cfg: PipelineConfig, dataset: str, row, out) -> dict:
    pred = out.prediction or {}
    pred_lat, pred_lon = pred.get("lat"), pred.get("lon")
    predicted_latlon = None if pred_lat is None or pred_lon is None else [pred_lat, pred_lon]
    return {
        "item_id": item_id(row["text"]),
        "dataset": dataset,
        "config_name": cfg.name,
        "config_hash": cfg.config_hash,
        "paradigm": cfg.paradigm,
        "text": row["text"],
        "place_qid": _clean(row.get("qid")),
        "condition": _clean(row.get("signal_type")),
        "gold_latlon": _latlon(row),
        "gold_country": _clean(row.get("country")),
        "gold_continent": _clean(row.get("continent")),
        "ner_spans": out.ner_spans,
        "linked_entities": out.linked_entities,
        "candidate_places": out.candidate_places,
        "routing_branch": out.routing_branch,
        "routing_rule": out.routing_rule,
        "predicted_qid": _clean(pred.get("qid")),
        "predicted_latlon": predicted_latlon,
        "predicted_country": _clean(pred.get("country")),
        "predicted_continent": _clean(pred.get("continent")),
        "llm_raw_response": out.llm_raw_response,
        "error": out.error,
    }


def _load_existing(path: Path, config_hash: str, overwrite: bool):
    """Return {item_id: record} already on disk, for resume."""
    if overwrite or not path.exists():
        return {}
    existing = {}
    stale = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("config_hash") != config_hash:
                stale += 1
                continue
            existing[rec["item_id"]] = rec
    if stale:
        print(f"  {stale} record(s) in {path.name} were written by a different "
              f"config_hash and will be replaced")
    return existing


def run(config_name: str, dataset_name: str, limit=None, overwrite=False,
        out_dir: Path = PREDICTIONS_DIR, checkpoint_every: int = 10):
    cfg = load_config(config_name)
    datasets = load_datasets()
    if dataset_name not in datasets:
        raise SystemExit(f"unknown dataset {dataset_name!r}; "
                         f"known: {sorted(datasets)}")
    spec = datasets[dataset_name]

    df = pd.read_csv(spec.resolved_path())
    if limit is not None:
        df = df.head(limit)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.name}__{dataset_name}.jsonl"

    print(f"config   : {cfg.name} ({cfg.paradigm}) hash={cfg.config_hash[:12]}")
    print(f"dataset  : {dataset_name} -> {spec.path} ({len(df)} items)")
    print(f"output   : {out_path.relative_to(REPO_ROOT)}")

    existing = _load_existing(out_path, cfg.config_hash, overwrite)
    if existing:
        print(f"resume   : {len(existing)} item(s) already done")

    pipe = Pipeline(cfg)
    records = []
    from wikidata import save_caches

    n_err = 0
    for i, row in df.iterrows():
        iid = item_id(row["text"])
        if iid in existing:
            records.append(existing[iid])
            continue
        out = pipe.run(row["text"])
        rec = build_record(cfg, dataset_name, row, out)
        records.append(rec)
        if rec["error"]:
            n_err += 1
        status = rec["predicted_qid"] or ("ERROR" if rec["error"] else "no-result")
        branch = f" [{rec['routing_branch']}]" if rec["routing_branch"] else ""
        print(f"[{len(records)}/{len(df)}] {iid} -> {status}{branch}", flush=True)
        if len(records) % checkpoint_every == 0:
            save_caches()
            pipe.save_llm_cache()
            _write(out_path, records)

    save_caches()
    pipe.save_llm_cache()
    _write(out_path, records)

    n_pred = sum(1 for r in records if r["predicted_qid"] or r["predicted_latlon"]
                 or r["predicted_continent"])
    print(f"\nwrote {len(records)} record(s) to {out_path.relative_to(REPO_ROOT)}")
    print(f"  with a prediction : {n_pred}")
    print(f"  no result         : {len(records) - n_pred - n_err}")
    print(f"  errors            : {n_err}")
    print("no metrics computed here -- run scripts/score.py")
    return out_path


def _write(path: Path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main():
    from config import load_all_configs
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True,
                   help="configuration name or path to a configs/*.yaml file")
    p.add_argument("--dataset", required=True, help="dataset name from configs/datasets.yaml")
    p.add_argument("--limit", type=int, default=None, help="only the first N items")
    p.add_argument("--overwrite", action="store_true",
                   help="recompute every item instead of resuming")
    p.add_argument("--out-dir", default=str(PREDICTIONS_DIR))
    p.add_argument("--debug", action="store_true", help="print tracebacks for item failures")
    p.add_argument("--list", action="store_true", help="list configurations and datasets, then exit")
    args = p.parse_args()

    if args.list:
        for name, cfg in load_all_configs().items():
            print(f"{name:22} {cfg.paradigm:14} {cfg.config_hash[:12]}")
        print()
        for name, spec in load_datasets().items():
            print(f"{name:22} {spec.path}")
        return

    set_debug(args.debug)
    run(args.config, args.dataset, limit=args.limit, overwrite=args.overwrite,
        out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()
