"""Distance-only evaluation for a CSV that has ground-truth coordinates
but no QIDs (e.g. raw GeoCorpora rows).

Reads `tweet_text`, `latitude`, `longitude` (or falls back to `text`/`true_lat`/`true_lon`)
from the input CSV, runs the pipeline in each requested mode, computes
Haversine distance between predicted and ground-truth coordinates, and
reports accuracy at 161 km (~100 mi).

Usage:
    python eval_acc161.py src/geocorpora_144_unique_tweets.csv
    python eval_acc161.py src/geocorpora_144_unique_tweets.csv --modes refined refined_gemma

Outputs (per mode):
    results/raw/eval_acc161_<mode>_<stem>.csv
    results/logs/log_acc161_<mode>_<stem>.txt
"""
import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
LOG_DIR = PROJECT_ROOT / "results" / "logs"

EARTH_RADIUS_KM = 6371.0
ACC_RADIUS_KM = 161.0  # ~100 mi

ALL_MODES = ["two_stage", "refined", "refined_search",
             "refined_gemma", "gemma",
             "refined_groq_oss", "groq_oss",
             "refined_groq_llama", "groq_llama"]


def haversine_km(lat1, lon1, lat2, lon2):
    for v in (lat1, lon1, lat2, lon2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


def run_mode(csv_in, mode, out_csv, log_path):
    """Spawn a child process that runs the named pipeline mode on every row
    of csv_in, writing distance results to out_csv. We shell out so each
    mode's wikidata cache lifecycle (load → run → save) matches how
    evaluate.py / run_chapter4.py treat it."""
    runner = Path(__file__).resolve().parent / "_eval_acc161_worker.py"
    cmd = [sys.executable, str(runner), str(csv_in), mode, str(out_csv)]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    with open(log_path, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.run(cmd, cwd=str(SRC), env=env,
                              stdout=logf, stderr=subprocess.STDOUT, check=False)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Input CSV with tweet_text + latitude/longitude")
    parser.add_argument("--modes", nargs="+", default=["refined", "refined_search",
                                                      "refined_gemma", "gemma"],
                        choices=ALL_MODES,
                        help="Which pipeline modes to run (default: 4 non-two_stage modes)")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise SystemExit(f"input CSV not found: {csv_path}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stem = csv_path.stem
    summary_rows = []
    for i, mode in enumerate(args.modes, start=1):
        out_csv = RAW_DIR / f"eval_acc161_{mode}_{stem}.csv"
        log_path = LOG_DIR / f"log_acc161_{mode}_{stem}.txt"
        if out_csv.exists():
            print(f"[{i}/{len(args.modes)}] SKIP {mode} (output exists: {out_csv.name})")
        else:
            print(f"[{i}/{len(args.modes)}] running {mode}...", flush=True)
            rc = run_mode(csv_path, mode, out_csv, log_path)
            if rc != 0 or not out_csv.exists():
                print(f"   FAILED (exit={rc}) — see {log_path.relative_to(PROJECT_ROOT)}")
                continue

        df = pd.read_csv(out_csv)
        n = len(df)
        within = int(df["within_161km"].fillna(False).astype(bool).sum())
        no_pred = int(df["pred_lat"].isna().sum())
        avg_dist = df["dist_km"].dropna().mean() if df["dist_km"].notna().any() else float("nan")
        median_dist = df["dist_km"].dropna().median() if df["dist_km"].notna().any() else float("nan")
        print(f"   acc@161km = {within}/{n} ({within/n*100:.1f}%)  "
              f"no_pred={no_pred}  median_dist={median_dist:.1f}km  mean_dist={avg_dist:.1f}km")
        summary_rows.append(dict(mode=mode, n=n, within_161km=within,
                                 acc_161km_pct=within / n * 100,
                                 no_prediction=no_pred,
                                 median_dist_km=median_dist,
                                 mean_dist_km=avg_dist))

    if summary_rows:
        summary = pd.DataFrame(summary_rows)
        summary_path = PROJECT_ROOT / "results" / "summary" / f"acc161_summary_{stem}.csv"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path, index=False, float_format="%.1f")
        print()
        print(summary.to_string(index=False))
        print()
        print(f"Summary written to {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
