"""Enrich the eval_FINAL_*_{clean_sm,clean_journ,noisy_sm}.csv files that
were produced before evaluate.py started writing the within_161km column.

For each row we add:
  true_lat, true_lon  — joined from the source dataset by 'text' (with a
                        sanity check that the source qid equals the eval's
                        true_qid)
  pred_lat, pred_lon  — from Wikidata P625 of pred_qid (cached on disk)
  dist_km             — Haversine between truth and prediction
  within_161km        — True iff dist_km <= 161 km

Numerically identical to a fresh re-run because acc@161km is purely a
function of (true coords, pred_qid -> P625), neither of which changes
between the original run and now.
"""
import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
RAW = PROJECT_ROOT / "results" / "raw"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_coords import fetch_coords

EARTH_RADIUS_KM = 6371.0
ACC_RADIUS_KM = 161.0

MODES = ["two_stage", "refined", "refined_search", "refined_gemma", "gemma"]
DATASETS = {
    "clean_sm":    SRC / "social_media_dataset.csv",
    "clean_journ": SRC / "journalistic_dataset.csv",
    "noisy_sm":    SRC / "social_media_noisy.csv",
}

# Final column order matches what evaluate.py writes when it produces a
# fresh CSV (so the enriched files line up with eval_FINAL_*_geocorpora.csv).
FINAL_COLS = [
    "text", "signal_type", "true_qid", "true_country", "true_continent",
    "true_lat", "true_lon",
    "pred_qid", "pred_country", "pred_continent",
    "pred_lat", "pred_lon",
    "exact", "country_match", "continent_match",
    "dist_km", "within_161km",
]


def haversine_km(lat1, lon1, lat2, lon2):
    for v in (lat1, lon1, lat2, lon2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


def load_source(path):
    """{text: (qid, lat, lon)} dict from a source dataset."""
    df = pd.read_csv(path)
    if not {"text", "qid", "latitude", "longitude"}.issubset(df.columns):
        raise SystemExit(f"{path}: missing required columns")
    out = {}
    for _, r in df.iterrows():
        out[r["text"]] = (r["qid"], r["latitude"], r["longitude"])
    if len(out) != len(df):
        print(f"  WARN {path.name}: {len(df) - len(out)} duplicate texts collapsed")
    return out


def main():
    # Phase 1: collect every pred_qid we need to look up, across all 15 CSVs.
    targets = []
    pred_qids = set()
    for ds, src_path in DATASETS.items():
        for mode in MODES:
            csv = RAW / f"eval_FINAL_{mode}_{ds}.csv"
            if not csv.exists():
                print(f"  SKIP {csv.name} (does not exist)")
                continue
            df = pd.read_csv(csv)
            if "within_161km" in df.columns:
                print(f"  SKIP {csv.name} (already has within_161km)")
                continue
            targets.append((csv, ds, src_path, df))
            for q in df["pred_qid"].dropna().unique():
                if isinstance(q, str) and q.startswith("Q"):
                    pred_qids.add(q)

    print(f"\n{len(targets)} files to enrich, {len(pred_qids)} unique pred_qids to resolve")
    pred_coords = fetch_coords(sorted(pred_qids))
    resolved = sum(1 for v in pred_coords.values() if v)
    print(f"resolved P625 for {resolved}/{len(pred_qids)} pred_qids")

    # Phase 2: enrich each CSV.
    src_cache = {}
    for csv, ds, src_path, df in targets:
        if src_path not in src_cache:
            src_cache[src_path] = load_source(src_path)
        src_lookup = src_cache[src_path]

        true_lats, true_lons = [], []
        pred_lats, pred_lons = [], []
        dists, within = [], []
        text_misses = 0
        qid_mismatches = 0

        for _, r in df.iterrows():
            entry = src_lookup.get(r["text"])
            if entry is None:
                text_misses += 1
                t_lat = t_lon = None
            else:
                src_qid, t_lat, t_lon = entry
                if src_qid != r["true_qid"]:
                    qid_mismatches += 1
                    print(f"  WARN {csv.name} row text={r['text'][:40]!r}: "
                          f"true_qid={r['true_qid']} but source qid={src_qid}")
            true_lats.append(t_lat)
            true_lons.append(t_lon)

            pq = r["pred_qid"]
            if isinstance(pq, str) and pq.startswith("Q"):
                pc = pred_coords.get(pq)
                p_lat = pc[0] if pc else None
                p_lon = pc[1] if pc else None
            else:
                p_lat = p_lon = None
            pred_lats.append(p_lat)
            pred_lons.append(p_lon)

            d = haversine_km(t_lat, t_lon, p_lat, p_lon)
            dists.append(d)
            within.append(False if d is None else d <= ACC_RADIUS_KM)

        df["true_lat"] = true_lats
        df["true_lon"] = true_lons
        df["pred_lat"] = pred_lats
        df["pred_lon"] = pred_lons
        df["dist_km"] = dists
        df["within_161km"] = within

        # Reorder columns to match evaluate.py's canonical schema.
        df = df[[c for c in FINAL_COLS if c in df.columns]]

        df.to_csv(csv, index=False)
        n = len(df)
        wn = sum(within)
        no_pred = sum(1 for v in pred_lats if v is None or (isinstance(v, float) and math.isnan(v)))
        print(f"  {csv.name}: acc@161km = {wn}/{n} ({wn/n*100:.1f}%)  "
              f"no_pred_coords={no_pred}  text_misses={text_misses}  "
              f"qid_mismatches={qid_mismatches}")


if __name__ == "__main__":
    main()
