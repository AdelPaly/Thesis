"""Worker invoked by eval_acc161.py — runs a single pipeline mode on every
row of the input CSV and writes the distance-only results.

Run via subprocess from src/ so the existing relative imports in main.py
and wikidata.py resolve correctly.
"""
import math
import sys
from pathlib import Path

# Ensure src/ is on the import path so the pipeline modules (main, wikidata,
# llm_gemma, ...) resolve regardless of which directory the launcher set as
# cwd. The launcher already chdirs into src/, but sys.path[0] is set from
# the script location, not cwd, so the import would fail without this.
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(SRC_DIR / ".env")

from main import (geolocate, geolocate_refined, geolocate_refined_search,
                  geolocate_refined_gemma, geolocate_refined_groq_oss,
                  geolocate_refined_groq_llama)
from wikidata import save_caches

EARTH_RADIUS_KM = 6371.0
ACC_RADIUS_KM = 161.0


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


def _pick_text_col(df):
    for c in ("tweet_text", "text"):
        if c in df.columns:
            return c
    raise SystemExit("input CSV needs a 'tweet_text' or 'text' column")


def _pick_coord_cols(df):
    # Prefer the harness-style names, fall back to GeoCorpora-style names.
    if "true_lat" in df.columns and "true_lon" in df.columns:
        return "true_lat", "true_lon"
    if "latitude" in df.columns and "longitude" in df.columns:
        return "latitude", "longitude"
    raise SystemExit("input CSV needs latitude/longitude (or true_lat/true_lon) columns")


MODE_FN = {
    "two_stage":          geolocate,
    "refined":            geolocate_refined,
    "refined_search":     geolocate_refined_search,
    "refined_gemma":      geolocate_refined_gemma,
    "refined_groq_oss":   geolocate_refined_groq_oss,
    "refined_groq_llama": geolocate_refined_groq_llama,
    # *_only modes are loaded lazily because llm_gemma / llm_groq init
    # an API client at module-import time.
    "gemma":              None,
    "groq_oss":           None,
    "groq_llama":         None,
}


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: _eval_acc161_worker.py <csv_in> <mode> <csv_out>")
    csv_in, mode, csv_out = sys.argv[1], sys.argv[2], sys.argv[3]

    df = pd.read_csv(csv_in)
    text_col = _pick_text_col(df)
    lat_col, lon_col = _pick_coord_cols(df)
    print(f"Input: {len(df)} rows, text='{text_col}', truth=('{lat_col}','{lon_col}')")

    llm_save = None
    if mode == "gemma":
        from llm_gemma import geolocate_gemma, save_cache as llm_save
        predict = geolocate_gemma
    elif mode == "groq_oss":
        from llm_groq import geolocate_groq_oss, save_cache_oss as llm_save
        predict = geolocate_groq_oss
    elif mode == "groq_llama":
        from llm_groq import geolocate_groq_llama, save_cache_llama as llm_save
        predict = geolocate_groq_llama
    else:
        predict = MODE_FN[mode]

    rows_out = []
    for i, row in df.iterrows():
        if i > 0 and i % 10 == 0:
            save_caches()
            if llm_save is not None:
                llm_save()

        text = row[text_col]
        true_lat = row[lat_col]
        true_lon = row[lon_col]

        try:
            pred = predict(text)
        except Exception as e:
            print(f"[{i+1}/{len(df)}] ERROR ({type(e).__name__}: {e})")
            pred = None

        if pred is None:
            dist = None
            within = False
            pred_lat = None
            pred_lon = None
            pred_qid = ""
            print(f"[{i+1}/{len(df)}] NO RESULT")
        else:
            pred_lat = pred.get("lat")
            pred_lon = pred.get("lon")
            pred_qid = pred.get("qid", "")
            dist = haversine_km(true_lat, true_lon, pred_lat, pred_lon)
            within = (dist is not None) and (dist <= ACC_RADIUS_KM)
            dist_str = f"{dist:.1f}km" if dist is not None else "no-coords"
            mark = "WITHIN" if within else "OUTSIDE"
            print(f"[{i+1}/{len(df)}] {mark} {dist_str} | pred={pred_qid} | text={text[:60]!r}")

        rows_out.append({
            "tweet_text":   text,
            "true_lat":     true_lat,
            "true_lon":     true_lon,
            "pred_qid":     pred_qid,
            "pred_lat":     pred_lat,
            "pred_lon":     pred_lon,
            "dist_km":      dist,
            "within_161km": within,
        })

    save_caches()
    if llm_save is not None:
        llm_save()

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(csv_out, index=False)
    n = len(out_df)
    within = int(out_df["within_161km"].sum())
    print(f"\nDone: acc@161km = {within}/{n} ({within/n*100:.1f}%)  ->  {csv_out}")


if __name__ == "__main__":
    main()
