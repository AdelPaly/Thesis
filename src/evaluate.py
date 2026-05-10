import math

import pandas as pd
import argparse
from dotenv import load_dotenv
load_dotenv()
from main import (geolocate, geolocate_refined, geolocate_refined_search,
                  geolocate_refined_gemma, geolocate_refined_groq_oss,
                  geolocate_refined_groq_llama)
from wikidata import save_caches

EARTH_RADIUS_KM = 6371.0
# 161 km is the standard ~100 mile threshold used in the geoparsing literature
# (e.g. Gritta et al.); we report this as acc@161km.
ACC_RADIUS_KM = 161.0


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Returns None if any coord is missing."""
    for v in (lat1, lon1, lat2, lon2):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return EARTH_RADIUS_KM * 2 * math.asin(min(1.0, math.sqrt(a)))


def _print_summary(label, sub_df):
    total = len(sub_df)
    if total == 0:
        print(f"{label}: (no rows)")
        return
    no_result = (sub_df["pred_qid"] == "").sum()
    exact = sub_df["exact"].sum()
    country = sub_df["country_match"].sum()
    continent = sub_df["continent_match"].sum()
    print(f"{label} (n={total})")
    print(f"  Exact location:  {exact:>3}/{total}  ({exact/total*100:5.1f}%)")
    print(f"  Country level:   {country:>3}/{total}  ({country/total*100:5.1f}%)")
    print(f"  Continent level: {continent:>3}/{total}  ({continent/total*100:5.1f}%)")
    if "within_161km" in sub_df.columns:
        within = int(sub_df["within_161km"].sum())
        print(f"  Within 161 km:   {within:>3}/{total}  ({within/total*100:5.1f}%)")
    print(f"  No result:       {no_result:>3}/{total}  ({no_result/total*100:5.1f}%)")

def evaluate(dataset_path, limit=None, mode="two_stage", output="eval_results.csv"):
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows_out = []
    llm_save = None
    # LLM-backed modes share their per-model cache; load the saver lazily so
    # non-LLM modes don't pay for client init.
    geolocate_gemma = None
    geolocate_groq_oss = None
    geolocate_groq_llama = None
    if mode in ("gemma", "refined_gemma"):
        from llm_gemma import geolocate_gemma, save_cache as llm_save
    elif mode in ("groq_oss", "refined_groq_oss"):
        from llm_groq import geolocate_groq_oss, save_cache_oss as llm_save
    elif mode in ("groq_llama", "refined_groq_llama"):
        from llm_groq import geolocate_groq_llama, save_cache_llama as llm_save

    predict = {
        "two_stage":          geolocate,
        "refined":            geolocate_refined,
        "refined_search":     geolocate_refined_search,
        "refined_gemma":      geolocate_refined_gemma,
        "refined_groq_oss":   geolocate_refined_groq_oss,
        "refined_groq_llama": geolocate_refined_groq_llama,
        "gemma":              geolocate_gemma,
        "groq_oss":           geolocate_groq_oss,
        "groq_llama":         geolocate_groq_llama,
    }[mode]

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['location_name']}...", end=" ", flush=True)
        if i > 0 and i % 10 == 0:
            save_caches()
            if llm_save:
                llm_save()

        try:
            pred = predict(row["text"])
        except Exception as e:
            print(f"ERROR ({type(e).__name__})")
            pred = None

        true_lat = row.get("latitude") if "latitude" in df.columns else None
        true_lon = row.get("longitude") if "longitude" in df.columns else None

        if pred is None:
            print("NO RESULT")
            rows_out.append({
                "text": row["text"],
                "signal_type": row.get("signal_type", ""),
                "true_qid": row["qid"],
                "true_country": row["country"],
                "true_continent": row["continent"],
                "true_lat": true_lat,
                "true_lon": true_lon,
                "pred_qid": "",
                "pred_country": "",
                "pred_continent": "",
                "pred_lat": None,
                "pred_lon": None,
                "exact": False,
                "country_match": False,
                "continent_match": False,
                "dist_km": None,
                "within_161km": False,
            })
            continue

        exact = pred.get("qid", "") == row["qid"]
        country_match = pred.get("country", "").lower() == row["country"].lower() or exact
        continent_match = (
            pred.get("continent", "").lower() == row["continent"].lower()
            or country_match
        )

        pred_lat = pred.get("lat")
        pred_lon = pred.get("lon")
        dist_km = _haversine_km(true_lat, true_lon, pred_lat, pred_lon)
        within_161km = dist_km is not None and dist_km <= ACC_RADIUS_KM

        dist_str = f"{dist_km:.1f}km" if dist_km is not None else "no-coords"
        status = "EXACT" if exact else ("COUNTRY" if country_match else ("CONTINENT" if continent_match else "MISS"))
        print(f"{status} | {dist_str} | pred={pred.get('qid','')} ({pred.get('country','')}) | true={row['qid']} ({row['country']})")

        rows_out.append({
            "text": row["text"],
            "signal_type": row.get("signal_type", ""),
            "true_qid": row["qid"],
            "true_country": row["country"],
            "true_continent": row["continent"],
            "true_lat": true_lat,
            "true_lon": true_lon,
            "pred_qid": pred.get("qid", ""),
            "pred_country": pred.get("country", ""),
            "pred_continent": pred.get("continent", ""),
            "pred_lat": pred_lat,
            "pred_lon": pred_lon,
            "exact": exact,
            "country_match": country_match,
            "continent_match": continent_match,
            "dist_km": dist_km,
            "within_161km": within_161km,
        })

    results_df = pd.DataFrame(rows_out)

    print("\n" + "=" * 60)
    print("Pipeline performance overview")
    print("=" * 60)
    _print_summary("OVERALL", results_df)

    if "signal_type" in results_df.columns:
        print("\n" + "-" * 60)
        print("Per-signal breakdown")
        print("-" * 60)
        for signal in sorted(s for s in results_df["signal_type"].unique() if s != ""):
            _print_summary(f"Signal {signal}", results_df[results_df["signal_type"] == signal])
    print("=" * 60)

    if llm_save:
        llm_save()
    results_df.to_csv(output, index=False)
    print(f"Detailed results saved to {output}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sample_100.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows")
    parser.add_argument("--mode", choices=["two_stage", "refined", "refined_search",
                                            "refined_gemma", "gemma",
                                            "refined_groq_oss", "groq_oss",
                                            "refined_groq_llama", "groq_llama"],
                        default="two_stage")
    parser.add_argument("--output", default="eval_results.csv", help="Path to write detailed results CSV")
    args = parser.parse_args()
    evaluate(args.dataset, limit=args.limit, mode=args.mode, output=args.output)
