"""Correctness check: run the new scorer over retired result CSVs.

    python scripts/check_scorer.py
    python scripts/check_scorer.py --files eval_FINAL_refined_clean_sm.csv ...

The point is NOT to reproduce the retired numbers -- they are retired. It is to
catch sign errors and column mix-ups in the new scorer, which are otherwise
invisible: a haversine with a flipped sign, lat and lon swapped, or gold and
predicted transposed all still produce plausible-looking tables.

Each retired row is converted into the new record format and scored by
score.py. The new verdicts are then compared field by field against the
booleans the old pipeline stored in the same row.

For configurations with no LLM in them, every field should agree exactly. Any
disagreement there is a bug in the new scorer, because both are reading the
same coordinates.

For LLM-backed configurations, the distance fields are EXPECTED to disagree,
on exactly the rows where the old code returned the continent-only candidate at
lat/lon 0.0. The conversion applies the Null Island fix: a row whose predicted
coordinate is exactly 0.0/0.0 with no predicted QID had no coordinate, and is
converted to `predicted_latlon: null`. The old row calls that a ~6,000 km
error; the new scorer calls it no distance. That disagreement is the fix
working, and the check labels it as such rather than counting it as a failure.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import ACC_RADIUS_KM, aggregate, score_record   # noqa: E402

RAW = REPO_ROOT / "results" / "raw"

# Configurations whose predictions can pass through the continent-only
# fallback, so whose distance fields may legitimately disagree.
LLM_MARKERS = ("gemma", "groq")

DEFAULT_FILES = [
    "eval_FINAL_refined_clean_sm.csv",      # no LLM: must agree exactly
    "eval_FINAL_two_stage_clean_journ.csv",  # no LLM: must agree exactly
    "eval_FINAL_groq_llama_noisy_sm.csv",   # LLM, most Null Island rows
]


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and not v.strip():
        return None
    return v


def convert(df, config_name):
    """Retired CSV rows -> new-format records."""
    records = []
    null_island = 0
    for _, row in df.iterrows():
        pred_lat, pred_lon = _clean(row.get("pred_lat")), _clean(row.get("pred_lon"))
        pred_qid = _clean(row.get("pred_qid"))

        # The Null Island conversion, stated explicitly.
        is_null_island = (pred_lat == 0.0 and pred_lon == 0.0 and not pred_qid)
        if is_null_island:
            null_island += 1
            pred_latlon = None
        elif pred_lat is None or pred_lon is None:
            pred_latlon = None
        else:
            pred_latlon = [float(pred_lat), float(pred_lon)]

        true_lat, true_lon = _clean(row.get("true_lat")), _clean(row.get("true_lon"))
        gold_latlon = (None if true_lat is None or true_lon is None
                       else [float(true_lat), float(true_lon)])

        records.append({
            "item_id": hashlib.sha256(str(row["text"]).encode()).hexdigest()[:16],
            "dataset": "retired",
            "config_name": config_name,
            "config_hash": "retired",
            "paradigm": "retired",
            "text": row["text"],
            "place_qid": _clean(row.get("true_qid")),
            "condition": _clean(row.get("signal_type")),
            "gold_latlon": gold_latlon,
            "gold_country": _clean(row.get("true_country")),
            "gold_continent": _clean(row.get("true_continent")),
            "ner_spans": [], "linked_entities": [], "candidate_places": [],
            "routing_branch": None, "routing_rule": None,
            "predicted_qid": pred_qid,
            "predicted_latlon": pred_latlon,
            "predicted_country": _clean(row.get("pred_country")),
            "predicted_continent": _clean(row.get("pred_continent")),
            "llm_raw_response": None, "error": None,
            # kept alongside for the comparison, not part of the format
            "_old": {
                "exact": bool(row.get("exact")),
                "country": bool(row.get("country_match")),
                "continent": bool(row.get("continent_match")),
                "within_161km": bool(row.get("within_161km"))
                if "within_161km" in df.columns else None,
                "dist_km": _clean(row.get("dist_km")),
                "null_island": is_null_island,
            },
        })
    return records, null_island


def compare(records, is_llm):
    """Field-by-field agreement between the new scorer and the stored booleans."""
    counts = {k: 0 for k in ("exact", "country", "continent", "within_161km",
                             "dist_km")}
    explained = {k: 0 for k in counts}
    examples = []
    for rec in records:
        old = rec["_old"]
        new = score_record(rec)
        checks = {
            "exact": (new["exact"], old["exact"]),
            "country": (new["country_correct"], old["country"]),
            "continent": (new["continent_correct"], old["continent"]),
        }
        if old["within_161km"] is not None:
            checks["within_161km"] = (new["within_161km"], old["within_161km"])

        nd, od = new["distance_km"], old["dist_km"]
        if nd is None and od is None:
            dist_agrees = True
        elif nd is None or od is None:
            dist_agrees = False
        else:
            dist_agrees = abs(nd - od) < 0.1
        checks["dist_km"] = (dist_agrees, True)

        for field, (a, b) in checks.items():
            if bool(a) != bool(b):
                counts[field] += 1
                if old["null_island"]:
                    explained[field] += 1
                elif len(examples) < 8:
                    examples.append((field, rec["item_id"], a, b, nd, od,
                                     rec["predicted_qid"], rec["place_qid"]))
    return counts, explained, examples


def check_file(name, boot_note=True):
    path = RAW / name
    if not path.exists():
        print(f"{name}: not found, skipping\n")
        return None
    config = name.replace("eval_FINAL_", "").replace(".csv", "")
    is_llm = any(m in config for m in LLM_MARKERS)
    df = pd.read_csv(path)
    records, n_ni = convert(df, config)
    counts, explained, examples = compare(records, is_llm)

    print("=" * 92)
    print(f"{name}   ({len(df)} rows, {'LLM-backed' if is_llm else 'no LLM'})")
    print(f"  Null Island rows converted to 'no coordinate': {n_ni}")
    print("=" * 92)
    print(f"  {'field':14} {'disagreements':>14} {'of which Null Island':>22} "
          f"{'unexplained':>12}")
    ok = True
    for field in ("exact", "country", "continent", "within_161km", "dist_km"):
        n = counts[field]
        exp = explained[field]
        unexplained = n - exp
        if unexplained:
            ok = False
        print(f"  {field:14} {n:>14} {exp:>22} {unexplained:>12}")

    new_metrics = aggregate([score_record(r) for r in records])
    old_exact = df["exact"].mean()
    old_country = df["country_match"].mean()
    old_cont = df["continent_match"].mean()
    old_dist = df["dist_km"].mean()
    print()
    print(f"  {'metric':14} {'old':>10} {'new':>10}")
    print(f"  {'exact':14} {old_exact * 100:>9.1f}% {new_metrics['exact'] * 100:>9.1f}%")
    print(f"  {'country':14} {old_country * 100:>9.1f}% {new_metrics['country'] * 100:>9.1f}%")
    print(f"  {'continent':14} {old_cont * 100:>9.1f}% {new_metrics['continent'] * 100:>9.1f}%")
    print(f"  {'mean km':14} {old_dist:>10.1f} {new_metrics['mean_km']:>10.1f}")
    print(f"  {'n w/ distance':14} {df['dist_km'].notna().sum():>10} "
          f"{new_metrics['n_with_distance']:>10}")

    if examples:
        print("\n  UNEXPLAINED disagreements (these would be scorer bugs):")
        for field, iid, a, b, nd, od, pq, gq in examples:
            print(f"    {field:12} item={iid} new={a} old={b} "
                  f"new_dist={nd} old_dist={od} pred={pq} gold={gq}")
    print()
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = p.parse_args()

    results = {}
    for name in args.files:
        results[name] = check_file(name)

    print("=" * 92)
    print("VERDICT")
    print("=" * 92)
    for name, ok in results.items():
        if ok is None:
            print(f"  {name}: skipped")
        elif ok:
            print(f"  {name}: every disagreement accounted for")
        else:
            print(f"  {name}: UNEXPLAINED disagreements -- investigate the scorer")
    if all(v for v in results.values() if v is not None):
        print("\nNo sign errors or column mix-ups: on the non-LLM files the new scorer")
        print("reproduces the retired verdicts exactly, and on the LLM file every")
        print("disagreement falls on a Null Island row.")


if __name__ == "__main__":
    main()
