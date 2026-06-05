"""Reconstruct src/geocorpora_eval.csv from the ground-truth columns of an
existing eval_FINAL_*_geocorpora.csv. This guarantees two_stage gets
evaluated on the exact same row set as the four modes already run, instead
of a fresh prepare_geocorpora.py sample that might land on different rows.

Schema produced (matches what evaluate.py expects via the harness):
    text, qid, location_name, country, continent, signal_type,
    latitude, longitude

We don't have location_name in the eval_FINAL CSV, so we leave that column
blank — the harness only uses it for log messages, not for matching.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
RAW = PROJECT_ROOT / "results" / "raw"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_coords import fetch_coords

SOURCE_FINAL = RAW / "eval_FINAL_refined_geocorpora.csv"
OUTPUT_CSV = SRC / "geocorpora_eval.csv"


def main():
    if not SOURCE_FINAL.exists():
        raise SystemExit(f"missing {SOURCE_FINAL}")
    df = pd.read_csv(SOURCE_FINAL)

    # Cross-check: every other eval_FINAL_*_geocorpora.csv should have the
    # same texts in the same order. If they don't, restoring from any single
    # one is unsafe.
    for other in RAW.glob("eval_FINAL_*_geocorpora.csv"):
        if other == SOURCE_FINAL:
            continue
        d2 = pd.read_csv(other)
        if list(d2["text"]) != list(df["text"]) or list(d2["true_qid"]) != list(df["true_qid"]):
            raise SystemExit(
                f"row alignment mismatch between {SOURCE_FINAL.name} and {other.name}; "
                "the geocorpora ground truth is not identical across the 4 existing runs"
            )
    print(f"verified all eval_FINAL_*_geocorpora.csv share identical (text, true_qid)")

    # Verify ground-truth qid -> coords match Wikidata P625 (same check as on
    # the other source datasets).
    qids = df["true_qid"].dropna().unique().tolist()
    coords = fetch_coords(qids)
    TOL = 0.01
    mismatches = 0
    no_p625 = 0
    for _, r in df.iterrows():
        wd = coords.get(r["true_qid"])
        if wd is None:
            no_p625 += 1
            continue
        if abs(r["true_lat"] - wd[0]) > TOL or abs(r["true_lon"] - wd[1]) > TOL:
            print(f"  MISMATCH {r['true_qid']}: csv=({r['true_lat']:.4f},{r['true_lon']:.4f}) "
                  f"wd=({wd[0]:.4f},{wd[1]:.4f})")
            mismatches += 1
    print(f"WD verification: {len(qids) - mismatches}/{len(qids)} unique qids match within {TOL}deg "
          f"({no_p625} rows had no P625)")
    if mismatches:
        raise SystemExit("aborting — fix qid/coord mismatches first")

    out = pd.DataFrame({
        "text":          df["text"].values,
        "qid":           df["true_qid"].values,
        "location_name": "",
        "country":       df["true_country"].values,
        "continent":     df["true_continent"].values,
        "signal_type":   df["signal_type"].fillna("").values,
        "latitude":      df["true_lat"].values,
        "longitude":     df["true_lon"].values,
    })
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"wrote {len(out)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
