"""NER recall diagnostic — attributes pipeline misses to the Flair stage.

Runs stock Flair (ner-large, as used in src/ner.py) once per row across the
three signal-typed datasets (clean SM, journalistic, noisy SM). For each row
it records what Flair extracted and at what type, so the per-signal A/B/C
gap can be split into:

  - "guaranteed-MISS" rows (Flair returned zero LOC/PER/ORG; the refined and
    two_stage modes return None on these without ever calling ReFinED or the
    reranker — see src/main.py:30,59),
  - "no-LOC" rows (Flair found no LOC; refined_gemma's routing gate skips
    ReFinED entirely on these — src/main.py:98,115),
  - rows where Flair did extract something but the pipeline still failed
    (downstream attribution: linking or reranking, not NER).

Outputs:
  results/raw/ner_recall_per_row.csv      — one row per dataset row
  results/summary/table_FINAL_ner_recall.csv — per dataset x signal summary

Usage:
  python diagnose_ner_recall.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
RAW_DIR = PROJECT_ROOT / "results" / "raw"
SUMMARY_DIR = PROJECT_ROOT / "results" / "summary"

sys.path.insert(0, str(SRC))

DATASETS = [
    ("clean_sm",    "social_media_dataset.csv"),
    ("clean_journ", "journalistic_dataset.csv"),
    ("noisy_sm",    "social_media_noisy.csv"),
]

GEO_TYPES = ("LOC", "PER", "ORG")
SIGNALS = ["A", "B", "C"]


def per_row_records(dataset_short, dataset_path, extract_fn):
    df = pd.read_csv(dataset_path)
    rows = []
    for i, row in df.iterrows():
        text = row["text"]
        ents = extract_fn(text)
        by_type = {"LOC": [], "PER": [], "ORG": [], "OTHER": []}
        for e in ents:
            t = e["type"] if e["type"] in by_type else "OTHER"
            by_type[t].append(e["text"])
        n_loc, n_per, n_org = len(by_type["LOC"]), len(by_type["PER"]), len(by_type["ORG"])
        n_geo = n_loc + n_per + n_org
        rows.append({
            "dataset": dataset_short,
            "signal_type": row.get("signal_type", ""),
            "true_qid": row.get("qid", ""),
            "location_name": row.get("location_name", ""),
            "text": text,
            "n_entities": len(ents),
            "n_loc": n_loc,
            "n_per": n_per,
            "n_org": n_org,
            "n_geo": n_geo,
            "zero_geo": n_geo == 0,
            "zero_loc": n_loc == 0,
            "loc_surface": "|".join(by_type["LOC"]),
            "per_surface": "|".join(by_type["PER"]),
            "org_surface": "|".join(by_type["ORG"]),
            "other_surface": "|".join(by_type["OTHER"]),
        })
        print(f"  [{i+1}/{len(df)}] {row.get('location_name','')[:30]:<30} "
              f"sig={row.get('signal_type','')} "
              f"L={n_loc} P={n_per} O={n_org}", flush=True)
    return rows


def summarise(per_row_df):
    out = []
    for ds in [d[0] for d in DATASETS]:
        sub_ds = per_row_df[per_row_df["dataset"] == ds]
        for sig in SIGNALS:
            sub = sub_ds[sub_ds["signal_type"] == sig]
            n = len(sub)
            if n == 0:
                out.append(dict(dataset=ds, signal_type=sig, n=0))
                continue
            out.append(dict(
                dataset=ds,
                signal_type=sig,
                n=n,
                pct_zero_geo=sub["zero_geo"].mean() * 100,
                pct_zero_loc=sub["zero_loc"].mean() * 100,
                avg_n_loc=sub["n_loc"].mean(),
                avg_n_per=sub["n_per"].mean(),
                avg_n_org=sub["n_org"].mean(),
                pct_loc_only=((sub["n_loc"] > 0) & (sub["n_per"] == 0) & (sub["n_org"] == 0)).mean() * 100,
                pct_no_loc_has_per_org=((sub["n_loc"] == 0) & ((sub["n_per"] + sub["n_org"]) > 0)).mean() * 100,
            ))
    return pd.DataFrame(out)


def print_summary(summary_df):
    print()
    print("=" * 78)
    print("NER recall by dataset x signal type")
    print("=" * 78)
    print("zero_geo = Flair found no LOC/PER/ORG => refined and two_stage return None")
    print("zero_loc = Flair found no LOC        => refined_gemma routing skips ReFinED")
    print()
    for ds in [d[0] for d in DATASETS]:
        sub = summary_df[summary_df["dataset"] == ds]
        print(f"\n{ds}")
        print("-" * len(ds))
        for _, r in sub.iterrows():
            if r["n"] == 0:
                print(f"  Signal {r['signal_type']}: (no rows)")
                continue
            print(f"  Signal {r['signal_type']} (n={int(r['n']):>3})  "
                  f"zero_geo={r['pct_zero_geo']:5.1f}%  "
                  f"zero_loc={r['pct_zero_loc']:5.1f}%  "
                  f"avg L/P/O={r['avg_n_loc']:.2f}/{r['avg_n_per']:.2f}/{r['avg_n_org']:.2f}")
    print("=" * 78)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Flair ner-large...", flush=True)
    from ner import extract_entities

    all_rows = []
    for ds_short, ds_file in DATASETS:
        ds_path = SRC / ds_file
        if not ds_path.exists():
            print(f"\nSKIP {ds_short}: {ds_path} not found")
            continue
        print(f"\n=== {ds_short} ({ds_file}) ===")
        all_rows.extend(per_row_records(ds_short, ds_path, extract_entities))

    per_row_df = pd.DataFrame(all_rows)
    per_row_path = RAW_DIR / "ner_recall_per_row.csv"
    per_row_df.to_csv(per_row_path, index=False)
    print(f"\nWrote {per_row_path}")

    summary_df = summarise(per_row_df)
    summary_path = SUMMARY_DIR / "table_FINAL_ner_recall.csv"
    summary_df.to_csv(summary_path, index=False, float_format="%.2f")
    print(f"Wrote {summary_path}")

    print_summary(summary_df)


if __name__ == "__main__":
    main()
