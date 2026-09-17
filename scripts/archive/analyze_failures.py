"""Failure-mode analysis across all chapter 4 evals.

Three questions:
  1. Where in the granularity stack do failures cluster?
       continent wrong | continent ok, country wrong |
       country ok, > 161 km | within 161 km but not exact | exact
  2. How does that breakdown vary by signal type (A/B/C)?
  3. For continent-wrong rows, what does the *predicted* continent look like
     — i.e. is the model committing to the wrong rough region, or is it
     refusing/missing?
"""
import os
from pathlib import Path

import pandas as pd

RAW = str(Path(__file__).resolve().parent.parent / "results" / "raw")
MODES = ["two_stage", "refined", "gemma", "groq_llama", "groq_oss",
         "refined_search", "refined_gemma"]
DATASETS = ["clean_sm", "clean_journ", "noisy_sm", "geocorpora"]


def stage(row):
    if pd.isna(row["pred_qid"]) or row["pred_qid"] == "":
        return "no_result"
    if not row["continent_match"]:
        return "continent_wrong"
    if not row["country_match"]:
        return "country_wrong"
    if not row["within_161km"]:
        return "country_ok_far"
    if not row["exact"]:
        return "within_161km_not_exact"
    return "exact"


def load(config, dataset):
    p = os.path.join(RAW, f"eval_FINAL_{config}_{dataset}.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["stage"] = df.apply(stage, axis=1)
    df["config"] = config
    df["dataset"] = dataset
    return df


def all_runs():
    parts = []
    for c in MODES:
        for d in DATASETS:
            df = load(c, d)
            if df is not None:
                parts.append(df)
    return pd.concat(parts, ignore_index=True)


STAGE_ORDER = ["exact", "within_161km_not_exact", "country_ok_far",
               "country_wrong", "continent_wrong", "no_result"]


def stage_breakdown(df, group_cols):
    pct = (df.groupby(group_cols)["stage"]
             .value_counts(normalize=True)
             .unstack(fill_value=0) * 100)
    cols = [s for s in STAGE_ORDER if s in pct.columns]
    return pct[cols].round(1)


def continent_confusion(df):
    sub = df[df["stage"] == "continent_wrong"].copy()
    sub["pred_continent"] = sub["pred_continent"].fillna("(none)")
    return (sub.groupby(["config", "true_continent", "pred_continent"])
              .size().rename("n").reset_index()
              .sort_values(["config", "n"], ascending=[True, False]))


if __name__ == "__main__":
    df = all_runs()

    print("=" * 80)
    print("Q1. Where do failures land? (rows = config, % across stages)")
    print("    pooled across all 4 datasets")
    print("=" * 80)
    print(stage_breakdown(df, ["config"]).to_string())
    print()

    print("=" * 80)
    print("Q2. Same breakdown, split by signal type (A/B/C)")
    print("    pooled across all 4 datasets")
    print("=" * 80)
    print(stage_breakdown(df, ["signal_type", "config"]).to_string())
    print()

    print("=" * 80)
    print("Q2b. Same, but only failures (drop 'exact')")
    print("=" * 80)
    failed = df[df["stage"] != "exact"]
    print(stage_breakdown(failed, ["signal_type", "config"]).to_string())
    print()

    print("=" * 80)
    print("Q3. Continent confusion — true -> predicted (counts)")
    print("    pooled across all datasets, only continent_wrong rows")
    print("=" * 80)
    cc = continent_confusion(df)
    print(cc.to_string(index=False))
    print()

    print("=" * 80)
    print("Q3b. For continent_wrong rows: signal mix (% A/B/C within failures)")
    print("=" * 80)
    cw = df[df["stage"] == "continent_wrong"]
    if len(cw):
        mix = (cw.groupby("config")["signal_type"]
                 .value_counts(normalize=True).unstack(fill_value=0) * 100).round(1)
        print(mix.to_string())
    print()

    print("=" * 80)
    print("Q3c. For continent_wrong rows: did the model refuse (pred is None)")
    print("    or commit to a wrong continent?")
    print("=" * 80)
    cw = df[df["stage"] == "continent_wrong"].copy()
    cw["mode"] = cw["pred_continent"].fillna("").apply(
        lambda s: "no_continent" if not s else "wrong_continent")
    print((cw.groupby("config")["mode"].value_counts(normalize=True)
             .unstack(fill_value=0) * 100).round(1).to_string())
