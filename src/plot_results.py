"""Render evaluation comparison as PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODES = ["original", "refined", "gated", "coverage"]
COLORS = {"original": "#4C72B0", "refined": "#55A868", "gated": "#C44E52", "coverage": "#8172B2"}

sample = pd.read_csv("sample_100.csv")[["text", "source"]]
dfs = {m: pd.read_csv(f"eval_sample_{m}.csv").merge(sample, on="text", how="left") for m in MODES}


def pct(df, col):
    return df[col].sum() / len(df) * 100 if len(df) else 0


def noresult_pct(df):
    return (df["pred_qid"].fillna("") == "").sum() / len(df) * 100 if len(df) else 0


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Blend comparison — 100-row stratified sample",
             fontsize=14, fontweight="bold")

# 1. Overall metrics
ax = axes[0, 0]
metrics = ["Exact", "Country", "Continent", "No-result"]
cols = ["exact", "country_match", "continent_match", None]
x = np.arange(len(metrics))
w = 0.2
for i, m in enumerate(MODES):
    vals = [pct(dfs[m], c) if c else noresult_pct(dfs[m]) for c in cols]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("% of samples")
ax.set_title("Overall (n=100)")
ax.set_ylim(0, 80)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 2. Per-signal exact match
ax = axes[0, 1]
signals = ["A (unambiguous)", "B (moderate)", "C (highly ambiguous)"]
sig_keys = ["A", "B", "C"]
x = np.arange(len(signals))
for i, m in enumerate(MODES):
    vals = [pct(dfs[m][dfs[m]["signal_type"] == s], "exact") for s in sig_keys]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(signals, fontsize=9)
ax.set_ylabel("Exact match %")
ax.set_title("Exact match by signal type")
ax.set_ylim(0, 90)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 3. Per-source country-level
ax = axes[1, 0]
sources = ["journalistic", "social_media"]
x = np.arange(len(sources))
for i, m in enumerate(MODES):
    vals = [pct(dfs[m][dfs[m]["source"] == s], "country_match") for s in sources]
    bars = ax.bar(x + (i - 1.5) * w, vals, w, label=m, color=COLORS[m])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(sources)
ax.set_ylabel("Country-level match %")
ax.set_title("Country match by source")
ax.set_ylim(0, 80)
ax.legend()
ax.grid(axis="y", alpha=0.3)

# 4. Head-to-head: where does coverage beat / match / lose vs refined?
ax = axes[1, 1]
ref_df = dfs["refined"][["text", "exact", "country_match"]].rename(
    columns={"exact": "ref_exact", "country_match": "ref_country"})
cov_df = dfs["coverage"][["text", "exact", "country_match"]].rename(
    columns={"exact": "cov_exact", "country_match": "cov_country"})
h2h = ref_df.merge(cov_df, on="text")

buckets = [
    ("Both exact",    (h2h["ref_exact"]) & (h2h["cov_exact"])),
    ("Coverage only", (~h2h["ref_exact"]) & (h2h["cov_exact"])),
    ("Refined only",  (h2h["ref_exact"]) & (~h2h["cov_exact"])),
    ("Both country",  (~h2h["ref_exact"]) & (~h2h["cov_exact"])
                       & (h2h["ref_country"]) & (h2h["cov_country"])),
    ("Cov country only", (~h2h["ref_exact"]) & (~h2h["cov_exact"])
                          & (~h2h["ref_country"]) & (h2h["cov_country"])),
    ("Ref country only", (~h2h["ref_exact"]) & (~h2h["cov_exact"])
                          & (h2h["ref_country"]) & (~h2h["cov_country"])),
    ("Both miss",     (~h2h["ref_exact"]) & (~h2h["cov_exact"])
                       & (~h2h["ref_country"]) & (~h2h["cov_country"])),
]
labels = [b[0] for b in buckets]
counts = [mask.sum() for _, mask in buckets]
colors_list = ["#2ecc71", "#8172B2", "#55A868", "#a4cfa0",
               "#c5b7d9", "#95c093", "#7f7f7f"]
ax.barh(labels, counts, color=colors_list)
for i, c in enumerate(counts):
    ax.text(c + 0.3, i, str(c), va="center", fontsize=9)
ax.set_xlabel("# rows (of 100)")
ax.set_title("Coverage vs ReFinED — head-to-head")
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("results_comparison.png", dpi=140, bbox_inches="tight")
print("Wrote results_comparison.png")
