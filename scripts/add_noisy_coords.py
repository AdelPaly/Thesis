"""Add latitude/longitude to social_media_noisy.csv by joining
social_media_dataset.csv on qid (the noisy dataset is row-aligned with the
clean one — same 144 entries, same qids, just rewritten with noise). Then
verify the resulting coords match Wikidata P625 like the clean datasets do.
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from add_coords import fetch_coords

CLEAN = SRC / "social_media_dataset.csv"
NOISY = SRC / "social_media_noisy.csv"

clean = pd.read_csv(CLEAN)
noisy = pd.read_csv(NOISY)

print(f"clean rows: {len(clean)}, noisy rows: {len(noisy)}")

# Sanity: are the qid lists identical, in order?
clean_qids = clean["qid"].tolist()
noisy_qids = noisy["qid"].tolist()
if clean_qids == noisy_qids:
    print("qid sequences match exactly (row-aligned)")
else:
    only_clean = set(clean_qids) - set(noisy_qids)
    only_noisy = set(noisy_qids) - set(clean_qids)
    print(f"qid sets differ: only_clean={len(only_clean)}, only_noisy={len(only_noisy)}")
    if only_noisy:
        print(f"  noisy has qids not in clean: {sorted(only_noisy)[:10]}")

# Build lookup from clean.qid -> (lat, lon). If a qid appears multiple times
# we take the first; they should be identical anyway.
lookup = (clean.drop_duplicates(subset=["qid"])
                .set_index("qid")[["latitude", "longitude"]])

noisy["latitude"] = noisy["qid"].map(lookup["latitude"])
noisy["longitude"] = noisy["qid"].map(lookup["longitude"])

missing = noisy["latitude"].isna().sum()
print(f"after join: {len(noisy) - missing}/{len(noisy)} rows have coords")
if missing:
    gaps = noisy.loc[noisy["latitude"].isna(), "qid"].tolist()
    print(f"  missing qids: {gaps}")
    raise SystemExit("aborting — every noisy row should resolve to a clean qid")

# Verify against Wikidata P625
qids = noisy["qid"].dropna().unique().tolist()
coords = fetch_coords(qids)
TOL = 0.01
mismatches = 0
for _, row in noisy.iterrows():
    q = row["qid"]
    wd = coords.get(q)
    if wd is None:
        print(f"  no P625 for {q}")
        continue
    if abs(row["latitude"] - wd[0]) > TOL or abs(row["longitude"] - wd[1]) > TOL:
        print(f"  MISMATCH {q}: csv=({row['latitude']:.4f},{row['longitude']:.4f}) "
              f"wd=({wd[0]:.4f},{wd[1]:.4f})")
        mismatches += 1
print(f"WD verification: {len(qids) - mismatches}/{len(qids)} match within {TOL}deg")

if mismatches == 0:
    noisy.to_csv(NOISY, index=False)
    print(f"wrote {NOISY}")
else:
    raise SystemExit("not writing — fix mismatches first")
