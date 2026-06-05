"""Verify that latitude/longitude columns in eval CSVs match Wikidata P625
for the row's qid. Read-only — writes nothing, just prints mismatches.

Usage:
    python verify_coords.py src/social_media_dataset.csv src/journalistic_dataset.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

# Reuse the SPARQL fetch + disk cache from add_coords.py
sys.path.insert(0, str(Path(__file__).parent.resolve()))
from add_coords import fetch_coords

TOL_DEG = 0.01  # ~1.1 km — flag anything beyond this


def verify(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "qid" not in df.columns:
        print(f"{csv_path.name}: no 'qid' column, skipping")
        return
    if "latitude" not in df.columns or "longitude" not in df.columns:
        print(f"{csv_path.name}: no latitude/longitude columns, skipping")
        return

    qids = df["qid"].dropna().unique().tolist()
    coords = fetch_coords(qids)

    mismatches = []
    no_p625 = []
    matched = 0
    for _, row in df.iterrows():
        q = row["qid"]
        if not isinstance(q, str) or not q.startswith("Q"):
            continue
        wd = coords.get(q)
        if wd is None:
            no_p625.append((q, row.get("location_name", ""), row["latitude"], row["longitude"]))
            continue
        wd_lat, wd_lon = wd
        d_lat = abs(float(row["latitude"]) - wd_lat)
        d_lon = abs(float(row["longitude"]) - wd_lon)
        if d_lat > TOL_DEG or d_lon > TOL_DEG:
            mismatches.append((q, row.get("location_name", ""),
                               (row["latitude"], row["longitude"]),
                               (wd_lat, wd_lon),
                               (d_lat, d_lon)))
        else:
            matched += 1

    print(f"\n=== {csv_path.name} ===")
    print(f"  rows checked:  {len(df)}")
    print(f"  matches:       {matched}")
    print(f"  no P625 in WD: {len(no_p625)}")
    print(f"  mismatches (>{TOL_DEG}deg): {len(mismatches)}")
    for q, name, csv_pt, wd_pt, d in mismatches:
        print(f"    {q} ({name}): csv=({csv_pt[0]:.4f},{csv_pt[1]:.4f}) "
              f"vs wd=({wd_pt[0]:.4f},{wd_pt[1]:.4f}) "
              f"delta=({d[0]:.4f},{d[1]:.4f})")
    if no_p625:
        print(f"  QIDs without P625:")
        for q, name, lat, lon in no_p625:
            print(f"    {q} ({name}): csv=({lat},{lon})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csvs", nargs="+", help="Paths to CSVs with qid + latitude + longitude")
    args = parser.parse_args()
    for c in args.csvs:
        verify(c)
