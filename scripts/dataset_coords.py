"""Dataset coordinate tool: fill in latitude/longitude from each row's QID, or
check the ones already there.

    python scripts/dataset_coords.py add    src/social_media_dataset.csv
    python scripts/dataset_coords.py verify src/social_media_dataset.csv src/journalistic_dataset.csv

`add` sources coordinates from the QID currently in the CSV, so manual QID
edits flow through, and writes the file in place. `verify` is read-only and
prints rows whose stored coordinates disagree with Wikidata.

This replaces add_coords.py and verify_coords.py, which reached Wikidata
through a SPARQL endpoint with their own on-disk cache
(src/.qid_coords_cache.pkl) to read the same P625 fact the pipeline reads
through wbgetentities. There is now one coordinate client
(src/wikidata.coords_for_qids) and one cache (src/.wikidata_cache), shared
with the pipeline: a QID resolved by either side is cached for both.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wikidata import coords_for_qids, save_caches   # noqa: E402

# ~0.01 deg is ~1.1 km -- flag anything beyond this.
TOL_DEG = 0.01


def _resolve(path_arg):
    p = Path(path_arg)
    if p.exists():
        return p
    candidate = PROJECT_ROOT / "src" / path_arg
    return candidate if candidate.exists() else p


def add(csv_path):
    csv_path = _resolve(csv_path)
    df = pd.read_csv(csv_path)
    if "qid" not in df.columns:
        raise SystemExit(f"{csv_path}: no 'qid' column")

    coords = coords_for_qids(df["qid"].dropna().unique().tolist())
    save_caches()

    df["latitude"] = df["qid"].map(lambda q: (coords.get(q) or (None, None))[0])
    df["longitude"] = df["qid"].map(lambda q: (coords.get(q) or (None, None))[1])

    missing = int(df["latitude"].isna().sum())
    print(f"{csv_path.name}: {len(df) - missing}/{len(df)} rows have coordinates "
          f"({missing} QIDs without P625)")
    if missing:
        print(f"  QIDs missing coords: {df.loc[df['latitude'].isna(), 'qid'].tolist()}")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


def verify(csv_path):
    csv_path = _resolve(csv_path)
    df = pd.read_csv(csv_path)
    for required in ("qid", "latitude", "longitude"):
        if required not in df.columns:
            print(f"{csv_path.name}: no '{required}' column, skipping")
            return

    coords = coords_for_qids(df["qid"].dropna().unique().tolist())
    save_caches()

    mismatches, no_p625, matched = [], [], 0
    for _, row in df.iterrows():
        q = row["qid"]
        if not isinstance(q, str) or not q.startswith("Q"):
            continue
        wd = coords.get(q)
        if wd is None:
            no_p625.append((q, row.get("location_name", ""), row["latitude"], row["longitude"]))
            continue
        d_lat = abs(float(row["latitude"]) - wd[0])
        d_lon = abs(float(row["longitude"]) - wd[1])
        if d_lat > TOL_DEG or d_lon > TOL_DEG:
            mismatches.append((q, row.get("location_name", ""),
                               (row["latitude"], row["longitude"]), wd, (d_lat, d_lon)))
        else:
            matched += 1

    print(f"\n=== {csv_path.name} ===")
    print(f"  rows checked:  {len(df)}")
    print(f"  matches:       {matched}")
    print(f"  no P625 in WD: {len(no_p625)}")
    print(f"  mismatches (>{TOL_DEG}deg): {len(mismatches)}")
    for q, name, csv_pt, wd_pt, d in mismatches:
        print(f"    {q} ({name}): csv=({csv_pt[0]:.4f},{csv_pt[1]:.4f}) "
              f"vs wd=({wd_pt[0]:.4f},{wd_pt[1]:.4f}) delta=({d[0]:.4f},{d[1]:.4f})")
    for q, name, lat, lon in no_p625:
        print(f"    no P625: {q} ({name}) csv=({lat},{lon})")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["add", "verify"])
    p.add_argument("csvs", nargs="+", help="CSVs with a 'qid' column")
    args = p.parse_args()
    for csv_path in args.csvs:
        (add if args.action == "add" else verify)(csv_path)


if __name__ == "__main__":
    main()
