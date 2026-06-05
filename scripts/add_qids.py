"""Add a `qid` column to a CSV with a `geoNameId` column by resolving each
GeoNames ID to its Wikidata QID via P1566.

Usage:
    python add_qids.py src/geocorpora_144.csv

Reuses the SPARQL resolver and on-disk cache from prepare_geocorpora.py, so
GeoNames IDs already resolved on previous runs cost nothing. Idempotent:
re-runs only fetch rows whose geoNameId has been added or changed.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prepare_geocorpora import (
    EXCLUDED_GEONAMES,
    EXCLUDED_QIDS,
    filter_alive_qids,
    resolve_geonames_to_qids,
)


def add_qids_to(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)
    if "geoNameId" not in df.columns:
        raise SystemExit(f"{csv_path}: no 'geoNameId' column")

    geonames = df["geoNameId"].dropna().tolist()
    qid_map = resolve_geonames_to_qids(geonames)

    resolved_qids = {q for q in qid_map.values() if q}
    alive = filter_alive_qids(resolved_qids)
    dangling = resolved_qids - alive
    if dangling:
        print(f"  dropping {len(dangling)} dangling QIDs (deleted Wikidata entities): "
              f"{sorted(dangling)}")
        qid_map = {gn: (q if q in alive else None) for gn, q in qid_map.items()}

    def _lookup(g):
        if pd.isna(g):
            return None
        gn = str(int(g))
        if gn in EXCLUDED_GEONAMES:
            return None
        q = qid_map.get(gn)
        if q in EXCLUDED_QIDS:
            return None
        return q

    df["qid"] = df["geoNameId"].map(_lookup)
    resolved = df["qid"].notna().sum()
    print(f"{csv_path.name}: {resolved}/{len(df)} rows have a live QID "
          f"({len(df) - resolved} unresolved)")
    if resolved < len(df):
        gaps = df.loc[df["qid"].isna(), "geoNameId"].astype("Int64").tolist()
        print(f"  geoNameIds without a QID: {gaps}")

    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to a CSV with a 'geoNameId' column")
    args = parser.parse_args()
    add_qids_to(Path(args.csv))
