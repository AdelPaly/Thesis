"""Add latitude/longitude columns to an eval CSV by looking up Wikidata
P625 (coordinate location) for each row's QID.

Usage:
    python add_coords.py src/geocorpora_eval.csv
    python add_coords.py src/social_media_dataset.csv
    python add_coords.py src/journalistic_dataset.csv

Always sources from the QID currently in the CSV (so manual QID edits flow
through to the coordinates), batched and disk-cached at
src/.qid_coords_cache.pkl. Idempotent: re-runs only fetch QIDs that have
been added or changed since the last run.
"""
import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
COORDS_CACHE = SRC / ".qid_coords_cache.pkl"

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
WDQS_USER_AGENT = "GeoThesisBot/1.0 (student research; contact via GitHub)"
SPARQL_BATCH = 50
SPARQL_PAUSE_S = 1.0


def _load_cache():
    if COORDS_CACHE.exists():
        try:
            with open(COORDS_CACHE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  cache load failed ({e}), starting empty")
    return {}


def _save_cache(cache):
    tmp = COORDS_CACHE.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as f:
        pickle.dump(cache, f)
    tmp.replace(COORDS_CACHE)


def _parse_point_wkt(wkt):
    # WDQS returns coordinates as 'Point(LON LAT)' (note: lon-then-lat).
    if not (wkt.startswith("Point(") and wkt.endswith(")")):
        return None
    parts = wkt[6:-1].split()
    if len(parts) != 2:
        return None
    try:
        lon, lat = float(parts[0]), float(parts[1])
    except ValueError:
        return None
    return lat, lon


def fetch_coords(qids):
    """Return {qid: (lat, lon)_or_None}. Disk-cached across runs."""
    cache = _load_cache()
    qids = sorted({q for q in qids if isinstance(q, str) and q.startswith("Q")})
    missing = [q for q in qids if q not in cache]
    if missing:
        print(f"  fetching P625 for {len(missing)} QIDs "
              f"(cached: {len(qids) - len(missing)})")
        sparql = SPARQLWrapper(WDQS_ENDPOINT, agent=WDQS_USER_AGENT)
        sparql.setReturnFormat(JSON)
        for i in range(0, len(missing), SPARQL_BATCH):
            batch = missing[i:i + SPARQL_BATCH]
            values_clause = " ".join(f"wd:{q}" for q in batch)
            query = (
                f"SELECT ?item ?coord WHERE {{ "
                f"  VALUES ?item {{ {values_clause} }} "
                f"  ?item wdt:P625 ?coord . "
                f"}}"
            )
            sparql.setQuery(query)
            for attempt in range(4):
                try:
                    results = sparql.query().convert()
                    break
                except Exception as e:
                    if attempt == 3:
                        print(f"  batch {i//SPARQL_BATCH+1} failed permanently: {e}")
                        results = {"results": {"bindings": []}}
                        break
                    print(f"  retry {attempt+1} after {type(e).__name__}; sleeping")
                    time.sleep(2 ** attempt)
            # Mark every QID in this batch as "tried" so a second pass doesn't
            # re-query the ones with no P625.
            for q in batch:
                cache.setdefault(q, None)
            for binding in results["results"]["bindings"]:
                qid = binding["item"]["value"].rsplit("/", 1)[-1]
                if cache.get(qid) is not None:
                    continue  # keep first coordinate when an item has multiple
                parsed = _parse_point_wkt(binding["coord"]["value"])
                if parsed:
                    cache[qid] = parsed
            _save_cache(cache)
            resolved = sum(1 for q in batch if cache.get(q))
            print(f"  ... batch {i//SPARQL_BATCH+1} done ({resolved}/{len(batch)} resolved)")
            time.sleep(SPARQL_PAUSE_S)
    return {q: cache.get(q) for q in qids}


def add_coords_to(csv_path):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if "qid" not in df.columns:
        raise SystemExit(f"{csv_path}: no 'qid' column")
    qids = df["qid"].dropna().unique().tolist()
    coords = fetch_coords(qids)

    def _lat(q):
        c = coords.get(q)
        return c[0] if c else None

    def _lon(q):
        c = coords.get(q)
        return c[1] if c else None

    df["latitude"] = df["qid"].map(_lat)
    df["longitude"] = df["qid"].map(_lon)
    missing = df["latitude"].isna().sum()
    print(f"{csv_path.name}: {len(df) - missing}/{len(df)} rows have coordinates "
          f"({missing} QIDs without P625)")
    if missing:
        gaps = df.loc[df["latitude"].isna(), "qid"].tolist()
        print(f"  QIDs missing coords: {gaps}")
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to an eval CSV with a 'qid' column")
    args = parser.parse_args()
    add_coords_to(args.csv)
