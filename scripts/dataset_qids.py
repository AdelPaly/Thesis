"""Dataset QID tool: check that each row's `qid` really is the place the row
claims, and optionally repair the ones that are not.

    python scripts/dataset_qids.py validate src/social_media_dataset.csv
    python scripts/dataset_qids.py repair   src/geocorpora_eval.csv [--out fixed.csv]

Both actions run the same three checks against Wikidata:

  1. the `qid` exists;
  2. its English label matches `entity_name` if the row has one, otherwise
     `location_name`;
  3. its country (P17) is consistent with the `country` column.

`validate` is read-only and reports failures. `repair` additionally searches
Wikidata for a candidate whose P17 matches the expected country and rewrites
`qid`, writing the CSV in place unless --out is given.

This replaces validate_dataset.py and fix_qids.py, which were the same
capability split by dataset -- validate_dataset.py described itself as "the
validation script for the synthetic datasets, analogous to the GeoCorpora
validator in fix_qids.py". They ran the same label/country check through two
copies of the same SPARQL boilerplate, and only one of them could repair. The
split is now by capability, and either action runs on any dataset.

The run log of the old GeoCorpora repair pass is kept at
results/logs/fix_qids_geocorpora_144.log; `repair` prints the same per-row
shape.

No API key is required -- the Wikidata SPARQL and search endpoints are public.
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper, JSON

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "GeoProject/1.0 (dataset-qid-tool)"
DELAY_BETWEEN_QUERIES = 1.5   # respect Wikidata rate limits

_sparql = SPARQLWrapper(SPARQL_ENDPOINT)
_sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
_sparql.setReturnFormat(JSON)


def _query(query, retries=1):
    """Execute a SPARQL query, returning bindings ([] on permanent failure)."""
    _sparql.setQuery(query)
    for attempt in range(retries + 1):
        try:
            return _sparql.query().convert()["results"]["bindings"]
        except Exception as e:                        # noqa: BLE001
            if attempt < retries:
                print(f"    query failed ({e}), retrying...")
                time.sleep(DELAY_BETWEEN_QUERIES)
            else:
                return []


def _label_and_country(qid):
    """({labels}, {country labels}) for a QID, both lowercased."""
    bindings = _query(f"""
    SELECT ?label ?countryLabel WHERE {{
        wd:{qid} rdfs:label ?label .
        FILTER(LANG(?label) = "en")
        OPTIONAL {{
            wd:{qid} wdt:P17 ?country .
            ?country rdfs:label ?countryLabel .
            FILTER(LANG(?countryLabel) = "en")
        }}
    }}
    LIMIT 10
    """)
    labels = {b["label"]["value"].lower() for b in bindings if "label" in b}
    countries = {b["countryLabel"]["value"].lower() for b in bindings if "countryLabel" in b}
    return labels, countries


def _country_matches(expected, found):
    expected = (expected or "").lower()
    return any(expected in c or c in expected for c in found)


def check_row(row):
    """The shared check. Returns a list of failure reasons ([] if it passes)."""
    qid = (row.get("qid") or "").strip()
    expected_name = (row.get("entity_name") or row.get("location_name") or "").strip()
    country = (row.get("country") or "").strip()

    if not qid:
        return ["no qid in row"]

    labels, countries = _label_and_country(qid)
    if not labels:
        return ["QID does not exist in Wikidata"]

    failures = []
    if expected_name and not any(lbl == expected_name.lower() for lbl in labels):
        failures.append(f"Label mismatch: expected '{expected_name}', Wikidata has {labels}")
    if not countries:
        failures.append(f"No country (P17) found in Wikidata for {qid}")
    elif not _country_matches(country, countries):
        failures.append(f"Country mismatch: expected '{country}', Wikidata has {countries}")
    return failures


def search_for_qid(location_name, country):
    """Search Wikidata by name and return the first hit whose P17 matches the
    expected country, or None."""
    params = urllib.parse.urlencode({
        "action": "wbsearchentities", "search": location_name, "language": "en",
        "type": "item", "limit": 10, "format": "json",
    })
    req = urllib.request.Request(f"{WIKIDATA_SEARCH_API}?{params}",
                                 headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    for candidate in data.get("search", []):
        time.sleep(DELAY_BETWEEN_QUERIES)
        _, countries = _label_and_country(candidate["id"])
        if _country_matches(country, countries):
            return candidate["id"]
    return None


def _read(path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def _resolve(path_arg):
    p = Path(path_arg)
    if p.exists():
        return p
    candidate = PROJECT_ROOT / "src" / path_arg
    return candidate if candidate.exists() else p


def validate(path):
    path = _resolve(path)
    _, rows = _read(path)
    failed = []
    print(f"Validating {len(rows)} entries in {path.name}...\n")
    for i, row in enumerate(rows):
        name = row.get("entity_name") or row.get("location_name", "N/A")
        print(f"[{i+1}/{len(rows)}] Checking {row.get('qid', 'N/A')} ({name})...", end=" ")
        failures = check_row(row)
        if failures:
            print("FAILED")
            for reason in failures:
                print(f"    - {reason}")
            failed.append((i, row.get("qid"), name, failures))
        else:
            print("OK")
        if i < len(rows) - 1:
            time.sleep(DELAY_BETWEEN_QUERIES)

    print("\n" + "=" * 60)
    print(f"Validation complete: {len(rows) - len(failed)}/{len(rows)} passed")
    for i, qid, name, failures in failed:
        print(f"  Index {i} | {qid} | {name}")
        for reason in failures:
            print(f"    - {reason}")
    return failed


def repair(path, out=None):
    path = _resolve(path)
    out = Path(out) if out else path
    fieldnames, rows = _read(path)
    fixed = not_found = 0

    for i, row in enumerate(rows):
        name = row.get("location_name", "")
        country = row.get("country", "")
        old_qid = row.get("qid", "")
        print(f"[{i+1}/{len(rows)}] {name}, {country} ({old_qid})...", end=" ")

        if not check_row(row):
            print("OK")
            time.sleep(DELAY_BETWEEN_QUERIES)
            continue

        print("searching...", end=" ")
        time.sleep(DELAY_BETWEEN_QUERIES)
        new_qid = search_for_qid(name, country)
        if new_qid and new_qid != old_qid:
            print(f"FIXED {old_qid} -> {new_qid}")
            row["qid"] = new_qid
            fixed += 1
        elif new_qid == old_qid:
            print("QID already correct (check may be too strict)")
        else:
            print("NOT FOUND")
            not_found += 1
        time.sleep(DELAY_BETWEEN_QUERIES)

    with open(out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 60)
    print(f"Done: {fixed} fixed, {not_found} not found, "
          f"{len(rows) - fixed - not_found} already correct")
    print(f"Output written to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["validate", "repair"])
    p.add_argument("csv", help="dataset CSV with qid / location_name / country columns")
    p.add_argument("--out", default=None,
                   help="repair only: write here instead of in place")
    args = p.parse_args()
    if args.action == "validate":
        validate(args.csv)
    else:
        repair(args.csv, args.out)


if __name__ == "__main__":
    main()
