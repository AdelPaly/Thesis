"""Validate a generated dataset (social media or journalistic) against Wikidata.

This is the validation script for the *synthetic* datasets, analogous to the
GeoCorpora validator in `fix_qids.py` (whose run log is committed at
results/logs/fix_qids_geocorpora_144.log).

For every row it checks, via the Wikidata SPARQL endpoint:
  1. the `qid` actually exists in Wikidata;
  2. the English label of that item matches `entity_name` (if present) or
     otherwise `location_name`; and
  3. the item's country (P17) is consistent with the `country` column.

Rows that fail any check are reported. During dataset construction the failing
rows were regenerated (see DATASETS.md for the first-pass failure counts).

Usage:
    python scripts/validate_dataset.py [path-to-csv]

With no argument it defaults to src/social_media_dataset.csv. Run it on each
dataset, e.g.:
    python scripts/validate_dataset.py src/social_media_dataset.csv
    python scripts/validate_dataset.py src/journalistic_dataset.csv

No API key is required — the Wikidata SPARQL endpoint is public.
"""
import csv
import sys
import time
from pathlib import Path

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "GeoProject/1.0 (dataset-validator)"
DELAY_BETWEEN_QUERIES = 1.5  # seconds, to respect Wikidata rate limits

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
sparql.setReturnFormat(JSON)


def run_query(query, retries=1):
    """Execute a SPARQL query with one retry on failure."""
    sparql.setQuery(query)
    for attempt in range(retries + 1):
        try:
            results = sparql.query().convert()
            return results["results"]["bindings"]
        except (SPARQLExceptions.EndPointInternalError,
                SPARQLExceptions.EndPointNotFound,
                SPARQLExceptions.QueryBadFormed,
                Exception) as e:
            if attempt < retries:
                print(f"    Query failed ({e}), retrying...")
                time.sleep(DELAY_BETWEEN_QUERIES)
            else:
                raise


def validate_entry(entry):
    """Validate a single dataset entry against Wikidata.

    Returns a list of failure reasons (empty if all checks pass).
    """
    qid = entry["qid"]
    entity_name = entry.get("entity_name")
    location_name = entry["location_name"]
    country = entry["country"]

    # Lightweight query: fetch English label and country (P17) only
    query = f"""
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
    """

    try:
        results = run_query(query)
    except Exception as e:
        return [f"SPARQL query error: {e}"]

    failures = []

    # Check 1: QID exists (query returned results with a label)
    if not results:
        failures.append("QID does not exist in Wikidata")
        return failures

    labels = {r["label"]["value"] for r in results if "label" in r}

    # Check 2: English label matches entity_name or location_name
    if entity_name:
        if not any(label.lower() == entity_name.lower() for label in labels):
            failures.append(
                f"Label mismatch: expected '{entity_name}', "
                f"Wikidata has {labels}"
            )
    else:
        if not any(label.lower() == location_name.lower() for label in labels):
            failures.append(
                f"Label mismatch: expected '{location_name}', "
                f"Wikidata has {labels}"
            )

    # Check 3: Country matches
    country_labels = set()
    for r in results:
        if "countryLabel" in r:
            country_labels.add(r["countryLabel"]["value"].lower())

    country_lower = country.lower()

    if country_labels:
        if not any(country_lower in c or c in country_lower
                   for c in country_labels):
            failures.append(
                f"Country mismatch: expected '{country}', "
                f"Wikidata has {country_labels}"
            )
    else:
        failures.append(
            f"No country (P17) found in Wikidata for {qid}"
        )

    return failures


def load_dataset(filepath):
    """Load a dataset from a CSV file."""
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def validate_dataset(filepath):
    """Validate all entries in the dataset file."""
    dataset = load_dataset(filepath)

    failed_entries = []
    total = len(dataset)

    print(f"Validating {total} entries...\n")

    for i, entry in enumerate(dataset):
        qid = entry.get("qid", "N/A")
        # Prefer entity_name when present, otherwise show the location name.
        entity = entry.get("entity_name") or entry.get("location_name", "N/A")
        print(f"[{i + 1}/{total}] Checking {qid} ({entity})...", end=" ")

        failures = validate_entry(entry)

        if failures:
            print("FAILED")
            for reason in failures:
                print(f"    - {reason}")
            failed_entries.append({
                "index": i,
                "qid": qid,
                "entity_name": entity,
                "reasons": failures,
            })
        else:
            print("OK")

        if i < total - 1:
            time.sleep(DELAY_BETWEEN_QUERIES)

    # Summary
    print("\n" + "=" * 60)
    print(f"Validation complete: {total - len(failed_entries)}/{total} passed")

    if failed_entries:
        print(f"\n{len(failed_entries)} failed entries:\n")
        for entry in failed_entries:
            print(f"  Index {entry['index']} | {entry['qid']} | "
                  f"{entry['entity_name']}")
            for reason in entry["reasons"]:
                print(f"    - {reason}")
    else:
        print("All entries passed validation.")

    return failed_entries


def _resolve(path_arg):
    """Resolve a CLI path. An existing absolute/relative path is used as given;
    a bare filename is looked up under the project's src/ directory."""
    p = Path(path_arg)
    if p.exists():
        return p
    candidate = PROJECT_ROOT / "src" / path_arg
    return candidate if candidate.exists() else p


if __name__ == "__main__":
    if len(sys.argv) < 2:
        filepath = PROJECT_ROOT / "src" / "social_media_dataset.csv"
    else:
        filepath = _resolve(sys.argv[1])

    validate_dataset(filepath)
