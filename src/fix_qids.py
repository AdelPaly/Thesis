import csv
import sys
import time
import urllib.parse
import urllib.request
import json

from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_SEARCH_API = "https://www.wikidata.org/w/api.php"
USER_AGENT = "GeoProject/1.0 (qid-fixer)"
DELAY_BETWEEN_QUERIES = 1.5

sparql = SPARQLWrapper(SPARQL_ENDPOINT)
sparql.addCustomHttpHeader("User-Agent", USER_AGENT)
sparql.setReturnFormat(JSON)


def search_wikidata(location_name, country):
    """Search Wikidata for a location by name and verify it belongs to the expected country.

    Returns the correct QID or None if not found.
    """
    params = urllib.parse.urlencode({
        "action": "wbsearchentities",
        "search": location_name,
        "language": "en",
        "type": "item",
        "limit": 10,
        "format": "json",
    })
    url = f"{WIKIDATA_SEARCH_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    candidates = data.get("search", [])
    if not candidates:
        return None

    # Check each candidate's country (P17) against expected country
    for candidate in candidates:
        qid = candidate["id"]
        time.sleep(DELAY_BETWEEN_QUERIES)

        query = f"""
        SELECT ?countryLabel WHERE {{
            wd:{qid} wdt:P17 ?country .
            ?country rdfs:label ?countryLabel .
            FILTER(LANG(?countryLabel) = "en")
        }}
        LIMIT 5
        """
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()["results"]["bindings"]
        except Exception:
            continue

        country_lower = country.lower()
        for r in results:
            if "countryLabel" in r:
                c = r["countryLabel"]["value"].lower()
                if country_lower in c or c in country_lower:
                    return qid

    return None


def fix_qids(input_path, output_path=None):
    """Read the CSV, find correct QIDs for failed entries, and write a fixed CSV."""
    if output_path is None:
        output_path = input_path

    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    total = len(rows)
    fixed = 0
    failed = 0

    for i, row in enumerate(rows):
        location_name = row["location_name"]
        country = row["country"]
        old_qid = row["qid"]

        print(f"[{i + 1}/{total}] {location_name}, {country} ({old_qid})...", end=" ")

        # First check if the current QID is already correct
        query = f"""
        SELECT ?label ?countryLabel WHERE {{
            wd:{old_qid} rdfs:label ?label .
            FILTER(LANG(?label) = "en")
            OPTIONAL {{
                wd:{old_qid} wdt:P17 ?country .
                ?country rdfs:label ?countryLabel .
                FILTER(LANG(?countryLabel) = "en")
            }}
        }}
        LIMIT 10
        """
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()["results"]["bindings"]
        except Exception:
            results = []

        labels = {r["label"]["value"].lower() for r in results if "label" in r}
        countries = {r["countryLabel"]["value"].lower() for r in results if "countryLabel" in r}

        label_ok = location_name.lower() in labels
        country_ok = any(country.lower() in c or c in country.lower() for c in countries)

        if label_ok and country_ok:
            print("OK")
            time.sleep(DELAY_BETWEEN_QUERIES)
            continue

        # Search for the correct QID
        print("searching...", end=" ")
        time.sleep(DELAY_BETWEEN_QUERIES)
        new_qid = search_wikidata(location_name, country)

        if new_qid and new_qid != old_qid:
            print(f"FIXED {old_qid} -> {new_qid}")
            row["qid"] = new_qid
            fixed += 1
        elif new_qid == old_qid:
            print("QID already correct (validation may be too strict)")
        else:
            print("NOT FOUND")
            failed += 1

        time.sleep(DELAY_BETWEEN_QUERIES)

    # Write the fixed CSV
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'=' * 60}")
    print(f"Done: {fixed} fixed, {failed} not found, {total - fixed - failed} already correct")
    print(f"Output written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        input_path = ("social_media_dataset.csv")
    else:
        input_path = sys.argv[1]

    output_path = input_path if len(sys.argv) < 3 else sys.argv[2]
    fix_qids(input_path, output_path)
