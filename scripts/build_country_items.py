"""Build src/country_items.json: the Wikidata country items, with English labels
and aliases, that rule v4 of src/resolver.py matches a country name against.

    python scripts/build_country_items.py

A country item is an item with a P31 statement of country (Q6256) or
sovereign state (Q3624078) that is not deprecated and has no end time (P582),
and that has no dissolution date (P576). Statements, not truthy wdt:P31: China
(Q148) has a preferred-rank P31 of another class, so wdt:P31 misses it.
One QLever query; the result is committed so a run does not depend on the
endpoint, and the file records the query and the date.
"""
from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "src" / "country_items.json"
URL = "https://qlever.dev/api/wikidata"
QUERY = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT ?c ?label (GROUP_CONCAT(DISTINCT ?alias; separator="|") AS ?aliases) WHERE {
  VALUES ?class { wd:Q6256 wd:Q3624078 }
  ?c p:P31 ?st . ?st ps:P31 ?class .
  FILTER NOT EXISTS { ?st wikibase:rank wikibase:DeprecatedRank }
  FILTER NOT EXISTS { ?st pq:P582 ?stEnd }
  FILTER NOT EXISTS { ?c wdt:P576 ?end }
  ?c rdfs:label ?label . FILTER(LANG(?label) = "en")
  OPTIONAL { ?c skos:altLabel ?alias . FILTER(LANG(?alias) = "en") }
}
GROUP BY ?c ?label"""


def main() -> None:
    for attempt in range(6):
        resp = requests.get(URL, params={"query": QUERY}, timeout=300,
                            headers={"Accept": "application/sparql-results+json",
                                     "User-Agent": os.environ.get("WDQS_CONTACT", "geo-thesis/0.1")})
        if resp.status_code != 429 and resp.status_code < 500:
            break
        wait = 30 * 2 ** attempt
        print(f"HTTP {resp.status_code}, retrying in {wait}s", flush=True)
        time.sleep(wait)
    resp.raise_for_status()
    items = {}
    for b in resp.json()["results"]["bindings"]:
        qid = b["c"]["value"].rsplit("/", 1)[-1]
        aliases = [a for a in (b.get("aliases", {}).get("value") or "").split("|") if a]
        items[qid] = {"label": b["label"]["value"], "aliases": sorted(set(aliases))}
    OUT.write_text(json.dumps({"built": datetime.date.today().isoformat(), "source": URL,
                               "query": QUERY, "items": dict(sorted(items.items()))},
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(items)} country items)")


if __name__ == "__main__":
    main()
