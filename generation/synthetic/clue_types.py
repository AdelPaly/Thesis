#!/usr/bin/env python3
"""
What kind of thing each planned clue and distractor is, from Wikidata.

    python clue_types.py        # writes output/clue_types.json

The plan knows a clue only by its name and a broad family ("a place or
organisation"), so the writer treated railway stations, parks and a boat as
companies with offices and trucks, and the judge, told no more, let it pass.
This looks up "instance of" (P31) for every clue, spare clue and distractor
entity in the plan that is not a person and keeps the English labels, so the
prompt can say "a railway station".

A label with a capital letter in it is dropped ("commune of France"): it would
tell the writer the country. At most two labels are kept per entity, the
shortest first. The plan itself is not changed; generate.attach_clue_types()
adds the words to the items in memory.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import generate as gen
import plan

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://qlever.dev/api/wikidata"
BATCH = 150
PREFIXES = ("PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> ")


def query(qids: list[str]) -> dict[str, list[str]]:
    sparql = (PREFIXES + "SELECT ?item ?label WHERE { VALUES ?item { " + " ".join("wd:" + q for q in qids)
              + ' } ?item wdt:P31 ?class . ?class rdfs:label ?label . FILTER(LANG(?label) = "en") }')
    request = urllib.request.Request(ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql}),
                                     headers={"Accept": "application/sparql-results+json",
                                              "User-Agent": "clue-types/1.0"})
    with urllib.request.urlopen(request, timeout=120, context=gen.tls_context()) as response:
        rows = json.load(response)["results"]["bindings"]
    found: dict[str, list[str]] = {}
    for row in rows:
        found.setdefault(row["item"]["value"].rsplit("/", 1)[-1], []).append(row["label"]["value"])
    return found


def words(labels: list[str]) -> str:
    plain = sorted({l for l in labels if l == l.lower()}, key=lambda l: (len(l), l))
    return " / ".join(plain[:2])


def main() -> int:
    plan.configure_stdout()
    config = gen.load_config(HERE / "config.yaml")
    out = Path(config["paths"]["clue_types"])
    qids: set[str] = set()
    for item in gen.read_jsonl(Path(config["paths"]["item_plan"])):
        entities = (item.get("clues") or []) + (item.get("spare_clues") or [])
        if (item.get("noise_entity") or {}).get("kind") == "entity":
            entities.append(item["noise_entity"])
        qids |= {e["clue_qid"] for e in entities if not e.get("is_human") and e.get("clue_qid")}
    ordered = sorted(qids)
    types: dict[str, str] = {}
    for start in range(0, len(ordered), BATCH):
        for qid, labels in query(ordered[start:start + BATCH]).items():
            if words(labels):
                types[qid] = words(labels)
        print(f"  {min(start + BATCH, len(ordered))}/{len(ordered)}", flush=True)
        time.sleep(0.5)
    plan.atomic_write_text(out, json.dumps(dict(sorted(types.items())), ensure_ascii=False, indent=0) + "\n")
    print(f"{len(types)} of {len(ordered)} entities have a type; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
