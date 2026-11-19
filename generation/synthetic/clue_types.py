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

Queries go through ../reform_places.py (QLever, retried, cached in
../out/cache): a rerun reads the cache, so delete it to ask Wikidata again.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import reform_places as rp   # noqa: E402  (sparql: the cached, retried query function)

import generate as gen       # noqa: E402
import plan                  # noqa: E402

CACHE = str(HERE.parent / "out" / "cache")
BATCH = 150


def query(qids: list[str]) -> dict[str, list[str]]:
    rows = rp.sparql("SELECT ?item ?label WHERE { VALUES ?item { " + rp.V(qids)
                     + ' } ?item wdt:P31 ?class . ?class rdfs:label ?label . FILTER(LANG(?label) = "en") }', CACHE)
    found: dict[str, list[str]] = {}
    for row in rows:                    # rp.sparql gives {variable: value}, an entity as its QID
        found.setdefault(row["item"], []).append(row["label"])
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
    os.makedirs(CACHE, exist_ok=True)
    types: dict[str, str] = {}
    for start in range(0, len(ordered), BATCH):
        for qid, labels in query(ordered[start:start + BATCH]).items():
            if words(labels):
                types[qid] = words(labels)
        print(f"  {min(start + BATCH, len(ordered))}/{len(ordered)}", flush=True)
    plan.atomic_write_text(out, json.dumps(dict(sorted(types.items())), ensure_ascii=False, indent=0) + "\n")
    print(f"{len(types)} of {len(ordered)} entities have a type; wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
