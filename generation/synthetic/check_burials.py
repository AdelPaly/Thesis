#!/usr/bin/env python3
"""
Posts that put a person's tomb where the person is not buried.

    python check_burials.py        # writes output/audit/burial_check.csv

For a clue who is a person from history the prompt offers "their tomb, statue,
house" as the way to bring them in, and the writer often picks the tomb even
when the person is buried somewhere else ("Karl Marx's tomb" in a post about
Trier; he is buried in London). In a post with no place name that points a
reader to the wrong city. Neither judge model catches it: they see the post and
a line about the clue, not where anyone is buried.

For every kept post whose clue is a person and whose text speaks of a tomb,
grave, burial or cemetery, this looks up the person's place of burial (P119)
and place of death (P20) on Wikidata and measures how far each is from the
post's own place. A post is listed as

    elsewhere            buried more than FAR_KM from the place
    elsewhere_by_death   Wikidata has no place of burial and the person died
                         more than FAR_KM away; taken as buried where they
                         died (2026-11-17, user)
    unknown              neither a place of burial nor a place of death with
                         coordinates
    here                 buried, or with no burial on record died, within
                         FAR_KM (not written to the file)

The distance is only a screen for this check. It takes no part in scoring.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import finalise
import generate as gen
import plan

HERE = Path(__file__).resolve().parent
ENDPOINT = "https://qlever.dev/api/wikidata"
FAR_KM = 100
WORDS = re.compile(r"\b(tomb|tombs|grave|graves|graveside|buried|burial|cemetery|mausoleum|resting place)\b", re.I)
COLUMNS = ["verdict", "item_id", "condition", "place", "country", "person", "buried_at", "buried_km",
           "died_at", "died_km", "text"]


def km(a: tuple[float, float], b: tuple[float, float]) -> float:
    (la1, lo1), (la2, lo2) = [(math.radians(x), math.radians(y)) for x, y in (a, b)]
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def lookup(qids: list[str]) -> dict[str, dict[str, tuple[str, tuple[float, float] | None]]]:
    """{person: {"buried": (label, coords), "died": (label, coords)}}."""
    sparql = (
        "PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "SELECT ?p ?kind ?label ?coord WHERE { VALUES ?p { " + " ".join("wd:" + q for q in qids) + " } "
        '{ ?p wdt:P119 ?x . BIND("buried" AS ?kind) } UNION { ?p wdt:P20 ?x . BIND("died" AS ?kind) } '
        '?x rdfs:label ?label . FILTER(LANG(?label) = "en") OPTIONAL { ?x wdt:P625 ?coord } }')
    request = urllib.request.Request(ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql}),
                                     headers={"Accept": "application/sparql-results+json",
                                              "User-Agent": "burial-check/1.0"})
    with urllib.request.urlopen(request, timeout=120, context=gen.tls_context()) as response:
        rows = json.load(response)["results"]["bindings"]
    found: dict[str, dict[str, tuple[str, tuple[float, float] | None]]] = {}
    for row in rows:
        coords = None
        match = re.match(r"POINT\(([-\d.eE]+) ([-\d.eE]+)\)", row.get("coord", {}).get("value", ""), re.I)
        if match:
            coords = (float(match.group(2)), float(match.group(1)))
        slot = found.setdefault(row["p"]["value"].rsplit("/", 1)[-1], {})
        if row["kind"]["value"] not in slot or (coords and not slot[row["kind"]["value"]][1]):
            slot[row["kind"]["value"]] = (row["label"]["value"], coords)
    return found


def main() -> int:
    plan.configure_stdout()
    config = gen.load_config(HERE / "config.yaml")
    paths = config["paths"]
    items = {i["item_id"]: i for i in gen.read_jsonl(Path(paths["item_plan"]))}
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))
    with Path(paths["dataset_csv"]).open(encoding="utf-8", newline="") as f:
        dataset = list(csv.DictReader(f))

    posts = []
    for row in dataset:
        if not WORDS.search(row["text"]):
            continue
        item = finalise.used_item(items[row["item_id"]], latest.get(row["item_id"]))
        for clue in item.get("clues") or []:
            if clue.get("is_human"):
                posts.append((row, clue))
    people = sorted({clue["clue_qid"] for _, clue in posts})
    facts: dict[str, dict] = {}
    for start in range(0, len(people), 100):
        facts.update(lookup(people[start:start + 100]))
        time.sleep(0.5)

    out = []
    for row, clue in posts:
        here = (float(row["latitude"]), float(row["longitude"]))
        fact = facts.get(clue["clue_qid"], {})
        buried, died = fact.get("buried"), fact.get("died")
        b_km = round(km(here, buried[1])) if buried and buried[1] else None
        d_km = round(km(here, died[1])) if died and died[1] else None
        if b_km is not None and b_km <= FAR_KM:
            continue
        if b_km is None and buried is None and d_km is not None and d_km <= FAR_KM:
            continue            # no burial place on record, but died here: no reason to doubt it
        out.append({"verdict": "elsewhere" if b_km is not None else
                    "elsewhere_by_death" if buried is None and d_km is not None else "unknown",
                    "item_id": row["item_id"], "condition": row["signal_type"], "place": row["location_name"],
                    "country": row["country"], "person": clue["clue_label"],
                    "buried_at": buried[0] if buried else "", "buried_km": b_km if b_km is not None else "",
                    "died_at": died[0] if died else "", "died_km": d_km if d_km is not None else "",
                    "text": row["text"]})
    out.sort(key=lambda r: (r["verdict"], r["condition"], r["person"]))
    path = Path(paths["dataset_csv"]).parent / "audit" / "burial_check.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    finalise.write_csv(path, COLUMNS, out)
    elsewhere = [r for r in out if r["verdict"] == "elsewhere"]
    by_death = [r for r in out if r["verdict"] == "elsewhere_by_death"]
    print(f"{len(posts)} posts speak of a person's tomb or grave ({len(people)} people); "
          f"{len(elsewhere)} are buried over {FAR_KM} km away, {len(by_death)} have no burial place on "
          f"Wikidata and died over {FAR_KM} km away, {len(out) - len(elsewhere) - len(by_death)} cannot be checked")
    for r in elsewhere + by_death:
        where = (f'buried at {r["buried_at"][:34]} ({r["buried_km"]} km)' if r["verdict"] == "elsewhere"
                 else f'died at {r["died_at"][:34]} ({r["died_km"]} km)')
        print(f'  {r["condition"]:<9}{r["place"][:20]:<21}{r["person"][:26]:<27}{where}')
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
