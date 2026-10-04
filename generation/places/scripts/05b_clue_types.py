#!/usr/bin/env python3
"""Classify every clue entity in 05_clue_sets.jsonl by whether it is itself a
PLACE NAME. Measurement only -- changes nothing downstream.

Why it matters. The no-toponym conditions (A1, A3, NOISE_A1) forbid any
settlement, region, island, mountain, park or street name in the post, and AB
allows exactly one place name, the ambiguous one. A clue that is itself a
village ("Rüttmannsdorf", reached through P131) or a mountain ("Cheopsfjellet",
reached through P138) makes those prompts contradict themselves: the post must
mention the clue by name and must not contain a place name. The smoke run
showed the model resolving that by inventing ("coffee from Rüttmannsdorf, a
group", "Cheopsfjellet's hot chocolate").

THE TEST. An entity is in a category if it is an instance (P31) of the
category's class or of any subclass of it (P279*). The categories mirror the
things the forbidden_toponym prompt names. An entity can fall in several; its
`toponym_query` is true if it falls in any. P31 labels are recorded alongside so
every classification can be read and argued with.

REVIEWED OVERRIDES. P279 chains overreach: a castle or a fort can reach
"administrative territorial entity" through a subclass path. Each such case,
once read and judged, goes in data/05b_clue_type_overrides.csv with its
reason. `toponym` is the query result with those overrides applied, and it is
what the planner reads; `toponym_query` and `override_reason` keep both the
raw answer and the judgement on record. An override for an entity no longer in
the clue sets is reported, not silently ignored.

Output:
    data/05b_clue_types.jsonl   one row per distinct clue entity
    data/05b_type_cache.json    query cache, same format as 04's

Usage:
    python3 scripts/05b_clue_types.py [--clues data/05_clue_sets.jsonl]
                                      [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_clues = import_module("02_clues")
_score = import_module("04_score_clues")

value_of = _clues.value_of
qid_from_uri = _clues.qid_from_uri
run_chunked = _score.run_chunked

ROOT = Path(__file__).resolve().parent.parent
CLUE_SETS = ROOT / "data" / "05_clue_sets.jsonl"
OUT_JSONL = ROOT / "data" / "05b_clue_types.jsonl"
CACHE_PATH = ROOT / "data" / "05b_type_cache.json"
OVERRIDES = ROOT / "data" / "05b_clue_type_overrides.csv"

CACHE_VERSION = 1
CHUNK = 100

# The things the forbidden_toponym prompt tells the model not to name.
CATEGORIES = {
    "settlement": "Q486972",        # human settlement
    "admin_area": "Q56061",         # administrative territorial entity
    "neighbourhood": "Q123705",     # neighborhood
    "island": "Q23442",
    "mountain": "Q8502",
    "hill": "Q54050",
    "park": "Q22698",
    "street": "Q79007",
    "body_of_water": "Q15324",
    "watercourse": "Q355304",
    "landform": "Q271669",
}


def category_query(qids: list[str]) -> str:
    values = " ".join(f"wd:{q}" for q in qids)
    classes = " ".join(f'(wd:{q} "{name}")' for name, q in CATEGORIES.items())
    return f"""
SELECT DISTINCT ?e ?cat WHERE {{
  VALUES ?e {{ {values} }}
  VALUES (?cls ?cat) {{ {classes} }}
  ?e wdt:P31/wdt:P279* ?cls .
}}
"""


def instance_query(qids: list[str]) -> str:
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT ?e ?typeLabel WHERE {{
  VALUES ?e {{ {values} }}
  ?e wdt:P31 ?type .
  ?type rdfs:label ?typeLabel FILTER(lang(?typeLabel) = "en")
}}
"""


def collect_multi(key: str):
    def collect(batch: list[str], rows: list[dict], out: dict) -> None:
        found: dict[str, list[str]] = {q: [] for q in batch}
        for row in rows:
            entity = value_of(row, "e")
            value = value_of(row, key)
            if not entity or value is None:
                continue
            qid = qid_from_uri(entity)
            if qid in found and value not in found[qid]:
                found[qid].append(value)
        out.update({q: sorted(v) for q, v in found.items()})
    return collect


def load_overrides(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out = {}
    for row in rows:
        value = (row.get("toponym") or "").strip().lower()
        if value not in ("true", "false"):
            raise SystemExit(f"{path}: toponym for {row.get('clue_qid')} is "
                             f"{row.get('toponym')!r}, expected true or false")
        if not (row.get("reason") or "").strip():
            raise SystemExit(f"{path}: override for {row.get('clue_qid')} has no reason")
        out[row["clue_qid"].strip()] = {"toponym": value == "true",
                                        "reason": row["reason"].strip()}
    return out


def load_cache(use_cache: bool) -> dict:
    if not use_cache or not CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("version") != CACHE_VERSION:
        return {}
    return payload.get("entries", {})


def save_cache(entries: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps({"version": CACHE_VERSION, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def fetch(qids, prefix, build, collect, label, cache, use_cache, failures) -> dict:
    result = {}
    want = []
    for qid in qids:
        key = f"{prefix}|{qid}"
        if use_cache and key in cache:
            result[qid] = cache[key]
        else:
            want.append(qid)
    print(f"  {label}: {len(qids)} entities, {len(want)} to query", flush=True)
    if want:
        got = run_chunked(want, CHUNK, build, collect, label, failures)
        for qid, value in got.items():
            result[qid] = value
            cache[f"{prefix}|{qid}"] = value
        if use_cache:
            save_cache(cache)
    return result


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description="Classify clue entities as place names.")
    parser.add_argument("--clues", type=Path, default=CLUE_SETS)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    entities: dict[str, dict] = {}
    for line in args.clues.open(encoding="utf-8"):
        if not line.strip():
            continue
        record = json.loads(line)
        for entry in record.get("best_sets") or []:
            for clue in entry.get("clues") or []:
                row = entities.setdefault(clue["clue_qid"], {
                    "clue_qid": clue["clue_qid"],
                    "clue_label": clue["clue_label"],
                    "properties": set(),
                    "places": set(),
                })
                row["properties"].add(clue["property"])
                row["places"].add(record["place_qid"])

    qids = sorted(entities)
    use_cache = not args.no_cache
    cache = load_cache(use_cache)
    failures: list[tuple[str, str]] = []

    print(f"clue entities      : {len(qids)}")
    categories = fetch(qids, "cat", category_query, collect_multi("cat"),
                       "categories", cache, use_cache, failures)
    instances = fetch(qids, "p31", instance_query, collect_multi("typeLabel"),
                      "instance-of labels", cache, use_cache, failures)

    overrides = load_overrides(OVERRIDES)
    stale = sorted(set(overrides) - set(qids))

    out = []
    for qid in qids:
        row = entities[qid]
        cats = categories.get(qid)
        # None when the lookup failed: unknown, not "not a place".
        raw = None if cats is None else bool(cats)
        override = overrides.get(qid)
        out.append({
            "clue_qid": qid,
            "clue_label": row["clue_label"],
            "properties": sorted(row["properties"]),
            "place_qids": sorted(row["places"]),
            "instance_of": instances.get(qid),
            "categories": cats,
            "toponym_query": raw,
            "toponym": override["toponym"] if override else raw,
            "override_reason": override["reason"] if override else None,
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    # ---- report ----
    toponyms = [r for r in out if r["toponym"]]
    unknown = [r for r in out if r["toponym"] is None]
    overridden = [r for r in out if r["override_reason"]]
    print("\n" + "=" * 70)
    print(f"toponym by query   : {sum(1 for r in out if r['toponym_query'])} of {len(out)}")
    print(f"overrides applied  : {len(overridden)} (from {OVERRIDES.name})")
    print(f"toponym clues      : {len(toponyms)} of {len(out)}"
          + (f"   ({len(unknown)} unknown - lookup failed)" if unknown else ""))
    if stale:
        print(f"!! {len(stale)} overrides name entities not in the clue sets: {stale[:5]}")
    print("\nby category, final toponyms (an entity can be in several)")
    for cat, n in Counter(c for r in toponyms for c in r["categories"]).most_common():
        print(f"  {cat:<15} {n:>5}")
    print("\nby property (toponym / all)")
    by_prop: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in out:
        for prop in r["properties"]:
            by_prop[prop][1] += 1
            by_prop[prop][0] += bool(r["toponym"])
    for prop, (hit, total) in sorted(by_prop.items(), key=lambda kv: -kv[1][0]):
        print(f"  {prop:<8} {hit:>5} / {total:<5}")

    if failures:
        print(f"\n!! {len(failures)} lookups failed; those entities are written with "
              f"toponym null. Re-run to retry just those.")
    print(f"\nwrote {args.out}   {len(out)} entities")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
