#!/usr/bin/env python3
"""
Build the planner's place inputs from a list of places.

    python build_inputs.py my_places.csv --out-dir data/my_run
    python build_inputs.py my_places.csv --out-dir data/my_run --per-continent 17

The input CSV needs one column, `wikidata_id`. These are filled from Wikidata
when blank or absent: place_name, country, continent, wikimedia_sitelinks,
places_sharing_name. `type` defaults to "town". wikidata_links (inbound link
count) is only carried as metadata and is left blank, because the query is
slow on large places.

It runs the clue stages in order, each into --out-dir:

    02_clues.py           clue entities that point at each place
    05_name_leakage.py    which clues contain their place's name
    03_filter_clues.py    drop leaking and unlabelled clues
    04_score_clues.py     how many places each clue could point to
    05_score_sets.py      sets of clues that together point to one place
    -- survivors --       places with at least one such set; optional
                          per-continent balancing
    04b_place_aliases.py  the surface forms of each surviving place
    05b_clue_types.py     which clue entities are themselves places

Many places do not survive: on the original run 99 of 147 did. Start with
about half as many places again as you want to end up with.

The five files the planner reads end up in --out-dir:

    01_places_final.csv  05_clue_sets.jsonl  04b_place_aliases.jsonl
    05_name_leakage.csv  05b_clue_types.jsonl

and the script prints the lines to paste into ../synthetic/config.yaml.
Query caches are shared in data/, so re-running is cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_places as vp  # noqa: E402  (SPARQL client and lookups)

COLUMNS = ["place_name", "type", "country", "continent", "wikidata_id",
           "wikimedia_sitelinks", "wikidata_links", "places_sharing_name"]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America",
              "Oceania"]
CHUNK = 50


# --------------------------------------------------------------------------
# step 0: fill in the place columns
# --------------------------------------------------------------------------

def _continent_name(label: str) -> str | None:
    """Map a Wikidata P30 label onto the six continents the datasets use."""
    text = label.casefold()
    if "oceania" in text or "australia" in text:
        return "Oceania"
    for name in CONTINENTS:
        if name.casefold() in text:
            return name
    return None


def query_labels(qids: list[str]) -> dict[str, str]:
    values = " ".join(f"wd:{q}" for q in qids)
    rows = vp.sparql(f"""
SELECT ?item ?label WHERE {{
  VALUES ?item {{ {values} }}
  ?item rdfs:label ?label FILTER(LANG(?label) = "en")
}}""")
    return {vp.qid_from_uri(vp.value_of(r, "item")): vp.value_of(r, "label")
            for r in rows if vp.value_of(r, "item")}


def query_continents(country_qids: list[str]) -> dict[str, str]:
    values = " ".join(f"wd:{q}" for q in country_qids)
    rows = vp.sparql(f"""
SELECT ?country ?label WHERE {{
  VALUES ?country {{ {values} }}
  ?country wdt:P30 ?continent .
  ?continent rdfs:label ?label FILTER(LANG(?label) = "en")
}}""")
    out: dict[str, str] = {}
    for r in rows:
        name = _continent_name(vp.value_of(r, "label") or "")
        if name:
            out.setdefault(vp.qid_from_uri(vp.value_of(r, "country")), name)
    return out


def enrich(rows: list[dict]) -> list[dict]:
    """Fill blank columns from Wikidata. Values already in the CSV are kept."""
    qids = [r["wikidata_id"] for r in rows]
    core, labels = {}, {}
    for i in range(0, len(qids), CHUNK):
        batch = qids[i:i + CHUNK]
        core.update(vp.query_core(batch))
        labels.update(query_labels(batch))

    country_of = {q: sorted(core[q]["countries"])[0] if core[q]["countries"] else None
                  for q in qids}
    country_qids = sorted({c for c in country_of.values() if c})
    country_names, continents = {}, {}
    for i in range(0, len(country_qids), CHUNK):
        batch = country_qids[i:i + CHUNK]
        country_names.update({q: v["label"] for q, v in vp.query_country_names(batch).items()})
        continents.update(query_continents(batch))

    out = []
    for row in rows:
        qid = row["wikidata_id"]
        info = core.get(qid, {})
        if not info.get("exists"):
            print(f"  ! {qid} does not exist on Wikidata, skipped", file=sys.stderr)
            continue
        country_qid = country_of.get(qid)
        filled = {
            "place_name": row.get("place_name") or labels.get(qid) or "",
            "type": row.get("type") or "town",
            "country": row.get("country") or country_names.get(country_qid) or "",
            "continent": row.get("continent") or continents.get(country_qid) or "",
            "wikidata_id": qid,
            "wikimedia_sitelinks": row.get("wikimedia_sitelinks") or info.get("sitelinks") or "",
            "wikidata_links": row.get("wikidata_links") or "",
            "places_sharing_name": row.get("places_sharing_name") or "",
        }
        if not filled["place_name"]:
            print(f"  ! {qid} has no English label, skipped", file=sys.stderr)
            continue
        out.append({**row, **filled})

    need = sorted({r["place_name"] for r in out if not str(r["places_sharing_name"]).strip()})
    counts: dict[str, int] = {}
    for i in range(0, len(need), CHUNK):
        counts.update(vp.query_label_counts(need[i:i + CHUNK]))
    for r in out:
        if not str(r["places_sharing_name"]).strip():
            # An item always shares its own label, so the floor is 1.
            r["places_sharing_name"] = max(1, counts.get(r["place_name"], 1))

    for r in out:
        missing = [c for c in ("country", "continent") if not r[c]]
        if missing:
            print(f"  ! {r['place_name']} ({r['wikidata_id']}): could not fill "
                  f"{', '.join(missing)} - set it in the input CSV", file=sys.stderr)
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = COLUMNS + [c for c in rows[0] if c not in COLUMNS] if rows else COLUMNS
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(script: str, *args: Path | str) -> None:
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"\n=== {script} " + "=" * max(0, 60 - len(script)), flush=True)
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"{script} failed (exit {result.returncode}); fix it and "
                         f"re-run - finished steps are cached")


def choose_survivors(places: list[dict], sets: list[dict],
                     per_continent: int | None) -> list[dict]:
    """Places with at least one clue set that resolves to them alone,
    optionally cut to the same number per continent (most resolving sets
    first, then name, so the cut is deterministic)."""
    strength = {r["place_qid"]: sum(1 for s in r.get("best_sets", []) if s.get("resolves"))
                for r in sets}
    alive = [p for p in places if strength.get(p["wikidata_id"], 0) > 0]
    if not per_continent:
        return alive
    by_continent: dict[str, list[dict]] = defaultdict(list)
    for p in alive:
        by_continent[p["continent"]].append(p)
    chosen = []
    for continent, group in sorted(by_continent.items()):
        group.sort(key=lambda p: (-strength[p["wikidata_id"]], p["place_name"]))
        chosen += group[:per_continent]
        if len(group) < per_continent:
            print(f"  ! {continent or '(no continent)'}: only {len(group)} usable "
                  f"places, wanted {per_continent}")
    return chosen


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("places", type=Path, help="CSV with at least a wikidata_id column")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="folder for every output, e.g. data/my_run")
    parser.add_argument("--per-continent", type=int, default=None,
                        help="keep at most this many surviving places per continent")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into a folder that already has outputs")
    args = parser.parse_args(argv)

    out = args.out_dir if args.out_dir.is_absolute() else (Path.cwd() / args.out_dir)
    out = out.resolve()
    if out == (HERE / "data").resolve():
        raise SystemExit("refusing to write into data/ itself - it holds the current "
                         "planner inputs. Pick a subfolder, e.g. data/my_run")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)

    rows = read_csv(args.places)
    if not rows or "wikidata_id" not in rows[0]:
        raise SystemExit(f"{args.places} needs a wikidata_id column")
    rows = [r for r in rows if r["wikidata_id"].startswith("Q")]
    dupes = [q for q, n in Counter(r["wikidata_id"] for r in rows).items() if n > 1]
    if dupes:
        raise SystemExit(f"duplicate wikidata_id in input: {dupes[:5]}")

    print(f"=== filling place columns for {len(rows)} places")
    places = enrich(rows)
    all_places = out / "01_places_input.csv"
    write_csv(all_places, places)
    print(f"wrote {all_places}")

    clues = out / "02_clues.jsonl"
    filtered = out / "02_clues_filtered.jsonl"
    scored = out / "04_clues_scored.jsonl"
    all_sets = out / "05_clue_sets_all.jsonl"
    run("02_clues.py", "--input", all_places, "--out", clues)
    run("05_name_leakage.py", "--clues", clues, "--out", out / "05_name_leakage.csv")
    run("03_filter_clues.py", "--clues", clues, "--out", filtered,
        "--removed", out / "03_removed_clues.jsonl", "--places", all_places)
    run("04_score_clues.py", "--clues", filtered, "--out", scored,
        "--summary", out / "04_score_summary.csv")
    run("05_score_sets.py", "--clues", scored, "--out", all_sets)

    sets = read_jsonl(all_sets)
    survivors = choose_survivors(places, sets, args.per_continent)
    if not survivors:
        raise SystemExit("no place has a clue set that points to it alone")
    keep = {p["wikidata_id"] for p in survivors}
    final_csv = out / "01_places_final.csv"
    write_csv(final_csv, survivors)
    final_sets = out / "05_clue_sets.jsonl"
    with final_sets.open("w", encoding="utf-8") as handle:
        for record in sets:
            if record["place_qid"] in keep:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    run("04b_place_aliases.py", "--places", final_csv, "--out", out / "04b_place_aliases.jsonl")
    run("05b_clue_types.py", "--clues", final_sets, "--out", out / "05b_clue_types.jsonl")

    print("\n=== result")
    print(f"input places   : {len(places)}")
    print(f"usable         : {sum(1 for r in sets if any(s.get('resolves') for s in r.get('best_sets', [])))}")
    print(f"kept           : {len(survivors)}")
    for continent, n in sorted(Counter(p["continent"] for p in survivors).items()):
        print(f"  {continent or '(none)':<15} {n}")

    rel = Path("..") / "places" / out.relative_to(HERE) if out.is_relative_to(HERE) else out
    print("\nto generate from these places, set in ../synthetic/config.yaml:")
    for key, name in (("places_csv", "01_places_final.csv"),
                      ("clue_sets", "05_clue_sets.jsonl"),
                      ("place_aliases", "04b_place_aliases.jsonl"),
                      ("name_leakage", "05_name_leakage.csv"),
                      ("clue_types", "05b_clue_types.jsonl")):
        print(f"  {key + ':':<18} {(rel / name).as_posix()}")
    print("then make a new plan: python plan.py --replace")
    return 0


if __name__ == "__main__":
    sys.exit(main())
