#!/usr/bin/env python3
"""
Build the planner's place inputs from a list of places.

    python build_inputs.py my_places.csv --out-dir data/my_run
    python build_inputs.py my_places.csv --out-dir data/my_run --per-continent 17

The input CSV needs one column, `wikidata_id`. These are filled from Wikidata
when blank or absent: place_name, country, continent, wikimedia_sitelinks,
places_sharing_name, area_km2. `type` defaults to "town". wikidata_links
(inbound link count) is only carried as metadata and is left blank, because the
query is slow on large places.

The output place CSVs use the schema the companion tools expect, so a rebuilt
set can be verified and measured like the shipped one:

    place_name, type, country, continent, wikidata_id, clue_count,
    wikimedia_sitelinks, wikidata_links, places_sharing_name, area_km2

`clue_count` is the number of clues 02_clues.py collected for the place (blank in
01_places_input.csv, which is written before 02 runs). `area_km2` is the largest
non-deprecated P2046 value from the same query that fills the other columns; it
is NOT rank-aware, so run scripts/04_fix_areas.py on the result for the
best-rank value. `clue_total` and `capped` are added by scripts/03_clue_totals.py
(which needs `clue_count`), as in the shipped file.

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

Many places do not survive: on the original run 99 of the 147 candidates reached
a resolving set. Start with about half as many places again as you want to end
up with.

Survivor rule (choose_survivors): a place survives if at least one of its kept
best sets (05's `best_sets`) intersects to that place alone ("resolves"). With
--per-continent N, each continent is cut to its N places with the most resolving
best sets, ties broken by place name.

The five files the planner reads end up in --out-dir:

    01_places_final.csv  05_clue_sets.jsonl  04b_place_aliases.jsonl
    05_name_leakage.csv  05b_clue_types.jsonl

and the script prints the lines to paste into ../synthetic/config.yaml.
Query caches are shared in data/, so re-running is cheap.

--out-dir is resolved against this script's folder (generation/places/), not the
current directory, so `--out-dir data/my_run` means the same wherever you run it
from. Every run also writes run_manifest.json (date, seed, arguments, git hash)
into --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / "scripts"
sys.path.insert(0, str(SCRIPTS))

import verify_places as vp  # noqa: E402  (batched place lookups: core, names, counts)
from clue_queries import DEFAULT_SEED  # noqa: E402
from io_utils import (  # noqa: E402
    read_jsonl, utf8_stdout, write_csv_atomic, write_jsonl_atomic, write_text_atomic,
)
from wdqs import qid_from_uri, sparql, value_of  # noqa: E402

COLUMNS = ["place_name", "type", "country", "continent", "wikidata_id", "clue_count",
           "wikimedia_sitelinks", "wikidata_links", "places_sharing_name", "area_km2"]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America",
              "Oceania"]
CHUNK = 50                  # QIDs per VALUES clause in the lookups below
RULE_WIDTH = 60             # width of the "=== script ===" banner
M2_PER_KM2 = 1_000_000
MANIFEST_NAME = "run_manifest.json"
GIT_TIMEOUT = 10            # seconds


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
    """English rdfs:label of each item (items without one are absent)."""
    values = " ".join(f"wd:{q}" for q in qids)
    rows = sparql(f"""
SELECT ?item ?label WHERE {{
  VALUES ?item {{ {values} }}
  ?item rdfs:label ?label FILTER(LANG(?label) = "en")
}}""")
    return {qid_from_uri(value_of(r, "item")): value_of(r, "label")
            for r in rows if value_of(r, "item")}


def query_continents(country_qids: list[str]) -> dict[str, str]:
    """Country QID -> one of the six continent names, via P30 (first match wins)."""
    values = " ".join(f"wd:{q}" for q in country_qids)
    rows = sparql(f"""
SELECT ?country ?label WHERE {{
  VALUES ?country {{ {values} }}
  ?country wdt:P30 ?continent .
  ?continent rdfs:label ?label FILTER(LANG(?label) = "en")
}}""")
    out: dict[str, str] = {}
    for r in rows:
        name = _continent_name(value_of(r, "label") or "")
        if name:
            out.setdefault(qid_from_uri(value_of(r, "country")), name)
    return out


def area_km2_of(info: dict) -> str:
    """The largest non-deprecated P2046 area in km2 as text, or "" when there is none.

    `info` is a verify_places.query_core record, whose `areas` are square metres.
    Not rank-aware; scripts/04_fix_areas.py refines it.
    """
    areas = info.get("areas") or ()
    return vp.format_number(max(areas) / M2_PER_KM2) if areas else ""


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
            "area_km2": row.get("area_km2") or area_km2_of(info),
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
    """Read a CSV into stripped-string dicts (BOM tolerated)."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()}
                for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write place rows atomically: COLUMNS first, then any extra input columns."""
    columns = COLUMNS + [c for c in rows[0] if c not in COLUMNS] if rows else COLUMNS
    write_csv_atomic(path, columns,
                     ({c: row.get(c, "") for c in columns} for row in rows),
                     encoding="utf-8-sig")


def clue_counts(path: Path) -> Counter:
    """Clues collected per place_qid in a 02_clues.jsonl file."""
    return Counter(r["place_qid"] for r in read_jsonl(path))


def resolve_out_dir(path: Path) -> Path:
    """Resolve --out-dir against this script's folder, never the current directory."""
    return (path if path.is_absolute() else HERE / path).resolve()


def git_hash() -> str | None:
    """The repository's HEAD commit, or None when git or a repository is unavailable."""
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                                capture_output=True, text=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def write_manifest(out: Path, args: argparse.Namespace, status: str,
                   started: datetime.datetime) -> None:
    """Record how this build was made: date, seed, arguments, git hash."""
    now = datetime.datetime.now(datetime.timezone.utc)
    manifest = {
        "status": status,
        "started_utc": started.isoformat(timespec="seconds"),
        "written_utc": now.isoformat(timespec="seconds"),
        "seed": DEFAULT_SEED,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "out_dir": str(out),
        "git_hash": git_hash(),
        "python": sys.version.split()[0],
    }
    write_text_atomic(out / MANIFEST_NAME, json.dumps(manifest, indent=2) + "\n")


def run(script: str, *args: Path | str) -> None:
    """Run one pipeline script in a subprocess; stop the build if it fails."""
    cmd = [sys.executable, str(SCRIPTS / script), *map(str, args)]
    print(f"\n=== {script} " + "=" * max(0, RULE_WIDTH - len(script)), flush=True)
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

def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("places", type=Path, help="CSV with at least a wikidata_id column")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="folder for every output, e.g. data/my_run (a relative path "
                             "is resolved against generation/places/, not the current "
                             "directory)")
    parser.add_argument("--per-continent", type=int, default=None,
                        help="keep at most this many surviving places per continent")
    parser.add_argument("--force", action="store_true",
                        help="allow writing into a folder that already has outputs")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build a place set into --out-dir; see the module docstring."""
    utf8_stdout()
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = build_parser().parse_args(argv)

    out = resolve_out_dir(args.out_dir)
    if out == (HERE / "data").resolve():
        raise SystemExit("refusing to write into data/ itself - it holds the current "
                         "planner inputs. Pick a subfolder, e.g. data/my_run")
    if out.exists() and any(out.iterdir()) and not args.force:
        raise SystemExit(f"{out} is not empty; pass --force to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    started = datetime.datetime.now(datetime.timezone.utc)
    write_manifest(out, args, "started", started)

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
    counts = clue_counts(clues)
    for place in survivors:
        place["clue_count"] = counts.get(place["wikidata_id"], 0)
    final_csv = out / "01_places_final.csv"
    write_csv(final_csv, survivors)
    final_sets = out / "05_clue_sets.jsonl"
    write_jsonl_atomic(final_sets, (r for r in sets if r["place_qid"] in keep))

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
    write_manifest(out, args, "complete", started)
    return 0


if __name__ == "__main__":
    sys.exit(main())
