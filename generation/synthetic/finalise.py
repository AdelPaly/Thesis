#!/usr/bin/env python3
"""
Turn the generated posts into a dataset the evaluation pipeline can read.

    python finalise.py

Reads output/item_plan.jsonl and output/raw_generations.jsonl, takes the
latest attempt for every planned item, checks it, balances the conditions,
and writes:

    output/finalise_report.csv     every planned item with its status and,
                                   when dropped, the reason. Written BEFORE
                                   the Wikidata lookup, so a network failure
                                   cannot lose it, and rewritten once the
                                   coordinate check has run.
    output/synthetic_dataset.csv   the posts that passed, in the same columns
                                   as the datasets in ../../src/ (text, qid,
                                   location_name, country, continent,
                                   signal_type, latitude, longitude) plus the
                                   plan's details for analysis. Not written if
                                   the Wikidata lookup fails.

A post is dropped when:

    error           generation failed (re-run generate.py --retry-errors)
    not_generated   the item has no row yet (generate.py has not reached it)
    empty           the model returned no text
    names_place     a no-place-name condition (A1, A3, NOISE_A1) wrote the
                    place, another of its names, its country or a demonym, or
                    a region it sits in
    wrong_toponym   B, AB or NOISE_B did not write the planned ambiguous name,
                    or also wrote the country, a region or another name of
                    the place that gives it away
    missing_name    CEILING did not write the place name
    clue_missing    a clue entity the plan asked for is not in the text
    noise_missing   a NOISE item does not contain its distractor
    no_coordinates  the place's QID has no P625 coordinate on Wikidata, so the
                    row could not be scored
    repeated_clue   the plan lists the same clue entity twice, so the item
                    carries fewer distinct clues than its condition promises
                    (a planning-stage defect; checked before anything else)
    balanced_out    the post passed every check but was not selected when the
                    conditions were cut to the same size (see below)

Balancing: the plan is deliberately uneven (A1 and A3 get two posts per place,
the others one) and oversized, so after the checks the conditions hold
different numbers of posts. Pooled and per-condition numbers are easier to
compare when every condition has the same n, so each condition is cut to N
posts, N being the size of the smallest condition after the checks (or
--per-condition). Within a condition the posts are spread as evenly as the
supply allows across continents, and which posts are chosen is decided only by
--seed, so the same inputs always give the same dataset. Every post cut this
way stays in finalise_report.csv as balanced_out. --no-balance keeps them all.

Name matching is whole-word and ignores case and accents, the same test the
planner uses to keep leaking clues out.

signal_type is the plan's condition (A1, A3, B, AB, CEILING, NOISE_A1,
NOISE_B), so `scripts/score.py --by condition` reports each one separately.
Gold coordinates come from Wikidata P625 (truthy), the same property the
other datasets were built from.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PLACES_SCRIPTS = HERE.parent / "places" / "scripts"

import generate as gen   # noqa: E402  (config, jsonl reading, latest_rows)
import plan              # noqa: E402  (contains_word, atomic_write_text, ...)

NO_NAME = {"A1", "A3", "NOISE_A1"}
AMBIGUOUS = {"B", "AB", "NOISE_B"}

BALANCE_SEED = 20261004    # default --seed: which posts survive balancing
SPARQL_BATCH = 50          # QIDs per VALUES clause
ERROR_REASON_CHARS = 200   # how much of a generation error goes in the report

DATASET_COLUMNS = ["text", "qid", "location_name", "country", "continent",
                   "signal_type", "latitude", "longitude",
                   "signal", "surface_form", "clue_qids", "clue_labels",
                   "noise_entity", "register", "length_band", "item_id", "model"]
REPORT_COLUMNS = ["item_id", "condition", "place_qid", "place_name", "status",
                  "reason", "text"]


def mentions(text: str, label: str) -> bool:
    """A clue counts as mentioned if its label, or the label without a
    trailing "(film)"-style qualifier, appears as a whole word."""
    if plan.contains_word(text, label):
        return True
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return bare != label and plan.contains_word(text, bare)


def load_aliases(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    for record in gen.read_jsonl(path):
        out[record["place_qid"]] = {
            "forms": [f["text"] for f in record.get("surface_forms", [])
                      if f.get("source") != "derived" and f.get("kind") != "unmatched"],
            "country": [f["text"] for f in record.get("country_forms", [])],
            "regions": [f["text"] for f in record.get("region_forms", [])],
        }
    return out


def check(item: dict[str, Any], text: str, names: dict[str, Any]) -> tuple[str, str] | None:
    """(status, reason) when the post must be dropped, None when it passes."""
    condition = item["condition"]
    place = item["place_name"]
    surface = (item.get("surface_form") or {}).get("text")
    contains = plan.contains_word

    if condition in NO_NAME:
        for form in [place, *names["forms"]]:
            if contains(text, form):
                return "names_place", f'wrote "{form}"'
        for form in names["country"]:
            if contains(text, form):
                return "names_place", f'wrote the country or demonym "{form}"'
        for form in names["regions"]:
            if contains(text, form):
                return "names_place", f'wrote the region "{form}"'

    elif condition in AMBIGUOUS:
        if not surface or not contains(text, surface):
            return "wrong_toponym", f'planned name "{surface}" is missing'
        for form in [place, *names["forms"]]:
            # Another name of the place that is not just part of the planned one.
            if not contains(surface, form) and contains(text, form):
                return "wrong_toponym", f'also wrote "{form}"'
        for form in names["country"]:
            if contains(text, form):
                return "wrong_toponym", f'also wrote the country or demonym "{form}"'
        for form in names["regions"]:
            if not contains(surface, form) and contains(text, form):
                return "wrong_toponym", f'also wrote the region "{form}"'

    elif condition == "CEILING":
        if not (contains(text, surface or place) or contains(text, place)):
            return "missing_name", f'"{surface or place}" is missing'

    for clue in item.get("clues") or []:
        if not mentions(text, clue["clue_label"]):
            return "clue_missing", f'"{clue["clue_label"]}" is missing'
    noise = item.get("noise_entity")
    if noise and not mentions(text, noise["clue_label"]):
        return "noise_missing", f'"{noise["clue_label"]}" is missing'
    return None


def repeated_clue(item: dict[str, Any]) -> str | None:
    """The first clue QID the plan lists more than once for this item, if any."""
    seen: set[str] = set()
    for clue in item.get("clues") or []:
        qid = clue["clue_qid"]
        if qid in seen:
            return qid
        seen.add(qid)
    return None


def verdict_for(item: dict[str, Any], row: dict[str, Any] | None,
                aliases: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    """The drop (status, reason) for one planned item, or None when it is kept."""
    duplicate = repeated_clue(item)
    if duplicate:
        return "repeated_clue", f"clue {duplicate} appears more than once in the plan"
    text = ((row or {}).get("post_text") or "").strip()
    if row is None:
        return "not_generated", ""
    if row.get("error"):
        return "error", str(row["error"])[:ERROR_REASON_CHARS]
    if not text:
        return "empty", ""
    names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
    return check(item, text, names)


def coordinates(qids: list[str]) -> dict[str, tuple[float, float]]:
    """
    P625 (truthy) per QID, lowest-sorted value when several are best rank.
    Makes live SPARQL calls; verify_places is imported here so the module
    (and check()) can be used without it.
    """
    if str(PLACES_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(PLACES_SCRIPTS))
    import verify_places as vp  # noqa: E402  (SPARQL client)

    found: dict[str, list[tuple[float, float]]] = {}
    for i in range(0, len(qids), SPARQL_BATCH):
        values = " ".join(f"wd:{q}" for q in qids[i:i + SPARQL_BATCH])
        for row in vp.sparql(f"SELECT ?item ?coord WHERE {{ VALUES ?item {{ {values} }} "
                             f"?item wdt:P625 ?coord }}"):
            match = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", vp.value_of(row, "coord") or "")
            if match:
                lon, lat = float(match.group(1)), float(match.group(2))
                found.setdefault(vp.qid_from_uri(vp.value_of(row, "item")), []).append((lat, lon))
    return {q: sorted(v)[0] for q, v in found.items()}


def split_by_coordinates(
    kept: list[tuple[dict[str, Any], dict[str, Any], str]],
    coords: dict[str, tuple[float, float]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], str]], list[dict[str, Any]]]:
    """(kept with coordinates, items dropped as no_coordinates)."""
    with_coords, without = [], []
    for entry in kept:
        (with_coords if entry[0]["place_qid"] in coords else without).append(entry)
    return with_coords, [item for item, _, _ in without]


def apply_no_coordinates(report: list[dict[str, Any]], dropped: list[dict[str, Any]]) -> None:
    """Mark the dropped items in the report, in place."""
    ids = {item["item_id"] for item in dropped}
    for entry in report:
        if entry["item_id"] in ids:
            entry["status"] = "no_coordinates"
            entry["reason"] = f'{entry["place_qid"]} has no P625 coordinate on Wikidata'


Entry = tuple[dict[str, Any], dict[str, Any], str]   # (plan item, raw row, text)


def _rank(seed: int, item_id: str) -> str:
    """Seeded, stable sort key: the same seed and ids always give the same order."""
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def allocate_even(available: dict[str, int], n: int) -> dict[str, int]:
    """
    Split n picks across groups as evenly as the supply allows: every group gets
    the same share until one runs out, and its unused share goes to the rest.
    The remainder after an even split goes to groups in sorted-name order.
    """
    take = dict.fromkeys(available, 0)
    remaining = min(n, sum(available.values()))
    open_groups = sorted(g for g, k in available.items() if k > 0)
    while remaining > 0 and open_groups:
        share, extra = divmod(remaining, len(open_groups))
        progressed = 0
        for index, group in enumerate(open_groups):
            want = share + (1 if index < extra else 0)
            got = min(want, available[group] - take[group])
            take[group] += got
            progressed += got
        remaining -= progressed
        open_groups = [g for g in open_groups if take[g] < available[g]]
        if not progressed:
            break
    return take


def balance(kept: list[Entry], seed: int,
            per_condition: int | None = None) -> tuple[list[Entry], list[Entry], int]:
    """
    Cut every condition to the same number of posts.

    N is the smallest condition's size unless per_condition is given (it may not
    exceed that size). Within a condition the picks are spread evenly over
    continents (allocate_even) and ordered by a seeded hash, so the choice
    depends on nothing but the seed. Returns (selected, excluded, N); selected
    keeps the order of `kept`.
    """
    by_condition: dict[str, list[Entry]] = {}
    for entry in kept:
        by_condition.setdefault(entry[0]["condition"], []).append(entry)
    smallest = min(len(v) for v in by_condition.values())
    n = smallest if per_condition is None else per_condition
    if n > smallest:
        raise ValueError(f"--per-condition {n} exceeds the smallest condition "
                         f"({smallest} posts)")
    chosen: set[str] = set()
    for entries in by_condition.values():
        by_continent: dict[str, list[Entry]] = {}
        for entry in entries:
            by_continent.setdefault(entry[0]["place_continent"], []).append(entry)
        take = allocate_even({c: len(v) for c, v in by_continent.items()}, n)
        for continent, group in by_continent.items():
            group.sort(key=lambda e: _rank(seed, e[0]["item_id"]))
            chosen.update(e[0]["item_id"] for e in group[:take[continent]])
    selected = [e for e in kept if e[0]["item_id"] in chosen]
    excluded = [e for e in kept if e[0]["item_id"] not in chosen]
    return selected, excluded, n


def apply_balanced_out(report: list[dict[str, Any]], excluded: list[Entry],
                       n: int, seed: int) -> None:
    """Mark the posts cut by balance() in the report, in place."""
    ids = {item["item_id"] for item, _, _ in excluded}
    for entry in report:
        if entry["item_id"] in ids:
            entry["status"] = "balanced_out"
            entry["reason"] = f"not selected: {n} posts per condition, seed {seed}"


def csv_text(columns: list[str], rows: list[dict[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\r\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Atomic CSV write (tmp file + os.replace); newline handling as csv's default."""
    plan.atomic_write_text(path, csv_text(columns, rows), newline="")


def dataset_row(item: dict[str, Any], row: dict[str, Any], text: str,
                coord: tuple[float, float]) -> dict[str, Any]:
    lat, lon = coord
    return {
        "text": text,
        "qid": item["place_qid"],
        "location_name": item["place_name"],
        "country": item["place_country"],
        "continent": item["place_continent"],
        "signal_type": item["condition"],
        "latitude": lat,
        "longitude": lon,
        "signal": item.get("signal"),
        "surface_form": (item.get("surface_form") or {}).get("text") or "",
        "clue_qids": ";".join(c["clue_qid"] for c in item.get("clues") or []),
        "clue_labels": ";".join(c["clue_label"] for c in item.get("clues") or []),
        "noise_entity": (item.get("noise_entity") or {}).get("clue_label") or "",
        "register": item["register"],
        "length_band": item["length_band"],
        "item_id": item["item_id"],
        "model": row.get("model"),
    }


def print_summary(items: list[dict[str, Any]], report: list[dict[str, Any]],
                  dataset: list[dict[str, Any]]) -> None:
    print(f"planned items : {len(items)}")
    for status, n in Counter(r["status"] for r in report).most_common():
        print(f"  {status:<14} {n:>5}")
    generated = [r for r in report if r["text"] or r["status"] == "empty"]
    if generated:
        print(f"\nof {len(generated)} generated posts, {len(dataset)} kept "
              f"({len(dataset) / len(generated):.0%})")
    print("\nkept, by condition:")
    for condition, n in sorted(Counter(d["signal_type"] for d in dataset).items()):
        print(f"  {condition:<10} {n:>5}")


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--no-balance", action="store_true",
                        help="keep every post that passes the checks instead of "
                             "cutting each condition to the same size")
    parser.add_argument("--per-condition", type=int, default=None, metavar="N",
                        help="posts to keep per condition (default: the size of "
                             "the smallest condition)")
    parser.add_argument("--seed", type=int, default=BALANCE_SEED,
                        help="decides which posts survive balancing "
                             f"(default {BALANCE_SEED})")
    args = parser.parse_args(argv)

    config = gen.load_config(args.config)
    paths = config["paths"]
    items = gen.read_jsonl(Path(paths["item_plan"]))
    if not items:
        raise SystemExit("no plan - run plan.py first")
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))
    aliases = load_aliases(Path(paths["place_aliases"]))
    report_path = Path(paths["finalise_report"])
    dataset_path = Path(paths["dataset_csv"])

    report, kept = [], []
    for item in items:
        row = latest.get(item["item_id"])
        verdict = verdict_for(item, row, aliases)
        text = ((row or {}).get("post_text") or "").strip()
        report.append({"item_id": item["item_id"], "condition": item["condition"],
                       "place_qid": item["place_qid"], "place_name": item["place_name"],
                       "status": verdict[0] if verdict else "kept",
                       "reason": verdict[1] if verdict else "", "text": text})
        if verdict is None:
            kept.append((item, row, text))

    # The report needs no network, so it is written first: a Wikidata failure
    # below cannot lose the per-item verdicts.
    write_csv(report_path, REPORT_COLUMNS, report)
    print(f"wrote {report_path} (before the coordinate lookup)")

    try:
        coords = coordinates(sorted({item["place_qid"] for item, _, _ in kept}))
    except Exception as exc:  # network, SPARQL or import failure: report stays
        print(f"\n! coordinate lookup failed ({type(exc).__name__}: {exc}).\n"
              f"  {report_path} is written; {dataset_path} was NOT written. "
              f"Re-run when Wikidata is reachable.", file=sys.stderr)
        return 1

    kept, no_coords = split_by_coordinates(kept, coords)
    if no_coords:
        apply_no_coordinates(report, no_coords)
        write_csv(report_path, REPORT_COLUMNS, report)

    balanced_n = None
    if not args.no_balance and kept:
        planned = {item["condition"] for item in items}
        missing = sorted(planned - {item["condition"] for item, _, _ in kept})
        if missing:
            print(f"\n! cannot balance: no post survived for {', '.join(missing)}. "
                  f"Generate more, or re-run with --no-balance.", file=sys.stderr)
            return 1
        try:
            kept, cut, balanced_n = balance(kept, args.seed, args.per_condition)
        except ValueError as exc:
            print(f"\n! {exc}", file=sys.stderr)
            return 1
        apply_balanced_out(report, cut, balanced_n, args.seed)
        write_csv(report_path, REPORT_COLUMNS, report)

    dataset = [dataset_row(item, row, text, coords[item["place_qid"]])
               for item, row, text in kept]
    write_csv(dataset_path, DATASET_COLUMNS, dataset)

    print_summary(items, report, dataset)
    if balanced_n is not None:
        print(f"\nbalanced to {balanced_n} per condition (seed {args.seed}); "
              f"kept, by continent:")
        for continent, n in sorted(Counter(d["continent"] for d in dataset).items()):
            print(f"  {continent:<15} {n:>5}")
    if no_coords:
        print(f"\n! {len(no_coords)} posts dropped as no_coordinates "
              f"(QIDs: {', '.join(sorted({i['place_qid'] for i in no_coords}))})")
    print(f"\nwrote {dataset_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
