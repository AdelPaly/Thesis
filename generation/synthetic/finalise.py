#!/usr/bin/env python3
"""
Turn the generated posts into a dataset the evaluation pipeline can read.

    python finalise.py

Reads output/item_plan.jsonl and output/raw_generations.jsonl, takes the
latest attempt for every planned item, checks it, and writes:

    output/synthetic_dataset.csv   the posts that passed, in the same columns
                                   as the datasets in ../../src/ (text, qid,
                                   location_name, country, continent,
                                   signal_type, latitude, longitude) plus the
                                   plan's details for analysis
    output/finalise_report.csv     every planned item with its status and,
                                   when dropped, the reason

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
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "places" / "scripts"))

import generate as gen   # noqa: E402  (config, jsonl reading, latest_rows)
import plan              # noqa: E402  (_contains_word: the planner's name test)
import verify_places as vp  # noqa: E402  (SPARQL client)

NO_NAME = {"A1", "A3", "NOISE_A1"}
AMBIGUOUS = {"B", "AB", "NOISE_B"}

DATASET_COLUMNS = ["text", "qid", "location_name", "country", "continent",
                   "signal_type", "latitude", "longitude",
                   "signal", "surface_form", "clue_qids", "clue_labels",
                   "noise_entity", "register", "length_band", "item_id", "model"]
REPORT_COLUMNS = ["item_id", "condition", "place_qid", "place_name", "status",
                  "reason", "text"]


def mentions(text: str, label: str) -> bool:
    """A clue counts as mentioned if its label, or the label without a
    trailing "(film)"-style qualifier, appears as a whole word."""
    if plan._contains_word(text, label):
        return True
    bare = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
    return bare != label and plan._contains_word(text, bare)


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

    if condition in NO_NAME:
        for form in [place, *names["forms"]]:
            if plan._contains_word(text, form):
                return "names_place", f'wrote "{form}"'
        for form in names["country"]:
            if plan._contains_word(text, form):
                return "names_place", f'wrote the country or demonym "{form}"'
        for form in names["regions"]:
            if plan._contains_word(text, form):
                return "names_place", f'wrote the region "{form}"'

    elif condition in AMBIGUOUS:
        if not surface or not plan._contains_word(text, surface):
            return "wrong_toponym", f'planned name "{surface}" is missing'
        for form in [place, *names["forms"]]:
            # Another name of the place that is not just part of the planned one.
            if not plan._contains_word(surface, form) and plan._contains_word(text, form):
                return "wrong_toponym", f'also wrote "{form}"'
        for form in names["country"]:
            if plan._contains_word(text, form):
                return "wrong_toponym", f'also wrote the country or demonym "{form}"'
        for form in names["regions"]:
            if not plan._contains_word(surface, form) and plan._contains_word(text, form):
                return "wrong_toponym", f'also wrote the region "{form}"'

    elif condition == "CEILING":
        if not (plan._contains_word(text, surface or place)
                or plan._contains_word(text, place)):
            return "missing_name", f'"{surface or place}" is missing'

    for clue in item.get("clues") or []:
        if not mentions(text, clue["clue_label"]):
            return "clue_missing", f'"{clue["clue_label"]}" is missing'
    noise = item.get("noise_entity")
    if noise and not mentions(text, noise["clue_label"]):
        return "noise_missing", f'"{noise["clue_label"]}" is missing'
    return None


def coordinates(qids: list[str]) -> dict[str, tuple[float, float]]:
    """P625 (truthy) per QID, lowest-sorted value when several are best rank."""
    found: dict[str, list[tuple[float, float]]] = {}
    for i in range(0, len(qids), 50):
        values = " ".join(f"wd:{q}" for q in qids[i:i + 50])
        for row in vp.sparql(f"SELECT ?item ?coord WHERE {{ VALUES ?item {{ {values} }} "
                             f"?item wdt:P625 ?coord }}"):
            match = re.match(r"Point\(([-\d.eE]+) ([-\d.eE]+)\)", vp.value_of(row, "coord") or "")
            if match:
                lon, lat = float(match.group(1)), float(match.group(2))
                found.setdefault(vp.qid_from_uri(vp.value_of(row, "item")), []).append((lat, lon))
    return {q: sorted(v)[0] for q, v in found.items()}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    args = parser.parse_args(argv)

    config = gen.load_config(args.config)
    paths = config["paths"]
    items = gen.read_jsonl(Path(paths["item_plan"]))
    if not items:
        raise SystemExit("no plan - run plan.py first")
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))
    aliases = load_aliases(Path(paths["place_aliases"]))

    report, kept = [], []
    for item in items:
        row = latest.get(item["item_id"])
        text = ((row or {}).get("post_text") or "").strip()
        if row is None:
            verdict = ("not_generated", "")
        elif row.get("error"):
            verdict = ("error", str(row["error"])[:200])
        elif not text:
            verdict = ("empty", "")
        else:
            names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
            verdict = check(item, text, names)
        report.append({"item_id": item["item_id"], "condition": item["condition"],
                       "place_qid": item["place_qid"], "place_name": item["place_name"],
                       "status": verdict[0] if verdict else "kept",
                       "reason": verdict[1] if verdict else "", "text": text})
        if verdict is None:
            kept.append((item, row, text))

    coords = coordinates(sorted({item["place_qid"] for item, _, _ in kept}))
    dataset = []
    for item, row, text in kept:
        lat, lon = coords.get(item["place_qid"], (None, None))
        dataset.append({
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
        })

    for path, columns, rows in ((Path(paths["dataset_csv"]), DATASET_COLUMNS, dataset),
                                (Path(paths["finalise_report"]), REPORT_COLUMNS, report)):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    print(f"planned items : {len(items)}")
    for status, n in Counter(r["status"] for r in report).most_common():
        print(f"  {status:<14} {n:>5}")
    generated = [r for r in report if r["status"] not in ("not_generated", "error")]
    if generated:
        print(f"\nof {len(generated)} generated posts, {len(dataset)} kept "
              f"({len(dataset) / len(generated):.0%})")
    print("\nkept, by condition:")
    for condition, n in sorted(Counter(d["signal_type"] for d in dataset).items()):
        print(f"  {condition:<10} {n:>5}")
    missing = sum(1 for d in dataset if d["latitude"] is None)
    if missing:
        print(f"\n! {missing} kept posts have no gold coordinate")
    print(f"\nwrote {paths['dataset_csv']}")
    print(f"wrote {paths['finalise_report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
