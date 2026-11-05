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
    wrong_toponym   B, NOISE_B or EXPLICIT also wrote the country, a region or
                    another name of the place that gives it away, or B or
                    NOISE_B did not write the planned ambiguous name
    missing_name    EXPLICIT did not write the place name
    clue_missing    a clue entity the plan asked for is not in the text, by its
                    label, a Wikidata alias or, for a person, the surname
    noise_missing   a NOISE item does not contain its distractor
    too_many_anchors  the plan gives the post more than MAX_ANCHORS (2) clues
    anchor_invalid  validate_anchors.py rejected a planned clue or distractor on
                    Wikidata (filed under another place too, dissolved, a dated
                    past event, a religious, military or atrocity entity, or a
                    name that contains the place's)
    format          a line break, an emoji or a hashtag
    unplanned_entity  a capitalised name that belongs to no planned name (not
                    counting the first word of a sentence)
    (2026-11-02, user, after smoke13; the judge's own reasons are in judge.py)
    no_coordinates  the place's QID has no P625 coordinate on Wikidata, so the
                    row could not be scored
    repeated_clue   the plan lists the same clue entity twice, so the item
                    carries fewer distinct clues than its condition promises
                    (a planning-stage defect; checked before anything else)
    no_explicit     an A1, NOISE_A1, B or NOISE_B post whose place has no kept
                    EXPLICIT post, so it has no reference to be read against
                    (2026-11-04, user: EXPLICIT covers every A and B place).
                    Applied before balancing, so balancing fills from the rest.
    balanced_out    the post passed every check but was not selected when the
                    conditions were cut to the same size (see below)

Conditions listed in finalise.unbalanced in config.yaml (EXPLICIT) are not cut:
every post that passes the checks is kept, so the condition covers as many
places as it can.

Balancing: the plan holds a reserve per condition (pool "reserve", generated
only to replace failed posts), so after the checks the conditions hold
different numbers of posts. Pooled and per-condition numbers are easier to
compare when every condition has the same n, so each condition is cut to N
posts: finalise.per_condition in config.yaml (150), or --per-condition, or the
size of the smallest condition after the checks when neither is set. Within a condition the posts are spread as evenly as the
supply allows across continents, and which posts are chosen is decided only by
--seed, so the same inputs always give the same dataset. Every post cut this
way stays in finalise_report.csv as balanced_out. --no-balance keeps them all.

Name matching is whole-word and ignores case and accents, the same test the
planner uses to keep leaking clues out.

signal_type is the plan's condition (A1, A3, B, EXPLICIT, NOISE_A1,
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
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent

import generate as gen   # noqa: E402  (config, jsonl reading, latest_rows)
import plan              # noqa: E402  (contains_word, atomic_write_text, ...)

NO_NAME = {"A1", "A3", "NOISE_A1"}
AMBIGUOUS = {"B", "NOISE_B"}

BALANCE_SEED = 20261004    # default --seed: which posts survive balancing
ERROR_REASON_CHARS = 200   # how much of a generation error goes in the report

# Prominence bins, for splitting results; they do not change which posts are
# kept. All three go by Wikipedia counts.
# place_prominence (every condition), by the place's own sitelinks (language
# versions): under 30, 30 to 99, 100 or more (100+ is the "famous" tier of the
# place selection). Over the 899 places that is 456 / 278 / 165.
PLACE_PROMINENCE_BINS = [(100, "high_100_plus"), (30, "mid_30_to_99"), (0, "low_under_30")]
# rival_prominence (B, NOISE_B, and EXPLICIT when other places share the name;
# name_ambiguous says which): (place sitelinks + 1) / (sitelinks + 1 of the
# best-known other place with the same name, the "rival"). place_leads_2x at a
# ratio of 2 or more, rival_leads_2x at 0.5 or less, and close in between,
# where the counts are too near to call either one more prominent.
RIVAL_RATIO = 2.0
RIVAL_PROMINENCE_BINS = (f"place_leads_{RIVAL_RATIO:g}x", "close",
                         f"rival_leads_{RIVAL_RATIO:g}x")
# clue_prominence (every condition with a clue), by the sitelinks of the post's
# best-known clue (for A3, one recognisable clue is enough to start from):
# under 10, 10 to 49, 50 or more. Clues need >= 3 sitelinks (reform_places.py
# CLUE_MIN_SL), so the low bin is in practice 3 to 9; the 10 and 50 cuts sit
# near the plan's median (14) and 75th percentile (36), so no bin is tiny.
CLUE_PROMINENCE_BINS = [(50, "high_50_plus"), (10, "mid_10_to_49"), (0, "low_under_10")]

DATASET_COLUMNS = ["text", "qid", "location_name", "country", "continent",
                   "signal_type", "latitude", "longitude",
                   "signal", "surface_form", "clue_qids", "clue_labels",
                   "clue_forms", "noise_entity", "register", "target_len", "len",
                   "place_sitelinks", "rival_qid", "rival_sitelinks",
                   "name_ambiguous", "place_is_most_famous", "place_prominence",
                   "rival_prominence", "clue_prominence", "topic", "scale",
                   "valence", "angle", "role", "tries", "clue_swapped", "swapped_from",
                   "split", "pool", "item_id", "model"]
REPORT_COLUMNS = ["item_id", "condition", "place_qid", "place_name", "status",
                  "reason", "text"]

# Trailing name parts that are not the surname ("Louis Jones Jr.", "Henry III").
NAME_SUFFIXES = {"jr", "jr.", "sr", "sr.", "junior", "senior", "ii", "iii", "iv", "v"}


def _bare(label: str) -> str:
    """The label without a trailing "(film)"-style qualifier."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()


def surname(label: str) -> str:
    """The last name part of a person's label, numbering and Jr./Sr. dropped:
    "Percival Fredrickstein von Musel Klossowski de Rolo III" -> "Rolo"."""
    words = [w.strip(",") for w in _bare(label).split()]
    while len(words) > 1 and words[-1].casefold() in NAME_SUFFIXES:
        words.pop()
    return words[-1] if words else ""


def mention_form(text: str, clue: dict[str, Any] | str) -> str | None:
    """
    The form in which the post mentions the clue, or None if it does not.

    The prompt lets the model write an entity the way people would: a person
    by the shorter name people use ("Percy Rolo" for Percival Fredrickstein
    von Musel Klossowski de Rolo III), anything else by its usual name. So a
    mention is the label, the label without a "(film)" qualifier, any
    Wikidata alias, or, for a person (is_human from the planner), the surname
    as a whole word. Nicknames that are not aliases still need the surname.
    """
    if isinstance(clue, str):
        clue = {"clue_label": clue}
    label = clue["clue_label"]
    for form in [label, _bare(label), *(clue.get("aliases") or [])]:
        if form and plan.contains_word(text, form):
            return form
    if clue.get("is_human"):
        last = surname(label)
        if len(plan.fold(last)) >= 2 and plan.contains_word(text, last):
            return last
    return None


def mentions(text: str, clue: dict[str, Any] | str) -> bool:
    """True when the post mentions the clue in any accepted form (mention_form)."""
    return mention_form(text, clue) is not None


def load_aliases(path: Path) -> dict[str, dict[str, Any]]:
    """Place QID -> its names, country names and demonyms, and regions, from
    place_inputs.jsonl (../place_inputs.py)."""
    return {record["place_qid"]: {"forms": record.get("name_forms") or [],
                                  "country": record.get("country_forms") or [],
                                  "regions": record.get("region_forms") or []}
            for record in gen.read_jsonl(path)}


def _bin(value: int, bins: list[tuple[int, str]]) -> str:
    """The label of the first (threshold, label) that value reaches."""
    return next(label for threshold, label in bins if value >= threshold)


def place_prominence(sitelinks: int | None) -> str:
    return "" if sitelinks is None else _bin(sitelinks, PLACE_PROMINENCE_BINS)


def rival_prominence(sitelinks: int | None, rival: dict[str, Any]) -> str:
    """The RIVAL_PROMINENCE_BINS label of the place's sitelinks against the
    best-known place with the same name (B only). Both counts get +1 so a rival
    with no sitelinks still compares."""
    if sitelinks is None or "sitelinks" not in rival:
        return ""
    ratio = (sitelinks + 1) / (rival["sitelinks"] + 1)
    place_leads, close, rival_leads = RIVAL_PROMINENCE_BINS
    if ratio >= RIVAL_RATIO:
        return place_leads
    return rival_leads if ratio <= 1 / RIVAL_RATIO else close


def name_ambiguous(surface: dict[str, Any]) -> str:
    """"true" when other places share the name the post uses, "false" when none
    does, "" for a post that names no place. B and NOISE_B are always "true";
    EXPLICIT is either, and then carries rival_prominence like B."""
    flag = surface.get("name_ambiguous")
    return "" if flag is None else str(bool(flag)).lower()


def clue_prominence(clues: list[dict[str, Any]]) -> str:
    """The bin of the best-known clue by sitelinks; "" for posts without clues."""
    known = [int(c["sitelinks"]) for c in clues if c.get("sitelinks") is not None]
    return _bin(max(known), CLUE_PROMINENCE_BINS) if known else ""


def environment_pattern(terms: list[str] | None) -> re.Pattern | None:
    """The finalise.environment_keywords terms as one pattern, each matched at the
    start of a word; None when there are no terms (the check is then off)."""
    terms = [t for t in (terms or []) if t]
    if not terms:
        return None
    return re.compile(r"\b(?:" + "|".join(f"(?:{t})" for t in terms) + ")", re.I)


def word_limits(finalise_cfg: dict[str, Any] | None) -> dict[str, tuple[int, int]] | None:
    """{condition: (min, max) words}, "default" from finalise.min_words and
    max_words, others from finalise.words_by_condition; None when either
    default is unset. plan.py draws its target lengths from the same ranges."""
    cfg = finalise_cfg or {}
    if cfg.get("min_words") is None or cfg.get("max_words") is None:
        return None
    return {"default": (int(cfg["min_words"]), int(cfg["max_words"])),
            **{c: (int(lo), int(hi)) for c, (lo, hi) in (cfg.get("words_by_condition") or {}).items()}}


MAX_ANCHORS = 2          # 2026-11-02 (user): at most two planned anchors per post
EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF️]")
HASHTAG = re.compile(r"(?<!\w)#\w")
CAPITALISED = re.compile(r"[A-Z][\w'’./&-]*")
# Capitalised words that are not names: the pronoun, days, months, family
# words and common abbreviations. Anything else capitalised that is not part
# of a planned name, and not the first word of a sentence, is an unplanned
# entity (2026-11-02, user: "Chris Smith got it moving" in an EXPLICIT post
# that was given no person).
NOT_NAMES = {plan.fold(w) for w in (
    "I I'm I'd I've I'll Im Ive OK Ok TV EV EVs PM AM CO2 LOL Lol OMG Omg Mum Mom Dad God Sir "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February March April May June July "
    "August September October November December Christmas Easter").split()}
ANCHOR_CHECKS = HERE / "output" / "anchor_checks.jsonl"
_anchor_verdicts: dict[tuple[str, str, bool], tuple[str, str]] | None = None


def anchor_verdicts() -> dict[tuple[str, str, bool], tuple[str, str]]:
    """(place QID, anchor QID, is distractor) -> (status, reason) for every anchor
    validate_anchors.py rejected; empty when it has not been run."""
    global _anchor_verdicts
    if _anchor_verdicts is None:
        rows = gen.read_jsonl(ANCHOR_CHECKS) if ANCHOR_CHECKS.exists() else []
        _anchor_verdicts = {(r["place_qid"], r["clue_qid"], bool(r["noise"])): (r["status"], r["reason"])
                            for r in rows if r["status"] != "ok"}
    return _anchor_verdicts


def _name_words(forms: Iterable[str]) -> set[str]:
    return {plan.fold(w).strip(".,'’") for f in forms if f for w in re.split(r"[\s/]+", f)}


def unplanned_names(text: str, allowed_forms: Iterable[str]) -> list[str]:
    """Capitalised words in the post that belong to no planned name, leaving out
    the first word of each sentence and NOT_NAMES."""
    allowed = _name_words(allowed_forms)
    found = []
    for m in CAPITALISED.finditer(text):
        before = text[:m.start()].rstrip(" \"'“‘(")
        if not before or before[-1] in ".!?:;":
            continue                                     # sentence start
        word = plan.fold(re.sub(r"['’]s$", "", m.group())).strip(".,'’")
        if word and word not in allowed and word not in NOT_NAMES and not word.isdigit():
            found.append(m.group())
    return found


def check(item: dict[str, Any], text: str, names: dict[str, Any],
          env_pattern: re.Pattern | None = None,
          limits: dict[str, tuple[int, int]] | None = None) -> tuple[str, str] | None:
    """(status, reason) when the post must be dropped, None when it passes.
    env_pattern, when given, drops a post with an env_theme and none of its terms.
    limits, when given (word_limits), drops a post with fewer or more words
    than its condition's (min, max)."""
    condition = item["condition"]
    place = item["place_name"]
    surface = (item.get("surface_form") or {}).get("text")
    contains = plan.contains_word

    if len(item.get("clues") or []) > MAX_ANCHORS:
        return "too_many_anchors", f'{len(item["clues"])} planned anchors, at most {MAX_ANCHORS}'
    bad = anchor_verdicts()
    for clue in item.get("clues") or []:
        v = bad.get((item.get("place_qid"), clue.get("clue_qid"), False))
        if v:
            return "anchor_invalid", f'"{clue["clue_label"]}": {v[0]} ({v[1]})'
    if item.get("noise_entity"):
        v = bad.get((item.get("place_qid"), item["noise_entity"].get("clue_qid"), True))
        if v:
            return "anchor_invalid", f'distractor "{item["noise_entity"]["clue_label"]}": {v[0]} ({v[1]})'
    if "\n" in text.strip() or EMOJI.search(text) or HASHTAG.search(text):
        what = "a line break" if "\n" in text.strip() else "an emoji" if EMOJI.search(text) else "a hashtag"
        return "format", f"has {what}; one line, no emoji, no hashtags"

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

    elif condition in AMBIGUOUS or condition == "EXPLICIT":
        # The bare planned name and nothing that would add context to it.
        # EXPLICIT's planned name is the place's label.
        status = "missing_name" if condition == "EXPLICIT" else "wrong_toponym"
        surface = surface or (place if condition == "EXPLICIT" else None)
        if not surface or not contains(text, surface):
            return status, f'planned name "{surface}" is missing'
        for form in [place, *names["forms"]]:
            # Another name of the place that is not just part of the planned one.
            if not contains(surface, form) and contains(text, form):
                return "wrong_toponym", f'also wrote "{form}"'
        for form in names["country"]:
            # A country that is part of the name itself ("Guatemala City") is fine.
            if not contains(surface, form) and contains(text, form):
                return "wrong_toponym", f'also wrote the country or demonym "{form}"'
        for form in names["regions"]:
            if not contains(surface, form) and contains(text, form):
                return "wrong_toponym", f'also wrote the region "{form}"'

    for clue in item.get("clues") or []:
        if not mentions(text, clue):
            return "clue_missing", f'"{clue["clue_label"]}" is missing'
    noise = item.get("noise_entity")
    if noise and not mentions(text, noise):
        return "noise_missing", f'"{noise["clue_label"]}" is missing'
    planned = [*(c for clue in (item.get("clues") or []) + ([noise] if noise else [])
                 for c in [clue["clue_label"], _bare(clue["clue_label"]), *(clue.get("aliases") or [])])]
    if surface:
        planned.append(surface)
    if condition == "EXPLICIT":
        planned.append(place)
    extra = unplanned_names(text, planned)
    if extra:
        return "unplanned_entity", "names not on the list: " + ", ".join(f'"{e}"' for e in dict.fromkeys(extra))
    if env_pattern is not None and item.get("env_theme") and not env_pattern.search(text):
        return "no_environment", f'no environmental wording (theme "{item["env_theme"]}")'
    if limits is not None:
        lo, hi = limits.get(condition, limits["default"])
        words = len(text.split())
        if not lo <= words <= hi:
            return "length", f"{words} words, allowed {lo} to {hi}"
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
                aliases: dict[str, dict[str, Any]],
                env_pattern: re.Pattern | None = None,
                limits: dict[str, tuple[int, int]] | None = None) -> tuple[str, str] | None:
    """The drop (status, reason) for one planned item, or None when it is kept.
    Rows from the tries loop (generate.run_item, with "accepted") were already
    judged: a failed one keeps its last try's reason, a kept one is checked by
    the rules again but not by the keyword list, which the judge replaced."""
    duplicate = repeated_clue(item)
    if duplicate:
        return "repeated_clue", f"clue {duplicate} appears more than once in the plan"
    text = ((row or {}).get("post_text") or "").strip()
    if row is None:
        return ("reserve_unused", "") if item.get("pool") == "reserve" else ("not_generated", "")
    if "accepted" in row:
        if not row["accepted"]:
            return row.get("final_status") or "failed", str(row.get("final_reason") or "")[:ERROR_REASON_CHARS]
        env_pattern = None
    if row.get("error"):
        return "error", str(row["error"])[:ERROR_REASON_CHARS]
    if not text:
        return "empty", ""
    names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
    return check(used_item(item, row), text, names, env_pattern, limits)


def used_item(item: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    """The plan item with the clues the post was actually written with: a
    retry may have swapped a clue for a spare (generate.swap_decorative)."""
    if row and row.get("clues_used"):
        return {**item, "clues": row["clues_used"]}
    return item


def coordinates(qids: list[str], places_csv: Path = HERE.parent / "out" / "places.csv"
                ) -> dict[str, tuple[float, float]]:
    """
    (lat, lon) per QID from out/places.csv, which reform_places.py filled from
    Wikidata P625 when it built the place set. No network.
    """
    wanted = set(qids)
    found: dict[str, tuple[float, float]] = {}
    with places_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["i"] in wanted and row["lat"] and row["lon"]:
                found[row["i"]] = (float(row["lat"]), float(row["lon"]))
    return found


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


allocate_even = plan.allocate_even  # one implementation, shared with the planner


PAIRED_WITH_EXPLICIT = {"A1", "NOISE_A1", "B", "NOISE_B"}


def split_by_explicit(kept: list[Entry]) -> tuple[list[Entry], list[Entry]]:
    """(kept, dropped): the PAIRED_WITH_EXPLICIT posts whose place has no kept
    EXPLICIT post are dropped; every other post stays."""
    covered = {item["place_qid"] for item, _, _ in kept if item["condition"] == "EXPLICIT"}
    unpaired = lambda item: item["condition"] in PAIRED_WITH_EXPLICIT and item["place_qid"] not in covered  # noqa: E731
    return [e for e in kept if not unpaired(e[0])], [e for e in kept if unpaired(e[0])]


def apply_no_explicit(report: list[dict[str, Any]], dropped: list[Entry]) -> None:
    """Mark the posts dropped by split_by_explicit() in the report, in place."""
    ids = {item["item_id"] for item, _, _ in dropped}
    for entry in report:
        if entry["item_id"] in ids:
            entry["status"] = "no_explicit"
            entry["reason"] = f'{entry["place_qid"]} has no kept EXPLICIT post'


def balance(kept: list[Entry], seed: int, per_condition: int | None = None,
            unbalanced: Iterable[str] = ()) -> tuple[list[Entry], list[Entry], int]:
    """
    Cut every condition to the same number of posts, except those in
    `unbalanced`, which keep every post that passed the checks.

    N is the smallest balanced condition's size unless per_condition is given
    (it may not exceed that size). Within a condition the picks are spread
    evenly over continents (allocate_even) and ordered by a seeded hash, so the
    choice depends on nothing but the seed. Returns (selected, excluded, N);
    selected keeps the order of `kept`.
    """
    skip = set(unbalanced)
    by_condition: dict[str, list[Entry]] = {}
    for entry in kept:
        by_condition.setdefault(entry[0]["condition"], []).append(entry)
    cut = {c: v for c, v in by_condition.items() if c not in skip}
    if not cut:
        return list(kept), [], 0
    smallest = min(len(v) for v in cut.values())
    n = smallest if per_condition is None else per_condition
    if n > smallest:
        raise ValueError(f"--per-condition {n} exceeds the smallest condition "
                         f"({smallest} posts)")
    chosen: set[str] = {e[0]["item_id"] for c in skip for e in by_condition.get(c, [])}
    for entries in cut.values():
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
    surface = item.get("surface_form") or {}
    namesake = surface.get("top_namesake") or {}
    place_links = item.get("place_sitelinks")
    most_famous = ("" if not namesake or place_links is None
                   else str(place_links > namesake["sitelinks"]).lower())
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
        "clue_forms": ";".join(mention_form(text, c) or "" for c in item.get("clues") or []),
        "noise_entity": (item.get("noise_entity") or {}).get("clue_label") or "",
        "register": item["register"],
        "target_len": item.get("target_len", ""),
        "len": len(text.split()),
        "place_sitelinks": "" if place_links is None else place_links,
        "rival_qid": namesake.get("qid", ""),
        "rival_sitelinks": namesake.get("sitelinks", ""),
        "name_ambiguous": name_ambiguous(surface),
        "place_is_most_famous": most_famous,
        "place_prominence": place_prominence(place_links),
        "rival_prominence": rival_prominence(place_links, namesake),
        "clue_prominence": clue_prominence(item.get("clues") or []),
        "topic": item.get("env_theme") or "",
        "scale": item.get("scale") or "",
        "valence": item.get("valence") or "",
        "angle": row.get("angle") or "",
        "role": ";".join(dict.fromkeys(c.get("role") or "other" for c in item.get("clues") or [])),
        "tries": row.get("tries") or 1,
        "clue_swapped": ";".join(f'{s["from_qid"]}>{s["to_qid"]}' for s in row.get("clue_swaps") or []),
        "swapped_from": row.get("swapped_from") or "",
        "split": item.get("place_split") or "",
        "pool": item.get("pool") or "",
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
                        help="posts to keep per condition (default: finalise."
                             "per_condition in the config, else the size of the "
                             "smallest condition)")
    parser.add_argument("--seed", type=int, default=BALANCE_SEED,
                        help="decides which posts survive balancing "
                             f"(default {BALANCE_SEED})")
    args = parser.parse_args(argv)

    config = gen.load_config(args.config)
    paths = config["paths"]
    per_condition = args.per_condition
    if per_condition is None:
        per_condition = (config.get("finalise") or {}).get("per_condition")
    items = gen.read_jsonl(Path(paths["item_plan"]))
    if not items:
        raise SystemExit("no plan - run plan.py first")
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))
    aliases = load_aliases(Path(paths["place_inputs"]))
    report_path = Path(paths["finalise_report"])
    dataset_path = Path(paths["dataset_csv"])

    env_pattern = environment_pattern((config.get("finalise") or {}).get("environment_keywords"))
    limits = word_limits(config.get("finalise"))

    # Places swapped out of a condition set for a reserve (generate.py, Task 4):
    # their rows in that set are dropped, whatever they say.
    swaps_path = Path(paths["raw_generations"]).parent / "swaps.jsonl"
    swapped_out = {(s["place_qid"], c) for s in (gen.read_jsonl(swaps_path) if swaps_path.exists() else [])
                   for c in s.get("conditions") or []}

    report, kept = [], []
    for item in items:
        row = latest.get(item["item_id"])
        verdict = verdict_for(item, row, aliases, env_pattern, limits)
        if verdict is None and (item["place_qid"], item["condition"]) in swapped_out:
            verdict = ("swapped_out", "place swapped for a reserve after a failed item in its set")
        item = used_item(item, row)
        text = ((row or {}).get("post_text") or "").strip()
        report.append({"item_id": item["item_id"], "condition": item["condition"],
                       "place_qid": item["place_qid"], "place_name": item["place_name"],
                       "status": verdict[0] if verdict else "kept",
                       "reason": verdict[1] if verdict else "", "text": text})
        if verdict is None:
            kept.append((item, row, text))

    # Written first, so a failure below cannot lose the per-item verdicts.
    write_csv(report_path, REPORT_COLUMNS, report)
    print(f"wrote {report_path} (before the coordinate lookup)")

    try:
        coords = coordinates(sorted({item["place_qid"] for item, _, _ in kept}))
    except Exception as exc:  # unreadable places.csv: report stays
        print(f"\n! coordinate lookup failed ({type(exc).__name__}: {exc}).\n"
              f"  {report_path} is written; {dataset_path} was NOT written. "
              f"Check ../out/places.csv.", file=sys.stderr)
        return 1

    kept, no_coords = split_by_coordinates(kept, coords)
    if no_coords:
        apply_no_coordinates(report, no_coords)
        write_csv(report_path, REPORT_COLUMNS, report)

    no_explicit: list[Entry] = []
    if "EXPLICIT" in {item["condition"] for item in items}:
        kept, no_explicit = split_by_explicit(kept)
        if no_explicit:
            apply_no_explicit(report, no_explicit)
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
            kept, cut, balanced_n = balance(
                kept, args.seed, per_condition,
                (config.get("finalise") or {}).get("unbalanced") or ())
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
    if no_explicit:
        print(f"\n! {len(no_explicit)} posts dropped as no_explicit (their place has no kept EXPLICIT post)")
    print(f"\nwrote {dataset_path}")
    print(f"wrote {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
