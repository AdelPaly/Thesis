#!/usr/bin/env python3
"""Re-derive every queryable value in data/01_places_final.csv from Wikidata.

Read-only verification: the input CSV is never written to. Disagreements go to
data/verify_report.csv, P31 labels for manual review go to data/verify_types.csv.

Checks (all rows):
  resolves        every wikidata_id returns an entity
  sitelinks       wikibase:sitelinks vs the file's sitelink column
  inbound_links   distinct Q-entity subjects pointing at the item vs wikidata_links
  name_sharing    items with rdfs:label "<place_name>"@en vs places_sharing_name
  country         P17 vs the country column
  coordinates     flags items with no P625
  area            normalised P2046 (m2 -> km2) vs area_km2, 1% tolerance
  duplicate_*     repeated wikidata_id / place_name within the file
  type sanity     P31 labels dumped for human scanning, never auto-validated

A failed query is reported as "<check>:query_failed", never as a mismatch, and
does not affect the exit code. Exit 0 if no disagreements, 1 otherwise.

The WDQS client lives in wdqs.py; this module re-exports its names (sparql,
QueryError, value_of, qid_from_uri, ...) so existing importers keep working.

Usage:
    py -3 scripts/verify_places.py [--limit N] [--seed N] [--strict-sitelinks]
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import wdqs
from io_utils import utf8_stdout, write_csv_atomic
from jsoncache import JsonCache
from wdqs import QueryError, escape_literal, qid_from_uri, value_of  # noqa: F401  (re-exported)

# Alias of the shared default. verify_usable.py / 04_fix_areas.py still assign
# `vp.USER_AGENT = ...`; `sparql` below reads it at call time so that works.
USER_AGENT = wdqs.USER_AGENT
ENDPOINT = wdqs.ENDPOINT
HTTP_TIMEOUT = wdqs.HTTP_TIMEOUT
ATTEMPTS = wdqs.ATTEMPTS

BATCH_SIZE = 40          # QIDs per VALUES clause
LABEL_BATCH_SIZE = 20    # labels per VALUES clause
SITELINK_BATCH_SIZE = 20 # smaller: this one aggregates over every sitelink
SLEEP_BETWEEN = 0.2      # seconds between requests
CACHE_SAVE_EVERY = 5     # inbound-link lookups between cache saves
PROGRESS_EVERY = 10      # non-interactive progress line every N inbound lookups
SPOT_CHECK_SIZE = 10     # rows printed for manual spot checking
EXIT_DISAGREE = 1
EXIT_BAD_INPUT = 2
EXIT_INTERRUPTED = 130

ENTITY_PREFIX = "http://www.wikidata.org/entity/Q"

# The spec calls this column wikimedia_sitelinks; the shipped file names it
# wikipedia_languages. Accept either.
SITELINK_COLUMNS = ("wikimedia_sitelinks", "wikipedia_languages", "sitelinks")

AREA_TOLERANCE = 0.01  # 1% relative
M2_PER_KM2 = 1_000_000
REPORT_FIELDS = ["wikidata_id", "place_name", "check", "file_value", "live_value"]
TYPES_FIELDS = ["place_name", "assigned_type", "p31_labels"]


def sparql(query: str, *, user_agent: str | None = None, attempts: int = ATTEMPTS,
           timeout: int = HTTP_TIMEOUT) -> list[dict]:
    """Run a query through the shared client, using this module's USER_AGENT by default."""
    return wdqs.sparql(query, user_agent=user_agent or USER_AGENT,
                       attempts=attempts, timeout=timeout)


def chunks(items: list, size: int):
    """Yield successive slices of `items` of at most `size`."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def query_core(qids: list[str]) -> dict[str, dict]:
    """sitelinks, P17, P625 presence and normalised P2046 for a batch of items."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?version ?redirect ?sitelinks ?country ?coord ?area WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item schema:version ?version }}
  OPTIONAL {{ ?item owl:sameAs ?redirect }}
  OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks }}
  OPTIONAL {{ ?item wdt:P17 ?country }}
  OPTIONAL {{ ?item wdt:P625 ?coord }}
  OPTIONAL {{
    ?item p:P2046 ?areaStatement .
    ?areaStatement psn:P2046/wikibase:quantityAmount ?area .
    FILTER NOT EXISTS {{ ?areaStatement wikibase:rank wikibase:DeprecatedRank }}
  }}
}}
"""
    out: dict[str, dict] = {
        q: {
            "exists": False,
            "redirect": None,
            "sitelinks": None,
            "countries": set(),
            "has_coord": False,
            "areas": set(),
        }
        for q in qids
    }
    for row in sparql(query):
        item = value_of(row, "item")
        if not item:
            continue
        record = out.get(qid_from_uri(item))
        if record is None:
            continue
        if value_of(row, "version"):
            record["exists"] = True
        redirect = value_of(row, "redirect")
        if redirect and redirect.startswith(ENTITY_PREFIX):
            record["redirect"] = qid_from_uri(redirect)
        sitelinks = value_of(row, "sitelinks")
        if sitelinks is not None:
            record["exists"] = True
            record["sitelinks"] = int(sitelinks)
        country = value_of(row, "country")
        if country:
            record["countries"].add(qid_from_uri(country))
        if value_of(row, "coord"):
            record["has_coord"] = True
        area = value_of(row, "area")
        if area is not None:
            record["areas"].add(float(area))
    return out


def query_wikipedia_sitelinks(qids: list[str]) -> dict[str, int]:
    """Sitelinks restricted to Wikipedia editions.

    Anchored on ?item via schema:about, so it stays bounded. Used only to
    explain a systematic offset against the file's column; failure is tolerated.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item (COUNT(DISTINCT ?site) AS ?count) WHERE {{
  VALUES ?item {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf ?site .
  FILTER(STRENDS(STR(?site), ".wikipedia.org/"))
}}
GROUP BY ?item
"""
    out: dict[str, int] = {}
    for row in sparql(query):
        item = value_of(row, "item")
        count = value_of(row, "count")
        if item and count is not None:
            out[qid_from_uri(item)] = int(count)
    return out


def query_label_counts(labels: list[str]) -> dict[str, int]:
    """How many Wikidata items carry each exact English label."""
    values = " ".join(f'"{escape_literal(label)}"@en' for label in labels)
    query = f"""
SELECT ?label (COUNT(DISTINCT ?entity) AS ?count) WHERE {{
  VALUES ?label {{ {values} }}
  ?entity rdfs:label ?label .
  FILTER(STRSTARTS(STR(?entity), "{ENTITY_PREFIX}"))
}}
GROUP BY ?label
"""
    counts = {label: 0 for label in labels}
    for row in sparql(query):
        label = value_of(row, "label")
        count = value_of(row, "count")
        if label is not None and count is not None:
            counts[label] = int(count)
    return counts


def query_p31_labels(qids: list[str]) -> dict[str, list[str]]:
    """English labels of each item's P31 (instance of) classes, in first-seen order."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?classLabel WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P31 ?class .
  ?class rdfs:label ?classLabel .
  FILTER(LANG(?classLabel) = "en")
}}
"""
    out: dict[str, list[str]] = defaultdict(list)
    for row in sparql(query):
        item = value_of(row, "item")
        label = value_of(row, "classLabel")
        if item and label:
            qid = qid_from_uri(item)
            if label not in out[qid]:
                out[qid].append(label)
    return out


def query_country_names(qids: list[str]) -> dict[str, dict]:
    """English label plus aliases for the country items we saw."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?label ?alias WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?item skos:altLabel ?alias FILTER(LANG(?alias) = "en") }}
}}
"""
    out: dict[str, dict] = {q: {"label": None, "aliases": set()} for q in qids}
    for row in sparql(query):
        item = value_of(row, "item")
        if not item:
            continue
        record = out.setdefault(qid_from_uri(item), {"label": None, "aliases": set()})
        label = value_of(row, "label")
        if label:
            record["label"] = label
        alias = value_of(row, "alias")
        if alias:
            record["aliases"].add(alias)
    return out


def query_inbound_links(qid: str) -> int:
    """Distinct Q-entity subjects with a statement pointing at this item.

    Slow: unbound predicate against one object. Cached by the caller.
    """
    query = f"""
SELECT (COUNT(DISTINCT ?subject) AS ?count) WHERE {{
  ?subject ?predicate wd:{qid} .
  FILTER(STRSTARTS(STR(?subject), "{ENTITY_PREFIX}"))
}}
"""
    rows = sparql(query)
    if not rows:
        raise QueryError("no rows returned")
    count = value_of(rows[0], "count")
    if count is None:
        raise QueryError("no count in result")
    return int(count)


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------

def normalise_name(text: str) -> str:
    """Fold a place or country name for comparison (case, accents, quotes, 'the')."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("’", "'").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("the "):
        text = text[4:]
    return text


def parse_int(text: str) -> int | None:
    """Parse '1,234' / '12.0' to an int, or None if blank or not a number."""
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(text: str) -> float | None:
    """Parse '1,234.5' to a float, or None if blank or not a number."""
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def within_tolerance(file_value: float, live_value: float) -> bool:
    """True if `live_value` is within AREA_TOLERANCE (relative) of `file_value`."""
    if file_value == 0:
        return abs(live_value) < 1e-9
    return abs(live_value - file_value) / abs(file_value) <= AREA_TOLERANCE


def format_number(value: float) -> str:
    """Compact human-readable number for report cells."""
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

class Cache:
    """Inbound-link counts, persisted as {"inbound_links": {qid: count}}."""

    SECTION = "inbound_links"

    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self._store = JsonCache(path, None, enabled=enabled, indent=1, sort_keys=True)
        section = self._store.entries.get(self.SECTION)
        self._store.entries = {self.SECTION: section if isinstance(section, dict) else {}}

    @property
    def data(self) -> dict[str, dict]:
        """The whole cache payload."""
        return self._store.entries

    def get_inbound(self, qid: str) -> int | None:
        """Cached inbound-link count, or None."""
        return self._store.entries[self.SECTION].get(qid)

    def set_inbound(self, qid: str, count: int) -> None:
        """Record a count (kept in memory even when persistence is disabled)."""
        self._store.entries[self.SECTION][qid] = count
        self._store.mark_dirty()

    def save(self, force: bool = False) -> None:
        """Atomically persist, unless disabled or unchanged."""
        self._store.save(force=force)


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------

@dataclass
class Findings:
    """Disagreements (data is wrong) and query failures (nothing was verified)."""

    disagreements: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)

    def disagree(self, row: dict, check: str, file_value, live_value) -> None:
        """Record that the file and Wikidata differ."""
        self.disagreements.append(_finding(row, check, file_value, live_value))

    def fail(self, row: dict, check: str, file_value, error) -> None:
        """Record that a check could not be run."""
        self.failures.append(_finding(row, f"{check}:query_failed", file_value, error))


def _finding(row: dict, check: str, file_value, live_value) -> dict:
    return {
        "wikidata_id": row.get("wikidata_id", ""),
        "place_name": row.get("place_name", ""),
        "check": check,
        "file_value": "" if file_value is None else str(file_value),
        "live_value": "" if live_value is None else str(live_value),
    }


@dataclass
class LiveData:
    """Everything fetched from Wikidata for one run."""

    core: dict[str, dict] = field(default_factory=dict)
    core_errors: dict[str, str] = field(default_factory=dict)
    wikipedia_counts: dict[str, int] = field(default_factory=dict)
    wikipedia_available: bool = False
    country_names: dict[str, dict] = field(default_factory=dict)
    country_name_error: str | None = None
    label_counts: dict[str, int] = field(default_factory=dict)
    label_errors: dict[str, str] = field(default_factory=dict)
    p31: dict[str, list[str]] = field(default_factory=dict)
    p31_errors: dict[str, str] = field(default_factory=dict)
    inbound: dict[str, int] = field(default_factory=dict)
    inbound_errors: dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_rows(csv_path: Path) -> list[dict]:
    """Rows of the input CSV (BOM tolerated)."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------
# Fetch (queries)
# --------------------------------------------------------------------------

def find_duplicates(rows: list[dict], findings: Findings) -> None:
    """Local check: repeated wikidata_id / place_name."""
    print("[1/6] duplicates")
    for column, check in (("wikidata_id", "duplicate_wikidata_id"), ("place_name", "duplicate_place_name")):
        seen = defaultdict(list)
        for index, row in enumerate(rows, start=2):  # +2: header is line 1
            seen[row.get(column, "").strip()].append(index)
        for value, lines in seen.items():
            if value and len(lines) > 1:
                first = next(r for r in rows if r.get(column, "").strip() == value)
                findings.disagree(
                    first,
                    check,
                    value,
                    f"{len(lines)} rows (csv lines {', '.join(map(str, lines))})",
                )
    print(f"      {sum(1 for d in findings.disagreements if d['check'].startswith('duplicate'))} duplicate(s)")


def fetch_core(unique_qids: list[str], live: LiveData) -> None:
    """Core batch: resolves / sitelinks / country / coords / area."""
    print(f"[2/6] core batch ({len(unique_qids)} items, {BATCH_SIZE} per query)")
    for index, batch in enumerate(chunks(unique_qids, BATCH_SIZE), start=1):
        try:
            live.core.update(query_core(batch))
        except QueryError as exc:
            for qid in batch:
                live.core_errors[qid] = str(exc)
            print(f"      batch {index}: FAILED ({exc})")
        else:
            print(f"      batch {index}: ok ({len(batch)} items)")
        time.sleep(SLEEP_BETWEEN)


def fetch_wikipedia_counts(rows: list[dict], sitelink_column: str, live: LiveData) -> None:
    """Wikipedia-only sitelink counts, as context for rows where the total disagrees.

    Queried lazily so a clean dataset costs nothing extra.
    """
    mismatched = sorted(
        {
            row["wikidata_id"].strip()
            for row in rows
            if (record := live.core.get(row["wikidata_id"].strip()))
            and record["sitelinks"] is not None
            and parse_int(row.get(sitelink_column, "")) != record["sitelinks"]
        }
    )
    live.wikipedia_available = bool(mismatched)
    if not mismatched:
        print("[3/6] wikipedia-only sitelink counts: not needed (all sitelink counts match)")
        return
    print(f"[3/6] wikipedia-only sitelink counts ({len(mismatched)} mismatching items)")
    for index, batch in enumerate(chunks(mismatched, SITELINK_BATCH_SIZE), start=1):
        try:
            live.wikipedia_counts.update(query_wikipedia_sitelinks(batch))
        except QueryError as exc:
            live.wikipedia_available = False
            print(f"      batch {index}: FAILED ({exc}) - falling back to total sitelinks only")
        time.sleep(SLEEP_BETWEEN)
    # Items with no Wikipedia article return no group, which means zero.
    if live.wikipedia_available:
        for qid in mismatched:
            live.wikipedia_counts.setdefault(qid, 0)


def fetch_country_names(live: LiveData) -> None:
    """English labels and aliases for every country item seen in the core batch."""
    country_qids = sorted({c for record in live.core.values() for c in record["countries"]})
    for batch in chunks(country_qids, BATCH_SIZE):
        try:
            live.country_names.update(query_country_names(batch))
        except QueryError as exc:
            live.country_name_error = str(exc)
        time.sleep(SLEEP_BETWEEN)


def fetch_name_sharing(rows: list[dict], live: LiveData) -> None:
    """How many items share each place name."""
    names = sorted({row["place_name"] for row in rows if row.get("place_name")})
    print(f"[4/6] name sharing ({len(names)} labels, {LABEL_BATCH_SIZE} per query)")
    for index, batch in enumerate(chunks(names, LABEL_BATCH_SIZE), start=1):
        try:
            live.label_counts.update(query_label_counts(batch))
        except QueryError as exc:
            # Retry one at a time: one pathological label should not blind the rest.
            print(f"      batch {index}: FAILED ({exc}) - retrying individually")
            for name in batch:
                try:
                    live.label_counts.update(query_label_counts([name]))
                except QueryError as inner:
                    live.label_errors[name] = str(inner)
                time.sleep(SLEEP_BETWEEN)
        time.sleep(SLEEP_BETWEEN)


def fetch_p31(unique_qids: list[str], live: LiveData) -> None:
    """P31 labels for the manual type check."""
    print(f"[5/6] P31 labels ({BATCH_SIZE} per query)")
    for index, batch in enumerate(chunks(unique_qids, BATCH_SIZE), start=1):
        try:
            live.p31.update(query_p31_labels(batch))
        except QueryError as exc:
            for qid in batch:
                live.p31_errors[qid] = str(exc)
            print(f"      batch {index}: FAILED ({exc})")
        time.sleep(SLEEP_BETWEEN)


def fetch_inbound(unique_qids: list[str], live: LiveData, cache: Cache, skip: bool) -> None:
    """Inbound-link counts: slow, cached, one query per item, saved as it goes."""
    if skip:
        print("[6/6] inbound links: skipped (--skip-inbound)")
        return
    todo = [q for q in unique_qids if cache.get_inbound(q) is None]
    cached = len(unique_qids) - len(todo)
    print(f"[6/6] inbound links ({len(unique_qids)} items, {cached} from cache)")
    interactive = sys.stdout.isatty()
    try:
        for position, qid in enumerate(unique_qids, start=1):
            hit = cache.get_inbound(qid)
            if hit is not None:
                live.inbound[qid] = hit
                continue
            record = live.core.get(qid)
            if record is not None and not record["exists"]:
                continue  # nothing to count for an item that does not resolve
            label = f"      [{position:>3}/{len(unique_qids)}] {qid}"
            if interactive:
                print(f"{label} ...", end="\r", flush=True)
            try:
                count = query_inbound_links(qid)
            except QueryError as exc:
                live.inbound_errors[qid] = str(exc)  # not cached: retry on re-run
                print(f"{label} FAILED ({exc})")
            else:
                live.inbound[qid] = count
                cache.set_inbound(qid, count)
                if interactive:
                    print(f"{label} {count}      ", end="\r", flush=True)
                elif position % PROGRESS_EVERY == 0:
                    print(f"{label} {count}", flush=True)
            if position % CACHE_SAVE_EVERY == 0:
                cache.save()
            time.sleep(SLEEP_BETWEEN)
    except KeyboardInterrupt:
        print("\n      interrupted; cache saved, re-run to resume")
        raise
    finally:
        cache.save(force=True)
    if interactive:
        print(" " * 60, end="\r")
    print(f"      done ({len(live.inbound)} counted, {len(live.inbound_errors)} failed)")


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def check_resolves(row: dict, qid: str, live: LiveData, findings: Findings) -> bool:
    """Record resolve problems. True if the QID is unusable for the checks below."""
    if qid in live.core_errors:
        findings.fail(row, "resolves", qid, live.core_errors[qid])
        return True
    record = live.core.get(qid)
    if record is None or not record["exists"]:
        findings.disagree(row, "resolves", qid, "no entity found")
        return True
    if record["redirect"]:
        findings.disagree(row, "resolves", qid, f"redirects to {record['redirect']}")
        return True
    return False


def check_name_sharing(row: dict, live: LiveData, findings: Findings) -> None:
    """places_sharing_name vs the live count of items with that label."""
    name = row.get("place_name", "")
    file_shared = parse_int(row.get("places_sharing_name", ""))
    if name in live.label_errors:
        findings.fail(row, "name_sharing", row.get("places_sharing_name", ""), live.label_errors[name])
    elif name in live.label_counts:
        live_shared = live.label_counts[name]
        if file_shared is None:
            findings.disagree(row, "name_sharing", row.get("places_sharing_name", ""), live_shared)
        elif file_shared != live_shared:
            findings.disagree(row, "name_sharing", file_shared, live_shared)


def check_sitelinks(row: dict, qid: str, record: dict, sitelink_column: str, strict: bool,
                    live: LiveData, findings: Findings) -> bool:
    """wikibase:sitelinks vs the file. True if it differs only by the Wikipedia-only definition."""
    file_sitelinks = parse_int(row.get(sitelink_column, ""))
    live_sitelinks = record["sitelinks"]
    if live_sitelinks is None:
        findings.fail(row, "sitelinks", row.get(sitelink_column, ""), "wikibase:sitelinks missing")
        return False
    if file_sitelinks == live_sitelinks:
        return False
    wikipedia_only = live.wikipedia_counts.get(qid) if live.wikipedia_available else None
    if not strict and wikipedia_only is not None and file_sitelinks == wikipedia_only:
        # Definitional, not a data error: the column counts Wikipedia editions,
        # wikibase:sitelinks counts every project. Summarised once, not per row.
        return True
    text = str(live_sitelinks)
    if wikipedia_only is not None:
        text += f" (wikipedia-only: {wikipedia_only})"
    findings.disagree(row, "sitelinks", row.get(sitelink_column, ""), text)
    return False


def check_inbound(row: dict, qid: str, live: LiveData, findings: Findings) -> None:
    """wikidata_links vs the live inbound-link count."""
    file_links = parse_int(row.get("wikidata_links", ""))
    if qid in live.inbound_errors:
        findings.fail(row, "inbound_links", row.get("wikidata_links", ""), live.inbound_errors[qid])
    elif qid in live.inbound and file_links != live.inbound[qid]:
        findings.disagree(row, "inbound_links", row.get("wikidata_links", ""), live.inbound[qid])


def check_country(row: dict, record: dict, live: LiveData, findings: Findings) -> None:
    """The country column vs P17 label/aliases."""
    file_country = (row.get("country", "") or "").strip()
    live_countries = sorted(record["countries"])
    if not live_countries:
        findings.disagree(row, "country", file_country, "no P17")
        return
    display, accepted = [], set()
    for country_qid in live_countries:
        info = live.country_names.get(country_qid)
        if not info or not info["label"]:
            display.append(country_qid)
        else:
            display.append(f"{info['label']} ({country_qid})")
            accepted.add(normalise_name(info["label"]))
            accepted.update(normalise_name(a) for a in info["aliases"])
    if not accepted and live.country_name_error:
        findings.fail(row, "country", file_country, live.country_name_error)
    elif normalise_name(file_country) not in accepted:
        findings.disagree(row, "country", file_country, ", ".join(display))


def check_area(row: dict, record: dict, findings: Findings) -> None:
    """area_km2 vs normalised P2046 (m2 -> km2), within AREA_TOLERANCE."""
    file_area = parse_float(row.get("area_km2", ""))
    live_areas_km2 = sorted(a / M2_PER_KM2 for a in record["areas"])
    live_text = " / ".join(format_number(a) for a in live_areas_km2) + " km2"
    if file_area is None:
        if live_areas_km2:
            findings.disagree(row, "area", "(blank)", live_text)
    elif not live_areas_km2:
        findings.disagree(row, "area", row.get("area_km2", ""), "no P2046")
    elif not any(within_tolerance(file_area, live) for live in live_areas_km2):
        findings.disagree(row, "area", row.get("area_km2", ""), live_text)


def type_row(row: dict, qid: str, live: LiveData) -> dict:
    """One P31-labels line for the manual type review."""
    classes = live.p31.get(qid, [])
    return {
        "place_name": row.get("place_name", ""),
        "assigned_type": row.get("type", ""),
        "p31_labels": " | ".join(classes)
        if classes
        else ("QUERY FAILED" if qid in live.p31_errors else "(none)"),
    }


def compare_rows(rows: list[dict], live: LiveData, findings: Findings, sitelink_column: str,
                 strict_sitelinks: bool) -> tuple[list[dict], int]:
    """Run every per-row check. Returns (type rows, rows matching only Wikipedia-only sitelinks)."""
    type_rows: list[dict] = []
    wikipedia_only_rows = 0

    for row in rows:
        qid = row["wikidata_id"].strip()
        unresolved = check_resolves(row, qid, live, findings)
        check_name_sharing(row, live, findings)  # independent of whether the QID resolves

        type_rows.append(type_row(row, qid, live))
        if qid in live.p31_errors:
            findings.fail(row, "p31_labels", row.get("type", ""), live.p31_errors[qid])

        if unresolved:
            continue  # everything below would just echo the unresolved item

        record = live.core[qid]
        if check_sitelinks(row, qid, record, sitelink_column, strict_sitelinks, live, findings):
            wikipedia_only_rows += 1
        check_inbound(row, qid, live, findings)
        check_country(row, record, live, findings)
        if not record["has_coord"]:
            findings.disagree(row, "coordinates", "expected P625", "no P625")
        check_area(row, record, findings)

    return type_rows, wikipedia_only_rows


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def write_reports(report_path: Path, types_path: Path, findings: Findings,
                  type_rows: list[dict]) -> list[dict]:
    """Write verify_report.csv and verify_types.csv. Returns the report rows."""
    report_rows = findings.disagreements + findings.failures
    write_csv_atomic(report_path, REPORT_FIELDS, report_rows, encoding="utf-8-sig")
    write_csv_atomic(types_path, TYPES_FIELDS, type_rows, encoding="utf-8-sig")
    return report_rows


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_type_sanity(type_rows: list[dict]) -> None:
    """Assigned type next to P31 labels, for human scanning."""
    print()
    print("=" * 78)
    print("TYPE SANITY - assigned type vs P31 (human judgement, not auto-validated)")
    print("=" * 78)
    for entry in sorted(type_rows, key=lambda e: (e["assigned_type"], e["place_name"])):
        print(f"{entry['place_name'][:28]:<28} {entry['assigned_type']:<18} | {entry['p31_labels']}")


def print_summary(rows: list[dict], unique_qids: list[str], findings: Findings,
                  wikipedia_only_rows: int, sitelink_column: str, report_path: Path,
                  types_path: Path, report_count: int, type_count: int, cache_path: Path | None) -> str:
    """Print the summary block. Returns the RESULT text."""
    by_check = Counter(d["check"] for d in findings.disagreements)
    by_failure = Counter(f["check"] for f in findings.failures)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"rows checked        : {len(rows)}")
    print(f"distinct QIDs       : {len(unique_qids)}")
    print(f"disagreements       : {len(findings.disagreements)}")
    print(f"query failures      : {len(findings.failures)}")
    print()
    if by_check:
        print("disagreements by check:")
        for check, count in sorted(by_check.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {check:<24} {count}")
    else:
        print("disagreements by check: none")
    if by_failure:
        print()
        print("query failures by check (not counted as mismatches):")
        for check, count in sorted(by_failure.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {check:<24} {count}")
    if wikipedia_only_rows:
        print()
        print(
            f"note: {wikipedia_only_rows} row(s) differ from wikibase:sitelinks but match the\n"
            f"      Wikipedia-only sitelink count exactly, i.e. column '{sitelink_column}' counts\n"
            f"      Wikipedia editions while wikibase:sitelinks counts every project. Not reported\n"
            f"      as disagreements; use --strict-sitelinks to report them anyway."
        )
    print()
    print(f"report              : {report_path} ({report_count} rows)")
    print(f"types               : {types_path} ({type_count} rows)")
    if cache_path is not None:
        print(f"cache               : {cache_path}")
    print()
    result = "FAIL" if findings.disagreements else "PASS"
    if findings.failures:
        # A failed query verified nothing, so a clean PASS would be a lie.
        result += f" (inconclusive: {len(findings.failures)} check(s) could not be queried)"
    print(f"RESULT: {result}")
    return result


def print_spot_check(rows: list[dict], seed: int | None) -> None:
    """A random sample of rows to open by hand."""
    rng = random.Random(seed)
    sample = rng.sample(rows, min(SPOT_CHECK_SIZE, len(rows)))
    print()
    print("=" * 78)
    print("SPOT CHECK - open these and confirm label, country and type match")
    print("=" * 78)
    for entry in sample:
        print(
            f"{entry.get('place_name', ''):<28} {entry.get('type', ''):<18} "
            f"{entry.get('country', ''):<24} "
            f"https://www.wikidata.org/wiki/{entry.get('wikidata_id', '').strip()}"
        )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(root: Path) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Verify data/01_places_final.csv against Wikidata.")
    parser.add_argument("--csv", default=str(root / "data" / "01_places_final.csv"), help="input CSV (read only)")
    parser.add_argument("--out-dir", default=str(root / "data"), help="where reports and cache go")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N rows (smoke test)")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random spot-check sample")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the inbound-link cache")
    parser.add_argument(
        "--strict-sitelinks",
        action="store_true",
        help="report every difference from wikibase:sitelinks, even when the file matches the Wikipedia-only count",
    )
    parser.add_argument("--skip-inbound", action="store_true", help="skip the slow inbound-link check")
    return parser.parse_args()


def main() -> int:
    """CLI entry: load -> fetch -> compare -> write -> report."""
    # utf-8: place and class labels are not ASCII and the Windows console codepage
    # is not either. line_buffering: this run takes minutes, so progress has to
    # stream even when stdout is redirected to a file.
    utf8_stdout()

    root = Path(__file__).resolve().parent.parent
    args = parse_args(root)

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    report_path = out_dir / "verify_report.csv"
    types_path = out_dir / "verify_types.csv"
    cache_path = out_dir / "verify_cache.json"

    # Hard guard: this script must never write over its own input.
    for output in (report_path, types_path, cache_path):
        if output == csv_path:
            print(f"refusing to write output over the input CSV: {csv_path}")
            return EXIT_BAD_INPUT

    if not csv_path.exists():
        print(f"input not found: {csv_path}")
        return EXIT_BAD_INPUT

    all_rows = load_rows(csv_path)
    if not all_rows:
        print(f"no rows in {csv_path}")
        return EXIT_BAD_INPUT

    columns = list(all_rows[0].keys())
    sitelink_column = next((c for c in SITELINK_COLUMNS if c in columns), None)
    if sitelink_column is None:
        print(f"no sitelink column found; looked for {SITELINK_COLUMNS} in {columns}")
        return EXIT_BAD_INPUT

    rows = all_rows[: args.limit] if args.limit else all_rows

    print(f"input      : {csv_path}")
    print(f"rows       : {len(rows)}" + (f" (of {len(all_rows)}, --limit)" if args.limit else ""))
    print(f"sitelinks  : comparing wikibase:sitelinks against column '{sitelink_column}'")
    print(f"outputs    : {report_path}, {types_path}")
    print()

    unique_qids = sorted({q for q in (row["wikidata_id"].strip() for row in rows) if q})

    findings = Findings()
    live = LiveData()
    cache = Cache(cache_path, enabled=not args.no_cache)

    find_duplicates(rows, findings)
    fetch_core(unique_qids, live)
    fetch_wikipedia_counts(rows, sitelink_column, live)
    fetch_country_names(live)
    fetch_name_sharing(rows, live)
    fetch_p31(unique_qids, live)
    fetch_inbound(unique_qids, live, cache, args.skip_inbound)

    type_rows, wikipedia_only_rows = compare_rows(rows, live, findings, sitelink_column,
                                                  args.strict_sitelinks)
    report_rows = write_reports(report_path, types_path, findings, type_rows)

    print_type_sanity(type_rows)
    shown_cache = None if (args.no_cache or args.skip_inbound) else cache_path
    print_summary(rows, unique_qids, findings, wikipedia_only_rows, sitelink_column,
                  report_path, types_path, len(report_rows), len(type_rows), shown_cache)
    print_spot_check(rows, args.seed)

    return EXIT_DISAGREE if findings.disagreements else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(EXIT_INTERRUPTED)
