#!/usr/bin/env python3
"""Re-derive every queryable value in data/01_places_final.csv from Wikidata.

READ-ONLY. The input CSV is opened for reading and never written to. Everything
this script produces goes to --out-dir (data/ by default) under its OWN file
names, so it cannot overwrite verify_places.py's reports:

    verify_usable_report.csv   one row per disagreement / gap / query failure
    verify_usable_types.csv    assigned_type next to the item's P31 labels
    verify_usable_cache.json   inbound-link counts (the slow query)

(verify_places.py writes verify_report.csv, verify_types.csv, verify_cache.json.)

The SPARQL plumbing, the batched queries and the comparison helpers are shared
with verify_places.py rather than copied; this script adds the two checks that
file does not have (sovereign_state, continent).

COLUMNS. The identity columns are required: place_name, type, country,
continent, wikidata_id. The columns a check compares against are optional, and a
check whose column is absent is skipped with a note rather than failing the run
-- so a place CSV straight out of build_inputs.py can be verified:

    wikimedia_sitelinks  -> sitelinks        wikidata_links     -> inbound_links
    places_sharing_name  -> name_sharing     area_km2           -> area

(03_clue_totals.py needs `clue_count`, and 04_fix_areas.py needs `area_km2`;
this script needs neither.)

ORIGIN IS OPTIONAL. An earlier 147-row candidate file carried an `origin` column
(retained / replacement). If the input has it, --origin works and the report and
summary are split by it; if not (the shipped 01_places_final.csv has none), the
split is dropped and --origin exits with an explanation.

CHECKS

  resolves        the wikidata_id returns an entity at all
  redirect        the wikidata_id resolves but is a redirect to another item
  sitelinks       wikibase:sitelinks vs wikimedia_sitelinks
  inbound_links   distinct Q-entity subjects pointing at the item vs wikidata_links
  name_sharing    items with rdfs:label "<place_name>"@en vs places_sharing_name
  country         P17 label/alias vs the country column
  coordinates     the item has P625
  area            normalised P2046 (m2 -> km2) vs area_km2, 1% tolerance;
                  blank in the file must mean no P2046 in Wikidata, and vice versa
  duplicate_qid   the same wikidata_id on more than one row
  duplicate_name  the same place_name on more than one row
  sovereign_state the item is P31 sovereign state (Q3624078) -- a country is not a place
  continent       the continent column against P30 of the item's P17 country

  type            NOT auto-validated. assigned_type and the item's P31 labels are
                  dumped side by side to verify_usable_types.csv for a human to read.

MISMATCH vs FAILURE. A query that times out or errors is recorded as
"<check>:query_failed" with the error as live_value. That is not a disagreement
and does not fail the run -- an endpoint having a bad minute is not a data defect.
Pass --strict to make query failures fail the run too. Either way the counts are
printed separately, so a run that verified nothing cannot read as a clean pass.

A blank wikimedia_sitelinks is reported as "sitelinks:missing_in_file" rather
than as a disagreement: the file asserting nothing is a gap in the file, not a
conflict with Wikidata.

wikidata_links is frozen -- see FROZEN_COLUMNS. Growth since the freeze date is
reported as "inbound_links:drift_since_freeze", not as a disagreement.

AREA TOLERANCE stays at 1%. It was only ever breached by values rounded to one
decimal place, and 04_fix_areas.py has since re-derived the column at full
precision, so the tolerance no longer needs widening to pass. Loosening it to 2%
now would buy nothing and would hide real errors of a few percent.

QUERY RULES. Descriptive User-Agent (USER_AGENT below, built from wdqs.CONTACT);
QIDs batched ~40 per VALUES clause with 0.2s between requests; sitelink counts
read from the precomputed wikibase:sitelinks rather than COUNT(DISTINCT) over
sitelinks, which times out; FILTER NOT EXISTS anchored on a subject already
bound from ?item, never left unbounded. This script's own two queries send
USER_AGENT; the shared lookups it borrows from verify_places.py send that
module's default User-Agent.

The inbound-link loop saves its cache every few items and in a `finally`, so an
interrupt keeps the progress.

Exit 0 if no disagreements, 1 otherwise, 2 for bad input.

Usage:
    py -3 scripts/verify_usable.py [--csv PATH] [--out-dir DIR] [--limit N]
                                   [--origin replacement] [--strict] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_places as vp  # noqa: E402
import wdqs  # noqa: E402
from io_utils import utf8_stdout, write_text_atomic  # noqa: E402
from wdqs import QueryError, qid_from_uri, sparql, value_of  # noqa: E402

# Distinct from verify_places.py so operators can tell the two apart in logs.
USER_AGENT = (
    f"places-verify-usable/1.0 (read-only verification of a place dataset; "
    f"contact: {wdqs.CONTACT}) python-urllib"
)

chunks = vp.chunks
normalise_name = vp.normalise_name
parse_int = vp.parse_int
parse_float = vp.parse_float
within_tolerance = vp.within_tolerance
format_number = vp.format_number

BATCH_SIZE = vp.BATCH_SIZE              # 40 QIDs per VALUES clause
LABEL_BATCH_SIZE = vp.LABEL_BATCH_SIZE  # 20 labels per VALUES clause
SLEEP_BETWEEN = vp.SLEEP_BETWEEN        # 0.2s between requests
CACHE_SAVE_EVERY = vp.CACHE_SAVE_EVERY  # inbound-link lookups between cache saves
PROGRESS_EVERY = 10                     # progress line every N inbound lookups
ERROR_CHARS = 300                       # characters of an error kept in a report cell
M2_PER_KM2 = vp.M2_PER_KM2
SUMMARY_WIDTH = 68
WORST_EXAMPLES = 5                      # biggest differences listed per drift table

SOVEREIGN_STATE = "Q3624078"

REPORT_NAME = "verify_usable_report.csv"
TYPES_NAME = "verify_usable_types.csv"
CACHE_NAME = "verify_usable_cache.json"

ORIGINS = ("retained", "replacement")

# wikidata_links is a FROZEN column. Inbound-link counts rise as Wikidata grows,
# so re-deriving them on a schedule produces a permanent trickle of one- and
# two-unit "failures" that mean nothing and train a reader to ignore the report.
# The values are fixed as of the date below and are not refreshed.
#
# The check still runs -- a column that silently rots is worse than one that is
# openly stale -- but ordinary growth is reported as informational rather than as
# a disagreement. Only a divergence too large to be growth still fails: more than
# FROZEN_DRIFT_REL relative AND more than FROZEN_DRIFT_ABS absolute, the second
# clause so a 10 -> 11 on a small count is not treated as a 10% error.
FROZEN_COLUMNS = {"wikidata_links": "2026-08-28"}
FROZEN_DRIFT_REL = 0.01
FROZEN_DRIFT_ABS = 2

REQUIRED_COLUMNS = ["place_name", "type", "country", "continent", "wikidata_id"]

# check -> the input column it compares against. A check whose column is absent
# from the input is skipped (with a note), not failed.
CHECK_COLUMNS = {
    "sitelinks": "wikimedia_sitelinks",
    "inbound_links": "wikidata_links",
    "name_sharing": "places_sharing_name",
    "area": "area_km2",
}

# Report order. Anything not listed still prints, just after these.
CHECK_ORDER = [
    "resolves", "redirect", "sitelinks", "inbound_links", "name_sharing",
    "country", "coordinates", "area", "duplicate_qid", "duplicate_name",
    "sovereign_state", "continent",
]


# --------------------------------------------------------------------------
# Queries this script adds
# --------------------------------------------------------------------------

def query_sovereign_states(qids: list[str]) -> set[str]:
    """Which of these items are instances of sovereign state.

    Anchored on ?item through VALUES, so the pattern is bounded by the batch.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P31 wd:{SOVEREIGN_STATE} .
}}
"""
    found = set()
    for row in sparql(query, user_agent=USER_AGENT):
        item = value_of(row, "item")
        if item:
            found.add(qid_from_uri(item))
    return found


def query_country_continents(qids: list[str]) -> dict[str, set[str]]:
    """P30 continent labels for each country item."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?continentLabel WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P30 ?continent .
  ?continent rdfs:label ?continentLabel .
  FILTER(LANG(?continentLabel) = "en")
}}
"""
    out: dict[str, set[str]] = defaultdict(set)
    for row in sparql(query, user_agent=USER_AGENT):
        item = value_of(row, "item")
        label = value_of(row, "continentLabel")
        if item and label:
            out[qid_from_uri(item)].add(label)
    return out


# --------------------------------------------------------------------------
# Batch driver
# --------------------------------------------------------------------------

def run_batched(label: str, items: list, size: int, fn, on_fail):
    """Apply fn to each chunk, sleeping between requests.

    A chunk that raises is handed to on_fail with its members rather than
    aborting the pass, so one bad batch does not blank out the whole check.
    """
    merged: list = []
    total = (len(items) + size - 1) // size
    for index, chunk in enumerate(chunks(items, size), start=1):
        print(f"  {label} batch {index}/{total} ({len(chunk)})", flush=True)
        try:
            merged.append(fn(chunk))
        except QueryError as exc:
            print(f"    ! {label} batch {index} failed: {exc}", flush=True)
            on_fail(chunk, str(exc))
        time.sleep(SLEEP_BETWEEN)
    return merged


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------

def qid_of(row: dict) -> str:
    """The row's wikidata_id, stripped."""
    return (row.get("wikidata_id") or "").strip()


def name_of(row: dict) -> str:
    """The row's place_name, stripped."""
    return (row.get("place_name") or "").strip()


@dataclass
class Findings:
    """Every finding of the run, plus how many checks could not be evaluated."""

    has_origin: bool
    rows: list[dict] = field(default_factory=list)
    failed_checks: Counter = field(default_factory=Counter)

    def record(self, row: dict, check: str, file_value, live_value) -> None:
        """Note one finding (a disagreement, a gap, or a query failure)."""
        self.rows.append({
            "wikidata_id": qid_of(row),
            "place_name": row.get("place_name") or "",
            "origin": (row.get("origin") or "").strip(),
            "check": check,
            "file_value": "" if file_value is None else str(file_value),
            "live_value": "" if live_value is None else str(live_value),
        })

    def fail(self, row: dict, check: str, error: str) -> None:
        """Note that `check` could not be evaluated for `row`."""
        self.failed_checks[check] += 1
        self.record(row, f"{check}:query_failed", "", error[:ERROR_CHARS])


@dataclass
class Live:
    """Everything fetched from Wikidata for the run."""

    core: dict[str, dict] = field(default_factory=dict)
    core_failed: set[str] = field(default_factory=set)
    sovereign: set[str] = field(default_factory=set)
    sovereign_failed: set[str] = field(default_factory=set)
    p31: dict[str, list[str]] = field(default_factory=dict)
    label_counts: dict[str, int] = field(default_factory=dict)
    label_failed: set[str] = field(default_factory=set)
    country_names: dict[str, dict] = field(default_factory=dict)
    continents: dict[str, set[str]] = field(default_factory=dict)
    inbound: dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_input(csv_path: Path) -> tuple[list[str], list[dict]]:
    """The input CSV as (column names, rows). BOM tolerated."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def enabled_checks(fieldnames: list[str]) -> set[str]:
    """The optional checks whose input column is present; notes the ones skipped."""
    enabled = set()
    for check, column in CHECK_COLUMNS.items():
        if column in fieldnames:
            enabled.add(check)
        else:
            print(f"note: no '{column}' column, so the '{check}' check is skipped")
    return enabled


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------

def fetch_core(qids: list[str], by_qid: dict[str, list[dict]], enabled: set[str],
               findings: Findings, live: Live) -> None:
    """Existence, sitelinks, P17, P625 and P2046 for every item."""
    print("\ncore facts (existence, sitelinks, country, coordinates, area)")
    failing = ["resolves", "country", "coordinates"] + [
        c for c in ("sitelinks", "area") if c in enabled]

    def on_fail(chunk, error):
        live.core_failed.update(chunk)
        for q in chunk:
            for row in by_qid[q]:
                for check in failing:
                    findings.fail(row, check, error)

    for part in run_batched("core", qids, BATCH_SIZE, vp.query_core, on_fail):
        live.core.update(part)


def fetch_sovereign(qids: list[str], by_qid: dict[str, list[dict]],
                    findings: Findings, live: Live) -> None:
    """Which items are sovereign states."""
    print("\nsovereign-state check")

    def on_fail(chunk, error):
        live.sovereign_failed.update(chunk)
        for q in chunk:
            for row in by_qid[q]:
                findings.fail(row, "sovereign_state", error)

    for part in run_batched("sovereign", qids, BATCH_SIZE, query_sovereign_states, on_fail):
        live.sovereign.update(part)


def fetch_p31(qids: list[str], live: Live) -> None:
    """P31 labels for the type dump."""
    print("\nP31 labels (for manual type review)")

    def on_fail(chunk, error):
        for q in chunk:
            live.p31[q] = [f"<query failed: {error[:80]}>"]

    for part in run_batched("p31", qids, BATCH_SIZE, vp.query_p31_labels, on_fail):
        live.p31.update(part)


def fetch_label_counts(rows: list[dict], findings: Findings, live: Live) -> None:
    """How many items share each place name."""
    print("\nshared-name counts")
    wanted_labels = sorted({name_of(r) for r in rows if name_of(r)})

    def on_fail(chunk, error):
        live.label_failed.update(chunk)
        for label in chunk:
            for row in rows:
                if name_of(row) == label:
                    findings.fail(row, "name_sharing", error)

    for part in run_batched("labels", wanted_labels, LABEL_BATCH_SIZE,
                            vp.query_label_counts, on_fail):
        live.label_counts.update(part)


def fetch_countries(live: Live) -> None:
    """Country labels and their P30 continents (failures leave the check silent)."""
    country_qids = sorted({q for rec in live.core.values() for q in rec["countries"]})
    if not country_qids:
        return
    print("\ncountry labels")
    for part in run_batched("countries", country_qids, BATCH_SIZE,
                            vp.query_country_names, lambda c, e: None):
        live.country_names.update(part)
    print("\ncountry continents (P30)")
    for part in run_batched("continents", country_qids, BATCH_SIZE,
                            query_country_continents, lambda c, e: None):
        live.continents.update(part)


def fetch_inbound(qids: list[str], by_qid: dict[str, list[dict]], findings: Findings,
                  live: Live, cache: vp.Cache) -> None:
    """Inbound links: one slow query per item, cached, saved as it goes."""
    print(f"\ninbound links ({len(qids)} queries, unbound predicate -- slow)")
    since_save = 0
    try:
        for index, qid in enumerate(qids, start=1):
            cached = cache.get_inbound(qid)
            if cached is not None:
                live.inbound[qid] = cached
                continue
            if index % PROGRESS_EVERY == 1:
                print(f"  [{index}/{len(qids)}]", flush=True)
            try:
                value = vp.query_inbound_links(qid)
            except QueryError as exc:
                for row in by_qid[qid]:
                    findings.fail(row, "inbound_links", str(exc))
                continue
            live.inbound[qid] = value
            cache.set_inbound(qid, value)
            since_save += 1
            if since_save >= CACHE_SAVE_EVERY:
                cache.save()
                since_save = 0
            time.sleep(SLEEP_BETWEEN)
    finally:
        cache.save()


# --------------------------------------------------------------------------
# Compare
# --------------------------------------------------------------------------

def check_sitelinks(row: dict, record_core: dict, findings: Findings) -> None:
    """wikimedia_sitelinks against wikibase:sitelinks."""
    file_sitelinks = parse_int(row.get("wikimedia_sitelinks", ""))
    live_sitelinks = record_core["sitelinks"]
    if file_sitelinks is None:
        findings.record(row, "sitelinks:missing_in_file", "",
                        "" if live_sitelinks is None else live_sitelinks)
    elif live_sitelinks is None:
        findings.record(row, "sitelinks", file_sitelinks, "no wikibase:sitelinks")
    elif file_sitelinks != live_sitelinks:
        findings.record(row, "sitelinks", file_sitelinks, live_sitelinks)


def check_inbound(row: dict, live_links: int | None, findings: Findings) -> None:
    """wikidata_links against the live inbound count, allowing frozen-column drift."""
    if live_links is None:
        return
    file_links = parse_int(row.get("wikidata_links", ""))
    if file_links is None:
        findings.record(row, "inbound_links", row.get("wikidata_links", ""), live_links)
    elif file_links != live_links:
        delta = abs(live_links - file_links)
        drift = (delta <= FROZEN_DRIFT_ABS
                 or (file_links and delta / abs(file_links) <= FROZEN_DRIFT_REL))
        findings.record(row, "inbound_links:drift_since_freeze" if drift else "inbound_links",
                        file_links, live_links)


def check_name_sharing(row: dict, live: Live, findings: Findings) -> None:
    """places_sharing_name against the live count of items with that label."""
    name = name_of(row)
    if not name or name in live.label_failed:
        return
    live_share = live.label_counts.get(name)
    if live_share is None:
        return
    file_share = parse_int(row.get("places_sharing_name", ""))
    if file_share is None:
        findings.record(row, "name_sharing", row.get("places_sharing_name", ""), live_share)
    elif file_share != live_share:
        findings.record(row, "name_sharing", file_share, live_share)


def check_country(row: dict, live_countries: set[str], live: Live, findings: Findings) -> None:
    """The country column against the label or an alias of any P17 country."""
    file_country = (row.get("country") or "").strip()
    if not live_countries:
        findings.record(row, "country", file_country, "no P17")
        return
    names = []
    matched = False
    for country_qid in sorted(live_countries):
        info = live.country_names.get(country_qid, {"label": None, "aliases": set()})
        candidates = {c for c in ({info["label"]} | set(info["aliases"])) if c}
        names.append(info["label"] or country_qid)
        if any(normalise_name(c) == normalise_name(file_country) for c in candidates):
            matched = True
    if not matched:
        findings.record(row, "country", file_country, " / ".join(names))


def check_area(row: dict, record_core: dict, findings: Findings) -> None:
    """area_km2: blank in the file must mean absent in Wikidata, and vice versa."""
    file_area = parse_float(row.get("area_km2", ""))
    live_areas = sorted(a / M2_PER_KM2 for a in record_core["areas"])
    live_text = (" / ".join(format_number(a) for a in live_areas) + " km2"
                 if live_areas else "no P2046")
    if file_area is None:
        if live_areas:
            findings.record(row, "area", "(blank)", live_text)
    elif not live_areas:
        findings.record(row, "area", row.get("area_km2", ""), "no P2046")
    elif not any(within_tolerance(file_area, live) for live in live_areas):
        findings.record(row, "area", row.get("area_km2", ""), live_text)


def check_continent(row: dict, live_countries: set[str], live: Live,
                    findings: Findings) -> None:
    """The continent column against P30 of the item's P17 country."""
    file_continent = (row.get("continent") or "").strip()
    live_continents: set[str] = set()
    for country_qid in live_countries:
        live_continents |= live.continents.get(country_qid, set())
    if live_continents and not any(
        normalise_name(c) == normalise_name(file_continent) for c in live_continents
    ):
        findings.record(row, "continent", file_continent, " / ".join(sorted(live_continents)))


def compare_row(row: dict, enabled: set[str], qid_counts: Counter, name_counts: Counter,
                live: Live, findings: Findings) -> None:
    """Run every enabled check on one row."""
    qid = qid_of(row)
    name = name_of(row)
    record_core = live.core.get(qid)

    if qid_counts[qid] > 1:
        findings.record(row, "duplicate_qid", qid, f"{qid_counts[qid]} rows share this wikidata_id")
    if name_counts[normalise_name(name)] > 1:
        findings.record(row, "duplicate_name", name,
                        f"{name_counts[normalise_name(name)]} rows share this place_name")

    if qid in live.core_failed or record_core is None:
        return

    if not record_core["exists"]:
        findings.record(row, "resolves", qid, "no entity")
        return
    if record_core["redirect"]:
        findings.record(row, "redirect", qid, f"redirects to {record_core['redirect']}")

    live_countries = record_core["countries"]
    if "sitelinks" in enabled:
        check_sitelinks(row, record_core, findings)
    if "inbound_links" in enabled:
        check_inbound(row, live.inbound.get(qid), findings)
    if "name_sharing" in enabled:
        check_name_sharing(row, live, findings)
    check_country(row, live_countries, live, findings)
    if not record_core["has_coord"]:
        findings.record(row, "coordinates", "", "no P625")
    if "area" in enabled:
        check_area(row, record_core, findings)
    if qid not in live.sovereign_failed and qid in live.sovereign:
        findings.record(row, "sovereign_state", row.get("type", ""),
                        f"P31 sovereign state ({SOVEREIGN_STATE})")
    check_continent(row, live_countries, live, findings)


# --------------------------------------------------------------------------
# Write
# --------------------------------------------------------------------------

def write_csv_lf(path: Path, header: list[str], rows) -> None:
    """Atomically write a UTF-8-with-BOM CSV with LF line endings."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    write_text_atomic(path, buffer.getvalue(), encoding="utf-8-sig")


def write_reports(report_path: Path, types_path: Path, findings: Findings,
                  rows: list[dict], live: Live) -> None:
    """The findings report and the type dump (never the input)."""
    def sort_key(f: dict):
        base = f["check"].split(":")[0]
        return (CHECK_ORDER.index(base) if base in CHECK_ORDER else len(CHECK_ORDER),
                f["check"], f["place_name"])

    fields = ["wikidata_id", "place_name"] + (["origin"] if findings.has_origin else []) + [
        "check", "file_value", "live_value"]
    write_csv_lf(report_path, fields,
                 ([f[c] for c in fields] for f in sorted(findings.rows, key=sort_key)))
    write_csv_lf(types_path, ["place_name", "assigned_type", "p31_labels"],
                 ([row.get("place_name", ""), row.get("type", ""),
                   "; ".join(live.p31.get(qid_of(row), []))] for row in rows))


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

def print_drift_sizes(findings: Findings) -> None:
    """Size the growth-type differences: a +3 drift and a 10x error are not the same thing.

    Counts that grow on their own (inbound links, sitelinks) disagree by a few
    units on almost every row simply because the file was built days ago. That
    is still a disagreement -- the column IS stale -- but a reviewer needs to
    see the two as different things.
    """
    for check in ("inbound_links", "sitelinks"):
        drift = [f for f in findings.rows if f["check"].split(":")[0] == check
                 and not f["check"].endswith(":query_failed")
                 and not f["check"].endswith(":missing_in_file")]
        if not drift:
            continue
        buckets: Counter = Counter()
        worst: list[tuple[float, dict]] = []
        for finding in drift:
            before, after = parse_float(finding["file_value"]), parse_float(finding["live_value"])
            if before is None or after is None or before == 0:
                buckets["unparseable"] += 1
                continue
            delta = abs(after - before) / abs(before)
            worst.append((delta, finding))
            if delta <= 0.01:
                buckets["<=1% (drift)"] += 1
            elif delta <= 0.05:
                buckets["1-5%"] += 1
            elif delta <= 0.25:
                buckets["5-25%"] += 1
            else:
                buckets[">25%"] += 1
        counted = "disagreements" if any(":" not in f["check"] for f in drift) else "informational"
        print(f"\n{check} difference sizes ({len(drift)} rows, {counted}):")
        for bucket in ("<=1% (drift)", "1-5%", "5-25%", ">25%", "unparseable"):
            if buckets[bucket]:
                print(f"  {bucket:<16} {buckets[bucket]:>4}")
        for delta, finding in sorted(worst, key=lambda w: -w[0])[:WORST_EXAMPLES]:
            where = f"{finding['origin']:<12} " if findings.has_origin else ""
            print(f"    {finding['place_name']:<28} {where}"
                  f"{finding['file_value']} -> {finding['live_value']} ({delta:+.0%})")


def print_summary(rows: list[dict], findings: Findings, report_path: Path,
                  types_path: Path, strict: bool) -> bool:
    """Print the run summary; return True when the run failed."""
    disagreements = [f for f in findings.rows if ":" not in f["check"]]
    query_failures = [f for f in findings.rows if f["check"].endswith(":query_failed")]
    informational = [f for f in findings.rows
                     if ":" in f["check"] and not f["check"].endswith(":query_failed")]

    counts: dict[str, Counter] = defaultdict(Counter)
    for finding in disagreements:
        counts[finding["check"]][finding["origin"]] += 1

    print("\n" + "=" * SUMMARY_WIDTH)
    rows_by_origin = Counter(r.get("origin", "") for r in rows)
    print(f"rows checked            : {len(rows)}"
          + ("".join(f"   {o} {rows_by_origin.get(o, 0)}" for o in ORIGINS)
             if findings.has_origin else ""))
    print(f"disagreements           : {len(disagreements)}")
    print(f"informational (gaps)    : {len(informational)}")
    print(f"query failures          : {len(query_failures)}")
    for column, frozen_on in FROZEN_COLUMNS.items():
        print(f"frozen column           : {column} as of {frozen_on} (not re-derived)")
    print(f"area tolerance          : {vp.AREA_TOLERANCE:.0%}")

    origin_heads = f" {'retained':>9} {'replacement':>12}" if findings.has_origin else ""
    print(f"\n{'check':<18} {'total':>6}{origin_heads}")
    print("-" * (48 if findings.has_origin else 25))
    ordered = [c for c in CHECK_ORDER if c in counts]
    ordered += sorted(c for c in counts if c not in CHECK_ORDER)
    for check in ordered:
        total = sum(counts[check].values())
        split = (f" {counts[check]['retained']:>9} {counts[check]['replacement']:>12}"
                 if findings.has_origin else "")
        print(f"{check:<18} {total:>6}{split}")
    if not ordered:
        print("(none)")

    print_drift_sizes(findings)

    if informational:
        by_check = Counter(f["check"] for f in informational)
        print("\ninformational, not counted as disagreements:")
        for check, n in by_check.most_common():
            print(f"  {check:<28} {n}")

    if query_failures:
        print(f"\n!! {len(query_failures)} checks could not be evaluated "
              f"(reported as <check>:query_failed, not as mismatches):")
        for check, n in findings.failed_checks.most_common():
            print(f"  {check:<28} {n}")
        print("  a value that could not be fetched has NOT been verified.")

    print(f"\nreport : {report_path}")
    print(f"types  : {types_path}  ({len(rows)} rows, manual review -- not auto-validated)")

    failed = bool(disagreements) or (strict and bool(query_failures))
    print("\n" + ("FAIL" if failed else "PASS"))
    return failed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser(root: Path) -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(description="Verify data/01_places_final.csv against Wikidata.")
    parser.add_argument("--csv", default=str(root / "data" / "01_places_final.csv"),
                        help="input CSV (read only, never written)")
    parser.add_argument("--out-dir", default=str(root / "data"),
                        help=f"where {REPORT_NAME}, {TYPES_NAME} and {CACHE_NAME} go")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N rows")
    parser.add_argument("--origin", choices=ORIGINS,
                        help="restrict the run to one origin (needs an `origin` column)")
    parser.add_argument("--strict", action="store_true",
                        help="query failures fail the run as well as disagreements")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and do not write the inbound-link cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Fetch, compare, write the reports and print the summary. See the module docstring."""
    utf8_stdout()

    root = Path(__file__).resolve().parent.parent
    args = build_parser(root).parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    report_path = out_dir / REPORT_NAME
    types_path = out_dir / TYPES_NAME

    if not csv_path.exists():
        print(f"input not found: {csv_path}", file=sys.stderr)
        return 2
    # Read-only guarantee: refuse to run if an output would land on the input.
    for path in (report_path, types_path):
        if path.resolve() == csv_path.resolve():
            print(f"refusing to write output over the input CSV: {path}", file=sys.stderr)
            return 2

    fieldnames, rows = load_input(csv_path)
    absent = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if absent:
        print(f"input is missing required columns: {', '.join(absent)}", file=sys.stderr)
        return 2
    has_origin = "origin" in fieldnames
    if args.origin and not has_origin:
        print(f"--origin {args.origin} needs an `origin` column and {csv_path.name} has "
              f"none; drop --origin (the per-origin split only applies to the older "
              f"candidate files)", file=sys.stderr)
        return 2
    known = set(REQUIRED_COLUMNS) | set(CHECK_COLUMNS.values()) | {"origin"}
    extra = [c for c in fieldnames if c not in known]
    if extra:
        print(f"note: ignoring extra columns: {', '.join(extra)}")
    enabled = enabled_checks(fieldnames)

    # Duplicate checks run over the whole file before any filtering, so a
    # --limit or --origin run still sees collisions with rows it skipped.
    qid_counts = Counter(qid_of(r) for r in rows)
    name_counts = Counter(normalise_name(r.get("place_name") or "") for r in rows)

    if args.origin:
        rows = [r for r in rows if (r.get("origin") or "").strip() == args.origin]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no rows to check", file=sys.stderr)
        return 2

    findings = Findings(has_origin=has_origin)
    live = Live()
    qids = [qid_of(r) for r in rows]
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qid[qid_of(row)].append(row)

    print(f"verifying {len(rows)} rows from {csv_path.name}")
    if has_origin:
        print(f"  retained    : {sum(1 for r in rows if r.get('origin') == 'retained')}")
        print(f"  replacement : {sum(1 for r in rows if r.get('origin') == 'replacement')}")

    fetch_core(qids, by_qid, enabled, findings, live)
    fetch_sovereign(qids, by_qid, findings, live)
    fetch_p31(qids, live)
    if "name_sharing" in enabled:
        fetch_label_counts(rows, findings, live)
    fetch_countries(live)
    if "inbound_links" in enabled:
        cache = vp.Cache(out_dir / CACHE_NAME, enabled=not args.no_cache)
        fetch_inbound(qids, by_qid, findings, live, cache)

    print("\ncomparing")
    for row in rows:
        compare_row(row, enabled, qid_counts, name_counts, live, findings)

    write_reports(report_path, types_path, findings, rows, live)
    failed = print_summary(rows, findings, report_path, types_path, args.strict)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
