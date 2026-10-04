#!/usr/bin/env python3
"""Re-derive every queryable value in data/01_places_final.csv from Wikidata.

READ-ONLY. The input CSV is opened for reading and never written to. Everything
this script produces goes to --out-dir (data/ by default).

The SPARQL plumbing, the batched queries and the comparison helpers are shared
with verify_places.py rather than copied; this script adds the two checks that
file does not have (sovereign_state, continent) and reports split by origin.

WHY ORIGIN MATTERS. The 90 origin=retained rows came through this repo's own
pipeline. The 57 origin=replacement rows were produced outside it and have had
less checking, so every count in the summary is broken out by origin and the
replacement side is what a reviewer should read first.

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
                  dumped side by side to verify_types.csv for a human to read.

MISMATCH vs FAILURE. A query that times out or errors is recorded as
"<check>:query_failed" with the error as live_value. That is not a disagreement
and does not fail the run -- an endpoint having a bad minute is not a data defect.
Pass --strict to make query failures fail the run too. Either way the counts are
printed separately, so a run that verified nothing cannot read as a clean pass.

A blank wikimedia_sitelinks is reported as "sitelinks:missing_in_file" rather
than as a disagreement: the file asserting nothing is a gap in the file, not a
conflict with Wikidata. (In the current file that is every retained row.)

wikidata_links is frozen -- see FROZEN_COLUMNS. Growth since the freeze date is
reported as "inbound_links:drift_since_freeze", not as a disagreement.

AREA TOLERANCE stays at 1%. It was only ever breached by values rounded to one
decimal place, and 04_fix_areas.py has since re-derived the column at full
precision, so the tolerance no longer needs widening to pass. Loosening it to 2%
now would buy nothing and would hide real errors of a few percent.

QUERY RULES. Descriptive User-Agent; QIDs batched ~40 per VALUES clause with
0.2s between requests; sitelink counts read from the precomputed
wikibase:sitelinks rather than COUNT(DISTINCT) over sitelinks, which times out;
FILTER NOT EXISTS anchored on a subject already bound from ?item, never left
unbounded.

Exit 0 if no disagreements, 1 otherwise.

Usage:
    py -3 scripts/verify_usable.py [--csv PATH] [--out-dir DIR] [--limit N]
                                   [--origin replacement] [--strict] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import verify_places as vp

# Distinct from verify_places.py so operators can tell the two apart in logs.
vp.USER_AGENT = (
    "places-verify-usable/1.0 (https://github.com/; read-only verification of a "
    "147-row place dataset; contact: shay@shaypunter.co.uk) python-urllib"
)

sparql = vp.sparql
QueryError = vp.QueryError
value_of = vp.value_of
qid_from_uri = vp.qid_from_uri
chunks = vp.chunks
normalise_name = vp.normalise_name
parse_int = vp.parse_int
parse_float = vp.parse_float
within_tolerance = vp.within_tolerance
format_number = vp.format_number

BATCH_SIZE = vp.BATCH_SIZE              # 40 QIDs per VALUES clause
LABEL_BATCH_SIZE = vp.LABEL_BATCH_SIZE  # 20 labels per VALUES clause
SLEEP_BETWEEN = vp.SLEEP_BETWEEN        # 0.2s between requests
ENTITY_PREFIX = vp.ENTITY_PREFIX

SOVEREIGN_STATE = "Q3624078"

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

REQUIRED_COLUMNS = [
    "place_name", "type", "country", "continent", "wikidata_id",
    "clue_count", "wikimedia_sitelinks", "wikidata_links",
    "places_sharing_name", "area_km2",
]

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
    for row in sparql(query):
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
    for row in sparql(query):
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
    import time

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
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Verify data/01_places_final.csv against Wikidata.")
    parser.add_argument("--csv", default=str(root / "data" / "01_places_final.csv"),
                        help="input CSV (read only, never written)")
    parser.add_argument("--out-dir", default=str(root / "data"))
    parser.add_argument("--limit", type=int, default=0, help="check only the first N rows")
    parser.add_argument("--origin", choices=("retained", "replacement"),
                        help="restrict the run to one origin")
    parser.add_argument("--strict", action="store_true",
                        help="query failures fail the run as well as disagreements")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and do not write the inbound-link cache")
    args = parser.parse_args(argv)

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    report_path = out_dir / "verify_report.csv"
    types_path = out_dir / "verify_types.csv"

    if not csv_path.exists():
        print(f"input not found: {csv_path}", file=sys.stderr)
        return 2
    # Read-only guarantee: refuse to run if an output would land on the input.
    for path in (report_path, types_path):
        if path.resolve() == csv_path.resolve():
            print(f"refusing to write output over the input CSV: {path}", file=sys.stderr)
            return 2

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    absent = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if absent:
        print(f"input is missing required columns: {', '.join(absent)}", file=sys.stderr)
        return 2
    extra = [c for c in fieldnames if c not in REQUIRED_COLUMNS]
    if extra:
        print(f"note: ignoring extra columns: {', '.join(extra)}")

    # Duplicate checks run over the whole file before any filtering, so a
    # --limit or --origin run still sees collisions with rows it skipped.
    qid_counts = Counter((r.get("wikidata_id") or "").strip() for r in rows)
    name_counts = Counter(normalise_name(r.get("place_name") or "") for r in rows)

    if args.origin:
        rows = [r for r in rows if (r.get("origin") or "").strip() == args.origin]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("no rows to check", file=sys.stderr)
        return 2

    findings: list[dict] = []
    failed_checks: Counter = Counter()

    def record(row: dict, check: str, file_value, live_value) -> None:
        findings.append({
            "wikidata_id": (row.get("wikidata_id") or "").strip(),
            "place_name": row.get("place_name") or "",
            "origin": (row.get("origin") or "").strip(),
            "check": check,
            "file_value": "" if file_value is None else str(file_value),
            "live_value": "" if live_value is None else str(live_value),
        })

    def fail(row: dict, check: str, error: str) -> None:
        failed_checks[check] += 1
        record(row, f"{check}:query_failed", "", error[:300])

    qids = [(r.get("wikidata_id") or "").strip() for r in rows]
    by_qid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_qid[(row.get("wikidata_id") or "").strip()].append(row)

    print(f"verifying {len(rows)} rows from {csv_path.name}")
    print(f"  retained    : {sum(1 for r in rows if r.get('origin') == 'retained')}")
    print(f"  replacement : {sum(1 for r in rows if r.get('origin') == 'replacement')}")

    # ---- core: existence, sitelinks, P17, P625, P2046 ----
    print("\ncore facts (existence, sitelinks, country, coordinates, area)")
    core: dict[str, dict] = {}
    core_failed: set[str] = set()

    def core_fail(chunk, error):
        core_failed.update(chunk)
        for q in chunk:
            for row in by_qid[q]:
                for check in ("resolves", "sitelinks", "country", "coordinates", "area"):
                    fail(row, check, error)

    for part in run_batched("core", qids, BATCH_SIZE, vp.query_core, core_fail):
        core.update(part)

    # ---- sovereign state ----
    print("\nsovereign-state check")
    sovereign: set[str] = set()
    sovereign_failed: set[str] = set()

    def sovereign_fail(chunk, error):
        sovereign_failed.update(chunk)
        for q in chunk:
            for row in by_qid[q]:
                fail(row, "sovereign_state", error)

    for part in run_batched("sovereign", qids, BATCH_SIZE, query_sovereign_states, sovereign_fail):
        sovereign.update(part)

    # ---- P31 labels for the type dump ----
    print("\nP31 labels (for manual type review)")
    p31: dict[str, list[str]] = {}

    def p31_fail(chunk, error):
        for q in chunk:
            p31[q] = [f"<query failed: {error[:80]}>"]

    for part in run_batched("p31", qids, BATCH_SIZE, vp.query_p31_labels, p31_fail):
        p31.update(part)

    # ---- shared-name counts ----
    print("\nshared-name counts")
    wanted_labels = sorted({(r.get("place_name") or "").strip() for r in rows
                            if (r.get("place_name") or "").strip()})
    label_counts: dict[str, int] = {}
    label_failed: set[str] = set()

    def label_fail(chunk, error):
        label_failed.update(chunk)
        for label in chunk:
            for row in rows:
                if (row.get("place_name") or "").strip() == label:
                    fail(row, "name_sharing", error)

    for part in run_batched("labels", wanted_labels, LABEL_BATCH_SIZE,
                            vp.query_label_counts, label_fail):
        label_counts.update(part)

    # ---- country labels and their continents ----
    country_qids = sorted({q for rec in core.values() for q in rec["countries"]})
    country_names: dict[str, dict] = {}
    continents: dict[str, set[str]] = {}
    if country_qids:
        print("\ncountry labels")
        for part in run_batched("countries", country_qids, BATCH_SIZE,
                                vp.query_country_names, lambda c, e: None):
            country_names.update(part)
        print("\ncountry continents (P30)")
        for part in run_batched("continents", country_qids, BATCH_SIZE,
                                query_country_continents, lambda c, e: None):
            continents.update(part)

    # ---- inbound links: one slow query per item, cached ----
    print(f"\ninbound links ({len(qids)} queries, unbound predicate -- slow)")
    cache = vp.Cache(out_dir / "verify_usable_cache.json", enabled=not args.no_cache)
    inbound: dict[str, int] = {}
    for index, qid in enumerate(qids, start=1):
        cached = cache.get_inbound(qid)
        if cached is not None:
            inbound[qid] = cached
            continue
        if index % 10 == 1:
            print(f"  [{index}/{len(qids)}]", flush=True)
        try:
            value = vp.query_inbound_links(qid)
        except QueryError as exc:
            for row in by_qid[qid]:
                fail(row, "inbound_links", str(exc))
            continue
        inbound[qid] = value
        cache.set_inbound(qid, value)
        import time
        time.sleep(SLEEP_BETWEEN)
    cache.save()

    # ---- compare ----
    print("\ncomparing")
    for row in rows:
        qid = (row.get("wikidata_id") or "").strip()
        name = (row.get("place_name") or "").strip()
        record_core = core.get(qid)

        if qid_counts[qid] > 1:
            record(row, "duplicate_qid", qid, f"{qid_counts[qid]} rows share this wikidata_id")
        if name_counts[normalise_name(name)] > 1:
            record(row, "duplicate_name", name,
                   f"{name_counts[normalise_name(name)]} rows share this place_name")

        if qid in core_failed or record_core is None:
            continue

        if not record_core["exists"]:
            record(row, "resolves", qid, "no entity")
            continue
        if record_core["redirect"]:
            record(row, "redirect", qid, f"redirects to {record_core['redirect']}")

        # sitelinks
        file_sitelinks = parse_int(row.get("wikimedia_sitelinks", ""))
        live_sitelinks = record_core["sitelinks"]
        if file_sitelinks is None:
            record(row, "sitelinks:missing_in_file", "",
                   "" if live_sitelinks is None else live_sitelinks)
        elif live_sitelinks is None:
            record(row, "sitelinks", file_sitelinks, "no wikibase:sitelinks")
        elif file_sitelinks != live_sitelinks:
            record(row, "sitelinks", file_sitelinks, live_sitelinks)

        # inbound links
        file_links = parse_int(row.get("wikidata_links", ""))
        live_links = inbound.get(qid)
        if live_links is not None:
            if file_links is None:
                record(row, "inbound_links", row.get("wikidata_links", ""), live_links)
            elif file_links != live_links:
                delta = abs(live_links - file_links)
                drift = (delta <= FROZEN_DRIFT_ABS
                         or (file_links and delta / abs(file_links) <= FROZEN_DRIFT_REL))
                record(row, "inbound_links:drift_since_freeze" if drift else "inbound_links",
                       file_links, live_links)

        # shared name
        if name and name not in label_failed:
            file_share = parse_int(row.get("places_sharing_name", ""))
            live_share = label_counts.get(name)
            if live_share is not None:
                if file_share is None:
                    record(row, "name_sharing", row.get("places_sharing_name", ""), live_share)
                elif file_share != live_share:
                    record(row, "name_sharing", file_share, live_share)

        # country
        file_country = (row.get("country") or "").strip()
        live_countries = record_core["countries"]
        if not live_countries:
            record(row, "country", file_country, "no P17")
        else:
            names = []
            matched = False
            for country_qid in sorted(live_countries):
                info = country_names.get(country_qid, {"label": None, "aliases": set()})
                candidates = {info["label"]} | set(info["aliases"])
                candidates = {c for c in candidates if c}
                names.append(info["label"] or country_qid)
                if any(normalise_name(c) == normalise_name(file_country) for c in candidates):
                    matched = True
            if not matched:
                record(row, "country", file_country, " / ".join(names))

        # coordinates
        if not record_core["has_coord"]:
            record(row, "coordinates", "", "no P625")

        # area: blank in the file must mean absent in Wikidata
        file_area = parse_float(row.get("area_km2", ""))
        live_areas = sorted(a / 1_000_000 for a in record_core["areas"])
        live_text = " / ".join(format_number(a) for a in live_areas) + " km2" if live_areas else "no P2046"
        if file_area is None:
            if live_areas:
                record(row, "area", "(blank)", live_text)
        elif not live_areas:
            record(row, "area", row.get("area_km2", ""), "no P2046")
        elif not any(within_tolerance(file_area, live) for live in live_areas):
            record(row, "area", row.get("area_km2", ""), live_text)

        # sovereign state
        if qid not in sovereign_failed and qid in sovereign:
            record(row, "sovereign_state", row.get("type", ""),
                   f"P31 sovereign state ({SOVEREIGN_STATE})")

        # continent, via the country's P30
        file_continent = (row.get("continent") or "").strip()
        live_continents: set[str] = set()
        for country_qid in live_countries:
            live_continents |= continents.get(country_qid, set())
        if live_continents and not any(
            normalise_name(c) == normalise_name(file_continent) for c in live_continents
        ):
            record(row, "continent", file_continent, " / ".join(sorted(live_continents)))

    # ---- write reports (never the input) ----
    out_dir.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["wikidata_id", "place_name", "origin", "check", "file_value", "live_value"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(sorted(
            findings,
            key=lambda f: (CHECK_ORDER.index(f["check"].split(":")[0])
                           if f["check"].split(":")[0] in CHECK_ORDER else len(CHECK_ORDER),
                           f["check"], f["place_name"]),
        ))

    with types_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["place_name", "assigned_type", "p31_labels"])
        for row in rows:
            qid = (row.get("wikidata_id") or "").strip()
            writer.writerow([
                row.get("place_name", ""),
                row.get("type", ""),
                "; ".join(p31.get(qid, [])),
            ])

    # ---- summary ----
    disagreements = [f for f in findings if ":" not in f["check"]]
    query_failures = [f for f in findings if f["check"].endswith(":query_failed")]
    informational = [f for f in findings
                     if ":" in f["check"] and not f["check"].endswith(":query_failed")]

    origins = ["retained", "replacement"]
    counts: dict[str, Counter] = defaultdict(Counter)
    for finding in disagreements:
        counts[finding["check"]][finding["origin"]] += 1

    rows_by_origin = Counter(r.get("origin", "") for r in rows)
    print("\n" + "=" * 68)
    print(f"rows checked            : {len(rows)}"
          + "".join(f"   {o} {rows_by_origin.get(o, 0)}" for o in origins))
    print(f"disagreements           : {len(disagreements)}")
    print(f"informational (gaps)    : {len(informational)}")
    print(f"query failures          : {len(query_failures)}")
    for column, frozen_on in FROZEN_COLUMNS.items():
        print(f"frozen column           : {column} as of {frozen_on} (not re-derived)")
    print(f"area tolerance          : {vp.AREA_TOLERANCE:.0%}")

    print(f"\n{'check':<18} {'total':>6} {'retained':>9} {'replacement':>12}")
    print("-" * 48)
    ordered = [c for c in CHECK_ORDER if c in counts]
    ordered += sorted(c for c in counts if c not in CHECK_ORDER)
    for check in ordered:
        total = sum(counts[check].values())
        print(f"{check:<18} {total:>6} {counts[check]['retained']:>9} "
              f"{counts[check]['replacement']:>12}")
    if not ordered:
        print("(none)")

    # Counts that grow on their own (inbound links, sitelinks) disagree by a few
    # units on almost every row simply because the file was built days ago. That
    # is still a disagreement -- the column IS stale -- but a reviewer needs to
    # see a +3 drift and a 10x error as different things, so size them here.
    for check in ("inbound_links", "sitelinks"):
        drift = [f for f in findings if f["check"].split(":")[0] == check
                 and not f["check"].endswith(":query_failed")
                 and not f["check"].endswith(":missing_in_file")]
        if not drift:
            continue
        buckets = Counter()
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
        for delta, finding in sorted(worst, key=lambda w: -w[0])[:5]:
            print(f"    {finding['place_name']:<28} {finding['origin']:<12} "
                  f"{finding['file_value']} -> {finding['live_value']} ({delta:+.0%})")

    if informational:
        by_check = Counter(f["check"] for f in informational)
        print("\ninformational, not counted as disagreements:")
        for check, n in by_check.most_common():
            print(f"  {check:<28} {n}")

    if query_failures:
        print(f"\n!! {len(query_failures)} checks could not be evaluated "
              f"(reported as <check>:query_failed, not as mismatches):")
        for check, n in failed_checks.most_common():
            print(f"  {check:<28} {n}")
        print("  a value that could not be fetched has NOT been verified.")

    print(f"\nreport : {report_path}")
    print(f"types  : {types_path}  ({len(rows)} rows, manual review -- not auto-validated)")

    failed = bool(disagreements) or (args.strict and bool(query_failures))
    print("\n" + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
