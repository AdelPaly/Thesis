#!/usr/bin/env python3
"""Score every surviving clue for ambiguity. This is where "distinct enough"
stops being a judgement and becomes a measurement.

Three scores per clue.

PLACE AMBIGUITY -- how many distinct places this clue entity links to by the
SAME relation that linked it here. The clue arrived because `clue -> place`
held; the question is how many other places that same arrow also hits. One is
strong signal ("Vinita Schools" is in exactly one town). Forty is noise (a film
with forty filming locations tells you nothing about any of them).

  The relation is inverted per property, and for the three patterns the whole
  multi-hop shape from 02_clues.py is inverted rather than approximated:
  SPORTS keeps its venue hop, EVENTS keeps P276|P131 but drops the date triple
  (that was a candidate net over ?clue, not part of the link), WORKS keeps
  P840|P915.

  The value is counted over PLACES, not over all values of the property --
  restricted to things carrying a coordinate (P625). That restriction is what
  makes the number mean what its name says, and it is load-bearing for P138:
  "named after" values are frequently people, and counting them raw would score
  a thing named after one town and two saints as ambiguity 3. Both counts are
  taken in the same query so the divergence is reported rather than hidden.

SURFACE AMBIGUITY -- how many distinct Wikidata entities carry the clue's
English label as a label or alias. A different question from place ambiguity,
asked of the string rather than the entity: it is what an entity linker sees
when the label shows up in a post. A clue can be perfectly unambiguous as an
entity and hopeless as a string.

  Aliases are collected alongside (P1449 nickname, P1705 native label, and
  Wikidata's own English aliases via skos:altLabel) because a post may use any
  of them. --no-altlabel restricts this to the two named properties.
  surface_ambiguity itself is measured on the primary label, as specified.

PROPERTY FAMILY -- person / place_or_org / work_or_event / named_after, per the
mapping the leakage measurement justified. P495 and P1435 are in the clue set
but absent from that mapping; they are tagged `other` and reported separately
rather than being quietly folded into a family they were never assigned to.

Queries are batched with VALUES and cached; a chunk that fails is split and
retried down to single entities, so one pathological clue costs one clue rather
than a hundred and fifty.

Output:
    data/04_clues_scored.jsonl   input fields + place_ambiguity,
                                 surface_ambiguity, aliases, property_family
    data/04_score_summary.csv    one row per place: clue count, min/median/max
                                 place ambiguity, clues by family

Usage:
    py -3 scripts/04_score_clues.py [--clues data/02_clues_filtered.jsonl]
                                    [--high N] [--limit N] [--no-altlabel]
                                    [--no-cache] [--dry-run]
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_clues = import_module("02_clues")

QueryError = _clues.QueryError
sparql = _clues.sparql
value_of = _clues.value_of
qid_from_uri = _clues.qid_from_uri
SLEEP_BETWEEN = _clues.SLEEP_BETWEEN
SPORTS_ORG_CLASS = _clues.SPORTS_ORG_CLASS

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "02_clues_filtered.jsonl"
OUT_JSONL = ROOT / "data" / "04_clues_scored.jsonl"
OUT_CSV = ROOT / "data" / "04_score_summary.csv"
CACHE_PATH = ROOT / "data" / "04_score_cache.json"

CACHE_VERSION = 1

QID_CHUNK = 150      # entities bound into one VALUES block for count queries
LABEL_CHUNK = 80     # label strings per surface-ambiguity query
ALIAS_CHUNK = 200    # entities per alias query

# The mapping as specified. P495 and P1435 are deliberately absent -- see
# UNMAPPED handling below; they are not silently assigned.
FAMILIES = {
    "P19": "person",
    "P20": "person",
    "P937": "person",
    "P131": "place_or_org",
    "P159": "place_or_org",
    "P276": "place_or_org",
    "SPORTS": "place_or_org",
    "WORKS": "work_or_event",
    "EVENTS": "work_or_event",
    "P1071": "work_or_event",
    "P138": "named_after",
}
FAMILY_ORDER = ["person", "place_or_org", "work_or_event", "named_after", "other"]

AMBIGUITY_BUCKETS = (
    ("1", lambda n: n == 1),
    ("2-4", lambda n: 2 <= n <= 4),
    ("5-9", lambda n: 5 <= n <= 9),
    ("10+", lambda n: n >= 10),
)

SURFACE_BUCKETS = (
    ("1", lambda n: n <= 1),
    ("2-4", lambda n: 2 <= n <= 4),
    ("5-19", lambda n: 5 <= n <= 19),
    ("20+", lambda n: n >= 20),
)


# --------------------------------------------------------------------------
# Inverted relations
# --------------------------------------------------------------------------

def place_where(prop: str) -> str:
    """The 02_clues.py WHERE clause with the place turned into a variable."""
    if prop == "SPORTS":
        return f"""{{
    {{ ?clue wdt:P115 ?venue . ?venue (wdt:P131|wdt:P276) ?place . }}
    UNION {{ ?clue wdt:P115 ?place . }}
    UNION {{
      ?clue (wdt:P131|wdt:P159|wdt:P276) ?place .
      ?clue wdt:P31/wdt:P279* wd:{SPORTS_ORG_CLASS} .
    }}
  }}"""
    if prop == "WORKS":
        return "?clue (wdt:P840|wdt:P915) ?place ."
    if prop == "EVENTS":
        # The date triple was a candidate net over ?clue, not part of the link.
        return "?clue (wdt:P276|wdt:P131) ?place ."
    return f"?clue wdt:{prop} ?place ."


def place_count_query(prop: str, qids: list[str]) -> str:
    """Both counts in one pass: all values, and values that are actually places.

    ?geo is bound only where the value carries a coordinate, so COUNT(?geo)
    is the geographic count and COUNT(?place) the raw one.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT ?clue (COUNT(DISTINCT ?place) AS ?n_all) (COUNT(DISTINCT ?geo) AS ?n_geo) WHERE {{
  VALUES ?clue {{ {values} }}
  {place_where(prop)}
  OPTIONAL {{ ?place wdt:P625 ?coord . BIND(?place AS ?geo) }}
}}
GROUP BY ?clue
"""


def escape_literal(text: str) -> str:
    out = text.replace("\\", "\\\\").replace('"', '\\"')
    return out.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")


def surface_count_query(labels: list[str]) -> str:
    """Entities carrying the string as an English label OR alias.

    Alias matching is the point: candidate generation in an entity linker looks
    at both, so counting only rdfs:label would understate what the string
    actually denotes.
    """
    values = " ".join(f'"{escape_literal(text)}"@en' for text in labels)
    return f"""
SELECT ?label (COUNT(DISTINCT ?e) AS ?n) WHERE {{
  VALUES ?label {{ {values} }}
  ?e rdfs:label|skos:altLabel ?label .
}}
GROUP BY ?label
"""


def alias_query(qids: list[str], altlabel: bool) -> str:
    values = " ".join(f"wd:{q}" for q in qids)
    blocks = [
        '{ ?clue wdt:P1449 ?alias . BIND("P1449" AS ?src) }',
        '{ ?clue wdt:P1705 ?alias . BIND("P1705" AS ?src) }',
    ]
    if altlabel:
        blocks.insert(
            0,
            '{ ?clue skos:altLabel ?alias FILTER(lang(?alias) = "en") '
            'BIND("altLabel" AS ?src) }',
        )
    return f"""
SELECT DISTINCT ?clue ?alias ?src WHERE {{
  VALUES ?clue {{ {values} }}
  {" UNION ".join(blocks)}
}}
"""


# --------------------------------------------------------------------------
# Batched fetching with split-on-failure
# --------------------------------------------------------------------------

def run_chunked(items: list, chunk: int, build, collect, label: str,
                failures: list[tuple[str, str]]) -> dict:
    """Query `items` in chunks; on failure split the chunk and retry.

    A chunk that dies takes only its own members down, and halving isolates the
    one entity that killed it instead of writing off the batch.
    """
    out: dict = {}
    pending = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    total = len(pending)
    done = 0
    while pending:
        batch = pending.pop(0)
        try:
            rows = sparql(build(batch))
        except QueryError as exc:
            if len(batch) > 1:
                middle = len(batch) // 2
                pending.insert(0, batch[middle:])
                pending.insert(0, batch[:middle])
                print(f"      ! {label} chunk of {len(batch)} failed ({exc}); splitting",
                      flush=True)
                continue
            failures.append((str(batch[0]), str(exc)))
            print(f"      ! {label} failed for {batch[0]}: {exc}", flush=True)
            continue
        collect(batch, rows, out)
        done += 1
        if done % 5 == 0 or done == total:
            print(f"      {label}: {done}/{max(total, done)} chunks", flush=True)
        time.sleep(SLEEP_BETWEEN)
    return out


def collect_place_counts(batch: list[str], rows: list[dict], out: dict) -> None:
    seen = {}
    for row in rows:
        clue = value_of(row, "clue")
        if not clue:
            continue
        seen[qid_from_uri(clue)] = (
            int(value_of(row, "n_all") or 0),
            int(value_of(row, "n_geo") or 0),
        )
    # A clue absent from a GROUP BY result matched nothing: that is a real 0.
    for qid in batch:
        out[qid] = seen.get(qid, (0, 0))


def collect_surface_counts(batch: list[str], rows: list[dict], out: dict) -> None:
    seen = {}
    for row in rows:
        text = value_of(row, "label")
        if text is not None:
            seen[text] = int(value_of(row, "n") or 0)
    for text in batch:
        out[text] = seen.get(text, 0)


def collect_aliases(batch: list[str], rows: list[dict], out: dict) -> None:
    found: dict[str, list] = {q: [] for q in batch}
    for row in rows:
        clue = value_of(row, "clue")
        alias = value_of(row, "alias")
        src = value_of(row, "src")
        if not clue or not alias:
            continue
        qid = qid_from_uri(clue)
        entry = [alias, src or ""]
        if qid in found and entry not in found[qid]:
            found[qid].append(entry)
    out.update(found)


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

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


def bucket_row(values: list[int], buckets) -> list[int]:
    return [sum(1 for v in values if test(v)) for _, test in buckets]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description="Score clues for ambiguity.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--summary", type=Path, default=OUT_CSV)
    parser.add_argument("--high", type=int, default=5,
                        help="place ambiguity at or above this counts as high (default 5)")
    parser.add_argument("--limit", type=int,
                        help="only score the first N places (partial output; for testing)")
    parser.add_argument("--no-altlabel", action="store_true",
                        help="aliases from P1449/P1705 only, ignoring skos:altLabel")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the query plan and exit without querying")
    args = parser.parse_args(argv)

    if not args.clues.exists():
        print(f"clue file not found: {args.clues}", file=sys.stderr)
        return 2

    records = [json.loads(line) for line in args.clues.open(encoding="utf-8") if line.strip()]
    if not records:
        print("no clue records", file=sys.stderr)
        return 2

    if args.limit:
        keep = []
        seen: list[str] = []
        for record in records:
            if record["place_qid"] not in seen:
                if len(seen) >= args.limit:
                    continue
                seen.append(record["place_qid"])
            keep.append(record)
        records = keep
        print(f"!! --limit {args.limit}: scoring {len(seen)} places, "
              f"{len(records)} records. Output is PARTIAL.\n")

    # ---- family tagging ----
    for record in records:
        record["property_family"] = FAMILIES.get(record["property"], "other")
    unmapped = sorted({r["property"] for r in records if r["property_family"] == "other"})

    # ---- work plan ----
    by_prop: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_prop[record["property"]].append(record["clue_qid"])
    pairs = {p: sorted(set(q)) for p, q in by_prop.items()}
    labels = sorted({r["clue_label"] for r in records if r["clue_label"]})
    all_qids = sorted({r["clue_qid"] for r in records})

    print(f"records            : {len(records)}")
    print(f"places             : {len({r['place_qid'] for r in records})}")
    print(f"distinct clues     : {len(all_qids)}")
    print(f"(clue, property)   : {sum(len(v) for v in pairs.values())} to score for place ambiguity")
    print(f"distinct labels    : {len(labels)} to score for surface ambiguity")
    print(f"aliases            : {len(all_qids)} entities "
          f"({'P1449/P1705 only' if args.no_altlabel else 'altLabel + P1449/P1705'})")
    if args.dry_run:
        return 0

    use_cache = not args.no_cache
    cache = load_cache(use_cache)
    failures: list[tuple[str, str]] = []

    # ---- place ambiguity, per property ----
    print("\npass 1: place ambiguity", flush=True)
    place_counts: dict[tuple[str, str], tuple[int, int]] = {}
    for prop in sorted(pairs, key=lambda p: -len(pairs[p])):
        want = []
        for qid in pairs[prop]:
            ck = f"pa|{qid}|{prop}"
            if use_cache and ck in cache:
                place_counts[(qid, prop)] = tuple(cache[ck])
            else:
                want.append(qid)
        print(f"  {prop:<8} {len(pairs[prop]):>5} clues, {len(want):>5} to query", flush=True)
        if not want:
            continue
        got = run_chunked(
            want, QID_CHUNK,
            lambda batch, p=prop: place_count_query(p, batch),
            collect_place_counts, f"{prop} place-ambiguity", failures,
        )
        for qid, value in got.items():
            place_counts[(qid, prop)] = value
            cache[f"pa|{qid}|{prop}"] = list(value)
        if use_cache:
            save_cache(cache)

    # ---- surface ambiguity ----
    print("\npass 2: surface ambiguity", flush=True)
    surface: dict[str, int] = {}
    want_labels = []
    for text in labels:
        ck = f"sa|{text}"
        if use_cache and ck in cache:
            surface[text] = cache[ck]
        else:
            want_labels.append(text)
    print(f"  {len(labels)} labels, {len(want_labels)} to query", flush=True)
    if want_labels:
        got = run_chunked(want_labels, LABEL_CHUNK, surface_count_query,
                          collect_surface_counts, "surface", failures)
        for text, value in got.items():
            surface[text] = value
            cache[f"sa|{text}"] = value
        if use_cache:
            save_cache(cache)

    # ---- aliases ----
    print("\npass 3: aliases", flush=True)
    aliases: dict[str, list] = {}
    suffix = "P" if args.no_altlabel else "A"
    want_qids = []
    for qid in all_qids:
        ck = f"al{suffix}|{qid}"
        if use_cache and ck in cache:
            aliases[qid] = cache[ck]
        else:
            want_qids.append(qid)
    print(f"  {len(all_qids)} entities, {len(want_qids)} to query", flush=True)
    if want_qids:
        got = run_chunked(
            want_qids, ALIAS_CHUNK,
            lambda batch: alias_query(batch, not args.no_altlabel),
            collect_aliases, "aliases", failures,
        )
        for qid, value in got.items():
            aliases[qid] = value
            cache[f"al{suffix}|{qid}"] = value
        if use_cache:
            save_cache(cache)

    if use_cache:
        save_cache(cache)

    # ---- attach scores ----
    for record in records:
        counts = place_counts.get((record["clue_qid"], record["property"]))
        record["place_ambiguity"] = None if counts is None else counts[1]
        record["_pa_all"] = None if counts is None else counts[0]
        label = record.get("clue_label")
        record["surface_ambiguity"] = surface.get(label) if label else None
        record["aliases"] = [a for a, _ in aliases.get(record["clue_qid"], [])]
        record["_alias_src"] = [s for _, s in aliases.get(record["clue_qid"], [])]

    if not args.dry_run:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                row = {k: v for k, v in record.items() if not k.startswith("_")}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(args.out)

    # ================================================================
    # Report
    # ================================================================
    scored = [r for r in records if r["place_ambiguity"] is not None]
    unscored = len(records) - len(scored)

    print("\n" + "=" * 70)
    print("PROPERTY FAMILY")
    print(f"\n{'family':<15} {'clues':>7} {'places':>7}  {'properties'}")
    print("-" * 70)
    for family in FAMILY_ORDER:
        group = [r for r in records if r["property_family"] == family]
        if not group:
            continue
        props = sorted({r["property"] for r in group})
        print(f"{family:<15} {len(group):>7} {len({r['place_qid'] for r in group}):>7}  "
              f"{', '.join(props)}")
    if unmapped:
        print(f"\n!! {', '.join(unmapped)} are in the clue set but absent from the "
              f"specified family mapping.")
        print(f"   Tagged 'other' "
              f"({sum(1 for r in records if r['property_family'] == 'other')} records) "
              f"rather than assigned to a family they were not given.")

    print("\n" + "=" * 70)
    print(f"PLACE AMBIGUITY   ({len(scored)} scored"
          + (f", {unscored} unscored" if unscored else "") + ")")
    values = [r["place_ambiguity"] for r in scored]
    heads = [name for name, _ in AMBIGUITY_BUCKETS]
    print(f"\n{'family':<15} {'clues':>7} " + " ".join(f"{h:>6}" for h in heads)
          + f" {'median':>7} {'max':>6}")
    print("-" * 66)
    for family in FAMILY_ORDER + ["ALL"]:
        group = ([r for r in scored if r["property_family"] == family]
                 if family != "ALL" else scored)
        if not group:
            continue
        vals = [r["place_ambiguity"] for r in group]
        row = bucket_row(vals, AMBIGUITY_BUCKETS)
        if family == "ALL":
            print("-" * 66)
        print(f"{family:<15} {len(group):>7} " + " ".join(f"{c:>6}" for c in row)
              + f" {statistics.median(vals):>7.1f} {max(vals):>6}")

    zero = [r for r in scored if r["place_ambiguity"] == 0]
    if zero:
        print(f"\n!! {len(zero)} clues scored 0 -- every value of their relation lacks a "
              f"coordinate,")
        print(f"   so the place they came from is not counted as a place. "
              f"Affects {len({r['place_qid'] for r in zero})} places.")
        for record in zero[:8]:
            print(f"     {record['place_name']:<20} {record['property']:<8} "
                  f"{(record['clue_label'] or '')[:40]}")

    diverge = [r for r in scored if r["_pa_all"] != r["place_ambiguity"]]
    print(f"\ncoordinate restriction changed the count for {len(diverge)}/{len(scored)} clues")
    print(f"{'family':<15} {'changed':>8} {'raw median':>11} {'geo median':>11}")
    print("-" * 48)
    for family in FAMILY_ORDER:
        group = [r for r in scored if r["property_family"] == family]
        if not group:
            continue
        changed = sum(1 for r in group if r["_pa_all"] != r["place_ambiguity"])
        print(f"{family:<15} {changed:>8} "
              f"{statistics.median([r['_pa_all'] for r in group]):>11.1f} "
              f"{statistics.median([r['place_ambiguity'] for r in group]):>11.1f}")

    print("\n" + "=" * 70)
    surf = [r for r in records if r["surface_ambiguity"] is not None]
    print(f"SURFACE AMBIGUITY   ({len(surf)} scored)")
    heads = [name for name, _ in SURFACE_BUCKETS]
    print(f"\n{'family':<15} {'clues':>7} " + " ".join(f"{h:>6}" for h in heads)
          + f" {'median':>7} {'max':>7}")
    print("-" * 68)
    for family in FAMILY_ORDER + ["ALL"]:
        group = ([r for r in surf if r["property_family"] == family]
                 if family != "ALL" else surf)
        if not group:
            continue
        vals = [r["surface_ambiguity"] for r in group]
        row = bucket_row(vals, SURFACE_BUCKETS)
        if family == "ALL":
            print("-" * 68)
        print(f"{family:<15} {len(group):>7} " + " ".join(f"{c:>6}" for c in row)
              + f" {statistics.median(vals):>7.1f} {max(vals):>7}")

    both = [r for r in scored if r["surface_ambiguity"] is not None]
    if both:
        clean = sum(1 for r in both
                    if r["place_ambiguity"] == 1 and r["surface_ambiguity"] <= 1)
        entity_only = sum(1 for r in both
                          if r["place_ambiguity"] == 1 and r["surface_ambiguity"] >= 5)
        print(f"\nunique entity AND unique string : {clean} "
              f"({clean/len(both):.1%})")
        print(f"unique entity but string >= 5   : {entity_only} "
              f"({entity_only/len(both):.1%})  -- clean clue, ambiguous surface")

    with_alias = [r for r in records if r["aliases"]]
    src_counts = Counter(s for r in records for s in r["_alias_src"])
    print(f"\naliases: {len(with_alias)}/{len(records)} clues have at least one "
          f"({sum(len(r['aliases']) for r in records)} total)")
    for src, n in src_counts.most_common():
        print(f"  {src:<10} {n:>6}")

    # ---- per place ----
    places: list[str] = []
    for record in records:
        if record["place_qid"] not in places:
            places.append(record["place_qid"])
    name_of = {r["place_qid"]: r["place_name"] for r in records}

    print("\n" + "=" * 70)
    print(f"PER-PLACE PLACE-AMBIGUITY DISTRIBUTION   (high = >= {args.high})")
    heads = [name for name, _ in AMBIGUITY_BUCKETS]
    print(f"\n{'place':<24} {'qid':<11} {'n':>4} " + " ".join(f"{h:>5}" for h in heads)
          + f" {'min':>4} {'med':>6} {'max':>5}  flag")
    print("-" * 88)
    summary_rows = []
    all_high_places = []
    for qid in places:
        group = [r for r in records if r["place_qid"] == qid]
        vals = [r["place_ambiguity"] for r in group if r["place_ambiguity"] is not None]
        fam = Counter(r["property_family"] for r in group)
        row = bucket_row(vals, AMBIGUITY_BUCKETS) if vals else [0] * len(heads)
        high = bool(vals) and all(v >= args.high for v in vals)
        if high:
            all_high_places.append(qid)
        print(f"{name_of[qid][:24]:<24} {qid:<11} {len(group):>4} "
              + " ".join(f"{c:>5}" for c in row)
              + (f" {min(vals):>4} {statistics.median(vals):>6.1f} {max(vals):>5}"
                 if vals else f" {'-':>4} {'-':>6} {'-':>5}")
              + ("  ALL-HIGH" if high else ""))
        summary_rows.append({
            "place_qid": qid,
            "place_name": name_of[qid],
            "clue_count": len(group),
            "scored": len(vals),
            "pa_min": min(vals) if vals else "",
            "pa_median": f"{statistics.median(vals):.1f}" if vals else "",
            "pa_max": max(vals) if vals else "",
            "all_high_ambiguity": str(high).lower(),
            **{f"n_{f}": fam.get(f, 0) for f in FAMILY_ORDER},
        })

    print("\n" + "=" * 70)
    print(f"PLACES WHOSE CLUES ARE ALL HIGH-AMBIGUITY (>= {args.high}): "
          f"{len(all_high_places)}")
    for qid in all_high_places:
        group = [r for r in records if r["place_qid"] == qid]
        vals = [r["place_ambiguity"] for r in group if r["place_ambiguity"] is not None]
        print(f"  {name_of[qid]:<22} {qid:<11} {len(group):>3} clues, "
              f"min {min(vals)}, median {statistics.median(vals):.1f}")
        for record in sorted(group, key=lambda r: r["place_ambiguity"] or 0)[:3]:
            print(f"      {record['property']:<8} pa={record['place_ambiguity']:<4} "
                  f"{(record['clue_label'] or '')[:44]}")

    # ---- summary CSV ----
    if not args.dry_run:
        fields = (["place_qid", "place_name", "clue_count", "scored",
                   "pa_min", "pa_median", "pa_max", "all_high_ambiguity"]
                  + [f"n_{f}" for f in FAMILY_ORDER])
        with args.summary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(summary_rows)

    if failures:
        print(f"\n!! {len(failures)} lookups failed after splitting; "
              f"their scores are null. Re-run to retry just those.")
        for item, err in failures[:10]:
            print(f"   {item}: {err[:110]}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
    else:
        print(f"\nwrote {args.out}       {len(records)} records")
        print(f"wrote {args.summary}  {len(summary_rows)} places")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
