#!/usr/bin/env python3
"""Score clue SETS, not clues. A clue is only as good as what it narrows to
once the other clues in the set are also true, and that is a different number
from any per-clue score in 04.

THE INTERSECTION IS THE MEASUREMENT. Every clue is compatible with a set of
places -- the places its relation actually reaches. 04 counted the size of that
set and called it `place_ambiguity`; it did not record WHICH places. A set of
clues is scored by the size of the intersection of those sets: how many places
in the world satisfy all of them at once. That is the number a solver faces.

  The count cannot be reconstructed from the per-clue scores. Two clues with
  place_ambiguity 3 apiece intersect anywhere between 1 and 3 depending on
  whether they are ambiguous about the SAME three places. min(place_ambiguity)
  is an upper bound and nothing more, so this script re-runs 04's inverted
  relation returning ?place instead of COUNT(?place), and intersects the
  identities. The gap between the bound and the truth is reported, because that
  gap is the entire argument for combining clues.

  --offline skips the queries and scores every set at the bound instead. The
  output is then marked `intersection_method: "bound"` and the numbers are
  upper bounds, not intersections. It is a fallback, not an equivalent.

WHICH SETS EXIST. Enumerating every 3-to-5 subset of a 208-clue place is 2.9
billion sets, so eligibility is applied to CLUES first and the survivors are
capped into a pool:

  eligible clue  surface_ambiguity <= --max-surface (20, as specified), a
                 resolvable non-empty place set, and a label. One record per
                 clue ENTITY: a clue reached by two properties (P131 and P276,
                 say) is two records in 04's output, and only the better one
                 (see clue_quality) is kept, because a set may not name the
                 same entity twice. Counted as `duplicate_clue_entity`.
  the pool       --pool clues drawn round-robin across property families, best
                 first, so a place holding four families cannot produce a pool
                 of nothing but P19. This is where "prefer sets that mix
                 property families" is enforced -- at pool construction, where
                 it changes which sets exist, rather than as a filter that
                 would delete every set for the 45 single-family places.
  a set          any 3-, 4- or 5-subset of that pool.

  So `sets_eligible` counts sets over the pool, not over the place's whole clue
  list. With the default pool of 12 that is 1507 sets per place. Raising
  --pool raises it as C(n,3)+C(n,4)+C(n,5); the pool, not the corpus, is the
  thing to turn up.

RESOLVES means the intersection is exactly {this place} -- size 1 AND the right
one. A set can intersect to a single WRONG place if the place itself has
dropped out of one clue's relation since 04 ran; that is counted separately and
loudly rather than being folded into the success number.

Output:
    data/05_clue_sets.jsonl   one record per place: eligibility counts, set
                              count by size, the distribution of set-level
                              ambiguity, flags, and the best --keep sets in
                              full
    data/05_set_cache.json    resolved place sets per (clue, property)

Usage:
    py -3 scripts/05_score_sets.py [--clues data/04_clues_scored.jsonl]
                                   [--max-surface 20] [--pool 12]
                                   [--min-size 3] [--max-size 5] [--keep 5]
                                   [--offline] [--limit N]
                                   [--no-cache] [--dry-run]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from importlib import import_module
from itertools import combinations
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from io_utils import read_jsonl, utf8_stdout, write_jsonl_atomic  # noqa: E402
from jsoncache import JsonCache, cache_key  # noqa: E402
from wdqs import qid_from_uri, value_of  # noqa: E402

# The inverted relations, the chunked query runner and the family mapping are
# 04's, imported rather than restated. If 04's notion of "the places this clue
# reaches" ever changes, the intersections here change with it instead of
# drifting quietly out of step. The module name starts with a digit, so it cannot
# be a plain `import`; this is the only place the numbered module is reached.
_score = import_module("04_score_clues")
place_where = _score.place_where
run_chunked = _score.run_chunked
FAMILY_ORDER = _score.FAMILY_ORDER

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "04_clues_scored.jsonl"
OUT_JSONL = ROOT / "data" / "05_clue_sets.jsonl"
CACHE_PATH = ROOT / "data" / "05_set_cache.json"

CACHE_VERSION = 1
CACHE_PREFIX = "ps"      # cache keys are "ps|<clue_qid>|<property>"

QID_CHUNK = 120          # entities bound into one VALUES block per place-set query

DEFAULT_MAX_SURFACE = 20  # as specified: drop clues whose label denotes more than 20 things
DEFAULT_POOL = 12         # eligible clues per place enumerated over
DEFAULT_MIN_SIZE = 3
DEFAULT_MAX_SIZE = 5
DEFAULT_KEEP = 5          # best sets written out per place
REPORT_WIDTH = 78         # rule width of the console report sections
LIST_EXAMPLES = 8         # example rows shown in each warning list
FAILURE_EXAMPLES = 10     # failed lookups shown at the end
LABEL_CHARS = 40          # clue label characters shown in warnings

# Set-level ambiguity: how many places satisfy the whole set at once.
SET_BUCKETS = (
    ("1", lambda n: n == 1),
    ("2", lambda n: n == 2),
    ("3-4", lambda n: 3 <= n <= 4),
    ("5+", lambda n: n >= 5),
)


# --------------------------------------------------------------------------
# Resolving the compatible places
# --------------------------------------------------------------------------

def place_set_query(prop: str, qids: list[str]) -> str:
    """04's place-ambiguity query with the COUNT taken off.

    The coordinate restriction is kept, so the set returned here is exactly the
    set 04 counted as place_ambiguity -- same relation, same P625 filter. Only
    then is |resolved| == place_ambiguity a check worth reporting rather than a
    comparison of two different questions.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT DISTINCT ?clue ?place WHERE {{
  VALUES ?clue {{ {values} }}
  {place_where(prop)}
  ?place wdt:P625 ?coord .
}}
"""


def collect_place_sets(batch: list[str], rows: list[dict], out: dict) -> None:
    """Fold ?clue/?place rows into out[clue_qid] = sorted place QIDs, for every clue in batch."""
    found: dict[str, list[str]] = {q: [] for q in batch}
    for row in rows:
        clue = value_of(row, "clue")
        place = value_of(row, "place")
        if not clue or not place:
            continue
        qid = qid_from_uri(clue)
        target = qid_from_uri(place)
        if qid in found and target not in found[qid]:
            found[qid].append(target)
    # A clue absent from the result reaches no place carrying a coordinate.
    # That is an empty set, and an empty set is a real answer, not a gap.
    out.update({q: sorted(v) for q, v in found.items()})


def resolve_place_sets(records: list[dict], cache: JsonCache,
                       failures: list[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """(clue_qid, property) -> the places that relation reaches."""
    by_prop: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_prop[record["property"]].add(record["clue_qid"])

    resolved: dict[tuple[str, str], list[str]] = {}

    for prop in sorted(by_prop, key=lambda p: -len(by_prop[p])):
        qids = sorted(by_prop[prop])
        want = []
        for qid in qids:
            key = cache_key(CACHE_PREFIX, qid, prop)
            if key in cache:
                resolved[(qid, prop)] = cache.get(key)
            else:
                want.append(qid)
        print(f"  {prop:<8} {len(qids):>5} clues, {len(want):>5} to query", flush=True)
        if not want:
            continue
        got = run_chunked(
            want, QID_CHUNK,
            lambda batch, p=prop: place_set_query(p, batch),
            collect_place_sets, f"{prop} place-sets", failures,
        )
        for qid, places in got.items():
            resolved[(qid, prop)] = places
            cache.set(cache_key(CACHE_PREFIX, qid, prop), places)
        cache.save()

    cache.save()
    return resolved


# --------------------------------------------------------------------------
# Eligibility
# --------------------------------------------------------------------------

def over_surface(record: dict, max_surface: int) -> bool:
    """True when the clue's label is too ambiguous (or has no surface count at all)."""
    value = record.get("surface_ambiguity")
    return value is None or value > max_surface


def unscored(record: dict) -> bool:
    """True when 04 left the clue without a place_ambiguity or a label."""
    return record.get("place_ambiguity") is None or not record.get("clue_label")


def clue_quality(record: dict) -> tuple:
    """Best clue first: fewest places, cleanest surface, most recognisable.

    sitelinks breaks ties downward-of-ambiguity only. A famous clue is easier
    to recognise but no less ambiguous, so it never outranks a tighter one.
    """
    return (
        record["place_ambiguity"],
        record["surface_ambiguity"],
        -(record.get("sitelinks") or 0),
        record["clue_label"] or "",
        record["clue_qid"],
    )


def dedupe_by_clue_qid(eligible: list[dict]) -> tuple[list[dict], int]:
    """One record per clue entity, keeping the best by clue_quality (PLC-01).

    Eligibility is per (clue, property) record, so an entity reached by two
    properties appears twice. A set must not contain the same entity twice: it
    inflates the set size and n_families and can turn a "3-clue" set into a
    2-clue one. Order of first appearance is preserved. Returns (kept, n_dropped).
    """
    best: dict[str, dict] = {}
    for record in eligible:
        current = best.get(record["clue_qid"])
        if current is None or clue_quality(record) < clue_quality(current):
            best[record["clue_qid"]] = record
    return list(best.values()), len(eligible) - len(best)


@dataclass
class Eligibility:
    """What survived eligibility, and what was dropped along the way."""

    by_place: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    duplicates_by_place: dict[str, int] = field(default_factory=dict)
    place_sets: dict[tuple[str, str], frozenset] = field(default_factory=dict)
    dropped_empty: int = 0
    dropped_failed: int = 0
    drift_size: int = 0
    missing_self: list[dict] = field(default_factory=list)


def classify_eligibility(prefiltered: list[dict],
                         resolved: dict[tuple[str, str], list[str]],
                         offline: bool, dedupe: bool = True) -> Eligibility:
    """Pass B: what resolution disqualifies, then one record per clue entity per place.

    `dedupe=False` reproduces the pre-PLC-01 behaviour (for before/after comparison).
    """
    result = Eligibility()
    for record in prefiltered:
        key = (record["clue_qid"], record["property"])
        if offline:
            result.by_place[record["place_qid"]].append(record)
            continue
        if key not in resolved:
            result.dropped_failed += 1
            continue
        found = resolved[key]
        if not found:
            result.dropped_empty += 1
            continue
        result.place_sets[key] = frozenset(found)
        if len(found) != record["place_ambiguity"]:
            result.drift_size += 1
        if record["place_qid"] not in found:
            result.missing_self.append(record)
        result.by_place[record["place_qid"]].append(record)

    if dedupe:
        for qid, group in list(result.by_place.items()):
            kept, dropped = dedupe_by_clue_qid(group)
            result.by_place[qid] = kept
            result.duplicates_by_place[qid] = dropped
    return result


# --------------------------------------------------------------------------
# Pool construction
# --------------------------------------------------------------------------

def build_pool(eligible: list[dict], size: int) -> list[dict]:
    """Round-robin across families, best first within each.

    Taking a flat top-N would hand a place with 180 birthplaces and 4 landmarks
    a pool of 12 birthplaces, and every set drawn from it would be
    single-family. Interleaving is what makes mixed sets available to
    enumerate at all.
    """
    by_family: dict[str, list[dict]] = defaultdict(list)
    for record in eligible:
        by_family[record["property_family"]].append(record)
    for group in by_family.values():
        group.sort(key=clue_quality)

    families = sorted(by_family, key=lambda f: (FAMILY_ORDER.index(f)
                                                if f in FAMILY_ORDER else len(FAMILY_ORDER)))
    pool: list[dict] = []
    depth = 0
    while len(pool) < size:
        added = False
        for family in families:
            if depth < len(by_family[family]):
                pool.append(by_family[family][depth])
                added = True
                if len(pool) >= size:
                    break
        if not added:
            break
        depth += 1
    pool.sort(key=clue_quality)
    return pool


# --------------------------------------------------------------------------
# Set scoring
# --------------------------------------------------------------------------

def score_set(combo: tuple[dict, ...], place_sets: dict[tuple[str, str], frozenset],
              place_qid: str, bound_only: bool) -> dict:
    """Score one set of clues: intersection size (or the bound), resolves, families."""
    surfaces = [r["surface_ambiguity"] for r in combo]
    pas = [r["place_ambiguity"] for r in combo]
    families = {r["property_family"] for r in combo}
    bound = min(pas)

    if bound_only:
        intersection = bound
        resolves = None
    else:
        shared = place_sets[(combo[0]["clue_qid"], combo[0]["property"])]
        for record in combo[1:]:
            shared = shared & place_sets[(record["clue_qid"], record["property"])]
            if not shared:
                break
        intersection = len(shared)
        resolves = shared == {place_qid}

    return {
        "size": len(combo),
        "intersection": intersection,
        "bound": bound,
        "resolves": resolves,
        "families": sorted(families),
        "n_families": len(families),
        "surface_max": max(surfaces),
        "surface_mean": round(sum(surfaces) / len(surfaces), 1),
        "pa_max": max(pas),
        "clues": [
            {
                "clue_qid": r["clue_qid"],
                "clue_label": r["clue_label"],
                "property": r["property"],
                "property_label": r.get("property_label"),
                "property_family": r["property_family"],
                "place_ambiguity": r["place_ambiguity"],
                "surface_ambiguity": r["surface_ambiguity"],
            }
            for r in combo
        ],
    }


def set_rank(scored: dict) -> tuple:
    """Tightest first, then mixed families, then cleanest surfaces, then small."""
    return (
        scored["intersection"],
        0 if scored["resolves"] is not False else 1,
        -scored["n_families"],
        scored["surface_max"],
        scored["size"],
        scored["pa_max"],
    )


def bucket_row(values: list[int], buckets=SET_BUCKETS) -> list[int]:
    """Counts of `values` per SET_BUCKETS bucket."""
    return [sum(1 for v in values if test(v)) for _, test in buckets]


@dataclass
class ScoreStats:
    """Totals gathered while scoring every place."""

    all_intersections: list[int] = field(default_factory=list)
    all_scored_sets: int = 0
    beat_bound: int = 0
    wrong_singleton: list[str] = field(default_factory=list)


def score_place(qid: str, name: str, group: list[dict], eligible: list[dict],
                duplicates: int, place_sets: dict, sizes: list[int], pool_size: int,
                keep: int, min_size: int, max_surface: int, offline: bool,
                stats: ScoreStats) -> dict:
    """Build the output record for one place and update `stats`."""
    pool = build_pool(eligible, pool_size) if eligible else []
    families_available = sorted({r["property_family"] for r in eligible})

    scored_sets = []
    for k in sizes:
        if len(pool) < k:
            continue
        for combo in combinations(pool, k):
            scored_sets.append(score_set(combo, place_sets, qid, offline))

    intersections = [s["intersection"] for s in scored_sets]
    stats.all_intersections.extend(intersections)
    stats.all_scored_sets += len(scored_sets)
    stats.beat_bound += sum(1 for s in scored_sets if s["intersection"] < s["bound"])

    unique = [s for s in scored_sets if s["intersection"] == 1]
    resolving = [s for s in scored_sets if s["resolves"]]
    if unique and not offline and not resolving:
        stats.wrong_singleton.append(qid)

    scored_sets.sort(key=set_rank)
    buckets = dict(zip([bucket for bucket, _ in SET_BUCKETS], bucket_row(intersections)))

    flags = []
    if len(eligible) < min_size:
        flags.append("too_few_eligible_clues")
    if not scored_sets:
        flags.append("no_sets")
    elif not (resolving if not offline else unique):
        flags.append("no_unique_set")
    if len(families_available) < 2:
        flags.append("single_family")
    if offline:
        flags.append("bound_only")

    carried = sum(1 for r in group if not over_surface(r, max_surface) and not unscored(r))
    return {
        "place_qid": qid,
        "place_name": name,
        "intersection_method": "bound" if offline else "intersection",
        "clues_total": len(group),
        "clues_eligible": len(eligible),
        "excluded": {
            "surface_over_max": sum(1 for r in group if over_surface(r, max_surface)),
            "unscored_or_unlabelled": sum(1 for r in group if unscored(r)),
            # The same entity reached by more than one property: kept once.
            "duplicate_clue_entity": duplicates,
            # Whatever survived the file-level cuts but died at resolution.
            "empty_or_unresolved_place_set": carried - len(eligible) - duplicates,
        },
        "families_available": families_available,
        "pool_size": len(pool),
        "sets_eligible": len(scored_sets),
        "sets_by_size": {str(k): sum(1 for s in scored_sets if s["size"] == k)
                         for k in sizes},
        "sets_mixed_family": sum(1 for s in scored_sets if s["n_families"] >= 2),
        "ambiguity_distribution": buckets,
        "intersection_min": min(intersections) if intersections else None,
        "intersection_median": (round(statistics.median(intersections), 1)
                                if intersections else None),
        "intersection_max": max(intersections) if intersections else None,
        "sets_resolving_to_place": len(resolving) if not offline else None,
        "flags": flags,
        "best_sets": scored_sets[:keep],
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def report_header(records: list[dict], places: list[str], args: argparse.Namespace,
                  sizes: list[int]) -> None:
    """The plan: what will be scored and how."""
    per_pool = sum(comb(args.pool, k) for k in sizes if args.pool >= k)
    print(f"clues              : {len(records)}")
    print(f"places             : {len(places)}")
    print(f"eligibility        : surface_ambiguity <= {args.max_surface}, "
          f"non-empty place set")
    print(f"pool               : <= {args.pool} clues/place, round-robin across families")
    print(f"set sizes          : {', '.join(str(k) for k in sizes)} "
          f"-> <= {per_pool} sets/place, <= {per_pool * len(places)} total")
    print(f"intersection       : "
          + ("min(place_ambiguity) BOUND -- --offline" if args.offline
             else "resolved place sets, intersected"))


def report_resolution(eligibility: Eligibility) -> None:
    """Resolution drops, drift and clues that no longer reach their own place."""
    print(f"\ndropped: empty place set: {eligibility.dropped_empty}   "
          f"unresolved after splitting: {eligibility.dropped_failed}")
    print(f"resolved set size disagrees with stored place_ambiguity for "
          f"{eligibility.drift_size}/{len(eligibility.place_sets)} clues "
          f"(Wikidata moved since 04 ran)")
    missing_self = eligibility.missing_self
    if missing_self:
        print(f"!! {len(missing_self)} clues no longer reach the place they were "
              f"collected for.")
        print(f"   Any set built on them can only intersect to somewhere else. "
              f"Affects {len({r['place_qid'] for r in missing_self})} places.")
        for record in missing_self[:LIST_EXAMPLES]:
            print(f"     {record['place_name'][:20]:<20} {record['property']:<8} "
                  f"{(record['clue_label'] or '')[:LABEL_CHARS]}")


def report_totals(out_records: list[dict], places: list[str], stats: ScoreStats,
                  offline: bool) -> None:
    """Set counts, the set-level ambiguity table, and how often intersection beat the bound."""
    with_sets = [r for r in out_records if r["sets_eligible"]]
    print(f"\n{stats.all_scored_sets} sets scored across {len(with_sets)}/{len(places)} places")

    if stats.all_intersections:
        heads = [name for name, _ in SET_BUCKETS]
        row = bucket_row(stats.all_intersections)
        print("\nSET-LEVEL AMBIGUITY (all sets)")
        print(f"{'places compatible':<20} " + " ".join(f"{h:>8}" for h in heads))
        print("-" * 56)
        print(f"{'sets':<20} " + " ".join(f"{c:>8}" for c in row))
        print(f"{'share':<20} "
              + " ".join(f"{c/len(stats.all_intersections):>7.1%}" for c in row))
        print(f"\nmedian {statistics.median(stats.all_intersections):.1f}   "
              f"max {max(stats.all_intersections)}")

    if not offline:
        print(f"\nintersection beat the min(place_ambiguity) bound for "
              f"{stats.beat_bound}/{stats.all_scored_sets} sets "
              f"({stats.beat_bound/max(stats.all_scored_sets,1):.1%})")
        print("  -- sets where the clues are ambiguous about DIFFERENT places, "
              "which is the")
        print("     whole reason to combine them. The rest were already pinned by "
              "one clue.")


def report_family_mix(out_records: list[dict]) -> None:
    """Does mixing property families tighten the intersection (best sets only)?"""
    print("\n" + "=" * REPORT_WIDTH)
    print("FAMILY MIX  (does mixing families actually tighten the intersection?)")
    by_mix: dict[int, list[int]] = defaultdict(list)
    for record in out_records:
        for scored in record["best_sets"]:
            by_mix[scored["n_families"]].append(scored["intersection"])
    print(f"\n{'families in set':<18} {'best sets':>10} {'median':>8} {'= 1':>8}")
    print("-" * 48)
    for n_fam in sorted(by_mix):
        vals = by_mix[n_fam]
        print(f"{n_fam:<18} {len(vals):>10} {statistics.median(vals):>8.1f} "
              f"{sum(1 for v in vals if v == 1):>8}")
    print("\n(best sets only -- the ranking already prefers mixed sets, so this is "
          "a\n check on what got picked, not an unbiased comparison.)")


def report_per_place(out_records: list[dict]) -> None:
    """One table row per place."""
    print("\n" + "=" * REPORT_WIDTH)
    print("PER PLACE")
    heads = [name for name, _ in SET_BUCKETS]
    print(f"\n{'place':<22} {'qid':<11} {'clue':>5} {'elig':>5} {'pool':>5} {'sets':>6} "
          + " ".join(f"{h:>6}" for h in heads) + f" {'med':>5}  flags")
    print("-" * 108)
    for record in out_records:
        buckets = [record["ambiguity_distribution"][h] for h in heads]
        median = record["intersection_median"]
        print(f"{record['place_name'][:22]:<22} {record['place_qid']:<11} "
              f"{record['clues_total']:>5} {record['clues_eligible']:>5} "
              f"{record['pool_size']:>5} {record['sets_eligible']:>6} "
              + " ".join(f"{c:>6}" for c in buckets)
              + (f" {median:>5.1f}" if median is not None else f" {'-':>5}")
              + ("  " + ",".join(record["flags"]) if record["flags"] else ""))


def report_unresolved(out_records: list[dict], places: list[str], min_size: int) -> None:
    """The flag that matters: places with no set that intersects to a single place."""
    unresolved = [r for r in out_records if "no_unique_set" in r["flags"]
                  or "no_sets" in r["flags"]]
    print("\n" + "=" * REPORT_WIDTH)
    print(f"PLACES WITH NO SET THAT INTERSECTS TO A SINGLE PLACE: "
          f"{len(unresolved)}/{len(places)}")
    if not unresolved:
        return
    print(f"\n{'place':<22} {'qid':<11} {'elig':>5} {'sets':>6} {'best':>5}  why")
    print("-" * REPORT_WIDTH)
    for record in unresolved:
        best = record["best_sets"][0] if record["best_sets"] else None
        if "no_sets" in record["flags"]:
            why = (f"only {record['clues_eligible']} eligible clues, "
                   f"need {min_size}")
        else:
            why = (f"tightest set still leaves {best['intersection']} places"
                   if best else "")
        print(f"{record['place_name'][:22]:<22} {record['place_qid']:<11} "
              f"{record['clues_eligible']:>5} {record['sets_eligible']:>6} "
              + (f"{best['intersection']:>5}" if best else f"{'-':>5}")
              + f"  {why}")


def report_starved(out_records: list[dict], min_size: int, max_surface: int) -> None:
    """Places short of eligible clues: starved of supply or cut by the surface filter."""
    starved = [r for r in out_records if "too_few_eligible_clues" in r["flags"]]
    if not starved:
        return
    # Two different problems wearing the same flag, and they have different
    # fixes. Blaming the surface cut for a place that arrived from 04 with
    # one clue would send someone to tune the wrong knob.
    thin = [r for r in starved if r["clues_total"] < min_size]
    cut = [r for r in starved if r["clues_total"] >= min_size]
    print(f"\n{len(starved)} of those are starved of clues rather than of "
          f"distinctness:")
    if thin:
        print(f"   {len(thin)} arrived from 04 with under {min_size} clues at "
              f"all. No filter here touched them;")
        print("     more clues from 02/03 is the only fix.")
    if cut:
        print(f"   {len(cut)} had enough clues until the {max_surface}-surface "
              f"cut took them under {min_size}:")
        for record in cut[:LIST_EXAMPLES]:
            print(f"     {record['place_name'][:22]:<22} "
                  f"{record['clues_total']} clues -> {record['clues_eligible']} "
                  f"eligible")
        print("     Loosening --max-surface would recover these, at the cost of "
              "clues whose\n     labels denote more than one thing.")


def report_tail(out_records: list[dict], places: list[str], stats: ScoreStats,
                name_of: dict[str, str], failures: list[tuple[str, str]],
                offline: bool) -> None:
    """Wrong singletons, single-family places, failed lookups and the offline warning."""
    if stats.wrong_singleton:
        print(f"\n!! {len(stats.wrong_singleton)} places have a set intersecting to "
              f"exactly one place that is NOT them:")
        for qid in stats.wrong_singleton[:LIST_EXAMPLES]:
            print(f"     {name_of[qid]:<22} {qid}")
        print("   Those sets identify somewhere else. Not counted as resolving.")

    single_family = [r for r in out_records if "single_family" in r["flags"]]
    print(f"\n{len(single_family)}/{len(places)} places have eligible clues from only "
          f"one property family,")
    print(f"   so no mixed set exists to prefer. "
          f"{sum(1 for r in out_records if r['sets_mixed_family'])} places have at "
          f"least one mixed set.")

    if failures:
        print(f"\n!! {len(failures)} place-set lookups failed after splitting; those "
              f"clues were dropped from eligibility. Re-run to retry just those.")
        for item, err in failures[:FAILURE_EXAMPLES]:
            print(f"   {item}: {err[:110]}")

    if offline:
        print("\n!! --offline: every 'intersection' above is min(place_ambiguity), an "
              "UPPER BOUND.")
        print("   Real intersections are smaller. Re-run without --offline before "
              "trusting any of it.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(description="Score candidate clue sets by intersection.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--max-surface", type=int, default=DEFAULT_MAX_SURFACE,
                        help=f"drop clues whose surface_ambiguity exceeds this "
                             f"(default {DEFAULT_MAX_SURFACE})")
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help=f"eligible clues per place to enumerate over "
                             f"(default {DEFAULT_POOL})")
    parser.add_argument("--min-size", type=int, default=DEFAULT_MIN_SIZE)
    parser.add_argument("--max-size", type=int, default=DEFAULT_MAX_SIZE)
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"best sets written out per place (default {DEFAULT_KEEP})")
    parser.add_argument("--offline", action="store_true",
                        help="do not query; score sets at the min(place_ambiguity) "
                             "upper bound instead of the true intersection")
    parser.add_argument("--limit", type=int,
                        help="only process the first N places (partial output; for testing)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the cache and do not write it")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and exit without querying or writing")
    return parser


def load_records(args: argparse.Namespace) -> tuple[list[dict], list[str]] | None:
    """The scored clue records and their places (in first-seen order), or None if unusable."""
    if not args.clues.exists():
        print(f"scored clue file not found: {args.clues}", file=sys.stderr)
        return None
    records = read_jsonl(args.clues)
    if not records:
        print("no clue records", file=sys.stderr)
        return None

    places = list(dict.fromkeys(r["place_qid"] for r in records))
    if args.limit:
        places = places[:args.limit]
        keep = set(places)
        records = [r for r in records if r["place_qid"] in keep]
        print(f"!! --limit {args.limit}: {len(places)} places, {len(records)} clues. "
              f"Output is PARTIAL.\n")
    return records, places


def prefilter(records: list[dict], max_surface: int) -> list[dict]:
    """Pass A: print what the file already disqualifies; return what is carried forward."""
    n_surface = sum(1 for r in records if over_surface(r, max_surface))
    n_unscored = sum(1 for r in records if unscored(r))
    carried = [r for r in records if not over_surface(r, max_surface) and not unscored(r)]
    print(f"\ndropped: surface > {max_surface}: {n_surface}   "
          f"unscored/unlabelled: {n_unscored}   "
          f"carried forward: {len(carried)}")
    return carried


def score_all(places: list[str], records: list[dict], eligibility: Eligibility,
              sizes: list[int], args: argparse.Namespace) -> tuple[list[dict], ScoreStats]:
    """Enumerate and score the sets of every place."""
    name_of = {r["place_qid"]: r["place_name"] for r in records}
    by_place: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_place[record["place_qid"]].append(record)

    stats = ScoreStats()
    out_records = [
        score_place(qid, name_of[qid], by_place[qid], eligibility.by_place.get(qid, []),
                    eligibility.duplicates_by_place.get(qid, 0), eligibility.place_sets,
                    sizes, args.pool, args.keep, args.min_size, args.max_surface,
                    args.offline, stats)
        for qid in places
    ]
    return out_records, stats


def main(argv: list[str] | None = None) -> int:
    """Resolve place sets, score every place's sets, write the JSONL and print the report."""
    utf8_stdout()
    args = build_parser().parse_args(argv)

    if args.min_size < 1 or args.max_size < args.min_size:
        print("--min-size/--max-size must satisfy 1 <= min <= max", file=sys.stderr)
        return 2
    loaded = load_records(args)
    if loaded is None:
        return 2
    records, places = loaded
    name_of = {r["place_qid"]: r["place_name"] for r in records}
    sizes = list(range(args.min_size, args.max_size + 1))

    report_header(records, places, args, sizes)
    prefiltered = prefilter(records, args.max_surface)

    if args.dry_run:
        print("\n--dry-run: nothing queried, nothing written")
        return 0

    # ---- resolve the compatible-place sets ----
    failures: list[tuple[str, str]] = []
    resolved: dict[tuple[str, str], list[str]] = {}
    if not args.offline:
        print("\nresolving compatible places", flush=True)
        cache = JsonCache(CACHE_PATH, CACHE_VERSION, enabled=not args.no_cache)
        resolved = resolve_place_sets(prefiltered, cache, failures)

    eligibility = classify_eligibility(prefiltered, resolved, args.offline)
    if not args.offline:
        report_resolution(eligibility)

    print("\n" + "=" * REPORT_WIDTH)
    print("SETS", flush=True)
    out_records, stats = score_all(places, records, eligibility, sizes, args)

    write_jsonl_atomic(args.out, out_records)

    report_totals(out_records, places, stats, args.offline)
    report_family_mix(out_records)
    report_per_place(out_records)
    report_unresolved(out_records, places, args.min_size)
    report_starved(out_records, args.min_size, args.max_surface)
    report_tail(out_records, places, stats, name_of, failures, args.offline)

    print(f"\nwrote {args.out}   {len(out_records)} places")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
