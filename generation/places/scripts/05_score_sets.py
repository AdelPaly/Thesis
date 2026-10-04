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
                 resolvable non-empty place set, and a label.
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
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module

_clues = import_module("02_clues")
_score = import_module("04_score_clues")

QueryError = _clues.QueryError
sparql = _clues.sparql
value_of = _clues.value_of
qid_from_uri = _clues.qid_from_uri

# The inverted relations and the family mapping are 04's, imported rather than
# restated. If 04's notion of "the places this clue reaches" ever changes, the
# intersections here change with it instead of drifting quietly out of step.
place_where = _score.place_where
run_chunked = _score.run_chunked
FAMILY_ORDER = _score.FAMILY_ORDER

ROOT = Path(__file__).resolve().parent.parent
CLUES = ROOT / "data" / "04_clues_scored.jsonl"
OUT_JSONL = ROOT / "data" / "05_clue_sets.jsonl"
CACHE_PATH = ROOT / "data" / "05_set_cache.json"

CACHE_VERSION = 1

QID_CHUNK = 120      # entities bound into one VALUES block per place-set query

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


def resolve_place_sets(records: list[dict], use_cache: bool,
                       failures: list[tuple[str, str]]) -> dict[tuple[str, str], list[str]]:
    """(clue_qid, property) -> the places that relation reaches."""
    by_prop: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_prop[record["property"]].add(record["clue_qid"])

    cache = load_cache(use_cache)
    resolved: dict[tuple[str, str], list[str]] = {}

    for prop in sorted(by_prop, key=lambda p: -len(by_prop[p])):
        qids = sorted(by_prop[prop])
        want = []
        for qid in qids:
            key = f"ps|{qid}|{prop}"
            if use_cache and key in cache:
                resolved[(qid, prop)] = cache[key]
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
            cache[f"ps|{qid}|{prop}"] = places
        if use_cache:
            save_cache(cache)

    if use_cache:
        save_cache(cache)
    return resolved


# --------------------------------------------------------------------------
# Pool construction
# --------------------------------------------------------------------------

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
    return [sum(1 for v in values if test(v)) for _, test in buckets]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover
        pass

    parser = argparse.ArgumentParser(description="Score candidate clue sets by intersection.")
    parser.add_argument("--clues", type=Path, default=CLUES)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--max-surface", type=int, default=20,
                        help="drop clues whose surface_ambiguity exceeds this (default 20)")
    parser.add_argument("--pool", type=int, default=12,
                        help="eligible clues per place to enumerate over (default 12)")
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=5)
    parser.add_argument("--keep", type=int, default=5,
                        help="best sets written out per place (default 5)")
    parser.add_argument("--offline", action="store_true",
                        help="do not query; score sets at the min(place_ambiguity) "
                             "upper bound instead of the true intersection")
    parser.add_argument("--limit", type=int,
                        help="only process the first N places (partial output; for testing)")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the plan and exit without querying or writing")
    args = parser.parse_args(argv)

    if args.min_size < 1 or args.max_size < args.min_size:
        print("--min-size/--max-size must satisfy 1 <= min <= max", file=sys.stderr)
        return 2
    if not args.clues.exists():
        print(f"scored clue file not found: {args.clues}", file=sys.stderr)
        return 2

    records = [json.loads(line) for line in args.clues.open(encoding="utf-8") if line.strip()]
    if not records:
        print("no clue records", file=sys.stderr)
        return 2

    places: list[str] = []
    for record in records:
        if record["place_qid"] not in places:
            places.append(record["place_qid"])
    if args.limit:
        places = places[:args.limit]
        keep = set(places)
        records = [r for r in records if r["place_qid"] in keep]
        print(f"!! --limit {args.limit}: {len(places)} places, {len(records)} clues. "
              f"Output is PARTIAL.\n")
    name_of = {r["place_qid"]: r["place_name"] for r in records}

    sizes = list(range(args.min_size, args.max_size + 1))
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

    # ---- eligibility, pass A: what the file already disqualifies ----
    def over_surface(record: dict) -> bool:
        value = record.get("surface_ambiguity")
        return value is None or value > args.max_surface

    def unscored(record: dict) -> bool:
        return record.get("place_ambiguity") is None or not record.get("clue_label")

    n_surface = sum(1 for r in records if over_surface(r))
    n_unscored = sum(1 for r in records if unscored(r))
    prefiltered = [r for r in records if not over_surface(r) and not unscored(r)]
    print(f"\ndropped: surface > {args.max_surface}: {n_surface}   "
          f"unscored/unlabelled: {n_unscored}   "
          f"carried forward: {len(prefiltered)}")

    if args.dry_run:
        print("\n--dry-run: nothing queried, nothing written")
        return 0

    # ---- resolve the compatible-place sets ----
    failures: list[tuple[str, str]] = []
    resolved: dict[tuple[str, str], list[str]] = {}
    if not args.offline:
        print("\nresolving compatible places", flush=True)
        resolved = resolve_place_sets(prefiltered, not args.no_cache, failures)

    # ---- eligibility, pass B: what resolution disqualifies ----
    eligible_by_place: dict[str, list[dict]] = defaultdict(list)
    dropped_empty = 0
    dropped_failed = 0
    place_sets: dict[tuple[str, str], frozenset] = {}
    drift_size = 0
    missing_self: list[dict] = []

    for record in prefiltered:
        key = (record["clue_qid"], record["property"])
        if args.offline:
            eligible_by_place[record["place_qid"]].append(record)
            continue
        if key not in resolved:
            dropped_failed += 1
            continue
        found = resolved[key]
        if not found:
            dropped_empty += 1
            continue
        place_sets[key] = frozenset(found)
        if len(found) != record["place_ambiguity"]:
            drift_size += 1
        if record["place_qid"] not in found:
            missing_self.append(record)
        eligible_by_place[record["place_qid"]].append(record)

    if not args.offline:
        print(f"\ndropped: empty place set: {dropped_empty}   "
              f"unresolved after splitting: {dropped_failed}")
        print(f"resolved set size disagrees with stored place_ambiguity for "
              f"{drift_size}/{len(place_sets)} clues (Wikidata moved since 04 ran)")
        if missing_self:
            print(f"!! {len(missing_self)} clues no longer reach the place they were "
                  f"collected for.")
            print(f"   Any set built on them can only intersect to somewhere else. "
                  f"Affects {len({r['place_qid'] for r in missing_self})} places.")
            for record in missing_self[:8]:
                print(f"     {record['place_name'][:20]:<20} {record['property']:<8} "
                      f"{(record['clue_label'] or '')[:40]}")

    # ================================================================
    # Enumerate and score
    # ================================================================
    print("\n" + "=" * 78)
    print("SETS", flush=True)

    out_records = []
    all_intersections: list[int] = []
    all_scored_sets = 0
    beat_bound = 0
    wrong_singleton: list[str] = []

    by_place: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_place[record["place_qid"]].append(record)

    for qid in places:
        group = by_place[qid]
        eligible = eligible_by_place.get(qid, [])
        pool = build_pool(eligible, args.pool) if eligible else []
        families_available = sorted({r["property_family"] for r in eligible})

        scored_sets = []
        for k in sizes:
            if len(pool) < k:
                continue
            for combo in combinations(pool, k):
                scored_sets.append(score_set(combo, place_sets, qid, args.offline))

        intersections = [s["intersection"] for s in scored_sets]
        all_intersections.extend(intersections)
        all_scored_sets += len(scored_sets)
        beat_bound += sum(1 for s in scored_sets if s["intersection"] < s["bound"])

        unique = [s for s in scored_sets if s["intersection"] == 1]
        resolving = [s for s in scored_sets if s["resolves"]]
        if unique and not args.offline and not resolving:
            wrong_singleton.append(qid)

        scored_sets.sort(key=set_rank)
        buckets = dict(zip([name for name, _ in SET_BUCKETS], bucket_row(intersections)))

        flags = []
        if len(eligible) < args.min_size:
            flags.append("too_few_eligible_clues")
        if not scored_sets:
            flags.append("no_sets")
        elif not (resolving if not args.offline else unique):
            flags.append("no_unique_set")
        if len(families_available) < 2:
            flags.append("single_family")
        if args.offline:
            flags.append("bound_only")

        out_records.append({
            "place_qid": qid,
            "place_name": name_of[qid],
            "intersection_method": "bound" if args.offline else "intersection",
            "clues_total": len(group),
            "clues_eligible": len(eligible),
            "excluded": {
                "surface_over_max": sum(1 for r in group if over_surface(r)),
                "unscored_or_unlabelled": sum(1 for r in group if unscored(r)),
                # Whatever survived the file-level cuts but died at resolution.
                "empty_or_unresolved_place_set": (
                    sum(1 for r in group if not over_surface(r) and not unscored(r))
                    - len(eligible)),
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
            "sets_resolving_to_place": (len(resolving) if not args.offline
                                        else None),
            "flags": flags,
            "best_sets": scored_sets[:args.keep],
        })

    # ---- write ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in out_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(args.out)

    # ================================================================
    # Report
    # ================================================================
    with_sets = [r for r in out_records if r["sets_eligible"]]
    print(f"\n{all_scored_sets} sets scored across {len(with_sets)}/{len(places)} places")

    if all_intersections:
        heads = [name for name, _ in SET_BUCKETS]
        row = bucket_row(all_intersections)
        print(f"\nSET-LEVEL AMBIGUITY (all sets)")
        print(f"{'places compatible':<20} " + " ".join(f"{h:>8}" for h in heads))
        print("-" * 56)
        print(f"{'sets':<20} " + " ".join(f"{c:>8}" for c in row))
        print(f"{'share':<20} "
              + " ".join(f"{c/len(all_intersections):>7.1%}" for c in row))
        print(f"\nmedian {statistics.median(all_intersections):.1f}   "
              f"max {max(all_intersections)}")

    if not args.offline:
        print(f"\nintersection beat the min(place_ambiguity) bound for "
              f"{beat_bound}/{all_scored_sets} sets "
              f"({beat_bound/max(all_scored_sets,1):.1%})")
        print("  -- sets where the clues are ambiguous about DIFFERENT places, "
              "which is the")
        print("     whole reason to combine them. The rest were already pinned by "
              "one clue.")

    # ---- family mix ----
    print("\n" + "=" * 78)
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

    # ---- per place ----
    print("\n" + "=" * 78)
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

    # ---- the flag that matters ----
    unresolved = [r for r in out_records if "no_unique_set" in r["flags"]
                  or "no_sets" in r["flags"]]
    print("\n" + "=" * 78)
    print(f"PLACES WITH NO SET THAT INTERSECTS TO A SINGLE PLACE: "
          f"{len(unresolved)}/{len(places)}")
    if unresolved:
        print(f"\n{'place':<22} {'qid':<11} {'elig':>5} {'sets':>6} {'best':>5}  why")
        print("-" * 78)
        for record in unresolved:
            best = record["best_sets"][0] if record["best_sets"] else None
            if "no_sets" in record["flags"]:
                why = (f"only {record['clues_eligible']} eligible clues, "
                       f"need {args.min_size}")
            else:
                why = (f"tightest set still leaves {best['intersection']} places"
                       if best else "")
            print(f"{record['place_name'][:22]:<22} {record['place_qid']:<11} "
                  f"{record['clues_eligible']:>5} {record['sets_eligible']:>6} "
                  + (f"{best['intersection']:>5}" if best else f"{'-':>5}")
                  + f"  {why}")

    starved = [r for r in out_records if "too_few_eligible_clues" in r["flags"]]
    if starved:
        # Two different problems wearing the same flag, and they have different
        # fixes. Blaming the surface cut for a place that arrived from 04 with
        # one clue would send someone to tune the wrong knob.
        thin = [r for r in starved if r["clues_total"] < args.min_size]
        cut = [r for r in starved if r["clues_total"] >= args.min_size]
        print(f"\n{len(starved)} of those are starved of clues rather than of "
              f"distinctness:")
        if thin:
            print(f"   {len(thin)} arrived from 04 with under {args.min_size} clues at "
                  f"all. No filter here touched them;")
            print(f"     more clues from 02/03 is the only fix.")
        if cut:
            print(f"   {len(cut)} had enough clues until the {args.max_surface}-surface "
                  f"cut took them under {args.min_size}:")
            for record in cut[:8]:
                print(f"     {record['place_name'][:22]:<22} "
                      f"{record['clues_total']} clues -> {record['clues_eligible']} "
                      f"eligible")
            print(f"     Loosening --max-surface would recover these, at the cost of "
                  f"clues whose\n     labels denote more than one thing.")

    if wrong_singleton:
        print(f"\n!! {len(wrong_singleton)} places have a set intersecting to exactly "
              f"one place that is NOT them:")
        for qid in wrong_singleton[:8]:
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
        for item, err in failures[:10]:
            print(f"   {item}: {err[:110]}")

    if args.offline:
        print("\n!! --offline: every 'intersection' above is min(place_ambiguity), an "
              "UPPER BOUND.")
        print("   Real intersections are smaller. Re-run without --offline before "
              "trusting any of it.")

    print(f"\nwrote {args.out}   {len(out_records)} places")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
