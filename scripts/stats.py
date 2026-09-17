"""Statistical primitives for the analysis, all clustered by place.

Items about the same place are not independent -- the datasets contain several
posts per place, and the synthetic sets were generated per place. Every test
and interval here therefore treats the PLACE as the unit of resampling or
pairing, never the item. A per-item bootstrap over this data would report
intervals that are too narrow, and a per-item McNemar would report p-values
that are too small, for the same reason.

Imports nothing from src/. Reads nothing from disk. Takes scored records (the
output of score.score_record) and returns numbers.
"""
from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict

from scipy import stats as sps


# --------------------------------------------------------------- pairing

def pair_records(a_records, b_records, key=("item_id", "dataset")):
    """Align two configurations' records on the same items.

    Returns [(place_key, rec_a, rec_b), ...] for items present in both.
    """
    def index(records):
        return {tuple(r.get(k) for k in key): r for r in records}

    ia, ib = index(a_records), index(b_records)
    shared = sorted(set(ia) & set(ib))
    out = []
    for k in shared:
        ra, rb = ia[k], ib[k]
        place = ra.get("place_key") or ra.get("place_qid") or f"item:{k[0]}"
        out.append((place, ra, rb))
    return out


def _by_place(pairs, value):
    """{place: [(value_a, value_b), ...]}"""
    grouped = defaultdict(list)
    for place, ra, rb in pairs:
        grouped[place].append((value(ra), value(rb)))
    return grouped


# --------------------------------------------------------------- McNemar

def mcnemar(pairs, correct, unit="place"):
    """Paired accuracy comparison.

    unit="place" (the default and what should be reported): each place
    contributes ONE outcome per system -- whether that system got the majority
    of the place's items right. Places where the two systems tie are
    concordant and drop out, exactly as in the item-level test. This is the
    clustered form: the pairs being counted are places, which are independent.

    unit="item": the textbook per-item test. Reported only for reference,
    because its pairs are not independent when a place has several items, so
    its p-value is anti-conservative.

    Returns a dict with the discordant counts b and c, the statistic, and an
    exact two-sided binomial p-value (exact rather than chi-square, because b+c
    is small at this sample size).
    """
    if unit == "item":
        outcomes = [(bool(correct(ra)), bool(correct(rb))) for _, ra, rb in pairs]
    elif unit == "place":
        outcomes = []
        for place, vals in _by_place(pairs, correct).items():
            a_rate = sum(bool(v[0]) for v in vals) / len(vals)
            b_rate = sum(bool(v[1]) for v in vals) / len(vals)
            if a_rate == b_rate:
                outcomes.append((True, True))       # concordant, drops out
            else:
                outcomes.append((a_rate > b_rate, b_rate > a_rate))
    else:
        raise ValueError(f"unit must be 'place' or 'item', got {unit!r}")

    b = sum(1 for a, bb in outcomes if a and not bb)     # a right, b wrong
    c = sum(1 for a, bb in outcomes if bb and not a)     # b right, a wrong
    n_disc = b + c
    if n_disc == 0:
        return {"unit": unit, "n": len(outcomes), "b": 0, "c": 0,
                "p": 1.0, "statistic": None,
                "note": "no discordant pairs"}
    p = sps.binomtest(b, n_disc, 0.5, alternative="two-sided").pvalue
    return {"unit": unit, "n": len(outcomes), "b": b, "c": c,
            "p": float(p),
            "statistic": (abs(b - c) - 1) ** 2 / n_disc if n_disc else None}


# --------------------------------------------------------------- Wilcoxon

def wilcoxon_distance(pairs, unit="place"):
    """Paired distance comparison, Wilcoxon signed-rank.

    Only items where BOTH systems produced a distance can be paired -- an
    item with no predicted coordinate has no distance, and substituting a
    number for it (0, or the antipode) is what the Null Island bug did.
    The count of usable pairs is returned so that a comparison resting on few
    pairs is visible rather than implied.

    unit="place" reduces each place to the median distance of its items per
    system before ranking, so the ranked values are independent.
    """
    def dist(r):
        return r.get("distance_km")

    if unit == "item":
        a_vals, b_vals = [], []
        for _, ra, rb in pairs:
            da, db = dist(ra), dist(rb)
            if da is not None and db is not None:
                a_vals.append(da)
                b_vals.append(db)
    else:
        a_vals, b_vals = [], []
        for place, vals in _by_place(pairs, dist).items():
            da = [v[0] for v in vals if v[0] is not None]
            db = [v[1] for v in vals if v[1] is not None]
            if da and db:
                a_vals.append(statistics.median(da))
                b_vals.append(statistics.median(db))

    n = len(a_vals)
    if n < 6:
        return {"unit": unit, "n_pairs": n, "p": None, "statistic": None,
                "median_diff": None,
                "note": f"only {n} usable pair(s); signed-rank not reported"}
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    if all(d == 0 for d in diffs):
        return {"unit": unit, "n_pairs": n, "p": 1.0, "statistic": 0.0,
                "median_diff": 0.0, "note": "identical distances"}
    res = sps.wilcoxon(a_vals, b_vals, zero_method="wilcox",
                       alternative="two-sided")
    return {"unit": unit, "n_pairs": n, "p": float(res.pvalue),
            "statistic": float(res.statistic),
            "median_diff": float(statistics.median(diffs))}


# ------------------------------------------------------------- bootstrap

def cluster_bootstrap_ci(records, metric, n_boot=10000, alpha=0.05, seed=20260917):
    """Percentile CI for a metric, resampling PLACES with replacement.

    `metric` takes a list of records and returns a float or None. Places, not
    items, are resampled: a place enters the replicate with all of its items,
    which is what preserves the within-place correlation instead of pretending
    it away.
    """
    by_place = defaultdict(list)
    for r in records:
        by_place[r.get("place_key") or r.get("place_qid") or r.get("item_id")].append(r)
    places = list(by_place)
    if len(places) < 2:
        return {"point": metric(records), "lo": None, "hi": None,
                "n_places": len(places), "note": "too few places to bootstrap"}

    rng = random.Random(seed)
    point = metric(records)
    replicates = []
    for _ in range(n_boot):
        sample = []
        for _ in range(len(places)):
            sample.extend(by_place[places[rng.randrange(len(places))]])
        value = metric(sample)
        if value is not None and not (isinstance(value, float) and math.isnan(value)):
            replicates.append(value)
    if not replicates:
        return {"point": point, "lo": None, "hi": None, "n_places": len(places)}
    replicates.sort()
    lo = replicates[int((alpha / 2) * len(replicates))]
    hi = replicates[min(len(replicates) - 1, int((1 - alpha / 2) * len(replicates)))]
    return {"point": point, "lo": lo, "hi": hi, "n_places": len(places),
            "n_boot": len(replicates)}


def paired_effect_ci(pairs, metric, n_boot=10000, alpha=0.05, seed=20260917):
    """CI for the PAIRED difference metric(a) - metric(b), resampling places.

    The effect size with an interval, which is what should be reported --
    a significance test alone says whether a difference is distinguishable
    from zero, not whether it is large enough to matter.
    """
    by_place = defaultdict(list)
    for place, ra, rb in pairs:
        by_place[place].append((ra, rb))
    places = list(by_place)
    if len(places) < 2:
        return {"point": None, "lo": None, "hi": None, "n_places": len(places)}

    def diff(sample_pairs):
        a = [ra for ra, _ in sample_pairs]
        b = [rb for _, rb in sample_pairs]
        ma, mb = metric(a), metric(b)
        if ma is None or mb is None:
            return None
        return ma - mb

    all_pairs = [p for place in places for p in by_place[place]]
    point = diff(all_pairs)

    rng = random.Random(seed)
    replicates = []
    for _ in range(n_boot):
        sample = []
        for _ in range(len(places)):
            sample.extend(by_place[places[rng.randrange(len(places))]])
        value = diff(sample)
        if value is not None:
            replicates.append(value)
    if not replicates:
        return {"point": point, "lo": None, "hi": None, "n_places": len(places)}
    replicates.sort()
    lo = replicates[int((alpha / 2) * len(replicates))]
    hi = replicates[min(len(replicates) - 1, int((1 - alpha / 2) * len(replicates)))]
    return {"point": point, "lo": lo, "hi": hi, "n_places": len(places),
            "n_boot": len(replicates)}


# ------------------------------------------------------------- metric fns

def rate(field):
    """Metric function: the share of records where `field` is true."""
    def _metric(records):
        if not records:
            return None
        return sum(1 for r in records if r.get(field)) / len(records)
    return _metric


def median_distance(records):
    d = [r["distance_km"] for r in records if r.get("distance_km") is not None]
    return statistics.median(d) if d else None


def mean_distance(records):
    d = [r["distance_km"] for r in records if r.get("distance_km") is not None]
    return sum(d) / len(d) if d else None


def acc_at(radius_km=161.0):
    def _metric(records):
        scorable = [r for r in records if r.get("scorable_distance")]
        if not scorable:
            return None
        hit = sum(1 for r in scorable
                  if r.get("distance_km") is not None and r["distance_km"] <= radius_km)
        return hit / len(scorable)
    return _metric


def exact_rate(records):
    """Exact match over the records that HAVE a gold QID."""
    have = [r for r in records if r.get("scorable_exact")]
    if not have:
        return None
    return sum(1 for r in have if r.get("exact")) / len(have)


def fmt_ci(ci, pct=False, unit=""):
    if ci is None or ci.get("point") is None:
        return "-"
    scale = 100 if pct else 1
    point = ci["point"] * scale
    if ci.get("lo") is None:
        return f"{point:.1f}{unit}"
    return (f"{point:.1f} [{ci['lo'] * scale:.1f}, "
            f"{ci['hi'] * scale:.1f}]{unit}")


def fmt_p(p):
    if p is None:
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
