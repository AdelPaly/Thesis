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

from collections import defaultdict
from typing import Callable

import numpy as np
from scipy import stats as sps

DEFAULT_SEED = 20260917
DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.8


# --------------------------------------------------------------- pairing

def _place_of(r: dict):
    return r.get("place_key") or r.get("place_qid") or f"item:{r.get('item_id')}"


def pair_records(a_records: list[dict], b_records: list[dict],
                 key: tuple[str, ...] = ("item_id", "dataset")) -> list[tuple]:
    """Align two configurations' records on the same items.

    Returns [(place_key, rec_a, rec_b), ...] for items present in both.
    """
    def index(records):
        return {tuple(r.get(k) for k in key): r for r in records}

    ia, ib = index(a_records), index(b_records)
    return [(_place_of(ia[k]), ia[k], ib[k]) for k in sorted(set(ia) & set(ib))]


def _by_place(pairs: list[tuple], value: Callable) -> dict:
    """{place: [(value_a, value_b), ...]}"""
    grouped = defaultdict(list)
    for place, ra, rb in pairs:
        grouped[place].append((value(ra), value(rb)))
    return grouped


# --------------------------------------------------------------- McNemar

def mcnemar(pairs: list[tuple], correct: Callable, unit: str = "place") -> dict:
    """Paired accuracy comparison.

    unit="place" (the default and what should be reported): each place
    contributes ONE outcome per system -- whether that system got the majority
    of the place's items right. Places where the two systems tie are
    concordant and drop out, exactly as in the item-level test. This is the
    clustered form: the pairs being counted are places, which are independent.

    unit="item": the textbook per-item test. Reported only for reference,
    because its pairs are not independent when a place has several items, so
    its p-value is anti-conservative.

    Returns a dict with the discordant counts b and c and an exact two-sided
    binomial p-value (exact rather than chi-square, because b+c is small at
    this sample size).
    """
    if unit not in ("place", "item"):
        raise ValueError(f"unit must be 'place' or 'item', got {unit!r}")
    if unit == "item":
        outcomes = [(bool(correct(ra)), bool(correct(rb))) for _, ra, rb in pairs]
    else:
        outcomes = []
        for vals in _by_place(pairs, correct).values():
            a_rate = sum(bool(v[0]) for v in vals) / len(vals)
            b_rate = sum(bool(v[1]) for v in vals) / len(vals)
            if a_rate == b_rate:
                outcomes.append((True, True))       # concordant, drops out
            else:
                outcomes.append((a_rate > b_rate, b_rate > a_rate))

    b = sum(1 for a, bb in outcomes if a and not bb)     # a right, b wrong
    c = sum(1 for a, bb in outcomes if bb and not a)     # b right, a wrong
    if b + c == 0:
        return {"unit": unit, "n": len(outcomes), "b": 0, "c": 0,
                "p": 1.0, "note": "no discordant pairs"}
    p = sps.binomtest(b, b + c, 0.5, alternative="two-sided").pvalue
    return {"unit": unit, "n": len(outcomes), "b": b, "c": c, "p": float(p)}


# ------------------------------------------------------------- bootstrap

# A ratio metric is a function of a record list that ALSO carries a `terms`
# attribute mapping one record to (numerator, denominator). The bootstrap then
# precomputes per-place sums once and resamples with numpy index arrays.
# Metrics without `terms` fall back to re-evaluating on resampled records.

def _terms_of(metric: Callable):
    return getattr(metric, "terms", None)


def _place_sums(records: list[dict], terms: Callable) -> tuple[list, np.ndarray, np.ndarray]:
    """Per-place (numerator, denominator) sums; places sorted for determinism."""
    num, den = defaultdict(float), defaultdict(float)
    for r in records:
        n, d = terms(r)
        p = _place_of(r)
        num[p] += n
        den[p] += d
    places = sorted(num, key=str)
    return (places, np.array([num[p] for p in places]),
            np.array([den[p] for p in places]))


def _interval(point, replicates: np.ndarray, n_places: int, n_boot: int,
              alpha: float) -> dict:
    out = {"point": point, "lo": None, "hi": None, "se": None, "n_places": n_places,
           "n_boot": int(len(replicates)),
           "n_dropped": n_boot - int(len(replicates))}
    if len(replicates):
        lo, hi = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["lo"], out["hi"] = float(lo), float(hi)
        out["se"] = float(np.std(replicates, ddof=1)) if len(replicates) > 1 else None
    return out


def _generic_replicates(groups: list[list], idx: np.ndarray, value: Callable) -> np.ndarray:
    vals = []
    for row in idx:
        v = value([x for i in row for x in groups[i]])
        if v is not None and not np.isnan(v):
            vals.append(v)
    return np.array(vals)


def cluster_bootstrap_ci(records: list[dict], metric: Callable,
                         n_boot: int = DEFAULT_N_BOOT, alpha: float = DEFAULT_ALPHA,
                         seed: int = DEFAULT_SEED) -> dict:
    """Percentile CI for a metric, resampling PLACES with replacement.

    `metric` takes a list of records and returns a float or None. Places, not
    items, are resampled: a place enters the replicate with all of its items,
    which is what preserves the within-place correlation instead of pretending
    it away. Replicates where the metric is undefined are dropped and counted
    in `n_dropped`.
    """
    terms = _terms_of(metric)
    point = metric(records)
    if terms:
        places, num, den = _place_sums(records, terms)
    else:
        grouped = defaultdict(list)
        for r in records:
            grouped[_place_of(r)].append(r)
        places = sorted(grouped, key=str)
    if len(places) < 2:
        return {"point": point, "lo": None, "hi": None,
                "n_places": len(places), "note": "too few places to bootstrap"}

    idx = np.random.default_rng(seed).integers(0, len(places), size=(n_boot, len(places)))
    if terms:
        n, d = num[idx].sum(axis=1), den[idx].sum(axis=1)
        reps = n[d > 0] / d[d > 0]
    else:
        reps = _generic_replicates([grouped[p] for p in places], idx, metric)
    return _interval(point, reps, len(places), n_boot, alpha)


def paired_effect_ci(pairs: list[tuple], metric: Callable,
                     n_boot: int = DEFAULT_N_BOOT, alpha: float = DEFAULT_ALPHA,
                     seed: int = DEFAULT_SEED) -> dict:
    """CI for the PAIRED difference metric(a) - metric(b), resampling places.

    The effect size with an interval, which is what should be reported --
    a significance test alone says whether a difference is distinguishable
    from zero, not whether it is large enough to matter.

    Both systems share each replicate's place draw; that pairing is what makes
    the interval a paired one. A replicate is dropped, and counted in
    `n_dropped`, if the metric is undefined for either system.
    """
    grouped = defaultdict(list)
    for place, ra, rb in pairs:
        grouped[place].append((ra, rb))
    places = sorted(grouped, key=str)
    if len(places) < 2:
        return {"point": None, "lo": None, "hi": None, "n_places": len(places)}

    def diff(sample_pairs):
        ma = metric([ra for ra, _ in sample_pairs])
        mb = metric([rb for _, rb in sample_pairs])
        return None if ma is None or mb is None else ma - mb

    point = diff([p for place in places for p in grouped[place]])
    idx = np.random.default_rng(seed).integers(0, len(places), size=(n_boot, len(places)))
    terms = _terms_of(metric)
    if terms:
        sums = np.zeros((4, len(places)))
        for i, place in enumerate(places):
            for ra, rb in grouped[place]:
                sums[0, i], sums[1, i] = (sums[0, i] + terms(ra)[0],
                                          sums[1, i] + terms(ra)[1])
                sums[2, i], sums[3, i] = (sums[2, i] + terms(rb)[0],
                                          sums[3, i] + terms(rb)[1])
        na, da, nb, db = (sums[k][idx].sum(axis=1) for k in range(4))
        ok = (da > 0) & (db > 0)
        reps = na[ok] / da[ok] - nb[ok] / db[ok]
    else:
        reps = _generic_replicates([grouped[p] for p in places], idx, diff)
    return _interval(point, reps, len(places), n_boot, alpha)


# ------------------------------------------------------------------ power

def min_detectable_effect(se: float | None, alpha: float = DEFAULT_ALPHA,
                          power: float = DEFAULT_POWER) -> float | None:
    """The smallest true difference a two-sided test at `alpha` would detect
    with probability `power`, given the standard error of the estimate:
    (z_{1-alpha/2} + z_{power}) * se.

    `se` is the place-resampled bootstrap standard error of the paired
    difference, so the effect is in the same units as that difference and
    carries the same clustering. A p-value above alpha next to an effect
    smaller than this says the comparison could not have detected it."""
    if se is None:
        return None
    return float((sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)) * se)


# ------------------------------------------------------------- multiplicity

def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda i: pvalues[i])):
        running = max(running, (m - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


# ------------------------------------------------------------- metric fns

def _exact_terms(r: dict) -> tuple[int, int]:
    if not r.get("scorable_exact"):
        return 0, 0
    return int(bool(r.get("exact"))), 1


def exact_rate(records: list[dict]) -> float | None:
    """Exact match over the records that HAVE a gold QID."""
    num = den = 0
    for r in records:
        n, d = _exact_terms(r)
        num += n
        den += d
    return num / den if den else None


exact_rate.terms = _exact_terms


def fmt_ci(ci: dict | None, pct: bool = False, unit: str = "") -> str:
    """point [lo, hi] with the number of place clusters the interval was
    resampled over: a cluster bootstrap over few clusters under-covers, so the
    count belongs next to every interval."""
    if ci is None or ci.get("point") is None:
        return "-"
    scale = 100 if pct else 1
    point = ci["point"] * scale
    clusters = f" ({ci['n_places']} places)" if ci.get("n_places") is not None else ""
    if ci.get("lo") is None:
        return f"{point:.1f}{unit}{clusters}"
    return (f"{point:.1f} [{ci['lo'] * scale:.1f}, "
            f"{ci['hi'] * scale:.1f}]{unit}{clusters}")


def fmt_p(p: float | None) -> str:
    if p is None:
        return "-"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"
