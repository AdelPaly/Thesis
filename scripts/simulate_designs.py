"""Offline designs: cascades, gates, fallbacks and oracles built from saved predictions.

    python scripts/simulate_designs.py                  # test split, all datasets
    python scripts/simulate_designs.py --split all
    python scripts/simulate_designs.py --out ''         # print only

No model, no network. Each design below combines the stored predictions of two
existing configurations item by item, so it costs nothing to try and needs no
new run. The handlers in src/pipeline.py stay as they are; this is the cheaper
of the two options in the supervisor's notes (2.3): simulate the composition
rules offline instead of building a rule-executing chain.

The designs are an EXPLORATORY family (dated below), not part of the
pre-registered comparisons.yaml family: they are reported with intervals and
are not Holm-corrected together with it.

Kinds of design
---------------
cascade   A, and only when A returns nothing, B.
gate      [mention found?] yes: A / no: B. The gate reads A's stored
          ner_spans (after A's type filter), never the gold condition.
override  [B's top candidate score >= t?] yes: B / no: A (B also when A is
          empty). t is tuned on the DEV split to maximise exact accuracy and
          applied once on test. The score is B's own scale (cosine for dense
          retrieval), so t is per design, never shared.
fallback  A; when A returns nothing, only B's country and continent (no QID,
          no coordinates): a partial answer instead of none.
oracle    correct when A or B is correct. NOT a design: the upper bound of
          what any rule choosing between A and B could reach, i.e. how far the
          two fail on different items.

Rules with a choice (override's t) see only dev; everything else has no free
parameter. Dev sets are small (20-45 items per dataset), so a tuned t is
noisy; it is reported next to its dev score.

For every design and dataset the table gives exact/country/continent and the
no-result rate, and the paired exact difference against the better of its two
inputs (chosen on dev, so test does not pick the baseline), with a
place-resampled 95% interval (scripts/stats.py).

Output: results/summary/table_simulated_designs_<split>.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from score import attach_split, find_prediction_files, load_predictions, score_record  # noqa: E402
import stats  # noqa: E402

FAMILY = "exploratory-2026-11-14"
DEFAULT_OUT = HERE.parent / "results" / "summary"
N_BOOT = 2000

# (name, kind, A, B, question)
DESIGNS = [
    ("cascade:refined>dense", "cascade", "refined", "dense_retrieval",
     "do linking and retrieval complement each other?"),
    ("cascade:blink>dense", "cascade", "blink", "dense_retrieval",
     "the same with the other linker"),
    ("cascade:refined>blink", "cascade", "refined", "blink",
     "do the two linkers fail on different items?"),
    ("cascade:refined>grounded_gemma", "cascade", "refined", "grounded_gemma",
     "linking, then the grounded LLM when linking finds nothing"),
    ("gate:refined|dense", "gate", "refined", "dense_retrieval",
     "when is mention detection unnecessary (no span -> retrieval)?"),
    ("override:dense>=t|refined", "override", "refined", "dense_retrieval",
     "does a confident retriever beat the linker?"),
    ("fallback:refined>dense_country", "fallback", "refined", "dense_retrieval",
     "is a country-level answer worth giving when linking abstains?"),
    ("oracle:refined|dense", "oracle", "refined", "dense_retrieval",
     "upper bound: linking or retrieval"),
    ("oracle:refined|blink", "oracle", "refined", "blink",
     "upper bound: either linker"),
    ("oracle:refined|geoparser", "oracle", "refined", "geoparser_irchel",
     "upper bound: linker or geoparser"),
]

PRED_FIELDS = ("predicted_qid", "predicted_latlon", "predicted_country", "predicted_continent")


def has_prediction(rec: dict) -> bool:
    return bool(rec.get("predicted_qid") or rec.get("predicted_latlon")
                or rec.get("predicted_continent"))


def top_score(rec: dict) -> float | None:
    scores = [c.get("score") for c in rec.get("candidate_places") or []
              if isinstance(c, dict) and isinstance(c.get("score"), (int, float))]
    return max(scores) if scores else None


def _take(base: dict, src: dict, name: str, fields=PRED_FIELDS) -> dict:
    out = {**base, "config_name": name}
    for f in PRED_FIELDS:
        out[f] = src.get(f) if f in fields else None
    out["error"] = None
    return out


def combine(kind: str, name: str, a: dict, b: dict, t: float | None = None) -> dict:
    """One simulated record from A's and B's records for the same item."""
    if kind == "cascade":
        return _take(a, a if has_prediction(a) else b, name)
    if kind == "gate":
        return _take(a, a if a.get("ner_spans") else b, name)
    if kind == "override":
        s = top_score(b)
        use_b = not has_prediction(a) or (s is not None and t is not None and s >= t)
        return _take(a, b if use_b else a, name)
    if kind == "fallback":
        if has_prediction(a):
            return _take(a, a, name)
        return _take(a, b, name, fields=("predicted_country", "predicted_continent"))
    if kind == "oracle":
        sa, sb = score_record(a), score_record(b)
        return _take(a, b if (sb["exact"] and not sa["exact"]) else a, name)
    raise ValueError(kind)


def exact(records: list[dict]) -> float | None:
    return stats.exact_rate([score_record(r) for r in records])


def tune_threshold(pairs: list[tuple[dict, dict]], name: str) -> tuple[float | None, float | None]:
    """The t (one of B's observed top scores, or 'never') maximising dev exact."""
    cands = sorted({s for _, b in pairs if (s := top_score(b)) is not None})
    best_t, best = None, exact([combine("override", name, a, b, None) for a, b in pairs])
    for t in cands:
        v = exact([combine("override", name, a, b, t) for a, b in pairs])
        if v is not None and (best is None or v > best):
            best_t, best = t, v
    return best_t, best


def metrics(scored: list[dict]) -> dict:
    n = len(scored)
    def rate(flag, scorable):
        den = sum(s[scorable] for s in scored)
        return sum(s[flag] for s in scored if s[scorable]) / den if den else None
    return {"n": n, "exact": rate("exact", "scorable_exact"),
            "country": rate("country_correct", "scorable_country"),
            "continent": rate("continent_correct", "scorable_continent"),
            "no_result_rate": sum(s["no_result"] for s in scored) / n if n else None}


def index(records: list[dict]) -> dict:
    by = defaultdict(dict)
    for r in records:
        by[r["config_name"]][(r["dataset"], r["item_id"])] = r
    return by


def run(split: str, out: Path | None) -> list[dict]:
    records = attach_split(load_predictions(find_prediction_files()))
    by = index(records)
    rows = []
    for name, kind, a_cfg, b_cfg, question in DESIGNS:
        if a_cfg not in by or b_cfg not in by:
            print(f"skip {name}: no predictions for {a_cfg if a_cfg not in by else b_cfg}")
            continue
        keys = sorted(set(by[a_cfg]) & set(by[b_cfg]))
        datasets = sorted({d for d, _ in keys})
        for dataset in datasets:
            ks = [k for k in keys if k[0] == dataset]
            dev = [(by[a_cfg][k], by[b_cfg][k]) for k in ks if by[a_cfg][k].get("split") == "dev"]
            ev = [(by[a_cfg][k], by[b_cfg][k]) for k in ks
                  if split == "all" or by[a_cfg][k].get("split") == split]
            if not ev:
                continue
            t = dev_score = None
            if kind == "override":
                t, dev_score = tune_threshold(dev, name)
            # baseline = the better input ON DEV (ties: A)
            ea, eb = exact([a for a, _ in dev]), exact([b for _, b in dev])
            base_cfg = b_cfg if (eb or 0) > (ea or 0) else a_cfg
            sim = [combine(kind, name, a, b, t) for a, b in ev]
            base = [a if base_cfg == a_cfg else b for a, b in ev]
            for condition in ["ALL"] + sorted({r.get("condition") or "-" for r in sim}):
                sel = [i for i, r in enumerate(sim)
                       if condition == "ALL" or (r.get("condition") or "-") == condition]
                s_sim = [score_record(sim[i]) for i in sel]
                s_base = [score_record(base[i]) for i in sel]
                pairs = [(r.get("place_key"), r, q) for r, q in zip(s_sim, s_base)]
                ci = stats.paired_effect_ci(pairs, stats.exact_rate, n_boot=N_BOOT)
                m, mb = metrics(s_sim), metrics(s_base)
                rows.append({
                    "family": FAMILY, "design": name, "kind": kind, "a": a_cfg, "b": b_cfg,
                    "question": question, "dataset": dataset, "split": split,
                    "condition": condition, **m,
                    "baseline": base_cfg, "baseline_exact": mb["exact"], "baseline_country": mb["country"],
                    "delta_exact": ci.get("point"), "delta_lo": ci.get("lo"),
                    "delta_hi": ci.get("hi"), "n_places": ci.get("n_places"),
                    "threshold": t, "threshold_dev_exact": dev_score,
                })
    print_overview(rows)
    if out is not None:
        path = Path(out) / f"table_simulated_designs_{split}.csv"
        write_csv(rows, path)
        print(f"\nwrote {path}")
    return rows


def _p(v, pct=True):
    return "-" if v is None else (f"{100 * v:5.1f}" if pct else f"{v:.3f}")


def print_overview(rows: list[dict]) -> None:
    print(f"SIMULATED DESIGNS ({FAMILY}); exact %, delta vs better input chosen on dev\n")
    print(f"{'design':<32}{'dataset':<20}{'n':>5}{'exact':>7}{'base':>7}"
          f"{'delta [95% CI]':>24}{'nores':>7}  baseline")
    for r in rows:
        if r["condition"] != "ALL":
            continue
        ci = (f"{_p(r['delta_exact'])} [{_p(r['delta_lo'])},{_p(r['delta_hi'])}]"
              if r["delta_exact"] is not None else "-")
        extra = f"  t={r['threshold']:.3f}" if r["threshold"] is not None else ""
        print(f"{r['design']:<32}{r['dataset']:<20}{r['n']:>5}{_p(r['exact']):>7}"
              f"{_p(r['baseline_exact']):>7}{ci:>24}{_p(r['no_result_rate']):>7}  "
              f"{r['baseline']}{extra}")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", choices=["all", "dev", "test"])
    p.add_argument("--out", default=str(DEFAULT_OUT), help="directory for the CSV ('' to skip)")
    args = p.parse_args()
    run(args.split, Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
