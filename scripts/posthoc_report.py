"""The correction of the LLM-only resolver: scores and failure reasons.

    python scripts/posthoc_report.py --split dev
    python scripts/posthoc_report.py --split test

Compares, on one split, for Gemma 4 and Qwen3-VL:

  first run     gemma4_first_run, qwen3_vl_first_run: prompt v1, first search
                result for the city (results/posthoc/original/)
  country only  the first run's stored answers resolved with the country
                (scripts/posthoc_llm_resolver.py -> results/posthoc/)
  reported      gemma4, qwen3_vl: prompt v2 with coordinates, resolved by
                country and coordinates, with the wider search of rule v3
                (results/predictions/)

with refined and grounded_gemma_linking for reference, and writes to
results/posthoc/:

  table_resolver_<split>.csv            exact match with a place-bootstrap
                                        interval, country, no answer, exact by condition
  table_resolver_differences_<split>.csv  paired differences with intervals
  table_resolution_reasons_<split>.csv  how each answer was resolved or why not

The reasons of the first runs are derived here from their stored replies with
the v1 rule (src/resolver.py), from the Wikidata cache; their predictions are
not touched. The intervals here are not adjusted for multiple comparisons; the
planned comparisons are in results/summary/table_stats_<split>.csv.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import provenance  # noqa: E402
import resolver  # noqa: E402
import stats as st  # noqa: E402
from score import attach_split, load_predictions, score_record, summarise_metrics  # noqa: E402

PREDICTIONS = REPO_ROOT / "results" / "predictions"
POSTHOC = REPO_ROOT / "results" / "posthoc"
CONDITIONS = ["EXPLICIT", "NOISE_EXPLICIT", "B", "NOISE_B", "A", "NOISE_A"]
MODELS = ["gemma4", "qwen3_vl"]
REFERENCES = ["refined", "grounded_gemma_linking"]


def load(path: Path, split: str) -> list[dict] | None:
    if not path.exists():
        return None
    records = [r for r in attach_split(load_predictions([path])) if r.get("split") == split]
    return [score_record(r) for r in records] or None


def variants(split: str) -> dict[str, list[dict]]:
    out = {}
    for model in MODELS:
        out[f"{model}_first_run"] = load(POSTHOC / "original" / f"{model}__synthetic.jsonl", split)
        out[f"{model}_country_resolver"] = load(
            POSTHOC / f"{model}_country_resolver__synthetic__{split}.jsonl", split)
        out[model] = load(PREDICTIONS / f"{model}__synthetic.jsonl", split)
    for name in REFERENCES:
        out[name] = load(PREDICTIONS / f"{name}__synthetic.jsonl", split)
    return {name: recs for name, recs in out.items() if recs}


def pct(x: float | None) -> str:
    return "" if x is None else f"{100 * x:.1f}"


def resolution_of(rec: dict) -> dict:
    """The stored resolution, or for a v1 record written before it existed the
    one the v1 rule gives for the stored reply."""
    if rec.get("resolution"):
        return rec["resolution"]
    return resolver.resolve_first_hit(rec.get("llm_raw_response"))[1]


def write(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([*header, *provenance.COLUMNS])
        w.writerows([*row, *provenance.values()] for row in rows)
    print(f"  -> {path.relative_to(REPO_ROOT)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    args = p.parse_args()
    data = variants(args.split)
    POSTHOC.mkdir(parents=True, exist_ok=True)

    print(f"\n{args.split} split: exact match, % (place-bootstrap interval)")
    rows = []
    for name, recs in data.items():
        m = summarise_metrics(recs)
        ci = st.cluster_bootstrap_ci(recs, st.exact_rate)
        by_cond = [pct(summarise_metrics([r for r in recs if r["condition"] == c]).get("exact"))
                   for c in CONDITIONS]
        rows.append([name, m["n"], pct(m["exact"]), pct(ci["lo"]), pct(ci["hi"]),
                     pct(m["country"]), pct(m["no_result_rate"]), *by_cond])
        print(f"  {name:30} n={m['n']:4}  exact {pct(m['exact']):>5} ({pct(ci['lo'])}-{pct(ci['hi'])})"
              f"  country {pct(m['country']):>5}  no answer {pct(m['no_result_rate']):>5}"
              f"  | {' '.join(f'{v:>5}' for v in by_cond)}")
    write(POSTHOC / f"table_resolver_{args.split}.csv",
          ["config", "n", "exact", "exact_lo", "exact_hi", "country", "no_answer",
           *[f"exact_{c}" for c in CONDITIONS]], rows)

    print("\npaired differences in exact match, points (a - b)")
    rows = []
    for model in MODELS:
        for a, b in ((model, f"{model}_first_run"), (model, f"{model}_country_resolver"),
                     (model, "refined"), (model, "grounded_gemma_linking")):
            if a not in data or b not in data:
                continue
            pairs = st.pair_records(data[a], data[b])
            ci = st.paired_effect_ci(pairs, st.exact_rate)
            gained = sum(ra["exact"] and not rb["exact"] for _, ra, rb in pairs)
            lost = sum(rb["exact"] and not ra["exact"] for _, ra, rb in pairs)
            rows.append([a, b, len(pairs), pct(ci["point"]), pct(ci["lo"]), pct(ci["hi"]),
                         gained, lost])
            print(f"  {a:10} - {b:30} {pct(ci['point']):>6} ({pct(ci['lo'])} to {pct(ci['hi'])})"
                  f"   a right, b wrong {gained:4}   b right, a wrong {lost:4}")
    write(POSTHOC / f"table_resolver_differences_{args.split}.csv",
          ["a", "b", "n_pairs", "diff", "diff_lo", "diff_hi", "a_only_correct", "b_only_correct"],
          rows)

    print("\nhow each answer was resolved")
    rows = []
    for name in (n for m in MODELS for n in (f"{m}_first_run", m)):
        if name not in data:
            continue
        counts, correct = Counter(), Counter()
        for rec in data[name]:
            res = resolution_of(rec)
            key = (res["status"], res.get("level") or "none", res.get("method") or "")
            counts[key] += 1
            correct[key] += rec["exact"]
        n = len(data[name])
        by_status = Counter()
        for (status, _, _), c in counts.items():
            by_status[status] += c
        print(f"  {name}: " + ", ".join(f"{s} {by_status[s]} ({100 * by_status[s] / n:.1f}%)"
                                         for s in resolver.STATUSES))
        for key in sorted(counts, key=lambda k: (resolver.STATUSES.index(k[0]), k[1], k[2])):
            rows.append([name, *key, counts[key], f"{100 * counts[key] / n:.1f}", correct[key]])
    write(POSTHOC / f"table_resolution_reasons_{args.split}.csv",
          ["config", "status", "answer_level", "method", "n", "share", "n_exact"], rows)


if __name__ == "__main__":
    main()
