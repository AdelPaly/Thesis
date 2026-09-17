"""Audit the retired results for defects unrelated to the method.

    python scripts/audit_retired_results.py

Reads only `results/raw/eval_FINAL_*.csv`, the committed run logs in
`results/logs/`, and the LLM response caches. Writes nothing. It exists so the
claims in METHODOLOGY.md ("Why the retired results are not comparable") can be
checked rather than taken on trust.

Three checks:

1. Null Island -- rows whose predicted coordinate is exactly 0.0/0.0, from the
   continent-only fallback that used to return a coordinate it did not have,
   and how much they inflate the reported mean error.

2. Transient gazetteer failures -- `Search error: 429 Too Many Requests after
   retries` in the run logs, attributed to the items they hit, and whether
   those items are over-represented among the Null Island rows. A 429 returned
   an empty search result, which is indistinguishable from "no such place", so
   the LLM path's city -> country -> continent cascade fell through a level.

3. Direction of that association -- the response caches still hold what each
   model said, keyed by the same text hash the runs used. If a Null Island
   row's response named a city or a country, the fallback was resolution
   failing, not the model declining to answer.

Check 3 is what makes check 2 causal rather than correlational, so it is the
one worth reading.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "results" / "raw"
LOGS = REPO_ROOT / "results" / "logs"
LLM_CACHE = REPO_ROOT / "src" / ".llm_cache"

ITEM_MARKER = re.compile(r"\[(\d+)/(\d+)\]")
JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

CACHE_BY_CONFIG = {
    "gemma": "gemma.pkl",
    "groq_oss": "groq_gpt_oss_120b.pkl",
    "groq_llama": "groq_llama4_scout_17b.pkl",
    "refined_gemma": "gemma.pkl",
    "refined_groq_oss": "groq_gpt_oss_120b.pkl",
    "refined_groq_llama": "groq_llama4_scout_17b.pkl",
}


def _null_island_mask(df):
    return (df["pred_lat"].fillna(9) == 0.0) & (df["pred_lon"].fillna(9) == 0.0)


def check_null_island():
    print("=" * 78)
    print("1. Null Island rows in the retired results")
    print("=" * 78)
    print(f"{'file':44} {'n':>4} {'NI':>4} {'mean km':>9} {'excl. NI':>9}")
    print("-" * 78)
    total_rows = total_ni = 0
    for path in sorted(RAW.glob("eval_FINAL_*.csv")):
        df = pd.read_csv(path)
        mask = _null_island_mask(df)
        total_rows += len(df)
        if not mask.any():
            continue
        total_ni += int(mask.sum())
        print(f"{path.name:44} {len(df):>4} {int(mask.sum()):>4} "
              f"{df['dist_km'].mean():>9.1f} {df.loc[~mask, 'dist_km'].mean():>9.1f}")
    print("-" * 78)
    print(f"{total_ni} Null Island row(s) across {total_rows} rows\n")
    return total_ni


def _items_hit_by_failures(log_text):
    """Item indices (0-based) whose processing logged a transient failure."""
    parts = ITEM_MARKER.split(log_text)
    hit = set()
    for k in range(1, len(parts) - 2, 3):
        chunk = parts[k + 2]
        if "Search error" in chunk or "API error" in chunk:
            hit.add(int(parts[k]) - 1)
    return hit


def check_transient_failures():
    print("=" * 78)
    print("2. Transient gazetteer failures, attributed to items")
    print("=" * 78)
    kinds = {}
    for path in LOGS.glob("log_FINAL_*.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.search(r"(?:Search|API) error: (.*)", line)
            if m:
                kinds[m.group(1).strip()[:60]] = kinds.get(m.group(1).strip()[:60], 0) + 1
    for kind, n in sorted(kinds.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>4}  {kind}")
    print()

    print(f"{'run':30} {'n':>4} {'hit':>4} {'NI':>4} {'NI & hit':>9}")
    print("-" * 78)
    tot_n = tot_hit = tot_ni = tot_both = 0
    for log_path in sorted(LOGS.glob("log_FINAL_*.txt")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if "Search error" not in text and "API error" not in text:
            continue
        csv_path = RAW / (log_path.stem.replace("log_FINAL_", "eval_FINAL_") + ".csv")
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        # Within one retired run the log and the CSV were written by the same
        # loop in the same order, so log index i maps to CSV row i-1.
        hit = _items_hit_by_failures(text)
        ni = set(df.index[_null_island_mask(df)])
        both = len(ni & hit)
        tot_n += len(df)
        tot_hit += len(hit)
        tot_ni += len(ni)
        tot_both += both
        print(f"{csv_path.stem.replace('eval_FINAL_', ''):30} {len(df):>4} "
              f"{len(hit):>4} {len(ni):>4} {both:>9}")
    print("-" * 78)
    print(f"{'TOTAL':30} {tot_n:>4} {tot_hit:>4} {tot_ni:>4} {tot_both:>9}")
    if tot_ni and tot_hit and tot_n:
        p_hit = tot_both / tot_hit
        p_base = tot_ni / tot_n
        print(f"\n  share of Null Island rows that hit a failure : "
              f"{tot_both / tot_ni:.0%}")
        print(f"  P(Null Island | hit a failure)               : {p_hit:.1%}")
        print(f"  P(Null Island | any item)                    : {p_base:.1%}")
        if p_base:
            print(f"  relative risk                                : "
                  f"{p_hit / p_base:.1f}x")
    print()


def _parse_json(raw):
    for cand in (raw.strip(), *(m.group(1) for m in JSON_FENCE.finditer(raw))):
        try:
            return json.loads(cand)
        except Exception:                                  # noqa: BLE001
            continue
    return None


def check_direction():
    print("=" * 78)
    print("3. What the model actually said on the Null Island rows")
    print("=" * 78)
    caches = {}
    for name in set(CACHE_BY_CONFIG.values()):
        path = LLM_CACHE / name
        if path.exists():
            with open(path, "rb") as f:
                caches[name] = pickle.load(f)
    if not caches:
        print("  no response caches on disk, skipping\n")
        return

    counts = {"city": 0, "country": 0, "continent_only": 0,
              "refused": 0, "unparsed": 0, "not_cached": 0}
    examples = []
    for csv_path in sorted(RAW.glob("eval_FINAL_*.csv")):
        stem = csv_path.stem.replace("eval_FINAL_", "")
        config = next((c for c in sorted(CACHE_BY_CONFIG, key=len, reverse=True)
                       if stem.startswith(c)), None)
        if config is None:
            continue
        cache = caches.get(CACHE_BY_CONFIG[config])
        if cache is None:
            continue
        df = pd.read_csv(csv_path)
        for _, row in df[_null_island_mask(df)].iterrows():
            raw = cache.get(hashlib.sha256(str(row["text"]).encode()).hexdigest())
            if raw is None:
                counts["not_cached"] += 1
                continue
            parsed = _parse_json(raw)
            if parsed is None:
                counts["unparsed"] += 1
            elif parsed.get("refused"):
                counts["refused"] += 1
            elif parsed.get("city"):
                counts["city"] += 1
                if len(examples) < 6:
                    examples.append((stem, parsed, row["true_country"]))
            elif parsed.get("country"):
                counts["country"] += 1
                if len(examples) < 6:
                    examples.append((stem, parsed, row["true_country"]))
            else:
                counts["continent_only"] += 1

    total = sum(counts.values())
    print(f"  named a CITY    : {counts['city']:>3}   <- resolution failed, not the model")
    print(f"  named a COUNTRY : {counts['country']:>3}   <- resolution failed, not the model")
    print(f"  continent only  : {counts['continent_only']:>3}   <- legitimate fallback")
    print(f"  refused         : {counts['refused']:>3}")
    print(f"  unparseable     : {counts['unparsed']:>3}")
    print(f"  not in cache    : {counts['not_cached']:>3}")
    print(f"  total           : {total:>3}")
    if total:
        forward = counts["city"] + counts["country"]
        print(f"\n  attributable to failed resolution: {forward}/{total} = "
              f"{forward / total:.0%}")
    print("\n  examples (run | model answer | gold country):")
    for stem, parsed, gold in examples:
        said = {k: parsed.get(k) for k in ("city", "country", "continent")}
        print(f"    {stem:28} {said} | gold={gold!r}")
    print()


if __name__ == "__main__":
    check_null_island()
    check_transient_failures()
    check_direction()
    print("See METHODOLOGY.md, 'Why the retired results are not comparable'.")
