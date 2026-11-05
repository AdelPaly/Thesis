#!/usr/bin/env python3
"""
Compare models on the post-writing step: the same planned items, the same
prompts, every model, then the same checks.

    python compare_models.py --models deepseek/deepseek-v4.1-flash vse:qwen3.5:122b ...
        [--plan output/item_plan.jsonl --plan other_plan.jsonl] [--per-condition 5]
        [--run output/model_test/run1]

Items: --per-condition items of each condition, split evenly over the --plan
files (old and new plans can be mixed), chosen by a seeded hash so a rerun
picks the same ones, one item per place where the supply allows. The sample
is saved as sample.jsonl in the run folder and reused on later runs, so a
model added later sees exactly the same items.

Per model it writes gen_<model>.jsonl (generate.py rows plus latency_s) and
raw_<model>.jsonl (prompts and raw replies). A rerun skips items a model has
already answered, so it resumes. Then it writes, in the run folder:

    report.md         pass rate (finalise.py checks) per model and condition,
                      drop reasons, cost and speed, diversity
    posts.csv         every post with its verdict, latency and tokens
    rating_sheet.csv  every post in random order with the model hidden, for
                      rating by hand (realism, naturalness, fit, 1-5)
    rating_key.csv    rating_id -> model; keep it closed while rating

Nothing here changes output/item_plan.jsonl, output/raw_generations.jsonl or
the random stream. The API key comes from .env, as for generate.py.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

import finalise          # noqa: E402  (check, environment_pattern, word_limits)
import generate as gen   # noqa: E402  (client, prompts, records, run_item)
import judge as judge_mod  # noqa: E402  (the model check)
import plan              # noqa: E402  (fold, configure_stdout)

SEED = 20261029            # which items are sampled, and the rating sheet order
CONDITIONS = list(plan.CONDITION_SPECS)   # narrowed to config plan.conditions in main()
NEAR_DUPLICATE = 0.5       # word-set Jaccard at or above this counts as a near-duplicate pair

# Phrasing the checks do not drop but the smoke run showed (2026-10-29): a
# named person used as an imagined judge ("X would be furious"), the clue's
# relation copied from the prompt ("born here"), the distractor marked as
# foreign ("from elsewhere"). Counted per model in the report.
FLAGS = {
    "imagined_judge": re.compile(
        r"\b(would|will)(n'?t|'ve| not| never| have| hardly)*( \w+)? (be|been|laugh|like|love|hate|cry|weep|"
        r"believe|recogni[sz]e|approve|care|mind|let|want|say|think|despair|smile|understand|know)\b"
        r"|\balways (hated|loved)\b|\bturning in (his|her|their) grave\b"
        r"|\bwhat would \w+( \w+)? (say|think)\b", re.I),
    # Any birth/death/upbringing wording, not just the prompt's own phrase:
    # the writer turned "born here" into "where X drew his first breath".
    "relation_echo": re.compile(r"\b(born|birth|birthplace|died|death|grew up|raised|buried|hometown|"
                                r"first breath|childhood home)\b|\bheadquartered here\b", re.I),
    "marked_foreign": re.compile(r"\bfrom elsewhere\b|\bsomewhere else\b|\bnot from (here|around here)\b", re.I),
    # A name brought in through media and then dropped ("was reading about X.
    # then saw..."): filler a real poster would cut (2026-10-29, user).
    # The device smoke7 fell back on, living people included.
    "statue": re.compile(r"\b(statue|memorial|monument|bust|mural|plaque)\b", re.I),
    "media_filler": re.compile(
        r"\b(reading|read|watched|watching|heard|hearing|listening to|replaying) "
        r"(about|up on|a (film|documentary|bit|clip|interview|book|bio)|the (bio|biography|bit|clips?|film|documentary))\b"
        r"|\b(finished|just finished) (the|that|a|reading)\b|\b(bio|biography|documentary|film) (on|about)\b", re.I),
}


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model)


def rank(*parts: Any) -> str:
    return hashlib.sha256(":".join(map(str, (SEED, *parts))).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# sample
# --------------------------------------------------------------------------- #

def sample_items(plans: list[Path], per_condition: int) -> list[dict[str, Any]]:
    """per_condition items of each condition, split evenly over the plans (the
    remainder goes to the first plans), each plan's items ordered by a seeded
    hash and a place taken at most once while others remain."""
    loaded = [(p, [i for i in gen.read_jsonl(p) if i.get("pool", "plan") == "plan"]) for p in plans]
    chosen: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        share, extra = divmod(per_condition, len(loaded))
        for n, (path, items) in enumerate(loaded):
            want = share + (n < extra)
            pool = sorted((i for i in items if i["condition"] == condition),
                          key=lambda i: rank(path.name, i["item_id"]))
            used = {i["place_qid"] for i in chosen}
            fresh = [i for i in pool if i["place_qid"] not in used]
            picked = (fresh + [i for i in pool if i not in fresh])[:want]
            chosen += [{**i, "source_plan": path.name} for i in picked]
    return chosen


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #

def model_config(spec: str, config: dict[str, Any]) -> dict[str, Any]:
    """The config for one model: "<provider>:<model>" takes that provider's
    settings from config.providers over generation:, anything else is an
    OpenRouter id. Only a known provider name counts as a prefix, so the
    colon in "qwen3.5:122b" is left alone."""
    cfg = copy.deepcopy(config)
    provider, _, rest = spec.partition(":")
    if rest and provider in (config.get("providers") or {}):
        cfg["generation"].update(config["providers"][provider])
        cfg["generation"]["model"] = rest
    else:
        cfg["generation"]["model"] = spec
    return cfg


def judge_config(config: dict[str, Any], spec: str | None) -> dict[str, Any]:
    """The config with the judge model replaced by spec ("<provider>:<model>"
    or an OpenRouter id), as model_config does for the writer; unchanged
    when spec is None."""
    if not spec:
        return config
    cfg = copy.deepcopy(config)
    as_writer = model_config(spec, config)["generation"]
    keep = {k: cfg["judge"]["model_settings"][k] for k in ("temperature", "send_seed")}
    cfg["judge"]["model_settings"] = {**cfg["judge"]["model_settings"],
                                      **{k: as_writer[k] for k in ("base_url", "model", "api_key_env",
                                                                   "extra_headers", "reasoning_effort",
                                                                   "exclude_reasoning", "reasoning_enabled",
                                                                   "extra_body", "timeout_seconds")
                                         if k in as_writer}, **keep, "max_tokens": 1500}
    return cfg


def run_model(model: str, items: list[dict[str, Any]], config: dict[str, Any],
              exemplars: dict[str, list[str]], run: Path, use_judge: bool = True,
              judge_spec: str | None = None) -> None:
    """Send every item not yet answered by this model through generate.run_item,
    the path generate.py takes: the rule checks, then the judge model, and up
    to judge.max_tries tries, each later try told why the last one failed.
    Rows are appended, so an interrupted run resumes."""
    cfg = model_config(model, config)
    out_path, raw_path = run / f"gen_{slug(model)}.jsonl", run / f"raw_{slug(model)}.jsonl"
    done = {r["item_id"] for r in gen.read_jsonl(out_path) if not r.get("error")} if out_path.exists() else set()
    todo = [i for i in items if i["item_id"] not in done]
    print(f"\n{model}: {len(todo)} of {len(items)} to send"
          + ("" if use_judge else " (no judge)"))
    if not todo:
        return
    client = gen.ChatClient(cfg)
    judge = judge_mod.Judge(judge_config(config, judge_spec)) if use_judge else None
    if judge and judge.model == cfg["generation"]["model"]:
        # 2026-11-02 smoke17/18: DeepSeek judging its own posts passed side-by-side ties that Qwen failed
        print(f"  ! {model} is judging its own posts; pass --judge with another family")
    config_hash = gen.config_fingerprint(cfg, use_exemplars=bool(exemplars))
    out_log, raw_log = gen.AppendLog(out_path), gen.AppendLog(raw_path)
    delay = float(gen.gen_setting(cfg["generation"], "delay_between_calls_seconds"))
    aliases = finalise.load_aliases(Path(config["paths"]["place_inputs"]))
    limits = finalise.word_limits(config.get("finalise"))
    try:
        for n, item in enumerate(todo, start=1):
            names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
            started = time.perf_counter()
            record = gen.run_item(item, client, judge, cfg, exemplars, config_hash, names, limits, raw_log)
            latency = time.perf_counter() - started
            record.update(latency_s=round(latency, 2), source_plan=item["source_plan"])
            out_log.write(record)
            steps = " > ".join(t["status"] for t in record["try_log"])
            print(f"  [{n}/{len(todo)}] {item['condition']:<9} {latency:5.1f}s  {item['place_name'][:26]:<26} {steps}",
                  flush=True)
            if delay and n < len(todo):
                time.sleep(delay)
    finally:
        out_log.close()
        raw_log.close()


# --------------------------------------------------------------------------- #
# checks and measures
# --------------------------------------------------------------------------- #

def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", plan.fold(text))


def diversity(texts: list[str]) -> dict[str, Any]:
    """Distinct-1 and distinct-2 (unique share of all words and word pairs),
    the share of posts opening with the most common first two words, and
    near-duplicate pairs (word-set Jaccard >= NEAR_DUPLICATE)."""
    toks = [words(t) for t in texts if t]
    if not toks:
        return {}
    unigrams = [w for t in toks for w in t]
    bigrams = [b for t in toks for b in zip(t, t[1:])]
    openings = Counter(" ".join(t[:2]) for t in toks if t)
    top_open, top_n = openings.most_common(1)[0]
    sets = [set(t) for t in toks]
    jac = [len(a & b) / len(a | b) for a, b in combinations(sets, 2) if a | b]
    return {"distinct_1": len(set(unigrams)) / max(len(unigrams), 1),
            "distinct_2": len(set(bigrams)) / max(len(bigrams), 1),
            "top_opening": top_open, "top_opening_share": top_n / len(toks),
            "mean_jaccard": statistics.mean(jac) if jac else 0.0,
            "near_duplicates": sum(j >= NEAR_DUPLICATE for j in jac),
            "mean_words": statistics.mean(len(t) for t in toks)}


def usages(row: dict[str, Any], who: str = "all") -> list[dict[str, Any]]:
    """The usage blocks of every call made for the item: each writer try and
    each judge call (who: "writer", "judge" or "all"). Rows from before the
    tries loop carry only response_metadata."""
    if not row.get("try_log"):
        return [u for u in [(row.get("response_metadata") or {}).get("usage")] if u and who != "judge"]
    keys = {"writer": ["writer_usage"], "judge": ["judge_usage"], "all": ["writer_usage", "judge_usage"]}[who]
    return [t[k] for t in row["try_log"] for k in keys if t.get(k)]


def cost_of(row: dict[str, Any], who: str = "all") -> float | None:
    """The billed cost (OpenRouter usage.cost, in USD) summed over every call
    for the item; None when the provider reported no cost (the VSE gateway)."""
    costs = [u.get("cost") for u in usages(row, who)]
    return sum(costs) if costs and all(c is not None for c in costs) else None


def tokens_of(row: dict[str, Any], key: str, who: str = "writer") -> int:
    return sum(int(u.get(key) or 0) for u in usages(row, who))


def evaluate(models: list[str], items: list[dict[str, Any]], config: dict[str, Any],
             run: Path) -> None:
    names: dict[str, dict[str, Any]] = {}
    for r in gen.read_jsonl(Path(config["paths"]["place_inputs"])):
        names[r["place_qid"]] = {"forms": r.get("name_forms") or [],
                                 "country": r.get("country_forms") or [],
                                 "regions": r.get("region_forms") or []}
    env = finalise.environment_pattern((config.get("finalise") or {}).get("environment_keywords"))
    limits = finalise.word_limits(config.get("finalise"))
    by_id = {i["item_id"]: i for i in items}

    posts: list[dict[str, Any]] = []
    for model in models:
        path = run / f"gen_{slug(model)}.jsonl"
        latest = gen.latest_rows(gen.read_jsonl(path)) if path.exists() else {}
        for item_id, item in by_id.items():
            row = latest.get(item_id)
            verdict = finalise.verdict_for(item, row, names, env, limits)
            posts.append({
                "model": model, "item_id": item_id, "source_plan": item["source_plan"],
                "condition": item["condition"], "place_name": item["place_name"],
                "status": verdict[0] if verdict else "kept",
                "reason": verdict[1] if verdict else "",
                "text": ((row or {}).get("post_text") or "").strip(),
                "words": len(((row or {}).get("post_text") or "").split()),
                "latency_s": (row or {}).get("latency_s"),
                "tries": (row or {}).get("tries") or (1 if row else 0),
                "try_statuses": " > ".join(t["status"] for t in (row or {}).get("try_log") or []),
                "off_theme": any(t["status"] == "judge_off_theme" for t in (row or {}).get("try_log") or []),
                "prompt_tokens": tokens_of(row or {}, "prompt_tokens"),
                "completion_tokens": tokens_of(row or {}, "completion_tokens"),
                "judge_tokens": tokens_of(row or {}, "total_tokens", "judge"),
                "writer_cost": cost_of(row or {}, "writer"),
                "judge_cost": cost_of(row or {}, "judge"),
                "cost": cost_of(row or {}),
                "flags": ";".join(k for k, rx in FLAGS.items()
                                  if rx.search((row or {}).get("post_text") or "")),
            })

    write_csv(run / "posts.csv", posts)
    write_rating_sheet(posts, by_id, run)
    (run / "report.md").write_text(report(models, posts), encoding="utf-8")
    print(f"\nwrote {run / 'report.md'}\nwrote {run / 'posts.csv'}\n"
          f"wrote {run / 'rating_sheet.csv'} (key: rating_key.csv)")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:   # BOM: opens cleanly in Excel
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_rating_sheet(posts: list[dict[str, Any]], by_id: dict[str, dict[str, Any]],
                       run: Path) -> None:
    """Every generated post, shuffled by a seeded hash, the model hidden. What
    the post was asked to contain is shown, so fit can be judged."""
    rows = sorted((p for p in posts if p["text"]), key=lambda p: rank("rate", p["model"], p["item_id"]))
    sheet, key = [], []
    for n, p in enumerate(rows, start=1):
        item = by_id[p["item_id"]]
        asked = [c["clue_label"] for c in item.get("clues") or []]
        if (item.get("surface_form") or {}).get("text"):
            asked.append(f'name: {item["surface_form"]["text"]}')
        if item.get("noise_entity"):
            asked.append(f'distractor: {item["noise_entity"]["clue_label"]}')
        sheet.append({"rating_id": n, "condition": p["condition"], "theme": item.get("env_theme") or "",
                      "valence": item.get("valence") or "", "asked_for": "; ".join(asked), "text": p["text"],
                      "realism_1to5": "", "naturalness_1to5": "", "fit_1to5": "", "notes": ""})
        key.append({"rating_id": n, "model": p["model"], "item_id": p["item_id"],
                    "auto_status": p["status"]})
    write_csv(run / "rating_sheet.csv", sheet)
    write_csv(run / "rating_key.csv", key)


def pct(n: int, d: int) -> str:
    return f"{100 * n / d:.0f}%" if d else "-"


def report(models: list[str], posts: list[dict[str, Any]]) -> str:
    out = ["# Model comparison", "",
           f"{len({p['item_id'] for p in posts})} items per model "
           f"({', '.join(sorted({p['source_plan'] for p in posts}))}). "
           "Pass = kept by the finalise.py checks.", ""]

    out += ["## Pass rate", "", "| model | " + " | ".join(CONDITIONS) + " | all |",
            "|---|" + "---|" * (len(CONDITIONS) + 1)]
    for m in models:
        mine = [p for p in posts if p["model"] == m]
        cells = []
        for c in CONDITIONS:
            sub = [p for p in mine if p["condition"] == c]
            cells.append(pct(sum(p["status"] == "kept" for p in sub), len(sub)))
        out.append(f"| {m} | " + " | ".join(cells) +
                   f" | **{pct(sum(p['status'] == 'kept' for p in mine), len(mine))}** |")

    out += ["", "## Drop reasons", "", "| model | reason | n |", "|---|---|---|"]
    for m in models:
        for status, n in Counter(p["status"] for p in posts
                                 if p["model"] == m and p["status"] != "kept").most_common():
            out.append(f"| {m} | {status} | {n} |")

    out += ["", "## Tries and theme", "",
            "Tries: how many writer calls an item took (2 and 3 were told why the last try failed). "
            "Off theme: items where the judge found a try off the planned topic, counted apart from "
            "judge_no_environment (no environmental claim at all).", "",
            "| model | kept on try 1 | kept on try 2 | kept on try 3 | failed all | off theme (any try) |",
            "|---|---|---|---|---|---|"]
    for m in models:
        mine = [p for p in posts if p["model"] == m and p["tries"]]
        kept_on = Counter(p["tries"] for p in mine if p["status"] == "kept")
        out.append(f"| {m} | {kept_on[1]} | {kept_on[2]} | {kept_on[3]} | "
                   f"{sum(p['status'] != 'kept' for p in mine)} | {sum(p['off_theme'] for p in mine)} |")

    out += ["", "## Phrasing flags", "",
            "Posts the checks keep or drop alike, but that read badly: a named person as an imagined "
            "judge (\"X would be furious\"), the clue's relation copied (\"born here\"), the distractor "
            "marked as foreign (\"from elsewhere\").", "",
            "| model | " + " | ".join(FLAGS) + " |", "|---|" + "---|" * len(FLAGS)]
    for m in models:
        mine = [p for p in posts if p["model"] == m and p["text"]]
        out.append(f"| {m} | " + " | ".join(
            f"{sum(k in p['flags'].split(';') for p in mine)} of {len(mine)}" for k in FLAGS) + " |")

    out += ["", "## Cost and speed", "",
            "Cost is OpenRouter's billed usage.cost summed over every call for an item: each writer try "
            "and each judge call. Tokens are the writer's, summed over its tries; latency covers the whole item.", "",
            "| model | median latency s | p90 latency s | prompt tok / post | output tok / post | writer $ / post | judge $ / post | cost / post | cost / 1,000 posts |",
            "|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        mine = [p for p in posts if p["model"] == m and p["text"]]
        lat = sorted(p["latency_s"] for p in mine if p["latency_s"] is not None)
        mean = lambda k: statistics.mean(c) if (c := [p[k] for p in mine if p[k] is not None]) else None  # noqa: E731
        money = lambda v: f"${v:.5f}" if v is not None else "not reported"  # noqa: E731
        mean_cost = mean("cost")
        out.append(f"| {m} | {statistics.median(lat):.1f} | {lat[int(0.9 * (len(lat) - 1))]:.1f} | "
                   f"{statistics.mean(p['prompt_tokens'] for p in mine):.0f} | "
                   f"{statistics.mean(p['completion_tokens'] for p in mine):.0f} | "
                   f"{money(mean('writer_cost'))} | {money(mean('judge_cost'))} | {money(mean_cost)} | "
                   + (f"${1000 * mean_cost:.2f} |" if mean_cost is not None else "- |")
                   if lat else f"| {m} | - | - | - | - | - | - | - | - |")

    out += ["", "## Diversity", "",
            "Distinct-n: unique share of all words / word pairs (higher is more varied). "
            f"Near-duplicates: pairs of posts sharing at least {NEAR_DUPLICATE:.0%} of their words.", "",
            "| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |",
            "|---|---|---|---|---|---|---|---|"]
    for m in models:
        d = diversity([p["text"] for p in posts if p["model"] == m])
        if d:
            out.append(f"| {m} | {d['distinct_1']:.2f} | {d['distinct_2']:.2f} | \"{d['top_opening']}\" | "
                       f"{d['top_opening_share']:.0%} | {d['mean_jaccard']:.2f} | {d['near_duplicates']} | "
                       f"{d['mean_words']:.1f} |")

    out += ["", "## Human rating", "",
            "Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening "
            "rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.", ""]
    return "\n".join(out) + "\n"


def score_ratings(run: Path) -> None:
    """Mean of each rating per model, joining the sheet to the key."""
    def read(name: str) -> list[dict[str, str]]:
        with (run / name).open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    model_of = {r["rating_id"]: r["model"] for r in read("rating_key.csv")}
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in read("rating_sheet.csv"):
        for col in ("realism_1to5", "naturalness_1to5", "fit_1to5"):
            if r[col].strip():
                scores[model_of[r["rating_id"]]][col].append(float(r[col]))
    print(f"{'model':<40} {'realism':>8} {'natural':>8} {'fit':>8} {'rated':>6}")
    for m, cols in sorted(scores.items()):
        mean = lambda c: f"{statistics.mean(cols[c]):.2f}" if cols[c] else "-"  # noqa: E731
        print(f"{m:<40} {mean('realism_1to5'):>8} {mean('naturalness_1to5'):>8} "
              f"{mean('fit_1to5'):>8} {len(cols['realism_1to5']):>6}")


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    help="OpenRouter model ids, or <provider>:<model> for config.providers")
    ap.add_argument("--plan", type=Path, action="append",
                    help="plan file to sample from, repeatable (default: the config's item_plan)")
    ap.add_argument("--per-condition", type=int, default=5)
    ap.add_argument("--run", type=Path, default=HERE / "output" / "model_test" / "run1")
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--env-file", type=Path, default=HERE / ".env")
    ap.add_argument("--report-only", action="store_true", help="re-check and re-report, send nothing")
    ap.add_argument("--no-judge", action="store_true", help="rule checks only, one try per item (the old smoke runs)")
    ap.add_argument("--judge", help="judge model, <provider>:<model> or an OpenRouter id (default: config judge:)")
    ap.add_argument("--score-ratings", action="store_true", help="summarise the filled rating sheet")
    args = ap.parse_args(argv)

    if args.score_ratings:
        score_ratings(args.run)
        return 0
    if not args.models:
        ap.error("--models is required")

    gen.load_dotenv(args.env_file)
    config = gen.load_config(args.config)
    CONDITIONS[:] = [c for c in CONDITIONS if c in config["plan"]["conditions"]]
    args.run.mkdir(parents=True, exist_ok=True)
    sample_path = args.run / "sample.jsonl"
    if sample_path.exists():
        items = gen.read_jsonl(sample_path)
        print(f"reusing {len(items)} sampled items from {sample_path}")
    else:
        items = sample_items(args.plan or [Path(config["paths"]["item_plan"])], args.per_condition)
        plan.write_jsonl(sample_path, items)
        print(f"sampled {len(items)} items: " + ", ".join(
            f"{c} {n}" for c, n in Counter(i["condition"] for i in items).items()))

    if not args.report_only:
        exemplars = gen.load_exemplars(Path(config["paths"]["style_exemplars"]))
        for model in args.models:
            run_model(model, items, config, exemplars, args.run, use_judge=not args.no_judge,
                      judge_spec=args.judge)
    evaluate(args.models, items, config, args.run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
