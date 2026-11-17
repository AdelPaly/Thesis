#!/usr/bin/env python3
"""
Audit the finished dataset with a model that took no part in making it.

    python audit.py --model vse:Qwen3.8-27B              # every post
    python audit.py --model vse:Qwen3.8-27B --sample 100 # a random sample
    python audit.py --model vse:Qwen3.8-27B --summary    # tables only, no calls
    python audit.py --list-models vse                    # what the endpoint offers

generate.py had every post checked while it was written: finalise.check for
what a string match can catch, judge.py (DeepSeek) for the rest. This script
runs both again on output/synthetic_dataset.csv, after the fact and with a
different judge model, so the posts are read by a model that neither wrote nor
accepted them. It changes nothing: the dataset, the plan and the generations
are only read.

The model is written "<provider>:<model>", the provider being a key of
providers: in the config (vse, openrouter). The questions, their wording and
the verdict order are judge.py's, so a difference between this audit and the
generation-time judge is a difference between the two models, not the rules.

One change to the verdict (2026-11-17, user): the distractor may be nonsense,
so "judge_noise_nonsense" is not counted unless --strict-noise is given. The
judge's full answer is kept either way.

Two more after reading Qwen's 305 flags by hand (2026-11-17), neither counted
unless asked for; the log is unchanged, so --summary re-scores without a call:

    --strict-theme   "judge_off_theme": the post is environmental but not on
                     the plan's exact sub-topic. No bearing on where it is.
    --strict-quotes  a clue that is not a person failed because the judge
                     paraphrased its link ("located here") instead of quoting
                     the post. A missing or visit-only link still counts.

Writes, next to the dataset (output/audit/):

    <model>.jsonl          one row per post: rule verdict, the judge's answers
                           and verdict. Append-only; a re-run skips the posts
                           already there, so a run can be stopped and resumed.
    <model>_flagged.csv    the posts with a verdict, to read by hand
    <model>_summary.csv    flagged posts per condition and reason
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
import sys
import threading
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import finalise
import generate as gen
import judge
import plan

HERE = Path(__file__).resolve().parent
SAMPLE_SEED = 20261117
FLAGGED_COLUMNS = ["item_id", "condition", "place_name", "country", "clues", "noise_entity",
                   "rule_status", "rule_reason", "judge_status", "judge_reason", "text"]


def split_model(spec: str, config: dict[str, Any]) -> tuple[str, str]:
    providers = config.get("providers") or {}
    provider, _, model = spec.partition(":")
    if not model or provider not in providers:
        raise SystemExit(f'--model is "<provider>:<model>", provider one of {sorted(providers)}; got {spec!r}')
    return provider, model


def audit_config(config: dict[str, Any], spec: str) -> dict[str, Any]:
    """The config with the audit model in the judge's place: the judge's own
    settings (temperature 0, its token budget), then the provider's, then the
    model name."""
    provider, model = split_model(spec, config)
    out = copy.deepcopy(config)
    out["judge"]["model_settings"] = {**config["judge"]["model_settings"],
                                      **config["providers"][provider], "model": model}
    return out


def list_models(config: dict[str, Any], provider: str) -> list[str]:
    import os
    settings = (config.get("providers") or {}).get(provider)
    if not settings:
        raise SystemExit(f"no provider {provider!r} in the config")
    request = urllib.request.Request(
        settings["base_url"].rstrip("/") + "/models",
        headers={"Authorization": "Bearer " + os.environ.get(settings["api_key_env"], "")})
    with urllib.request.urlopen(request, timeout=60, context=gen.tls_context()) as response:
        return sorted(m["id"] for m in json.load(response).get("data") or [])


def slug(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "_", spec).strip("_")


def load_posts(config: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, str]]]:
    """(plan item, dataset row) for every post in the dataset. A post whose
    clues differ from its plan item is refused: the judge would be told the
    wrong names."""
    items = {i["item_id"]: i for i in gen.read_jsonl(Path(config["paths"]["item_plan"]))}
    with Path(config["paths"]["dataset_csv"]).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    posts = []
    for row in rows:
        item = items.get(row["item_id"])
        if item is None:
            raise SystemExit(f'dataset post {row["item_id"]} is not in the plan')
        planned = "|".join(c["clue_qid"] for c in item.get("clues") or [])
        if planned != (row.get("clue_qids") or ""):
            raise SystemExit(f'dataset post {row["item_id"]}: clues {row.get("clue_qids")!r} '
                             f"differ from the plan's {planned!r}")
        posts.append((item, row))
    return posts


def select(posts: list, args: argparse.Namespace) -> list:
    if args.only_condition:
        posts = [p for p in posts if p[0]["condition"] in args.only_condition]
    if args.item_ids:
        wanted = {line.strip() for line in args.item_ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        posts = [p for p in posts if p[0]["item_id"] in wanted]
    if args.sample is not None and args.sample < len(posts):
        chosen = set(random.Random(args.seed).sample(range(len(posts)), args.sample))
        posts = [p for n, p in enumerate(posts) if n in chosen]
    return posts


def judge_verdict(item: dict[str, Any], answer: dict[str, Any] | None, text: str,
                  strict: frozenset[str]) -> tuple[str, str] | None:
    """judge.verdict, without the questions the audit does not count: `strict`
    holds those to count after all ("noise", "theme", "quotes")."""
    if answer is None:
        return None
    if "noise" not in strict:
        answer = {**answer, "distractor_makes_sense": True}
    if "theme" not in strict:
        answer = {**answer, "on_theme": True}
    found = judge.verdict(item, answer, text)
    if (found and found[0] == "judge_unconnected" and "quotes" not in strict
            and answer.get("claim_follows_from_names") is not False):
        people = tuple(c["clue_label"] + ":" for c in item.get("clues") or [] if c.get("is_human"))
        misses = judge.missing_evidence(item, answer, text)
        if all(("not in the post" in m or m.endswith("no evidence")) and not m.startswith(people)
               for m in misses):
            found = judge.verdict(item, answer, None)   # the other checks, without the quotes
    return found


def check_post(item: dict[str, Any], row: dict[str, str], checker: judge.Judge | None,
               aliases: dict[str, dict[str, Any]], limit: int | None) -> dict[str, Any]:
    text = row["text"].strip()
    names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
    rule = finalise.check(item, text, names, None, limit)
    record: dict[str, Any] = {
        "item_id": item["item_id"], "condition": item["condition"],
        "place_name": item["place_name"], "country": row.get("country") or "",
        "clues": [c["clue_label"] for c in item.get("clues") or []],
        "noise_entity": (item.get("noise_entity") or {}).get("clue_label") or "",
        "text": text, "rule": list(rule) if rule else None,
        "audited_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    }
    if checker is not None:
        result = checker.check(item, text)
        record.update(model=checker.model, answer=result.get("answer"), error=result.get("error"),
                      usage=result.get("usage"))
    return record


def summarise(records: list[dict[str, Any]], posts: dict[str, dict[str, Any]],
              strict: frozenset[str], out: Path) -> None:
    """Tables on the screen, and the flagged posts and counts as CSV."""
    flagged, by_reason, by_condition, errors = [], Counter(), Counter(), 0
    totals = Counter(r["condition"] for r in records)
    for r in records:
        item = posts.get(r["item_id"])
        verdict = judge_verdict(item, r.get("answer"), r["text"], strict) if item else None
        if r.get("error"):
            errors += 1
        rule = r.get("rule")
        if not (rule or verdict or r.get("error")):
            continue
        status = (verdict or ("judge_error", r.get("error") or "") if not rule else rule)[0]
        by_reason[(r["condition"], status)] += 1
        by_condition[r["condition"]] += 1
        flagged.append({
            "item_id": r["item_id"], "condition": r["condition"], "place_name": r["place_name"],
            "country": r.get("country", ""), "clues": "; ".join(r.get("clues") or []),
            "noise_entity": r.get("noise_entity", ""),
            "rule_status": rule[0] if rule else "", "rule_reason": rule[1] if rule else "",
            "judge_status": verdict[0] if verdict else ("judge_error" if r.get("error") else ""),
            "judge_reason": verdict[1] if verdict else (r.get("error") or ""),
            "text": r["text"]})

    print(f"\n{len(records)} posts audited, {len(flagged)} flagged "
          f"({100 * len(flagged) / max(1, len(records)):.1f}%), {errors} judge errors")
    print(f'\n{"condition":<16}{"audited":>8}{"flagged":>9}{"share":>8}')
    for condition in sorted(totals):
        n, bad = totals[condition], by_condition[condition]
        print(f"{condition:<16}{n:>8}{bad:>9}{100 * bad / n:>7.1f}%")
    reasons = Counter()
    for (_, status), n in by_reason.items():
        reasons[status] += n
    print(f'\n{"reason":<24}{"posts":>6}')
    for status, n in reasons.most_common():
        print(f"{status:<24}{n:>6}")

    stem = out.with_suffix("")
    finalise.write_csv(Path(f"{stem}_flagged.csv"), FLAGGED_COLUMNS, flagged)
    finalise.write_csv(Path(f"{stem}_summary.csv"), ["condition", "reason", "posts", "audited"],
                       [{"condition": c, "reason": s, "posts": n, "audited": totals[c]}
                        for (c, s), n in sorted(by_reason.items())])
    print(f"\nwrote {stem}_flagged.csv and {stem}_summary.csv")


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--model", help='the audit model, "<provider>:<model>" (vse:Qwen3.8-27B)')
    parser.add_argument("--list-models", metavar="PROVIDER", help="print the models a provider offers and stop")
    parser.add_argument("--out", type=Path, default=None,
                        help="the audit log (default output/audit/<model>.jsonl)")
    parser.add_argument("--sample", type=int, default=None, metavar="N", help="audit N random posts")
    parser.add_argument("--seed", type=int, default=SAMPLE_SEED, help=f"picks the sample (default {SAMPLE_SEED})")
    parser.add_argument("--only-condition", action="append", default=None, metavar="CONDITION")
    parser.add_argument("--item-ids", type=Path, default=None, help="a file of item ids, one per line")
    parser.add_argument("--rules-only", action="store_true", help="the string checks only, no model")
    parser.add_argument("--strict-noise", action="store_true",
                        help="count a distractor that makes no sense as a failure")
    parser.add_argument("--strict-theme", action="store_true",
                        help="count a post off the plan's sub-topic as a failure")
    parser.add_argument("--strict-quotes", action="store_true",
                        help="count a non-person clue whose link the judge did not quote as a failure")
    parser.add_argument("--summary", action="store_true", help="summarise the existing log, call nothing")
    parser.add_argument("--workers", type=int, default=1, help="posts checked at once (default 1)")
    args = parser.parse_args(argv)

    config = gen.load_config(args.config)
    gen.load_dotenv(args.env_file)
    if args.list_models:
        print("\n".join(list_models(config, args.list_models)))
        return 0
    if not args.model and not args.rules_only:
        raise SystemExit("give --model <provider>:<model>, or --rules-only")
    if args.model:
        config = audit_config(config, args.model)
    out = args.out or Path(config["paths"]["dataset_csv"]).parent / "audit" / (
        (slug(args.model) if args.model else "rules") + ".jsonl")

    posts = load_posts(config)
    by_id = {item["item_id"]: item for item, _ in posts}
    # A failed call is asked again, and so is a post rewritten since it was
    # audited (regenerate.py): the log's last row for an item counts only if it
    # is about the text the dataset holds now.
    current = {item["item_id"]: row["text"].strip() for item, row in posts}
    done = {r["item_id"]: r for r in (gen.read_jsonl(out) if out.exists() else [])
            if (args.rules_only or r.get("answer") is not None) and current.get(r["item_id"]) == r["text"]}
    if not args.summary:
        todo = [p for p in select(posts, args) if p[0]["item_id"] not in done]
        print(f"{len(posts)} posts in the dataset, {len(done)} already audited, {len(todo)} to do"
              + (f" with {args.model}" if args.model else ""))
        if todo:
            out.parent.mkdir(parents=True, exist_ok=True)
            checker = None if args.rules_only else judge.Judge(config)
            aliases = finalise.load_aliases(Path(config["paths"]["place_inputs"]))
            limit = finalise.char_limit(config.get("finalise"))
            delay = float(gen.gen_setting(config["generation"], "delay_between_calls_seconds"))
            lock, count = threading.Lock(), [0]

            def work(post: tuple[dict[str, Any], dict[str, str]]) -> None:
                record = check_post(post[0], post[1], checker, aliases, limit)
                with lock:
                    with out.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if record.get("answer") is not None or args.rules_only:
                        done[record["item_id"]] = record
                    count[0] += 1
                    if count[0] % 25 == 0 or count[0] == len(todo):
                        print(f"  {count[0]}/{len(todo)}", flush=True)
                if checker is not None:
                    time.sleep(delay)

            try:
                with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
                    list(pool.map(work, todo))
            except KeyboardInterrupt:
                print("\nstopped; run the same command again to go on from here")
    if not done:
        print("nothing audited yet")
        return 0
    strict = frozenset(k for k in ("noise", "theme", "quotes") if getattr(args, "strict_" + k))
    summarise(list(done.values()), by_id, strict, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
