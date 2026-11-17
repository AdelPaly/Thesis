#!/usr/bin/env python3
"""
Write a new post for the items taken out of the dataset by hand.

    python regenerate.py            # every item in output/manual_removed.csv
    python regenerate.py --dry-run  # show the prompts, send nothing
    python regenerate.py --failed NOISE_EXPLICIT   # see the last paragraph

A post in output/manual_removed.csv passed the rules and the judge and was
still wrong (read by hand after the audit, audit.py). Its item is sent to the
writer again, the same plan item and the same clue, with the removed post and
the reason it was removed as the retry feedback generate.py gives after a
failed try. The new post goes through the same checks as any other
(finalise.check, then judge.py), up to judge.max_tries tries.

An accepted post is appended to output/raw_generations.jsonl as the item's
latest row; its try_log holds the old tries, the removal and the new tries. The
item then leaves manual_removed.csv, so finalise.py takes the new post. An item
that fails every try stays in the list and out of the dataset.

--failed CONDITION does the same for the posts of that condition that failed
their tries or finalise.py's checks (output/finalise_report.csv) at a place
whose EXPLICIT post was kept: the failed post and its reason are the feedback,
and the item gets judge.max_tries further tries. Used on 2026-11-17 for
NOISE_EXPLICIT, which has no reserve, so that it covers the places EXPLICIT
does. manual_removed.csv is not touched.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
import time
from pathlib import Path

import finalise
import generate as gen
import judge as judge_module
import plan

HERE = Path(__file__).resolve().parent
COLUMNS = ["item_id", "condition", "place_name", "removed", "status", "reason"]
# finalise_report.csv statuses that are not a failed post
NOT_FAILED = {"kept", "balanced_out", "reserve_unused", "swapped_out", "no_explicit",
              "not_generated", "manual_removed", "no_coordinates"}


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and send nothing")
    parser.add_argument("--failed", metavar="CONDITION", default=None,
                        help="retry this condition's failed posts where the place has a kept EXPLICIT post")
    args = parser.parse_args(argv)

    gen.load_dotenv(args.env_file)
    config = gen.load_config(args.config)
    paths = config["paths"]
    out_path = Path(paths["raw_generations"])
    removed_path = Path(paths["dataset_csv"]).with_name("manual_removed.csv")
    if args.failed:
        with Path(paths["finalise_report"]).open(encoding="utf-8", newline="") as f:
            report = list(csv.DictReader(f))
        covered = {r["place_qid"] for r in report if r["condition"] == "EXPLICIT" and r["status"] == "kept"}
        removed = [r for r in report if r["condition"] == args.failed and r["place_qid"] in covered
                   and r["status"] not in NOT_FAILED]
    else:
        with removed_path.open(encoding="utf-8", newline="") as f:
            removed = list(csv.DictReader(f))
    if not removed:
        print("nothing to rewrite")
        return 0

    items = {i["item_id"]: i for i in gen.read_jsonl(Path(paths["item_plan"]))}
    latest = gen.latest_rows(gen.read_jsonl(out_path))
    exemplars = gen.load_exemplars(Path(paths["style_exemplars"]))
    aliases = finalise.load_aliases(Path(paths["place_inputs"]))
    limits = finalise.char_limit(config.get("finalise"))
    config_hash = gen.config_fingerprint(config)
    max_tries = int((config.get("judge") or {}).get("max_tries", 3))
    delay = float(gen.gen_setting(config["generation"], "delay_between_calls_seconds"))

    client = checker = out_log = raw_log = None
    if not args.dry_run:
        client, checker = gen.ChatClient(config), judge_module.Judge(config)
        out_log, raw_log = gen.AppendLog(out_path), gen.AppendLog(Path(paths["raw_response_log"]))
        print(f"model {client.model}, judge {checker.model}, config {config_hash}\n")

    still_out = []
    for entry in removed:
        item, old = items[entry["item_id"]], latest[entry["item_id"]]
        work = copy.deepcopy(item)
        if old.get("clues_used"):
            work["clues"] = old["clues_used"]
        names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
        try_log = list(old.get("try_log") or [])
        feedback = {"status": entry["status"], "reason": entry["reason"], "text": old.get("post_text")}
        if not args.failed:
            try_log.append({**feedback, "try": len(try_log) + 1, "manual_removed": entry["removed"]})
        first = len(try_log)
        record = None
        for n in range(first + 1, first + 1 + max_tries):
            seed = gen.try_seed(item["item_id"], n)
            messages = gen.build_messages(work, config, exemplars)
            last = try_log[-1] if len(try_log) > first else feedback
            messages[-1]["content"] += "\n\n" + gen.retry_feedback(last, config["prompt"])
            if args.dry_run:
                print("=" * 72, f'\n{item["item_id"]} {item["condition"]} {item["place_name"]}\n',
                      messages[-1]["content"], "\n")
                break
            content, metadata, attempts, api_error = None, {}, [], None
            try:
                content, metadata, attempts = gen.call_with_backoff(client, messages, config, seed=seed)
            except gen.ApiError as exc:
                api_error, attempts, metadata = str(exc), exc.attempt_log, {"http_status": exc.status}
            raw_log.write({**gen.build_raw_entry(work, client, config_hash, messages, content,
                                                 metadata, attempts, api_error), "try": n, "seed": seed})
            record, outcome = gen.build_record(work, client.model, config_hash, content, metadata, api_error)
            judgement = None
            if outcome != "ok":
                verdict = (outcome, str(record.get("error"))[:200])
            else:
                verdict = finalise.check(work, record["post_text"], names, None, limits)
                if verdict is None:
                    judgement = checker.check(work, record["post_text"])
                    verdict = tuple(judgement["verdict"]) if judgement["verdict"] else None
            try_log.append({"try": n, "seed": seed, "status": verdict[0] if verdict else "kept",
                            "reason": verdict[1] if verdict else "",
                            "clue_qids": [c["clue_qid"] for c in work["clues"]],
                            "text": record.get("post_text"),
                            "writer_usage": (metadata or {}).get("usage"),
                            "judge_usage": (judgement or {}).get("usage")})
            record["judge"] = judgement
            time.sleep(delay)
            if verdict is None:
                break
        if args.dry_run:
            continue
        accepted = try_log[-1]["status"] == "kept"
        steps = " > ".join(t["status"] for t in try_log[first:])   # the new tries
        print(f'{item["condition"]:<3} {item["place_name"][:24]:<24} {steps}')
        if not accepted:
            still_out.append(entry)
            continue
        record.update(tries=len(try_log), try_log=try_log, accepted=True, final_status="kept",
                      final_reason="", clue_swaps=old.get("clue_swaps") or [],
                      clues_used=old.get("clues_used"), judge_model=checker.model,
                      swapped_from=old.get("swapped_from"))
        out_log.write(record)
        print("    ", record["post_text"])

    if not args.dry_run:
        if not args.failed:
            finalise.write_csv(removed_path, COLUMNS, still_out)
        print(f"\n{len(removed) - len(still_out)} of {len(removed)} rewritten, "
              f"{len(still_out)} still out. Run finalise.py for the dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
