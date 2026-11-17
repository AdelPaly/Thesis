#!/usr/bin/env python3
"""
Write a new post for items whose kept post should be replaced.

    python regenerate.py                           # every item in output/manual_removed.csv
    python regenerate.py --failed NOISE_EXPLICIT   # that condition's failed posts
    python regenerate.py --stock-scenes            # posts that copy a stock scene
    python regenerate.py --burials                 # posts with a tomb in the wrong place
    python regenerate.py --dry-run ...             # show the prompts, send nothing

The item goes to the writer again: the same plan item and the same clue. The
new post goes through the same checks as any other (finalise.check, then
judge.py), up to judge.max_tries tries. An accepted post is appended to
output/raw_generations.jsonl as the item's latest row; its try_log holds the
old tries and the new ones, and "regenerated" says which of the three modes
below wrote it. An item that fails every try keeps what it had.

manual_removed.csv (default)
    A post that passed the rules and the judge and was still wrong, read by
    hand after the audit (audit.py). The removed post and the reason are the
    retry feedback generate.py gives after a failed try. On success the item
    leaves the list, so finalise.py takes the new post; on failure it stays in
    the list and out of the dataset.

--failed CONDITION
    The posts of that condition that failed their tries or finalise.py's
    checks (output/finalise_report.csv) at a place whose EXPLICIT post was
    kept: the failed post and its reason are the feedback. Used on 2026-11-17
    for NOISE_EXPLICIT, which has no reserve.

--stock-scenes
    The kept posts that follow one of the prompt's former worked examples
    (stock_scene() below: vehicles idling outside an office, a crowd leaving
    cups and wrappers). A third of the dataset shared those two scenes
    (2026-11-18). The examples are out of the prompt now, so the first new try
    is a fresh post from the current prompt with no feedback; a new post that
    is a stock scene again is rejected like any other failed try, unless its
    planned topic is about traffic fumes or litter. An item that fails keeps
    its old post, so the dataset does not shrink.

--burials
    The kept posts that check_burials.py lists as "elsewhere" or
    "elsewhere_by_death" in output/audit/burial_check.csv: the post speaks of
    a person's tomb or grave, and by Wikidata the person is buried, or with no
    burial on record died, over 100 km away. The "unknown" rows are rewritten
    too: a tomb that cannot be checked is not kept (2026-11-17, user). The post and where
    the person is buried are the feedback, and a new post that mentions a
    tomb, grave or burial again is rejected. An item that fails keeps its old
    post.
"""
from __future__ import annotations

import argparse
import copy
import csv
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import finalise
import generate as gen
import judge as judge_module
import plan

HERE = Path(__file__).resolve().parent
COLUMNS = ["item_id", "condition", "place_name", "removed", "status", "reason"]
# finalise_report.csv statuses that are not a failed post
NOT_FAILED = {"kept", "balanced_out", "reserve_unused", "swapped_out", "no_explicit",
              "not_generated", "manual_removed", "no_coordinates"}

# The two scenes the prompt's worked examples turned into templates.
_VEHICLE = re.compile(r"\b(trucks?|lorr(y|ies)|vans?|coaches|buses|engines?)\b", re.I)
_IDLE = re.compile(r"\bidl(e|es|ed|ing)\b", re.I)
_OFFICE = re.compile(r"\boffices?\b", re.I)
_LITTER = re.compile(r"\b(cups?|wrappers?|bottles|cans|crisp packets|packets)\b", re.I)
_CROWD = re.compile(r"\b(gigs?|crowds?|fans|tourists?|tour groups?|match|festival|fair|visitors)\b", re.I)
_LEFT = re.compile(r"\b(left|leave|leaves|leaving|covered|knee deep|buried|carpeted|piled|piling|"
                   r"overflow\w*|everywhere|all over)\b", re.I)
# planned topics where the scene is the topic itself
_TRAFFIC_TOPIC = re.compile(r"traffic|air|fumes|diesel|transport|road", re.I)
_LITTER_TOPIC = re.compile(r"litter|waste|rubbish|plastic|recycl|tourism|festival|clean-up|deposit|landfill", re.I)


def stock_scene(text: str) -> str | None:
    """Which stock scene the post is: "idling_vehicles", "vehicles_at_offices",
    "crowd_litter", "litter_left", or None."""
    if _VEHICLE.search(text) and _IDLE.search(text):
        return "idling_vehicles"
    if _VEHICLE.search(text) and _OFFICE.search(text):
        return "vehicles_at_offices"
    if _LITTER.search(text) and _LEFT.search(text):
        return "crowd_litter" if _CROWD.search(text) else "litter_left"
    return None


def stock_verdict(item: dict[str, Any], text: str) -> tuple[str, str] | None:
    """The try's verdict when a new post is a stock scene its topic does not ask for."""
    scene = stock_scene(text)
    topic = item.get("env_theme") or ""
    if scene in ("idling_vehicles", "vehicles_at_offices") and not _TRAFFIC_TOPIC.search(topic):
        return "stock_scene", "vehicles with engines running, or parked outside an office"
    if scene == "crowd_litter" and not _LITTER_TOPIC.search(topic):
        return "stock_scene", "a crowd or visitors leaving cups, bottles or wrappers"
    return None


BURIAL = re.compile(r"\b(tomb|tombs|grave|graves|graveside|buried|burial|cemetery|mausoleum|resting place)\b", re.I)


def entries_for(args: argparse.Namespace, paths: dict[str, str], latest: dict[str, Any],
                removed_path: Path) -> list[dict[str, str]]:
    """The items to rewrite: item_id, condition, place_name, status, reason."""
    if args.burials:
        with (Path(paths["dataset_csv"]).parent / "audit" / "burial_check.csv").open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))   # elsewhere, elsewhere_by_death and unknown (2026-11-17, user)
        return [{"item_id": r["item_id"], "condition": r["condition"], "place_name": r["place"],
                 "status": "false_burial",
                 "reason": (f'{r["person"]} is buried at {r["buried_at"]}, {r["buried_km"]} km from here'
                            if r["verdict"] == "elsewhere" else
                            f'{r["person"]} died at {r["died_at"]}, {r["died_km"]} km from here, and is not buried here'
                            if r["verdict"] == "elsewhere_by_death" else
                            f'nobody can say where {r["person"]} is buried, so the post must not speak of a tomb or grave')}
                for r in rows if BURIAL.search((latest.get(r["item_id"]) or {}).get("post_text") or "")]
    if args.stock_scenes:
        with Path(paths["dataset_csv"]).open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        return [{"item_id": r["item_id"], "condition": r["signal_type"], "place_name": r["location_name"],
                 "status": "stock_scene", "reason": stock_scene(r["text"]) or ""}
                for r in rows if stock_scene(r["text"])
                and (latest.get(r["item_id"]) or {}).get("regenerated") != "stock_scenes"]
    if args.failed:
        with Path(paths["finalise_report"]).open(encoding="utf-8", newline="") as f:
            report = list(csv.DictReader(f))
        covered = {r["place_qid"] for r in report if r["condition"] == "EXPLICIT" and r["status"] == "kept"}
        return [r for r in report if r["condition"] == args.failed and r["place_qid"] in covered
                and r["status"] not in NOT_FAILED]
    with removed_path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts and send nothing")
    parser.add_argument("--failed", metavar="CONDITION", default=None,
                        help="retry this condition's failed posts where the place has a kept EXPLICIT post")
    parser.add_argument("--stock-scenes", action="store_true",
                        help="rewrite the kept posts that copy a stock scene")
    parser.add_argument("--burials", action="store_true",
                        help="rewrite the posts with a tomb where the person is not buried")
    parser.add_argument("--only-condition", action="append", default=None, metavar="CONDITION")
    parser.add_argument("--limit", type=int, default=None, help="stop after this many items")
    parser.add_argument("--workers", type=int, default=1, help="items written at once (default 1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the accepted rows here instead of raw_generations.jsonl (a pilot)")
    args = parser.parse_args(argv)
    mode = ("stock_scenes" if args.stock_scenes else "burials" if args.burials
            else "failed" if args.failed else "manual")

    gen.load_dotenv(args.env_file)
    config = gen.load_config(args.config)
    paths = config["paths"]
    out_path = args.out or Path(paths["raw_generations"])
    raw_path = out_path.with_name("raw_responses.jsonl") if args.out else Path(paths["raw_response_log"])
    removed_path = Path(paths["dataset_csv"]).with_name("manual_removed.csv")
    plan_items = gen.read_jsonl(Path(paths["item_plan"]))
    typed = gen.attach_clue_types(plan_items, Path(paths["clue_types"])) if paths.get("clue_types") else 0
    items = {i["item_id"]: i for i in plan_items}
    latest = gen.latest_rows(gen.read_jsonl(Path(paths["raw_generations"])))

    entries = entries_for(args, paths, latest, removed_path)
    if args.only_condition:
        entries = [e for e in entries if e["condition"] in args.only_condition]
    entries.sort(key=lambda e: e["item_id"])       # a hash: mixes conditions and places
    if args.limit:
        entries = entries[:args.limit]
    if not entries:
        print("nothing to rewrite")
        return 0

    exemplars = gen.load_exemplars(Path(paths["style_exemplars"]))
    aliases = finalise.load_aliases(Path(paths["place_inputs"]))
    limits = finalise.char_limit(config.get("finalise"))
    config_hash = gen.config_fingerprint(config)
    max_tries = int((config.get("judge") or {}).get("max_tries", 3))
    delay = float(gen.gen_setting(config["generation"], "delay_between_calls_seconds"))

    client = checker = out_log = raw_log = None
    if not args.dry_run:
        client, checker = gen.ChatClient(config), judge_module.Judge(config)
        out_log, raw_log = gen.AppendLog(out_path), gen.AppendLog(raw_path)
        print(f"{len(entries)} items, model {client.model}, judge {checker.model}, config {config_hash}, "
              f"{typed} clue types\n", flush=True)
    lock, done, still_out = threading.Lock(), [0, 0], []

    def work(entry: dict[str, str]) -> None:
        item, old = items[entry["item_id"]], latest[entry["item_id"]]
        todo = copy.deepcopy(item)
        if old.get("clues_used"):
            todo["clues"] = old["clues_used"]
        names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
        try_log = list(old.get("try_log") or [])
        # What the first new try is told: the removed or failed post and why, or
        # nothing (stock scenes: a fresh post from the current prompt).
        feedback = None if mode == "stock_scenes" else {
            "status": entry["status"], "reason": entry["reason"], "text": old.get("post_text")}
        if mode == "manual":
            try_log.append({**feedback, "try": len(try_log) + 1, "manual_removed": entry["removed"]})
        first = len(try_log)
        record = None
        for n in range(first + 1, first + 1 + max_tries):
            seed = gen.try_seed(item["item_id"], n)
            messages = gen.build_messages(todo, config, exemplars)
            last = try_log[-1] if len(try_log) > first else feedback
            if last:
                messages[-1]["content"] += "\n\n" + gen.retry_feedback(last, config["prompt"])
            if args.dry_run:
                with lock:
                    print("=" * 72, f'\n{item["item_id"]} {item["condition"]} {item["place_name"]}\n',
                          messages[-1]["content"], "\n")
                return
            content, metadata, attempts, api_error = None, {}, [], None
            try:
                content, metadata, attempts = gen.call_with_backoff(client, messages, config, seed=seed)
            except gen.ApiError as exc:
                api_error, attempts, metadata = str(exc), exc.attempt_log, {"http_status": exc.status}
            with lock:
                raw_log.write({**gen.build_raw_entry(todo, client, config_hash, messages, content,
                                                     metadata, attempts, api_error), "try": n, "seed": seed})
            record, outcome = gen.build_record(todo, client.model, config_hash, content, metadata, api_error)
            judgement = None
            if outcome != "ok":
                verdict = (outcome, str(record.get("error"))[:200])
            else:
                verdict = finalise.check(todo, record["post_text"], names, None, limits)
                if verdict is None and mode == "stock_scenes":
                    verdict = stock_verdict(todo, record["post_text"])
                if verdict is None and mode == "burials" and BURIAL.search(record["post_text"]):
                    verdict = ("false_burial", entry["reason"])
                if verdict is None:
                    judgement = checker.check(todo, record["post_text"])
                    verdict = tuple(judgement["verdict"]) if judgement["verdict"] else None
            try_log.append({"try": n, "seed": seed, "status": verdict[0] if verdict else "kept",
                            "reason": verdict[1] if verdict else "",
                            "clue_qids": [c["clue_qid"] for c in todo["clues"]],
                            "text": record.get("post_text"),
                            "writer_usage": (metadata or {}).get("usage"),
                            "judge_usage": (judgement or {}).get("usage")})
            record["judge"] = judgement
            time.sleep(delay)
            if verdict is None:
                break
        accepted = try_log[-1]["status"] == "kept"
        steps = " > ".join(t["status"] for t in try_log[first:])   # the new tries
        with lock:
            done[0] += 1
            done[1] += accepted
            print(f'[{done[0]}/{len(entries)}] {item["condition"]:<15}{item["place_name"][:22]:<23}{steps}', flush=True)
            if not accepted:
                still_out.append(entry)
                return
            record.update(tries=len(try_log), try_log=try_log, accepted=True, final_status="kept",
                          final_reason="", clue_swaps=old.get("clue_swaps") or [],
                          clues_used=old.get("clues_used"), judge_model=checker.model,
                          swapped_from=old.get("swapped_from"), regenerated=mode)
            out_log.write(record)
            if len(entries) <= 60:
                print("    ", record["post_text"], flush=True)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            list(pool.map(work, entries))
    except KeyboardInterrupt:
        print("\nstopped; run the same command again to go on")
    if not args.dry_run:
        if mode == "manual":
            finalise.write_csv(removed_path, COLUMNS, still_out)
        print(f"\n{done[1]} of {done[0]} rewritten, {done[0] - done[1]} keep what they had. "
              f"Run finalise.py for the dataset.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
