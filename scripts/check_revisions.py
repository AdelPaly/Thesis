"""Did the code revision change any answer in the committed predictions?

    python scripts/check_revisions.py            # parts 1-3, from git and results/ only
    python scripts/check_revisions.py --replay   # also part 4, needs the Wikidata cache

The prediction files were written at more than one code revision (`code_rev`
on every record). This prints what is needed to judge whether that matters:

  1. records per revision, per prediction file, with the first and last run time;
  2. for a file written at two revisions on either side of the 512-token
     output limit (7cd7ba4): the longest stored reply written before the limit,
     in characters. A reply far below 512 tokens cannot have been cut by it;
  3. the files under src/ and configs/ that differ between consecutive
     revisions, so each difference can be set against the configurations that
     ran on either side of it;
  4. (--replay) the two LLM-only configurations resolved again from their
     stored replies, with the resolution code as it stood at the last run
     revision and with the current code: how many answers the two give
     differently, and how many stored answers differ from what the code gives
     now. The second number counts lookups that failed during the run, not a
     difference in code.

Reads git history and results/predictions/. Changes nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = REPO_ROOT / "results" / "predictions"
TOKEN_LIMIT_REV = "7cd7ba4"       # "Limit LLM answers to 512 output tokens"
LAST_RUN_LLM_GEO_REV = "44cacce"  # src/llm_geo.py as at the last revision a run was made at
LLM_ONLY = ["gemma4", "qwen3_vl"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout.strip()


def records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def is_ancestor(a: str, b: str) -> bool:
    return subprocess.run(["git", "merge-base", "--is-ancestor", a, b],
                          cwd=REPO_ROOT).returncode == 0


def part_1_and_3() -> list[str]:
    print("1. records per code revision\n")
    revisions, before_limit = set(), {}
    for path in sorted(PREDICTIONS.glob("*.jsonl")):
        by_rev = defaultdict(list)
        for rec in records(path):
            by_rev[rec.get("code_rev", "unknown")].append(rec)
        for rev, recs in sorted(by_rev.items(), key=lambda kv: min(r["run_utc"] for r in kv[1])):
            times = [r["run_utc"] for r in recs]
            print(f"  {path.stem:38} {rev:10} {len(recs):5}  {min(times)}  {max(times)}")
            base = rev.split("+")[0]
            revisions.add(base)
            if len(by_rev) > 1 and base != "unknown" and not is_ancestor(TOKEN_LIMIT_REV, base):
                replies = [r.get("llm_raw_response") or "" for r in recs]
                before_limit[(path.stem, rev)] = (len(recs), max(map(len, replies)))
    print("\n2. files written on both sides of the 512-token limit "
          f"({TOKEN_LIMIT_REV}): replies written before it\n")
    for (name, rev), (n, longest) in before_limit.items():
        print(f"  {name} at {rev}: {n} records, longest reply {longest} characters")
    if not before_limit:
        print("  none")
    order = git("rev-list", "--reverse", "HEAD").split("\n")
    full = {rev: git("rev-parse", rev) for rev in revisions if rev != "unknown"}
    return sorted(full, key=lambda rev: order.index(full[rev]))


def part_2(revisions: list[str]) -> None:
    print("\n3. files under src/ and configs/ that differ between consecutive revisions\n")
    chain = [*revisions, "HEAD"]
    for a, b in zip(chain, chain[1:]):
        for commit in git("log", "--reverse", "--format=%h %s", f"{a}..{b}").split("\n"):
            if not commit:
                continue
            rev = commit.split(" ")[0]
            files = git("diff", "--name-only", f"{rev}~1", rev, "--", "src", "configs")
            print(f"  {commit}")
            for name in files.split("\n") if files else ["(nothing under src/ or configs/)"]:
                print(f"      {name}")
        print(f"  -- runs were made at {b}" if b != "HEAD" else "  -- HEAD")


def part_4() -> None:
    print("\n4. LLM-only answers resolved again from the stored replies\n")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import resolver  # noqa: E402
    from llm_common import parse_json_object  # noqa: E402

    with tempfile.TemporaryDirectory() as tmp:
        old_path = Path(tmp) / "llm_geo_old.py"
        old_path.write_text(git("show", f"{LAST_RUN_LLM_GEO_REV}:src/llm_geo.py"), encoding="utf-8")
        spec = importlib.util.spec_from_file_location("llm_geo_old", old_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    old = module.LLMGeoPipeline.__new__(module.LLMGeoPipeline)

    for name in LLM_ONLY:
        n = code_differs = stored_differs = 0
        examples = []
        for rec in records(PREDICTIONS / f"{name}__synthetic.jsonl"):
            raw = rec.get("llm_raw_response")
            parsed = parse_json_object(raw) if raw else None
            then = None if parsed is None or parsed.get("refused") else old._resolve_to_candidate(parsed)
            now, _ = resolver.resolve_first_hit(raw)
            n += 1
            code_differs += then != now
            if (now or {}).get("qid") != rec.get("predicted_qid"):
                stored_differs += 1
                examples.append(f"{rec['item_id']} stored {rec.get('predicted_qid')} "
                                f"now {(now or {}).get('qid')}")
        print(f"  {name}: {n} records; code at {LAST_RUN_LLM_GEO_REV} and current code differ on "
              f"{code_differs}; stored answer differs from the current code's on {stored_differs}")
        for line in examples:
            print(f"      {line}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--replay", action="store_true",
                   help="part 4: resolve the LLM-only replies again (needs src/.wikidata_cache)")
    args = p.parse_args()
    revisions = part_1_and_3()
    part_2(revisions)
    if args.replay:
        part_4()


if __name__ == "__main__":
    main()
