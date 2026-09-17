"""Run a set of (configuration, dataset) pairs through src/runner.py, one at a
time, and log each one.

    python scripts/run_all.py                       # every runnable config x dataset
    python scripts/run_all.py --configs refined gemma
    python scripts/run_all.py --datasets social_media_clean

Sequential on purpose. Two runner processes would contend for the Wikidata
rate limiter, and each holds its cache in memory and writes the pickle on
exit, so concurrent runs lose whichever process finishes first. Runs are
resumable: a pair already complete in results/predictions/ costs nothing.

Per-run stdout goes to results/logs/run_<config>__<dataset>.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from config import load_all_configs, load_datasets   # noqa: E402

LOG_DIR = REPO_ROOT / "results" / "logs"

# Cheap configurations first, so the expensive ones are never what stands
# between you and a first look at the numbers.
ORDER = ["baseline_majority", "gemma", "groq_oss", "groq_llama",
         "baseline_population", "refined", "refined_search",
         "routed_gemma", "refined_gemma", "refined_groq_oss",
         "refined_groq_llama", "two_stage", "mordecai3"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip", nargs="*", default=["mordecai3"],
                   help="configurations to leave out (default: mordecai3, which "
                        "cannot run in this environment)")
    args = p.parse_args()

    all_configs = load_all_configs()
    names = args.configs or [c for c in ORDER if c in all_configs]
    names = [n for n in names if n not in (args.skip or [])]
    datasets = args.datasets or list(load_datasets())

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    total = len(names) * len(datasets)
    done = 0
    t_start = time.time()
    failures = []

    for name in names:
        for ds in datasets:
            done += 1
            log = LOG_DIR / f"run_{name}__{ds}.log"
            print(f"[{done}/{total}] {name} x {ds} -> {log.name}", flush=True)
            cmd = [sys.executable, str(REPO_ROOT / "src" / "runner.py"),
                   "--config", name, "--dataset", ds]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            t0 = time.time()
            with log.open("w", encoding="utf-8") as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=REPO_ROOT)
            mins = (time.time() - t0) / 60
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"      {status} in {mins:.1f} min", flush=True)
            if proc.returncode != 0:
                failures.append((name, ds, log))

    print(f"\ntotal {(time.time() - t_start) / 60:.1f} min")
    if failures:
        print("failures:")
        for name, ds, log in failures:
            print(f"  {name} x {ds} -- see {log.relative_to(REPO_ROOT)}")
    else:
        print("all runs completed")


if __name__ == "__main__":
    main()
