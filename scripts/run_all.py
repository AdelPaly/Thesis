"""Run a set of (configuration, dataset) pairs through src/runner.py, one at a
time, and log each one.

    python scripts/run_all.py                       # every runnable config x dataset
    python scripts/run_all.py --configs refined gemma4
    python scripts/run_all.py --datasets social_media_clean

Sequential on purpose. Two runner processes would contend for the Wikidata
rate limiter, and each holds its cache in memory and writes the pickle on
exit, so concurrent runs lose whichever process finishes first. Runs are
resumable: a pair already complete in results/predictions/ costs nothing.

Per-run stdout is appended to results/logs/run_<config>__<dataset>.log, each
run under a timestamped header, so re-running a finished pair never wipes the
log of the run that produced its predictions.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DATASET_REGISTRY = CONFIG_DIR / "datasets.yaml"
LOG_DIR = REPO_ROOT / "results" / "logs"

# Cheap first, so the expensive runs are never what stands between you and a
# first look at the numbers. Names absent from configs/ are skipped, and
# configurations present in configs/ but missing here still run (appended), so
# this is an ordering preference, not a registry.
ORDER = ["baseline_majority", "baseline_population",
         "gemma4",
         "dense_retrieval",
         "grounded_gemma", "grounded_gemma_linking",
         "two_stage", "refined"]


def config_names() -> list[str]:
    """Names of every configuration in configs/ (the `name:` key, else the file
    stem), read with yaml so run_all need not import src/."""
    names = []
    for p in sorted(CONFIG_DIR.glob("*.yaml")):
        if p == DATASET_REGISTRY:
            continue
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        names.append(raw.get("name", p.stem) if isinstance(raw, dict) else p.stem)
    return names


def dataset_names() -> list[str]:
    raw = yaml.safe_load(DATASET_REGISTRY.read_text(encoding="utf-8")) or {}
    return list(raw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip", nargs="*", default=[],
                   help="configurations to leave out")
    p.add_argument("--split", choices=["dev", "test"], default=None,
                   help="forwarded to runner.py: only items on this side of the split")
    p.add_argument("--overwrite", action="store_true",
                   help="forwarded to runner.py: redo pairs already complete")
    args = p.parse_args()

    all_configs = config_names()
    all_datasets = dataset_names()
    for label, given, known in (("configuration", (args.configs or []) + args.skip, all_configs),
                                ("dataset", args.datasets or [], all_datasets)):
        bad = [n for n in given if n not in known]
        if bad:
            p.error(f"unknown {label}(s): {', '.join(bad)}. Known: {', '.join(known)}")

    if args.configs:
        names = args.configs
    else:
        # ORDER first, then anything else configs/ contains, so adding a
        # configuration file is enough to get it run.
        names = [c for c in ORDER if c in all_configs]
        names += [c for c in sorted(all_configs) if c not in names]
    names = [n for n in names if n not in (args.skip or [])]
    datasets = args.datasets or all_datasets

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    total = len(names) * len(datasets)
    done = 0
    t_start = time.perf_counter()
    failures = []

    for name in names:
        for ds in datasets:
            done += 1
            log = LOG_DIR / f"run_{name}__{ds}.log"
            print(f"[{done}/{total}] {name} x {ds} -> {log.name}", flush=True)
            cmd = [sys.executable, str(REPO_ROOT / "src" / "runner.py"),
                   "--config", name, "--dataset", ds]
            if args.limit is not None:
                cmd += ["--limit", str(args.limit)]
            if args.split:
                cmd += ["--split", args.split]
            if args.overwrite:
                cmd += ["--overwrite"]
            t0 = time.perf_counter()
            with log.open("a", encoding="utf-8") as f:
                f.write(f"\n===== {datetime.now().isoformat(timespec='seconds')} "
                        f"{' '.join(cmd[1:])} =====\n")
                f.flush()
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                                      cwd=REPO_ROOT)
            mins = (time.perf_counter() - t0) / 60
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            print(f"      {status} in {mins:.1f} min", flush=True)
            if proc.returncode != 0:
                failures.append((name, ds, log))

    print(f"\ntotal {(time.perf_counter() - t_start) / 60:.1f} min")
    if failures:
        print("failures:")
        for name, ds, log in failures:
            print(f"  {name} x {ds} -- see {log.relative_to(REPO_ROOT)}")
    else:
        print("all runs completed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted; completed pairs are kept and a re-run resumes",
              file=sys.stderr)
        sys.exit(130)
