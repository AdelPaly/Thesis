"""Rescoring smoke test: do the committed tables reproduce from the saved predictions?

    python scripts/check_rescore.py              # temp dir, deleted afterwards
    python scripts/check_rescore.py --keep DIR   # write the fresh tables to DIR

Re-runs the evaluation layer -- scripts/analyse.py (which scores with the
functions in scripts/score.py) and scripts/score.py itself -- on
results/predictions/ into a temporary directory, never into results/summary.
Then:

  1. every CSV committed in results/summary/ is compared with the freshly
     written file of the same name (same columns, same rows, numbers equal to
     within --tol);
  2. the whole evaluation is run a second time and the two fresh runs are
     compared, which separates nondeterminism (e.g. an unseeded bootstrap) from
     stale committed tables.

Prints PASS/FAIL per table; exits 0 only if everything passes. Loads no model
and makes no network call.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SUMMARY_DIR = REPO_ROOT / "results" / "summary"
SPLIT = "test"

sys.path.insert(0, str(REPO_ROOT / "src"))
from keys import current_config_name  # noqa: E402  (stdlib-only rename map)


def run_evaluation(out_dir: Path) -> None:
    """Write every evaluation table into out_dir with the repo's own CLIs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmds = [
        [sys.executable, str(REPO_ROOT / "scripts" / "analyse.py"), "all",
         "--split", SPLIT, "--out", str(out_dir)],
        [sys.executable, str(REPO_ROOT / "scripts" / "score.py"), "--split", SPLIT,
         "--out", str(out_dir / f"score_{SPLIT}.csv")],
    ]
    for cmd in cmds:
        res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True,
                             encoding="utf-8", errors="replace")
        if res.returncode != 0:
            sys.stderr.write(res.stdout[-2000:] + res.stderr[-4000:])
            raise SystemExit(f"evaluation failed: {' '.join(cmd[1:3])}")


def _num(v: str):
    try:
        return float(v)
    except ValueError:
        return None


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Header and rows; a `config` column is mapped to current configuration
    names, so a table written before a rename is compared on its values."""
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return [], []
    header, body = rows[0], rows[1:]
    if "config" in header:
        i = header.index("config")
        for row in body:
            row[i] = current_config_name(row[i])
    return header, body


def compare(ref: Path, new: Path, tol: float) -> list[str]:
    """Problems found comparing two CSVs; empty means identical within tol."""
    if not new.exists():
        return [f"not produced by the current code ({new.name} missing)"]
    h_ref, r_ref = read_table(ref)
    h_new, r_new = read_table(new)
    problems = []
    if h_ref != h_new:
        problems.append(f"columns differ: only committed {sorted(set(h_ref) - set(h_new))}, "
                        f"only fresh {sorted(set(h_new) - set(h_ref))}")
    common = [c for c in h_ref if c in h_new]
    # Key = columns holding any non-numeric value (names, conditions, levels).
    key_cols = [c for c in common
                if any(row[h_ref.index(c)] and _num(row[h_ref.index(c)]) is None
                       for row in r_ref)]
    val_cols = [c for c in common if c not in key_cols]

    def index(header, rows):
        out = {}
        for row in rows:
            out.setdefault(tuple(row[header.index(c)] for c in key_cols), []).append(
                [row[header.index(c)] for c in val_cols])
        return out

    a, b = index(h_ref, r_ref), index(h_new, r_new)
    only_ref, only_new = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_ref:
        problems.append(f"{len(only_ref)} row(s) only in committed, e.g. {only_ref[0]}")
    if only_new:
        problems.append(f"{len(only_new)} row(s) only in fresh, e.g. {only_new[0]}")
    diffs = []
    for key in sorted(set(a) & set(b)):
        if len(a[key]) != len(b[key]):
            diffs.append(f"{key}: {len(a[key])} vs {len(b[key])} rows")
            continue
        for va, vb in zip(a[key], b[key]):
            for col, x, y in zip(val_cols, va, vb):
                nx, ny = _num(x), _num(y)
                same = (abs(nx - ny) <= tol) if nx is not None and ny is not None else x == y
                if not same:
                    diffs.append(f"{key} {col}: committed {x!r} fresh {y!r}")
    if diffs:
        problems.append(f"{len(diffs)} value difference(s), e.g. " + "; ".join(diffs[:3]))
    if not problems and len(r_ref) != len(r_new):
        problems.append(f"row count {len(r_ref)} vs {len(r_new)}")
    return problems


def report(label: str, problems: list[str]) -> bool:
    print(f"  {'PASS' if not problems else 'FAIL'}  {label}")
    for p in problems:
        print(f"        {p}")
    return not problems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tol", type=float, default=1e-9, help="numeric tolerance")
    p.add_argument("--keep", default=None,
                   help="directory to keep the fresh tables in (default: temp, deleted)")
    args = p.parse_args()

    base = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="rescore_"))
    if base.resolve() == SUMMARY_DIR.resolve() or SUMMARY_DIR.resolve() in base.resolve().parents:
        raise SystemExit("refusing to write inside results/summary")
    run1, run2 = base / "run1", base / "run2"
    try:
        print(f"rescoring results/predictions -> {base}")
        run_evaluation(run1)
        run_evaluation(run2)

        ok = True
        committed = sorted(SUMMARY_DIR.glob("*.csv"))
        print(f"\n1. committed tables vs fresh ({len(committed)} table(s), tol={args.tol})")
        if not committed:
            print("  (no committed tables)")
        for ref in committed:
            ok &= report(ref.name, compare(ref, run1 / ref.name, args.tol))

        fresh = sorted(run1.glob("*.csv"))
        print(f"\n2. determinism: fresh run 1 vs run 2 ({len(fresh)} table(s))")
        for t in fresh:
            ok &= report(t.name, compare(t, run2 / t.name, 0.0))
        uncommitted = sorted({t.name for t in fresh} - {r.name for r in committed})
        if uncommitted:
            print("\n  written by the current code but not committed: " + ", ".join(uncommitted))

        print(f"\n{'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
