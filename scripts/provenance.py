"""Which code and which comparison list a result table was made with.

Every table the evaluation writes ends in two columns, the same on every row:

  eval_code_rev       short git revision of the working tree when the table was
                      written, with `+dirty` if scripts/, src/keys.py or
                      configs/ had uncommitted changes
  comparisons_sha256  first 12 hex digits of the sha256 of
                      configs/comparisons.yaml (line endings normalised)

So a committed table names the revision of the evaluation code and the version
of the planned comparisons it came from. To get a clean stamp: commit the code,
write the tables, commit the tables. `check_rescore.py` leaves the two columns
out of its comparison, since they change with every commit by design.

Standard library only.
"""
from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
COMPARISONS_PATH = REPO_ROOT / "configs" / "comparisons.yaml"
EVALUATION_PATHS = ["scripts", "src/keys.py", "configs"]
COLUMNS = ["eval_code_rev", "comparisons_sha256"]


def _git(*args: str) -> str | None:
    try:
        res = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


@functools.lru_cache(maxsize=1)
def eval_code_rev() -> str:
    rev = _git("rev-parse", "--short", "HEAD")
    if not rev:
        return "unknown"
    dirty = _git("status", "--porcelain", "--", *EVALUATION_PATHS)
    return rev + ("+dirty" if dirty else "")


@functools.lru_cache(maxsize=1)
def comparisons_sha256() -> str:
    try:
        data = COMPARISONS_PATH.read_bytes().replace(b"\r\n", b"\n")
    except OSError:
        return "missing"
    return hashlib.sha256(data).hexdigest()[:12]


def values() -> list[str]:
    """The two values, in the order of COLUMNS."""
    return [eval_code_rev(), comparisons_sha256()]


def as_dict() -> dict[str, str]:
    return dict(zip(COLUMNS, values()))
