"""Small shared I/O helpers: UTF-8 console, JSONL and CSV/text atomic writes."""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def utf8_stdout() -> None:
    """Make stdout UTF-8 and line-buffered (labels are not ASCII; runs take minutes).

    Safe on stdout objects that cannot be reconfigured.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover - older/odd stdout objects
        pass


def read_jsonl(path: Path, strict: bool = True) -> list[dict]:
    """Read a JSONL file (file is closed). Blank lines are skipped.

    strict=True raises ValueError naming the file and line on a bad line;
    strict=False skips bad lines, warning once on stderr with the count.
    """
    rows: list[dict] = []
    skipped = 0
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError as exc:
                if strict:
                    raise ValueError(f"{path}:{number}: invalid JSON ({exc})") from exc
                skipped += 1
    if skipped:
        print(f"  ! {path}: skipped {skipped} unreadable line(s)", file=sys.stderr)
    return rows


def _replace_from_tmp(path: Path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_text_atomic(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a temp file then `os.replace` it over `path`."""
    def write(tmp: Path) -> None:
        with tmp.open("w", encoding=encoding, newline="\n") as handle:
            handle.write(text)
    _replace_from_tmp(path, write)


def write_jsonl_atomic(path: Path, rows: Iterable[dict]) -> int:
    """Write one JSON object per line atomically (LF newlines). Returns the row count."""
    count = 0

    def write(tmp: Path) -> None:
        nonlocal count
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    _replace_from_tmp(path, write)
    return count


def write_csv_atomic(path: Path, fieldnames: list[str], rows: Iterable[dict],
                     encoding: str = "utf-8") -> int:
    """Write a CSV with a header atomically. Returns the row count.

    Use encoding="utf-8-sig" for files Excel should open with the right charset.
    """
    count = 0

    def write(tmp: Path) -> None:
        nonlocal count
        with tmp.open("w", encoding=encoding, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                count += 1
    _replace_from_tmp(path, write)
    return count


def leaks_strict(place: str, label: str) -> bool:
    """The specified name-leak test: case-insensitive substring of the place name.

    Identical to the copies in 03_filter_clues.py and 05_name_leakage.py.
    """
    return place.casefold() in label.casefold()
