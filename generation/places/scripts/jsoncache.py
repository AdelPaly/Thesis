"""A small on-disk JSON cache with an atomic save.

Two on-disk shapes are supported, both matching what the scripts already wrote:

    versioned   {"version": V, "entries": {...}}      (02, 03, 04, 04b, 05, 05b caches)
    unversioned the file is the entries dict itself     (verify_cache.json, with version=None)

A file whose version differs from the expected one is ignored (a deliberate
bump). A file that cannot be parsed is ignored with a warning on stderr, never
silently. `save()` writes a temp file and `os.replace`s it, so an interrupt
cannot leave a truncated cache.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

KEY_SEPARATOR = "|"


def cache_key(*parts: object) -> str:
    """Join key parts the way the existing caches do: 'a|b|c'."""
    return KEY_SEPARATOR.join(str(part) for part in parts)


def _warn(message: str) -> None:
    print(f"  ! {message}", file=sys.stderr)


class JsonCache:
    """Key -> JSON value store backed by one file.

    `enabled=False` makes it inert: nothing is loaded, `get` always misses and
    `save` writes nothing (the `--no-cache` behaviour: ignore and do not write).
    """

    def __init__(self, path: Path, version: int | None, *, enabled: bool = True,
                 indent: int | None = None, sort_keys: bool = False) -> None:
        self.path = Path(path)
        self.version = version
        self.enabled = enabled
        self.indent = indent
        self.sort_keys = sort_keys
        self.entries: dict[str, Any] = {}
        self._dirty = False
        if enabled:
            self.entries = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            _warn(f"ignoring unreadable cache {self.path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            _warn(f"ignoring cache {self.path}: top level is not an object")
            return {}
        if self.version is None:
            return payload
        if payload.get("version") != self.version:
            _warn(f"ignoring cache {self.path}: version {payload.get('version')!r} "
                  f"!= expected {self.version!r}")
            return {}
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            _warn(f"ignoring cache {self.path}: 'entries' is not an object")
            return {}
        return entries

    def get(self, key: str, default: Any = None) -> Any:
        """The cached value, or `default` on a miss or when disabled."""
        if not self.enabled:
            return default
        return self.entries.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Store a value (a no-op when disabled)."""
        if not self.enabled:
            return
        self.entries[key] = value
        self._dirty = True

    def mark_dirty(self) -> None:
        """Flag in-place edits to a nested value so the next `save()` writes."""
        self._dirty = True

    def __contains__(self, key: object) -> bool:
        return self.enabled and key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def save(self, force: bool = False) -> None:
        """Atomically write the cache if it changed (or `force`)."""
        if not self.enabled or (not self._dirty and not force):
            return
        payload: Any = self.entries
        if self.version is not None:
            payload = {"version": self.version, "entries": self.entries}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False, indent=self.indent,
                           sort_keys=self.sort_keys),
                encoding="utf-8",
            )
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as exc:
            _warn(f"could not write cache {self.path}: {exc}")
