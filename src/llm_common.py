"""Shared plumbing for the LLM response caches and response parsing.

Cache file names and entry keys are owned by the callers (llm_geo, llm_grounded)
and are unchanged; this module only adds the load/save mechanics, a fingerprint
entry that notices prompt or generation-parameter changes, and one JSON parser.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import re
from typing import Any

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Real keys are sha256 hex digests, so this can never collide with one.
META_KEY = "__meta__"

# What a provider returns when the model blocked or emptied its answer. It
# parses as a refusal and is never written to the cache.
REFUSAL_SENTINEL = '{"refused": true}'

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{[^{}]*\"refused\"[^{}]*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict | None:
    """The JSON object in a model response, or None.

    Tries the whole string, then fenced blocks, then a bare `{... "refused" ...}`
    object buried in chatter. Valid JSON that is not an object (a list, a
    number, a string) is rejected."""
    if not isinstance(raw, str):
        return None
    for candidate in (
        raw.strip(),
        *(m.group(1) for m in _JSON_FENCE.finditer(raw)),
        *(m.group(0) for m in _JSON_OBJ.finditer(raw)),
    ):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def fingerprint(prompt: str, params: dict[str, Any]) -> str:
    """Hash of the prompt template plus generation parameters."""
    payload = json.dumps({"prompt": prompt, "params": params},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class PickleCache:
    """A dict of responses persisted as `<name>.pkl` in CACHE_DIR.

    The dict carries a `__meta__` entry holding the fingerprint of the prompt
    and generation parameters that produced it. A cache with no fingerprint
    (written before this existed) is assumed current and stamped; one whose
    fingerprint differs is discarded for this run and a warning is logged.
    """

    def __init__(self, name: str, meta: str):
        self.name = name
        self.meta = meta
        self.path = os.path.join(CACHE_DIR, f"{name}.pkl")
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    log.warning("LLM cache %s is not a dict; starting empty", self.path)
            except Exception as e:
                log.warning("LLM cache %s unreadable (%s: %s); starting empty",
                            self.path, type(e).__name__, e)
        stored = data.get(META_KEY)
        if stored is not None and stored != self.meta:
            log.warning("LLM cache %r was built with a different prompt or generation "
                        "settings; ignoring its %d entries (kept as %s.stale)",
                        self.name, len(data) - 1, os.path.basename(self.path))
            os.replace(self.path, f"{self.path}.stale")
            data = {}
        data[META_KEY] = self.meta
        return data

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a response, refusing empty ones and the reserved key."""
        if key == META_KEY or not isinstance(value, str) or not value.strip():
            return
        self.data[key] = value

    def __len__(self) -> int:
        return sum(1 for k in self.data if k != META_KEY)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)
