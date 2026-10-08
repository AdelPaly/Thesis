"""Client for the Irchel Geoparser worker (tier 1).

The geoparser runs in a separate environment (see src/geoparser_worker.py for
why). This module starts the worker once, sends it one text at a time and
reads back the toponyms it recognised and resolved against GeoNames.

The interpreter is GEOPARSER_PYTHON if set, else .venv-geoparser in the repo.
The installed geoparser version must equal `geoparser.version` in the
configuration; a mismatch is a configuration error, not a per-item one.
"""
from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
from pathlib import Path

from config import REPO_ROOT, ConfigError

WORKER = Path(__file__).resolve().parent / "geoparser_worker.py"


def _default_python() -> Path:
    venv = REPO_ROOT / ".venv-geoparser"
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


class GeoparserWorker:
    def __init__(self, gp_cfg):
        python = Path(os.environ.get("GEOPARSER_PYTHON") or _default_python())
        if not python.exists():
            raise ConfigError(f"geoparser interpreter not found at {python}; create "
                              ".venv-geoparser and `pip install geoparser` there")
        settings = {"recognizer": gp_cfg.recognizer,
                    "entity_types": list(gp_cfg.entity_types or []),
                    "resolver": gp_cfg.resolver, "gazetteer": gp_cfg.gazetteer}
        self._proc = subprocess.Popen(
            [str(python), str(WORKER), json.dumps(settings)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1)
        atexit.register(self.close)
        hello = self._read()
        if not hello.get("ready"):
            raise ConfigError(f"geoparser worker did not start: {hello}")
        if hello.get("version") != gp_cfg.version:
            self.close()
            raise ConfigError(f"geoparser {hello.get('version')} is installed but the "
                              f"configuration pins {gp_cfg.version}")
        self.version = hello["version"]

    def _read(self) -> dict:
        line = self._proc.stdout.readline()
        if not line:
            raise ConfigError("geoparser worker exited; its stderr is in the run log")
        return json.loads(line)

    def parse(self, text: str) -> list[dict]:
        """Toponyms in reading order: text, start, end, and the GeoNames
        `identifier` and feature `data` when the resolver found a referent."""
        self._proc.stdin.write(json.dumps({"text": text}) + "\n")
        self._proc.stdin.flush()
        reply = self._read()
        if "error" in reply:
            raise RuntimeError(f"geoparser: {reply['error']}")
        return reply["toponyms"]

    def close(self) -> None:
        if self._proc.poll() is None:
            self._proc.stdin.close()
            try:
                self._proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._proc.kill()
