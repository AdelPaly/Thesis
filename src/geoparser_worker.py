"""Irchel Geoparser worker (tier 1). Runs in its own environment.

geoparser 0.6 requires transformers>=5 and Flair requires transformers<5, so
the two cannot share an environment. This script runs under
.venv-geoparser's interpreter; src/geoparser_irchel.py starts it once per run
and talks to it over stdin/stdout. It imports nothing from src/.

    argv[1]  JSON settings: recognizer, entity_types, resolver, gazetteer
    stdin    one JSON object per line: {"text": "..."}
    stdout   one JSON object per line: {"toponyms": [...]} or {"error": "..."}

The first line written is {"ready": true, "version": "<geoparser version>"}.
Everything else written to stdout -- by the library, or by a subprocess it
starts -- goes to stderr: file descriptor 1 is moved, not just sys.stdout, so
stdout carries only the protocol.
"""
import json
import os
import sys


def _toponym(ref) -> dict:
    loc = ref.location
    out = {"text": ref.text, "start": ref.start, "end": ref.end,
           "identifier": None, "data": None}
    if loc is not None:
        out["identifier"] = str(loc.identifier)
        out["data"] = loc.data
    return out


def main() -> None:
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    settings = json.loads(sys.argv[1])
    from importlib.metadata import version

    from geoparser import Geoparser
    from geoparser.modules import SentenceTransformerResolver, SpacyRecognizer

    geoparser = Geoparser(
        recognizer=SpacyRecognizer(model_name=settings["recognizer"],
                                   entity_types=settings["entity_types"]),
        resolver=SentenceTransformerResolver(model_name=settings["resolver"],
                                             gazetteer_name=settings["gazetteer"]),
    )

    def send(obj) -> None:
        protocol.write(json.dumps(obj, default=str) + "\n")
        protocol.flush()

    send({"ready": True, "version": version("geoparser")})
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            document = geoparser.parse(json.loads(line)["text"])
            send({"toponyms": [_toponym(t) for t in document.toponyms]})
        except Exception as e:                      # noqa: BLE001 - reported per item
            send({"error": f"{type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
