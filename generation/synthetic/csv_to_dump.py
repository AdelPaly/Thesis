#!/usr/bin/env python3
"""
Turn a CSV of real posts (columns text, is_reply) into the JSONL shape
build_exemplars.py reads, so it can be passed with --dump.

    python csv_to_dump.py ../climate_en.csv data/environment_en.jsonl
    python build_exemplars.py --topic environment --dump data/environment_en.jsonl --dry-run

The CSV has no URIs, dates or image flags, so each post gets the uri
"csv:<file stem>:<row>", and has_images is false for all of them (so no
photo_caption exemplars can come from it). Literal "\\n" in the text becomes a
newline.
"""
import csv
import json
import sys
from pathlib import Path


def main(src: Path, dst: Path) -> None:
    with src.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with dst.open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows):
            record = {
                "uri": f"csv:{src.stem}:{i}",
                "text": (row.get("text") or "").replace("\\n", "\n"),
                "reply_to": "x" if (row.get("is_reply") or "").strip().lower() == "true" else None,
                "has_images": False,
                "created_at": None,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} posts to {dst}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]), Path(sys.argv[2]))
