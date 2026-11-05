#!/usr/bin/env python3
"""
Word-count distribution of the real Bluesky posts, for the post lengths.

    python length_dist.py

plan.py draws each item's target length from this distribution within the
condition's range in plan.length_ranges (Q1 to Q3, the middle half of real
posts, for most), and finalise.py drops a post outside that range. Reads
paths.bluesky_dump and writes paths.length_dist (sampler/length_dist.json):
the percentiles and the count of posts at every length.

Words are counted the way finalise.py counts them, on whitespace, after
removing links and @mentions (a generated post has neither). A post with no
words left is skipped.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

import plan

URL = re.compile(r"https?://\S+|\S+\.\S+/\S*|@\S+")


def percentile(counts: Counter, q: float) -> int:
    """The smallest length with at least q of the posts at or below it."""
    total, seen = sum(counts.values()), 0
    for length in sorted(counts):
        seen += counts[length]
        if seen >= q * total:
            return length
    raise ValueError("empty distribution")


def main() -> int:
    plan.configure_stdout()
    config = plan.load_config(plan.HERE / "config.yaml")
    src = plan.HERE / config["paths"]["bluesky_dump"]
    out = plan.HERE / config["paths"]["length_dist"]
    counts: Counter = Counter()
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            text = json.loads(line).get("text") or ""
            n = len(URL.sub(" ", text).split())
            if n:
                counts[n] += 1
    marks = {"p5": 0.05, "p10": 0.1, "q1": 0.25, "median": 0.5, "q3": 0.75, "p90": 0.9, "p95": 0.95}
    record = {
        "source": config["paths"]["bluesky_dump"],
        "posts": sum(counts.values()),
        **{name: percentile(counts, q) for name, q in marks.items()},
        "counts": {str(n): k for n, k in sorted(counts.items())},
    }
    plan.atomic_write_text(out, json.dumps(record, indent=1) + "\n")
    print(f"{record['posts']} posts; " + ", ".join(f"{name} {record[name]}" for name in marks)
          + f" words -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
