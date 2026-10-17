"""Filter the Bluesky post dump down to its English entries. Nothing else.

``alpindale/two-million-bluesky-posts`` is ~2M raw firehose posts. The snippet
everyone reaches for first,

    ds = load_dataset("alpindale/two-million-bluesky-posts",
                      "with-language-predictions", split="train")
    en = ds.filter(lambda x: x["language"] == "en")

cannot work: the dataset has exactly one config (``default``), and its schema is
``text, created_at, author, uri, has_images, reply_to`` -- there is no language
field to filter on. So the language has to be detected locally instead.

Detection is ``langdetect`` at ~2.2 ms/post, which is ~72 minutes single-core
over 2M posts, so the filter is parallelised with ``num_proc``. Posts that
langdetect cannot read at all (emoji-only, digits-only, empty) raise
LangDetectException and are treated as not-English: a post with no letters in it
has no language to detect.

Every row langdetect calls English is kept, with every column intact. No
deduplication, no length limits, no content filtering.

Usage:
    python scripts/fetch_bluesky_en.py
    python scripts/fetch_bluesky_en.py --format parquet --limit 50000
"""

import argparse
import pathlib

from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATASET = "alpindale/two-million-bluesky-posts"

# langdetect is randomised; without a fixed seed the same post can come back
# with different labels between runs. Set at import so every worker inherits it.
DetectorFactory.seed = 0


def is_english(row):
    """Row-level predicate for Dataset.filter. Must be module-level to pickle."""
    text = (row["text"] or "").strip()
    if not text:
        return False
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False  # no detectable features, so not confirmably English


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    help="output path (default: generation/synthetic/data/bluesky_en.<format>)")
    ap.add_argument("--format", choices=("jsonl", "parquet"), default="jsonl",
                    help="output format (default: jsonl, same shape as the source)")
    ap.add_argument("--num-proc", type=int, default=10,
                    help="parallel workers for the filter (default: 10)")
    ap.add_argument("--limit", type=int,
                    help="only process the first N posts (for a quick trial run)")
    args = ap.parse_args()

    out = args.out or (ROOT / "generation" / "synthetic" / "data" / f"bluesky_en.{args.format}")
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"loading {DATASET} (first run downloads ~2M posts)...", flush=True)
    ds = load_dataset(DATASET, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"{len(ds)} posts; detecting language with {args.num_proc} workers...", flush=True)

    en = ds.filter(is_english, num_proc=args.num_proc)

    if args.format == "jsonl":
        en.to_json(out, orient="records", lines=True, force_ascii=False)
    else:
        en.to_parquet(out)

    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    pct = len(en) / max(len(ds), 1) * 100
    print(f"\nkept {len(en)} of {len(ds)} posts ({pct:.1f}% English)")
    print(f"wrote {shown}")


if __name__ == "__main__":
    main()
