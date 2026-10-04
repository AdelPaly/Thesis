#!/usr/bin/env python3
"""
Build the style exemplars from the real Bluesky posts.

    python build_exemplars.py --dry-run     # count candidates, call nothing
    python build_exemplars.py               # label posts, write the exemplars

generate.py shows the model three real posts per prompt as tone references,
picked from data/style_exemplars.jsonl by register. This script fills that
file from data/bluesky_en.jsonl (about a million English posts).

Exemplars must not name anything. A real post that mentions a city, a
nationality, a team or a person is exactly what the model would copy into a
post whose prompt forbids place names. So a post has to pass two filters:

1. A cheap text filter over the whole dump: 4-60 words, no links, no
   @mentions or hashtags, no spam (../../scripts/filter_adult_spam.py), and
   no capitalised word except at the start of a sentence or "I" - which
   removes nearly every proper noun before anything is paid for.
2. The generation model reads the survivors in batches and gives each one a
   register (or none) and a yes/no for "names anything": a place, a
   nationality or language, a person, an organisation, a brand, a team, an
   event or a title. Only "no" is kept.

Two registers are also checked against the post's metadata: reply_in_thread
must be a real reply, photo_caption must have had an image.

Labels are cached in data/exemplar_labels.jsonl, so an interrupted run
resumes and a re-run with a larger --per-register only labels what it needs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generate as gen  # noqa: E402  (config, .env, OpenRouter client)

_spam_path = HERE.parent.parent / "scripts" / "filter_adult_spam.py"
_spec = importlib.util.spec_from_file_location("filter_adult_spam", _spam_path)
_spam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_spam)

LABEL_CACHE = HERE / "data" / "exemplar_labels.jsonl"

_URL = re.compile(r"https?://|www\.|\.(com|org|net|co|io|uk)\b|…$|\.\.\.$", re.I)
_HANDLE = re.compile(r"[@#]\w")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_ALLOWED_CAPS = {"I", "I'm", "I’m", "I've", "I’ve", "I'll", "I’ll", "I'd", "I’d",
                 "OK", "TV", "AM", "PM"}


def passes_text_filter(text: str) -> bool:
    words = text.split()
    if not 4 <= len(words) <= 60:
        return False
    if _URL.search(text) or _HANDLE.search(text) or "\n\n\n" in text:
        return False
    if _spam.classify(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 15 or sum(ch.isascii() for ch in text if ch.isalpha()) < letters:
        return False
    # No capitalised word except sentence-initially: a cheap proper-noun filter.
    sentence_start = True
    for token in re.split(r"(\s+)", text):
        if not token.strip():
            continue
        match = _WORD.search(token)
        if match:
            word = match.group(0)
            if (word[0].isupper() and not sentence_start
                    and word not in _ALLOWED_CAPS):
                return False
            if word.isupper() and len(word) > 1 and word not in _ALLOWED_CAPS:
                return False
        sentence_start = bool(re.search(r"[.!?:]['\")]*$", token))
    return True


def load_candidates(dump: Path) -> list[dict[str, Any]]:
    out = []
    with dump.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (record.get("text") or "").strip()
            if text and passes_text_filter(text):
                out.append({"uri": record.get("uri"),
                            "text": text,
                            "is_reply": bool(record.get("reply_to")),
                            "has_images": bool(record.get("has_images")),
                            "created_at": record.get("created_at")})
    return out


LABEL_PROMPT = """You are labelling real social media posts. For each numbered post give:

- "register": which ONE of these styles it is written in, or "none" if it fits none well:
{registers}
- "names_anything": true if the post mentions ANY of: a place (city, region, country, street, landmark, venue), a nationality or language, a person's name, an organisation, company, brand, sports team, event, or the title of a work. false only if it mentions none of these.

Answer with JSON only, in this shape, one entry per post in order:
{{"labels": [{{"n": 1, "register": "<name or none>", "names_anything": true}}]}}

Posts:
{posts}"""


def label_batch(client, config, registers, batch) -> dict[str, dict[str, Any]]:
    register_lines = "\n".join(f'    {r["name"]}: {" ".join(r["description"].split())}'
                               for r in registers)
    post_lines = "\n".join(f"{i}. {c['text'].replace(chr(10), ' / ')}"
                           for i, c in enumerate(batch, start=1))
    messages = [{"role": "user",
                 "content": LABEL_PROMPT.format(registers=register_lines, posts=post_lines)}]
    content, _meta, _log = gen.call_with_backoff(client, messages, config)
    raw = gen.extract_json_object(content or "") or "{}"
    labels = {}
    try:
        for entry in json.loads(raw).get("labels", []):
            n = int(entry.get("n", 0))
            if 1 <= n <= len(batch):
                labels[batch[n - 1]["uri"]] = {
                    "register": str(entry.get("register") or "none").strip(),
                    "names_anything": bool(entry.get("names_anything", True)),
                }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    if not labels:
        # Usually the reasoning used up max_tokens and no JSON arrived. Not
        # cached, so the batch is tried again on the next run.
        return {}
    # A post the model skipped inside an otherwise good answer is recorded as
    # unusable rather than retried forever.
    for cand in batch:
        labels.setdefault(cand["uri"], {"register": "none", "names_anything": True})
    return labels


# Phrases that make a post likely to be in a register. They only decide which
# posts are shown to the labeller first; the labeller still decides.
HINTS = {
    "event_mention": r"\b(this weekend|tonight|tomorrow|on (mon|tues|wednes|thurs|fri|satur|sun)day|"
                     r"next week|market|gig|concert|festival|fundraiser|open day|come along|"
                     r"tickets|starts at|kicks off)\b",
    "local_complaint": r"\b(roadworks|road works|traffic|queue|closed again|still broken|"
                       r"parking|bus (is|was) late|why is|every single|again\?|noise|"
                       r"can't believe|ridiculous|so annoying)\b",
    "recommendation": r"\b(recommend|highly|worth (it|a visit|going)|go to|try the|"
                      r"avoid|don't bother|best .{0,20} (around|nearby)|must try|10/10)\b",
}
HINT_RES = {name: re.compile(pattern, re.I) for name, pattern in HINTS.items()}


def bucket_of(cand: dict[str, Any]) -> list[str]:
    """Which registers a candidate is worth showing the labeller for."""
    out = [name for name, rx in HINT_RES.items() if rx.search(cand["text"])]
    if cand["has_images"]:
        out.append("photo_caption")
    if cand["is_reply"]:
        out.append("reply_in_thread")
    elif not out:
        out.append("personal_update")
    return out


def acceptable(cand: dict[str, Any], label: dict[str, Any], names: set[str]) -> str | None:
    """The register this post is an exemplar for, or None."""
    register = label["register"]
    if label["names_anything"] or register not in names:
        return None
    if register == "reply_in_thread" and not cand["is_reply"]:
        return None
    if register == "photo_caption" and not cand["has_images"]:
        return None
    return register


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--per-register", type=int, default=30,
                        help="exemplars to collect per register (default 30)")
    parser.add_argument("--batch", type=int, default=10, help="posts per labelling call")
    parser.add_argument("--max-calls", type=int, default=150,
                        help="stop after this many labelling calls")
    parser.add_argument("--seed", default="exemplars-2026",
                        help="shuffles the candidates; same seed, same order")
    parser.add_argument("--dry-run", action="store_true",
                        help="count candidates and show a few, call nothing")
    args = parser.parse_args(argv)

    gen.load_dotenv(args.env_file)
    config = gen.load_config(args.config)
    paths = config["paths"]
    dump = Path(paths["bluesky_dump"])
    out_path = Path(paths["style_exemplars"])
    registers = config["registers"]
    names = {r["name"] for r in registers}

    if not dump.exists():
        raise SystemExit(f"Bluesky dump not found: {dump}")
    print(f"reading {dump} ...", flush=True)
    candidates = load_candidates(dump)
    random.Random(args.seed).shuffle(candidates)
    print(f"{len(candidates)} posts pass the text filter "
          f"({sum(c['is_reply'] for c in candidates)} replies, "
          f"{sum(c['has_images'] for c in candidates)} with images)")

    if args.dry_run:
        for cand in candidates[:10]:
            print(f"  - {cand['text'][:110]}")
        print("\ndry run: nothing labelled, nothing written")
        return 0

    cached: dict[str, dict[str, Any]] = {}
    if LABEL_CACHE.exists():
        for row in gen.read_jsonl(LABEL_CACHE):
            cached[row["uri"]] = row["label"]

    chosen: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def take(cand, label):
        register = acceptable(cand, label, names)
        if register and len(chosen[register]) < args.per_register:
            chosen[register].append(cand)

    def full() -> bool:
        return all(len(chosen[n]) >= args.per_register for n in names)

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in candidates:
        if cand["uri"] in cached:
            take(cand, cached[cand["uri"]])
        else:
            for name in bucket_of(cand):
                buckets[name].append(cand)
    seen: set[str] = set()

    def next_batch() -> list[dict[str, Any]]:
        """Posts for the register that is furthest behind."""
        short = sorted((len(chosen[n]), n) for n in names
                       if len(chosen[n]) < args.per_register and buckets[n])
        if not short:
            return []
        bucket = buckets[short[0][1]]
        batch = []
        while bucket and len(batch) < args.batch:
            cand = bucket.pop()
            if cand["uri"] not in seen:
                seen.add(cand["uri"])
                batch.append(cand)
        return batch

    client = gen.ChatClient(config)
    client.structured_output = "json_object"
    client.temperature = 0.0
    # The model reasons before answering and the reasoning shares this budget;
    # at the generation default a batch can run out before the JSON arrives.
    client.max_tokens = max(client.max_tokens, 8000)
    calls = 0
    with LABEL_CACHE.open("a", encoding="utf-8") as cache_out:
        while not full() and calls < args.max_calls:
            batch = next_batch()
            if not batch:
                break
            try:
                labels = label_batch(client, config, registers, batch)
            except gen.ApiError as exc:
                print(f"  call failed ({exc}); stopping - re-run to resume", file=sys.stderr)
                break
            calls += 1
            if not labels:
                print(f"  call {calls}: no usable answer, batch skipped", flush=True)
                continue
            for cand in batch:
                cache_out.write(json.dumps({"uri": cand["uri"], "label": labels[cand["uri"]]},
                                           ensure_ascii=False) + "\n")
                take(cand, labels[cand["uri"]])
            cache_out.flush()
            counts = " ".join(f"{n}={len(chosen[n])}" for n in sorted(names))
            print(f"  call {calls}: {counts}", flush=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for register in sorted(chosen):
            for cand in chosen[register]:
                handle.write(json.dumps({"register": register, "text": cand["text"],
                                         "source": cand["uri"],
                                         "created_at": cand["created_at"]},
                                        ensure_ascii=False) + "\n")

    print(f"\nwrote {sum(len(v) for v in chosen.values())} exemplars to {out_path}")
    for name in sorted(names):
        flag = "" if len(chosen[name]) >= args.per_register else "   (short)"
        print(f"  {name:<16} {len(chosen[name]):>3}{flag}")
    if not full():
        print("\nsome registers are short; re-run (cached labels are reused) or "
              "raise --max-calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
