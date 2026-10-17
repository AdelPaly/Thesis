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

--topic climate or --topic misinformation builds a topical variant instead:
only posts matching that topic's keywords (TOPICS below) are candidates, the
labeller also gives each post a subtopic or "none" (off topic, which is
rejected), and the result goes to data/style_exemplars_<topic>.jsonl with its
own label cache, data/exemplar_labels_<topic>.jsonl. The plain exemplar file
is not touched.

Labels are cached in data/exemplar_labels.jsonl, so an interrupted run
resumes and a re-run with a larger --per-register only labels what it needs.
The exemplar file is only rewritten when at least one exemplar was chosen, so
a run that fails before labelling anything cannot wipe an existing file.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

import generate as gen  # noqa: E402  (config, .env, OpenRouter client)
import plan             # noqa: E402  (configure_stdout, atomic_write_text)

SPAM_FILTER_PATH = HERE.parent.parent / "scripts" / "filter_adult_spam.py"

# TODO: move to a paths: key in config.yaml (that file is edited elsewhere);
# --label-cache overrides it meanwhile.
LABEL_CACHE = HERE / "data" / "exemplar_labels.jsonl"

MIN_WORDS = 4              # shortest candidate post, in whitespace-separated words
MAX_WORDS = 60             # longest candidate post
MIN_LETTERS = 15           # fewest alphabetic characters in a candidate
LABEL_MAX_TOKENS = 8000    # floor for max_tokens: the model reasons before it answers
DEFAULT_PER_REGISTER = 30
DEFAULT_BATCH = 10
DEFAULT_MAX_CALLS = 150
DEFAULT_SEED = "exemplars-2026"
DRY_RUN_PREVIEW = 10       # candidates shown by --dry-run
PREVIEW_CHARS = 110

_URL = re.compile(r"https?://|www\.|\.(com|org|net|co|io|uk)\b|…$|\.\.\.$", re.I)
_HANDLE = re.compile(r"[@#]\w")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")
_ALLOWED_CAPS = {"I", "I'm", "I’m", "I've", "I’ve", "I'll", "I’ll", "I'd", "I’d",
                 "OK", "TV", "AM", "PM"}

_spam_module = None


def spam_filter():
    """scripts/filter_adult_spam.py, loaded on first use rather than at import
    (loading executes that script)."""
    global _spam_module
    if _spam_module is None:
        spec = importlib.util.spec_from_file_location("filter_adult_spam", SPAM_FILTER_PATH)
        _spam_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_spam_module)
    return _spam_module


def passes_text_filter(text: str) -> bool:
    words = text.split()
    if not MIN_WORDS <= len(words) <= MAX_WORDS:
        return False
    if _URL.search(text) or _HANDLE.search(text) or "\n\n\n" in text:
        return False
    if spam_filter().classify(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < MIN_LETTERS or sum(ch.isascii() for ch in text if ch.isalpha()) < letters:
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


# Topical variants. "keywords" only decides which posts are labelled at all
# (it is deliberately looser than the topic: "climate" also matches "the
# political climate"); the labeller's subtopic decides what is on topic. Terms
# that were mostly off topic in the 2024-11-27 dump are left out: "bots" and
# "lies"/"lying" (spam accounts, interpersonal rows), "gaslighting", and "ev".
TOPICS = {
    "climate": {
        "keywords": r"\bclimate\b|global warming|greenhouse|fossil fuels?\b|"
                    r"carbon (emissions?|footprint|tax|capture|dioxide|neutral)|\bco2\b|"
                    r"net[- ]zero|renewables?\b|solar (panels?|power|energy|farms?)|"
                    r"wind (turbines?|farms?|power|energy)|sea[- ]levels?|\bcop ?29\b|"
                    r"\bemissions\b|decarboni[sz]|heat ?waves?|wildfires?|\bdroughts?\b|"
                    r"\bflood(s|ing|ed)?\b|glaciers?|extreme weather|\bipcc\b|"
                    r"green energy|electric (vehicles?|cars?)|heat pumps?|"
                    r"environmental|deforestation|biodiversity|\bpollution\b",
        "description": "climate change, its causes and effects, energy and the "
                       "transition away from fossil fuels, climate policy or "
                       "activism, or extreme weather and environmental hazards",
        "subtopics": {
            "climate_change": "climate change itself: warming, its causes, science, denial, anxiety about it",
            "energy_transition": "emissions, fossil fuels, renewables, electric vehicles, heat pumps, climate policy or activism",
            "extreme_weather": "a heatwave, drought, flood, wildfire, storm or other weather or environmental hazard, experienced or discussed",
        },
    },
    "misinformation": {
        "keywords": r"misinformation|disinformation|malinformation|fake news|"
                    r"conspirac(y|ies)|propaganda|\bhoax(es)?\b|debunk|fact[- ]?check|"
                    r"false (claims?|information|narratives?|stories)|deep ?fakes?|"
                    r"misleading|\bpsyops?\b|troll farms?|\banti-?vax|flat earth|"
                    r"chemtrails?|post-truth|media literacy|propagandists?|"
                    r"\brumou?rs?\b|doctored|out of context|"
                    r"\bdo your (own )?research\b|\bsheeple\b",
        "description": "misinformation, disinformation, propaganda, conspiracy "
                       "theories, hoaxes, rumours, fake or misleading content, or "
                       "fact-checking and correcting false claims",
        "subtopics": {
            "spreading": "the post itself asserts or passes on a false, conspiratorial or unverified claim",
            "debunking": "the post corrects, fact-checks or pushes back on a specific false or misleading claim",
            "discussing": "the post talks about misinformation, propaganda, conspiracy theories or media literacy in general",
        },
    },
}


def load_candidates(dump: Path, keywords: re.Pattern | None = None) -> list[dict[str, Any]]:
    """Posts that pass the text filter and, when given, match the keywords."""
    out = []
    with dump.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (record.get("text") or "").strip()
            if keywords is not None and not keywords.search(text):
                continue
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
{topic_field}
Answer with JSON only, in this shape, one entry per post in order:
{{"labels": [{{"n": 1, "register": "<name or none>", "names_anything": true{topic_shape}}}]}}

Posts:
{posts}"""

TOPIC_FIELD = """- "subtopic": whether the post is about {description}. Give the ONE that fits best, or "none" if the topic words are used in another sense or the post is not really about this topic:
{subtopics}
"""


def parse_labels(raw: str, batch: list[dict[str, Any]],
                 topical: bool = False) -> dict[str, dict[str, Any]]:
    """
    The labels in a model answer, keyed by post uri. With topical, each label
    also carries the subtopic ("none" when missing). Raises JSONDecodeError,
    TypeError or ValueError when the answer is not the expected shape; the
    caller decides what to do about that.
    """
    labels = {}
    for entry in json.loads(raw).get("labels", []):
        n = int(entry.get("n", 0))
        if 1 <= n <= len(batch):
            label = {
                "register": str(entry.get("register") or "none").strip(),
                "names_anything": bool(entry.get("names_anything", True)),
            }
            if topical:
                label["subtopic"] = str(entry.get("subtopic") or "none").strip()
            labels[batch[n - 1]["uri"]] = label
    return labels


_parse_failure_logged = False   # log the first unusable answer only, not every one


def label_prompt(registers, batch, topic: dict[str, Any] | None = None) -> str:
    register_lines = "\n".join(f'    {r["name"]}: {" ".join(r["description"].split())}'
                               for r in registers)
    post_lines = "\n".join(f"{i}. {c['text'].replace(chr(10), ' / ')}"
                           for i, c in enumerate(batch, start=1))
    topic_field = topic_shape = ""
    if topic is not None:
        subtopic_lines = "\n".join(f"    {name}: {desc}"
                                   for name, desc in topic["subtopics"].items())
        topic_field = TOPIC_FIELD.format(description=topic["description"],
                                         subtopics=subtopic_lines)
        topic_shape = ', "subtopic": "<name or none>"'
    return LABEL_PROMPT.format(registers=register_lines, posts=post_lines,
                               topic_field=topic_field, topic_shape=topic_shape)


def label_batch(client, config, registers, batch,
                topic: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    global _parse_failure_logged
    messages = [{"role": "user", "content": label_prompt(registers, batch, topic)}]
    content, _meta, _log = gen.call_with_backoff(client, messages, config)
    raw = gen.extract_json_object(content or "") or "{}"
    try:
        labels = parse_labels(raw, batch, topical=topic is not None)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        labels = {}
        if not _parse_failure_logged:
            _parse_failure_logged = True
            print(f"  label answer unusable ({type(exc).__name__}: {exc}); "
                  f"later failures are not logged again", file=sys.stderr)
    if not labels:
        # Usually the reasoning used up max_tokens and no JSON arrived. Not
        # cached, so the batch is tried again on the next run.
        return {}
    # A post the model skipped inside an otherwise good answer is recorded as
    # unusable rather than retried forever.
    skipped = {"register": "none", "names_anything": True}
    if topic is not None:
        skipped["subtopic"] = "none"
        for label in labels.values():          # an invented subtopic is off topic
            if label["subtopic"] not in topic["subtopics"]:
                label["subtopic"] = "none"
    for cand in batch:
        labels.setdefault(cand["uri"], dict(skipped))
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
    """The register this post is an exemplar for, or None. A topical label
    (one with a subtopic) must also be on topic."""
    register = label["register"]
    if label["names_anything"] or register not in names:
        return None
    if label.get("subtopic", "") == "none":
        return None
    if register == "reply_in_thread" and not cand["is_reply"]:
        return None
    if register == "photo_caption" and not cand["has_images"]:
        return None
    return register


class ExemplarPool:
    """
    The exemplars chosen so far per register, and the candidates still waiting
    to be labelled, bucketed by the register they might fit.
    """

    def __init__(self, names: set[str], per_register: int, batch_size: int) -> None:
        self.names = names
        self.per_register = per_register
        self.batch_size = batch_size
        self.chosen: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.seen: set[str] = set()

    def take(self, cand: dict[str, Any], label: dict[str, Any]) -> None:
        """Keep the candidate if its label makes it an exemplar and there is room."""
        register = acceptable(cand, label, self.names)
        if register and len(self.chosen[register]) < self.per_register:
            if "subtopic" in label:
                cand = {**cand, "subtopic": label["subtopic"]}
            self.chosen[register].append(cand)

    def full(self) -> bool:
        return all(len(self.chosen[n]) >= self.per_register for n in self.names)

    def add_candidates(self, candidates: list[dict[str, Any]],
                       cached: dict[str, dict[str, Any]]) -> None:
        """Use cached labels where there are any, bucket the rest for labelling."""
        for cand in candidates:
            if cand["uri"] in cached:
                self.take(cand, cached[cand["uri"]])
            else:
                for name in bucket_of(cand):
                    self.buckets[name].append(cand)

    def next_batch(self) -> list[dict[str, Any]]:
        """Posts for the register that is furthest behind."""
        short = sorted((len(self.chosen[n]), n) for n in self.names
                       if len(self.chosen[n]) < self.per_register and self.buckets[n])
        if not short:
            return []
        bucket = self.buckets[short[0][1]]
        batch = []
        while bucket and len(batch) < self.batch_size:
            cand = bucket.pop()
            if cand["uri"] not in self.seen:
                self.seen.add(cand["uri"])
                batch.append(cand)
        return batch

    def progress(self) -> str:
        return " ".join(f"{n}={len(self.chosen[n])}" for n in sorted(self.names))


def make_label_client(config: dict[str, Any]):
    """
    A ChatClient set up for labelling: JSON object output, temperature 0, and
    a token budget of at least LABEL_MAX_TOKENS (the model reasons before it
    answers and the reasoning shares the budget). The overrides go into a copy
    of the config handed to the constructor, so the client is not mutated from
    outside. Returns (client, the overridden config).
    """
    label_config = copy.deepcopy(config)
    settings = label_config["generation"]
    settings["structured_output"] = "json_object"
    settings["temperature"] = 0.0
    settings["max_tokens"] = max(int(settings.get("max_tokens", 600)), LABEL_MAX_TOKENS)
    return gen.ChatClient(label_config), label_config


def label_candidates(pool: ExemplarPool, client, config, registers,
                     cache_path: Path, max_calls: int,
                     topic: dict[str, Any] | None = None) -> int:
    """Label batches until every register is full or max_calls is reached.
    Appends every label to the cache. Returns the number of calls made."""
    calls = 0
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as cache_out:
        while not pool.full() and calls < max_calls:
            batch = pool.next_batch()
            if not batch:
                break
            try:
                labels = label_batch(client, config, registers, batch, topic)
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
                pool.take(cand, labels[cand["uri"]])
            cache_out.flush()
            print(f"  call {calls}: {pool.progress()}", flush=True)
    return calls


def exemplar_lines(chosen: dict[str, list[dict[str, Any]]],
                   topic: str | None = None) -> str:
    """The style_exemplars.jsonl content for the chosen posts. A topical run
    adds the topic and each post's subtopic."""
    lines = []
    for register in sorted(chosen):
        for cand in chosen[register]:
            row = {"register": register, "text": cand["text"],
                   "source": cand["uri"], "created_at": cand["created_at"]}
            if topic is not None:
                row["topic"] = topic
                row["subtopic"] = cand.get("subtopic")
            lines.append(json.dumps(row, ensure_ascii=False) + "\n")
    return "".join(lines)


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env")
    parser.add_argument("--topic", choices=sorted(TOPICS),
                        help="build a topical variant (see TOPICS) instead of the plain set")
    parser.add_argument("--label-cache", type=Path,
                        help="where labels are cached (default data/exemplar_labels.jsonl, "
                             "or data/exemplar_labels_<topic>.jsonl)")
    parser.add_argument("--per-register", type=int, default=DEFAULT_PER_REGISTER,
                        help="exemplars to collect per register (default 30)")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH,
                        help="posts per labelling call")
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS,
                        help="stop after this many labelling calls")
    parser.add_argument("--seed", default=DEFAULT_SEED,
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
    topic = TOPICS[args.topic] if args.topic else None
    label_cache = args.label_cache or LABEL_CACHE
    if args.topic:
        out_path = out_path.with_name(f"{out_path.stem}_{args.topic}{out_path.suffix}")
        if args.label_cache is None:
            label_cache = LABEL_CACHE.with_name(f"{LABEL_CACHE.stem}_{args.topic}.jsonl")

    if not dump.exists():
        raise SystemExit(f"Bluesky dump not found: {dump}")
    print(f"reading {dump} ...", flush=True)
    candidates = load_candidates(dump, re.compile(topic["keywords"], re.I) if topic else None)
    random.Random(args.seed).shuffle(candidates)
    print(f"{len(candidates)} posts pass the text filter "
          f"({sum(c['is_reply'] for c in candidates)} replies, "
          f"{sum(c['has_images'] for c in candidates)} with images)")

    if args.dry_run:
        for cand in candidates[:DRY_RUN_PREVIEW]:
            print(f"  - {cand['text'][:PREVIEW_CHARS]}")
        print("\ndry run: nothing labelled, nothing written")
        return 0

    cached: dict[str, dict[str, Any]] = {}
    for row in gen.read_jsonl(label_cache):   # [] when the cache does not exist
        cached[row["uri"]] = row["label"]

    pool = ExemplarPool(names, args.per_register, args.batch)
    pool.add_candidates(candidates, cached)

    client, label_config = make_label_client(config)
    label_candidates(pool, client, label_config, registers, label_cache,
                     args.max_calls, topic)

    total = sum(len(v) for v in pool.chosen.values())
    if total == 0:
        print(f"\nnothing chosen, so {out_path} was left untouched", file=sys.stderr)
        return 1
    plan.atomic_write_text(out_path, exemplar_lines(pool.chosen, args.topic))

    print(f"\nwrote {total} exemplars to {out_path}")
    for name in sorted(names):
        flag = "" if len(pool.chosen[name]) >= args.per_register else "   (short)"
        print(f"  {name:<16} {len(pool.chosen[name]):>3}{flag}")
    if topic is not None:
        subtopics = Counter(c["subtopic"] for chosen in pool.chosen.values() for c in chosen)
        print("  by subtopic: " + ", ".join(f"{k}={v}" for k, v in sorted(subtopics.items())))
    if not pool.full():
        print("\nsome registers are short; re-run (cached labels are reused) or "
              "raise --max-calls")
    return 0


if __name__ == "__main__":
    sys.exit(main())
