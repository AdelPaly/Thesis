#!/usr/bin/env python3
"""
Stage 6b - generation.

WHAT THIS SCRIPT DOES
    Reads output/item_plan.jsonl, sends one request per planned item to a
    chat-completions API, and appends the result to
    output/raw_generations.jsonl.

    It decides nothing. Every choice was already made by plan.py. This
    script turns a planned item into a prompt, sends it, and records what came
    back.

FOUR RULES THAT ARE NOT NEGOTIABLE
    1. The raw response is logged before anything tries to parse it. A parser
       bug must never be able to destroy a response you have already paid for.
       The raw log is append-only and flushed on every write.
    2. Work is cached by item_id. A re-run skips items already present in the
       output and resumes where the last run stopped. Free-tier rate limits
       mean a full pass takes hours and will be interrupted.
    3. A failure is recorded, never skipped. A refused, truncated,
       rate-limited or unparseable item gets a row with an "error" field. A
       missing row and a failed row mean different things, and stage 7 needs to
       be able to tell them apart.
    4. No fallback changes content. Retries resend the same prompt. If a
       request keeps failing the item is written with an error and the run
       moves on. Nothing here rewrites a prompt, drops a constraint or
       substitutes a different clue to get a usable answer.

WHAT THIS SCRIPT MUST NEVER DO
    - validate the generated text, score it, or check whether the model leaked
      the place name (that is stage 7, and it needs the leaks on record)
    - choose places, clues, surface forms or slot values
    - hardcode a model, an endpoint, a path or an item count
    - tell the model to "be creative" or "vary the wording". Variation comes
      from the slot values and registers in the plan. Asking for variation
      instead produces the model's idea of variation, which is a small set of
      recognisable templates.

MODEL CHOICE
    The generating model must not be one of the models under evaluation.
    Generating the dataset with a system that later appears in the results
    table gives that system a private advantage that no experiment can measure
    away. The model is set in config.yaml.

INPUTS
    output/item_plan.jsonl        from plan.py
    data/style_exemplars.jsonl  real posts used as tone references only
    config.yaml         endpoint, model, prompt, rate limits

OUTPUTS
    output/raw_generations.jsonl  one row per attempt, append-only. An item
                                   that failed and was later retried has more
                                   than one row; the LAST row for an item_id is
                                   its current state, and stage 7 must load the
                                   file that way.
    output/raw_responses.jsonl    prompt and unparsed response body per call

USAGE
    python generate.py --dry-run --limit 5     # see the prompts, send nothing
    python generate.py --limit 20              # a paid smoke test
    python generate.py                         # the full pass
    python generate.py --only-condition A3
    python generate.py --retry-errors          # re-attempt failed rows only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


HERE = Path(__file__).resolve().parent

GENERATOR_VERSION = "1"

# The shape the model is asked to return. Sent as a JSON schema when the
# provider supports structured output, and restated in the prompt either way.
#
# The three fields beyond post_text are self-reports, not ground truth. They
# are cheap to collect and useful for triage in stage 7: a mismatch between
# naming_form_used and the surface form in the plan flags an item to inspect.
# They are never treated as evidence that an item is clean.
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "post_text": {
            "type": "string",
            "description": "The social media post. This is the only field that "
                           "becomes dataset content.",
        },
        "entities_referenced": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every named entity written into post_text, exactly "
                           "as it appears there.",
        },
        "naming_form_used": {
            "type": ["string", "null"],
            "description": "The place name written into post_text, exactly as it "
                           "appears, or null if no place name was written.",
        },
        "register": {
            "type": "string",
            "description": "The register that was followed.",
        },
    },
    "required": ["post_text", "entities_referenced", "naming_form_used", "register"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# config and inputs
# --------------------------------------------------------------------------- #

def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("paths", "generation", "prompt"):
        if section not in config:
            raise SystemExit(f"{path} is missing the {section!r} section")
    # Paths in the config are relative to the config file's folder, so the
    # scripts behave the same whatever directory they are started from.
    base = path.resolve().parent
    config["paths"] = {key: str((base / value).resolve())
                      for key, value in config["paths"].items()}
    return config


def load_dotenv(path: Path) -> None:
    """
    Read KEY=value lines from a .env file into the environment, so the API key
    can live in a file that is never committed rather than in the config.

    A variable already set in the shell wins over the file. No dependency on
    python-dotenv; the format supported is the plain one: KEY=value, optional
    quotes, # comments and blank lines ignored.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # A truncated last line happens when a run is killed mid-write.
                # Warn and carry on; the item will simply be regenerated.
                print(f"  warning: {path}:{line_no} is not valid JSON, skipped",
                      file=sys.stderr)
    return records


class AppendLog:
    """An append-only JSONL writer that flushes every record to disk."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.handle.close()


def load_exemplars(path: Path) -> dict[str, list[str]]:
    """
    Real posts grouped by register, used only as tone references.

    Expected shape, one object per line:
        {"register": "personal_update", "text": "...", "source": "..."}

    Exemplars must not contain place names or the entity types being tested,
    or the model copies them into the output. Filtering them is a job for
    whatever built this file, not for this script.
    """
    by_register: dict[str, list[str]] = defaultdict(list)
    if not path.exists():
        print(f"  note: no exemplar file at {path}, prompts will carry no "
              f"style references", file=sys.stderr)
        return by_register
    for record in read_jsonl(path):
        register = (record.get("register") or "").strip()
        text = (record.get("text") or "").strip()
        if register and text:
            by_register[register].append(text)
    return by_register


# --------------------------------------------------------------------------- #
# prompt construction
# --------------------------------------------------------------------------- #

def describe_clue(clue: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    """
    One line per clue, naming the entity in full and stating its relation to
    the place without naming the place.

    Entity names are always given in full and recognisable form. An indirect
    reference to an entity ("the anatomy professor who lived here") gives a
    reader nothing to resolve, so the task would stop being geolocation and
    start being guesswork. The ambiguity in this dataset lives in the place
    names only.

    entity_type in the plan is stage 4's property family code (place_or_org).
    The prompt shows the wording from prompt.entity_type_labels instead; a
    family with no wording there gets no type note at all, rather than the raw
    code.
    """
    relation = clue.get("property_label") or clue.get("property")
    labels = prompt_cfg.get("entity_type_labels") or {}
    entity_type = labels.get(clue.get("entity_type"))
    suffix = f" ({entity_type})" if entity_type else ""
    return f'- "{clue["clue_label"]}"{suffix} - relation to the location: {relation}'


def build_messages(
    item: dict[str, Any],
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
) -> list[dict[str, str]]:
    """
    Assemble the prompt in a fixed order:
        task -> entities -> hard constraint -> style references -> slots -> schema

    The constraint sits directly after the entities, before the style material.
    Putting the style examples last, nearest the generation, is what drags the
    output towards copying them; putting the constraint last makes the model
    treat it as a closing remark.
    """
    prompt_cfg = config["prompt"]
    condition = item["condition"]
    surface = item["surface_form"]
    clues = item["clues"]
    noise = item.get("noise_entity")

    parts: list[str] = []

    # 1. task
    parts.append(prompt_cfg["task"].strip())

    # 2. entities the post must mention
    mentions: list[str] = []
    if clues:
        mentions.append("Entities the post must mention, by name, in full:")
        mentions.extend(describe_clue(c, prompt_cfg) for c in clues)
    if noise:
        mentions.append(
            f'- "{noise["clue_label"]}" - mention by name. Do not connect it to '
            f"the writer's location in any way; it belongs somewhere else."
        )
    if surface.get("text"):
        if surface["kind"] == "ambiguous_alias":
            mentions.append(
                f'Place name the post must use, written exactly like this and '
                f'with nothing added to it: "{surface["text"]}"'
            )
        else:
            mentions.append(
                f'Place name the post must use, written exactly like this: '
                f'"{surface["text"]}"'
            )
    if mentions:
        parts.append("\n".join(mentions))

    # 3. hard constraint
    parts.append(build_constraint(item, prompt_cfg).strip())

    # 4. style references, labelled as tone only
    register = item["register"]
    pool = exemplars.get(register, [])
    n_exemplars = int(config["generation"].get("exemplars_per_prompt", 3))
    if pool:
        # Deterministic per item, so the same item always gets the same
        # references and a cache hit means the same prompt.
        rng = random.Random(item["item_id"])
        chosen = rng.sample(pool, min(n_exemplars, len(pool)))
        block = ["Tone and rhythm references. Copy nothing from these - not a "
                 "phrase, not a name, not a subject. They show how people in "
                 "this register punctuate, abbreviate and break lines:"]
        block.extend(f"  {i}. {text}" for i, text in enumerate(chosen, start=1))
        parts.append("\n".join(block))

    # 5. slot values from the plan
    detail: list[str] = [
        f"Register: {register}",
        f"Length: {prompt_cfg['length_bands'][item['length_band']]}",
    ]
    if item.get("register_description"):
        detail.append(f"Register notes: {item['register_description']}")
    for slot_name, slot_value in (item.get("slots") or {}).items():
        detail.append(f"{slot_name.replace('_', ' ').capitalize()}: {slot_value}")
    detail.append(
        "Use every value above. They are the specifics of this post and they "
        "are not suggestions."
    )
    parts.append("\n".join(detail))

    # 6. output shape
    parts.append(
        "Return one JSON object and nothing else, with these fields:\n"
        '  post_text            the post itself\n'
        '  entities_referenced  every named entity you wrote, exactly as written\n'
        '  naming_form_used     the place name you wrote, exactly as written, '
        'or null if you wrote none\n'
        '  register             the register you followed'
    )

    user_content = "\n\n".join(parts)
    system_content = prompt_cfg.get("system", "").strip()

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_constraint(item: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    """
    The constraint block, chosen by what the condition forbids.

    Templates live in the config so the wording can be revised without touching
    code. The banned-string list is built from the plan, not from the model's
    knowledge of the place.
    """
    surface = item["surface_form"]
    if surface["kind"] == "none":
        banned = prompt_cfg["forbidden_toponym"]
        return banned.format(
            place_name=item["place_name"],
            country=item["place_country"],
        )
    if surface["kind"] == "ambiguous_alias":
        return prompt_cfg["ambiguous_toponym"].format(
            surface=surface["text"],
            place_name=item["place_name"],
            country=item["place_country"],
        )
    return prompt_cfg["exact_toponym"].format(
        surface=surface["text"],
        place_name=item["place_name"],
    )


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

class ApiError(Exception):
    def __init__(self, message: str, status: int | None = None,
                 retryable: bool = False, body: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.body = body


class ChatClient:
    """
    A minimal client for an OpenAI-compatible /chat/completions endpoint.

    Compatible providers include Mistral, OpenRouter, Together and most
    free-tier gateways, which is why this shape is used rather than a
    provider-specific SDK: switching provider is a config change.

    429 and 5xx are retryable. 4xx other than 429 is not - a bad request does
    not become good by being resent.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gen = config["generation"]
        self.base_url = gen["base_url"].rstrip("/")
        self.model = gen["model"]
        self.temperature = float(gen.get("temperature", 1.0))
        self.top_p = float(gen.get("top_p", 1.0))
        self.max_tokens = int(gen.get("max_tokens", 600))
        self.timeout = float(gen.get("timeout_seconds", 120))
        self.structured_output = gen.get("structured_output", "json_schema")

        # Send a per-item seed when the model accepts one. Providers treat it
        # as best effort, not a guarantee, but it costs nothing and makes an
        # individual call closer to repeatable when you go back to inspect it.
        self.send_seed = bool(gen.get("send_seed", False))

        # Reasoning models put their thinking somewhere. Excluding it keeps the
        # content field clean and stops reasoning tokens eating max_tokens.
        self.exclude_reasoning = bool(gen.get("exclude_reasoning", False))

        # Excluding reasoning only hides it: the model still thinks, and the
        # thinking still counts against max_tokens. A reasoning model left at
        # its default effort can spend the whole budget thinking and return
        # empty content. The effort level is what actually shortens it.
        self.reasoning_effort = gen.get("reasoning_effort") or None

        # Optional attribution headers. OpenRouter recommends them and ignores
        # them if absent; other providers ignore them entirely.
        self.extra_headers: dict[str, str] = dict(gen.get("extra_headers") or {})

        key_env = gen["api_key_env"]
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key:
            raise SystemExit(
                f"no API key: set {key_env} in .env or in the environment. "
                f"The key is never read from the config file, so the config can "
                f"live in the repository."
            )

    def build_payload(
        self, messages: list[dict[str, str]], seed: int | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.send_seed and seed is not None:
            payload["seed"] = seed
        reasoning: dict[str, Any] = {}
        if self.reasoning_effort:
            reasoning["effort"] = self.reasoning_effort
        if self.exclude_reasoning:
            reasoning["exclude"] = True
        if reasoning:
            payload["reasoning"] = reasoning
        if self.structured_output == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "social_media_post",
                    "strict": True,
                    "schema": RESPONSE_SCHEMA,
                },
            }
        elif self.structured_output == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def call(
        self, messages: list[dict[str, str]], seed: int | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        One request. Returns (raw_content, response_metadata).
        Raises ApiError; retrying is the caller's job.
        """
        payload = self.build_payload(messages, seed=seed)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.extra_headers)
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                status = response.status
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retryable = exc.code == 429 or exc.code >= 500
            raise ApiError(
                f"HTTP {exc.code}", status=exc.code, retryable=retryable, body=body
            ) from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"network error: {exc.reason}", retryable=True) from exc
        except TimeoutError as exc:
            raise ApiError("request timed out", retryable=True) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError("response body was not JSON", status=status,
                           retryable=False, body=body) from exc

        try:
            choice = parsed["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ApiError("response held no message content", status=status,
                           retryable=False, body=body) from exc

        metadata = {
            "http_status": status,
            "finish_reason": choice.get("finish_reason"),
            "usage": parsed.get("usage"),
            "response_model": parsed.get("model"),
            "response_id": parsed.get("id"),
            "seed_sent": payload.get("seed"),
        }

        # Reasoning models return their thinking in a separate field. Recorded
        # for debugging, never used as content.
        reasoning = choice.get("message", {}).get("reasoning")
        if reasoning:
            metadata["reasoning_chars"] = len(str(reasoning))

        # A truncated response is the most common silent cause of a parse
        # failure on a reasoning model: the thinking consumed max_tokens and
        # the JSON never arrived. Surfaced here so it is visible in the row
        # rather than looking like a malformed answer.
        if choice.get("finish_reason") == "length":
            metadata["truncated"] = True

        return content if content is not None else "", metadata


def seed_for_item(item_id: str) -> int:
    """
    A stable per-item seed derived from the item_id, so the same item always
    asks for the same seed and two different items never share one.
    Kept inside 32 bits because some providers reject larger values.
    """
    return int(item_id[:8], 16)


def call_with_backoff(
    client: ChatClient,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    seed: int | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """
    Retry with exponential backoff and jitter. Returns
    (content, metadata, attempt_log) or raises the final ApiError.

    Rate limits are the single largest cost in wall-clock time on a free tier,
    and a 429 storm is the reason the earlier run produced 178 empty results
    that were indistinguishable from "no such place". Backoff is not an
    optimisation here, it is what keeps the failure modes distinguishable.
    """
    gen = config["generation"]
    max_attempts = int(gen.get("max_attempts", 6))
    base_delay = float(gen.get("backoff_base_seconds", 2.0))
    max_delay = float(gen.get("backoff_max_seconds", 120.0))

    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        try:
            content, metadata = client.call(messages, seed=seed)
            metadata["attempts"] = attempt
            return content, metadata, attempts
        except ApiError as exc:
            attempts.append({
                "attempt": attempt,
                "status": exc.status,
                "error": str(exc),
                "body_head": exc.body[:400],
            })
            if not exc.retryable or attempt == max_attempts:
                exc.attempt_log = attempts  # type: ignore[attr-defined]
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay += random.uniform(0, delay * 0.25)  # jitter, to desynchronise
            print(f"    {exc} - retry {attempt}/{max_attempts - 1} "
                  f"in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)

    raise ApiError("exhausted retries")  # unreachable, kept for the type checker


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def extract_json_object(text: str) -> str | None:
    """
    Pull the first balanced JSON object out of a string.

    Needed because not every model supports a JSON schema or a json_object
    response format. Without one, a model will happily return
        Sure, here is the post:
        ```json
        {...}
        ```
        Let me know if you want it shorter.
    and a reasoning model may put a paragraph of thinking in front of it.

    Scans for the first '{' and walks forward tracking brace depth, ignoring
    braces inside string literals and honouring backslash escapes. Returns the
    slice, or None if there is no balanced object.
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for i in range(start, len(text)):
        char = text[i]

        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def parse_response(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Turn the raw content into the four fields. Returns (parsed, error).

    Finds the first balanced JSON object anywhere in the response, so a model
    that cannot be held to a schema still parses as long as it emitted valid
    JSON somewhere. A response with no balanced object is recorded as a parse
    failure with the raw text preserved, and stage 7 decides what to do about
    it.

    The literal string "null" is treated as a missing value, not as the text
    "null". Nine of 1,646 cached responses in the previous run emitted it and
    were silently scored as a place called null.
    """
    if raw is None:
        return None, "content was null"

    text = raw.strip()
    if not text:
        return None, "content was empty"

    candidate = extract_json_object(text)
    if candidate is None:
        return None, "no balanced JSON object in the response"

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"not valid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"

    missing = [k for k in RESPONSE_SCHEMA["required"] if k not in parsed]
    if missing:
        return None, f"missing fields: {missing}"

    post_text = parsed.get("post_text")
    if not isinstance(post_text, str) or not post_text.strip():
        return None, "post_text was empty or not a string"

    naming_form = parsed.get("naming_form_used")
    if isinstance(naming_form, str) and naming_form.strip().lower() in {"null", "none", ""}:
        naming_form = None

    entities = parsed.get("entities_referenced")
    if not isinstance(entities, list):
        entities = []

    return {
        "post_text": post_text,
        "entities_referenced": [str(e) for e in entities],
        "naming_form_used": naming_form,
        "register_reported": parsed.get("register"),
    }, None


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #

def latest_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    The output file is append-only, so an item that failed and was later
    retried has two rows. The last row for an item_id is the current state of
    that item, and that is what resume must read.

    Stage 7 has to apply the same rule when it loads this file. Taking the
    first row, or all rows, double-counts every retried item.
    """
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        item_id = record.get("item_id")
        if item_id:
            latest[item_id] = record
    return latest


def select_items(
    plan: list[dict[str, Any]],
    done_ids: set[str],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    items = plan

    if args.only_condition:
        wanted = set(args.only_condition)
        items = [i for i in items if i["condition"] in wanted]
    if args.only_place:
        wanted = set(args.only_place)
        items = [i for i in items if i["place_qid"] in wanted]

    # Always skip what is already done. --retry-errors changes what counts as
    # done (failed rows stop counting), it does not switch the filter off.
    items = [i for i in items if i["item_id"] not in done_ids]

    # The plan is sorted by place, so working through it in file order means
    # any partial run - a --limit smoke test, the daily rate cap, an interrupt -
    # covers a few places completely and the rest not at all. item_id is a
    # hash, so its order interleaves places, conditions and continents, and it
    # is stable: the same plan always runs in the same order, which resume
    # relies on. Only the order of calls changes, never what is sent.
    items = sorted(items, key=lambda i: i["item_id"])

    if args.limit:
        items = items[: args.limit]
    return items


def main(argv: Sequence[str] | None = None) -> int:
    # Place names are not ASCII; the Windows console default codepage cannot
    # print them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Generate the planned items.")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--env-file", type=Path, default=HERE / ".env",
                        help="file holding the API key, read if present")
    parser.add_argument("--plan", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many items")
    parser.add_argument("--only-condition", action="append", default=None,
                        help="restrict to a condition, repeatable")
    parser.add_argument("--only-place", action="append", default=None,
                        help="restrict to a place QID, repeatable")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompts and send nothing")
    parser.add_argument("--no-exemplars", action="store_true",
                        help="generate without real-post style references "
                             "(normally built by build_exemplars.py)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="re-attempt items whose existing row has an error")
    args = parser.parse_args(argv)

    load_dotenv(args.env_file)
    config = load_config(args.config)
    paths = config["paths"]

    plan_path = args.plan or Path(paths.get("item_plan", "output/item_plan.jsonl"))
    out_path = args.out or Path(
        paths.get("raw_generations", "output/raw_generations.jsonl")
    )
    raw_log_path = Path(paths.get("raw_response_log", "output/raw_responses.jsonl"))
    exemplar_path = Path(paths.get("style_exemplars", "data/style_exemplars.jsonl"))

    plan = read_jsonl(plan_path)
    if not plan:
        raise SystemExit(f"no items in {plan_path} - run plan.py first")

    existing = latest_rows(read_jsonl(out_path))
    ok_ids = {i for i, r in existing.items() if not r.get("error")}
    error_ids = {i for i, r in existing.items() if r.get("error")}
    done_ids = ok_ids if args.retry_errors else ok_ids | error_ids

    exemplars = {} if args.no_exemplars else load_exemplars(exemplar_path)
    if not exemplars and not args.no_exemplars and not args.dry_run:
        raise SystemExit(
            f"no style exemplars at {exemplar_path}. Build them from the real "
            f"Bluesky posts with: python build_exemplars.py "
            f"(or pass --no-exemplars to generate without them)"
        )
    items = select_items(plan, done_ids, args)

    print(f"plan       {plan_path} ({len(plan)} items)")
    print(f"output     {out_path} ({len(ok_ids)} done, {len(error_ids)} failed)")
    print(f"to attempt {len(items)}")

    if args.retry_errors:
        retrying = [i for i in items if i["item_id"] in error_ids]
        print(f"           of which {len(retrying)} are previously failed rows")

    if not items:
        print("\nnothing to do")
        return 0

    if args.dry_run:
        for item in items:
            messages = build_messages(item, config, exemplars)
            print("\n" + "=" * 72)
            print(f"{item['item_id']}  {item['condition']}  "
                  f"{item['place_name']} ({item['place_qid']})  "
                  f"register={item['register']}")
            print("=" * 72)
            for message in messages:
                print(f"\n--- {message['role']} ---")
                print(message["content"])
        print(f"\ndry run: {len(items)} prompts shown, nothing sent")
        return 0

    client = ChatClient(config)
    delay_between = float(config["generation"].get("delay_between_calls_seconds", 1.0))

    print(f"model      {client.model} at {client.base_url}")
    print(f"structured {client.structured_output}\n")

    out_log = AppendLog(out_path)
    raw_log = AppendLog(raw_log_path)
    tally: Counter[str] = Counter()
    started = time.time()

    try:
        for n, item in enumerate(items, start=1):
            item_id = item["item_id"]
            messages = build_messages(item, config, exemplars)
            print(f"[{n}/{len(items)}] {item_id} {item['condition']:<9} "
                  f"{item['place_name']}", flush=True)

            content: str | None = None
            metadata: dict[str, Any] = {}
            attempt_log: list[dict[str, Any]] = []
            api_error: str | None = None

            try:
                content, metadata, attempt_log = call_with_backoff(
                    client, messages, config, seed=seed_for_item(item_id)
                )
            except ApiError as exc:
                api_error = str(exc)
                attempt_log = getattr(exc, "attempt_log", [])
                metadata = {"http_status": exc.status}

            # Rule 1: the raw response is on disk before anything reads it.
            raw_log.write({
                "item_id": item_id,
                "generator_version": GENERATOR_VERSION,
                "requested_utc": _utc_now(),
                "model": client.model,
                "base_url": client.base_url,
                "sampling": {
                    "temperature": client.temperature,
                    "top_p": client.top_p,
                    "max_tokens": client.max_tokens,
                    "reasoning_effort": client.reasoning_effort,
                },
                "messages": messages,
                "raw_content": content,
                "response_metadata": metadata,
                "attempt_log": attempt_log,
                "api_error": api_error,
            })

            record: dict[str, Any] = {
                "item_id": item_id,
                "generator_version": GENERATOR_VERSION,
                "plan_version": item.get("plan_version"),
                "generated_utc": _utc_now(),
                "model": client.model,
                "condition": item["condition"],
                "signal": item.get("signal"),
                "place_qid": item["place_qid"],
                "place_name": item["place_name"],
                "place_country": item["place_country"],
                "place_continent": item["place_continent"],
                "clue_qids": [c["clue_qid"] for c in item["clues"]],
                "surface_form": item["surface_form"],
                "noise_entity_qid": (item.get("noise_entity") or {}).get("clue_qid"),
                "register": item["register"],
                "length_band": item["length_band"],
                "response_metadata": metadata,
            }

            if api_error is not None:
                # Rule 3: a failure is a row, not a gap.
                record["error"] = api_error
                record["error_kind"] = "api"
                tally["api_error"] += 1
            else:
                parsed, parse_error = parse_response(content or "")
                if parse_error is not None:
                    record["error"] = parse_error
                    record["error_kind"] = "parse"
                    record["raw_content"] = content
                    tally["parse_error"] += 1
                else:
                    record.update(parsed)
                    tally["ok"] += 1

            out_log.write(record)

            if delay_between and n < len(items):
                time.sleep(delay_between)

    except KeyboardInterrupt:
        print("\ninterrupted - everything attempted so far is on disk; "
              "re-run to continue", file=sys.stderr)
    finally:
        out_log.close()
        raw_log.close()

    elapsed = time.time() - started
    print(f"\nattempted {sum(tally.values())} items in {elapsed / 60:.1f} min")
    for kind in ("ok", "parse_error", "api_error"):
        if tally[kind]:
            print(f"  {kind:<12} {tally[kind]:>5}")
    print(f"\nrows      {out_path}")
    print(f"raw log   {raw_log_path}")
    if tally["api_error"] or tally["parse_error"]:
        print("\nfailed rows are in the output with an error field. "
              "Re-attempt them with --retry-errors.")
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
