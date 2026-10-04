#!/usr/bin/env python3
"""
generate.py - send the planned items to the generating model.

WHAT THIS SCRIPT DOES
    Reads output/item_plan.jsonl, sends one request per planned item to a
    chat-completions API, and appends the result to
    output/raw_generations.jsonl.

    It decides nothing. Every choice was already made by plan.py. This
    script turns a planned item into a prompt, sends it, and records what came
    back. finalise.py then validates the rows and writes the dataset.

    Two OpenRouter clients exist in this repository on purpose: this
    stdlib-only one (urllib, no third-party packages, its own .env reader and
    backoff) and src/llm_openrouter.py (requests + python-dotenv) used by the
    evaluation code. Consolidating them is a deliberate non-goal for now.

FOUR RULES THAT ARE NOT NEGOTIABLE
    1. The raw response is logged before anything tries to parse it. A parser
       bug must never be able to destroy a response you have already paid for.
       The raw log is append-only and flushed on every write.
    2. Work is cached by item_id. A re-run skips items already present in the
       output and resumes where the last run stopped. A full pass can be
       interrupted (rate limits, network, Ctrl-C). Every row carries a
       config_hash (model, prompt wording, sampling settings); a resume whose
       existing rows were made with a different hash is refused unless
       --force-mixed is given, so one dataset cannot silently mix generators.
    3. A failure is recorded, never skipped. A refused, truncated,
       rate-limited or unparseable item gets a row with an "error" field. A
       missing row and a failed row mean different things, and finalise.py
       needs to be able to tell them apart.
    4. No fallback changes content. Retries resend the same prompt. If a
       request keeps failing the item is written with an error and the run
       moves on. Nothing here rewrites a prompt, drops a constraint or
       substitutes a different clue to get a usable answer.

WHAT THIS SCRIPT MUST NEVER DO
    - validate the generated text, score it, or check whether the model leaked
      the place name (that is finalise.py, and it needs the leaks on record)
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
    config.yaml         endpoint, model, prompt wording, rate limits

OUTPUTS
    output/raw_generations.jsonl  one row per attempt, append-only. An item
                                   that failed and was later retried has more
                                   than one row; the LAST row for an item_id is
                                   its current state, and finalise.py loads the
                                   file that way.
    output/raw_responses.jsonl    prompt and unparsed response body per call

USAGE
    python generate.py --dry-run --limit 5     # see the prompts, send nothing
    python generate.py --limit 20              # a paid smoke test
    python generate.py                         # the full pass
    python generate.py --only-condition A3
    python generate.py --retry-errors          # re-attempt failed rows only
    python generate.py --force-mixed           # resume although existing rows
                                               # came from a different config
"""

from __future__ import annotations

import argparse
import email.utils
import hashlib
import http.client
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
from typing import Any, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


HERE = Path(__file__).resolve().parent

GENERATOR_VERSION = "1"

# Defaults for optional generation: keys in config.yaml. This is the single
# place they are declared; read them through gen_setting() so the client, the
# backoff loop and the config hash all agree.
GENERATION_DEFAULTS: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 600,
    "timeout_seconds": 120,
    "structured_output": "json_schema",
    "send_seed": False,
    "exclude_reasoning": False,
    "reasoning_effort": None,
    "exemplars_per_prompt": 3,
    "delay_between_calls_seconds": 1.0,
    "max_attempts": 6,
    "backoff_base_seconds": 2.0,
    "backoff_max_seconds": 120.0,
    "jitter_fraction": 0.25,        # random extra delay, as a share of the delay
    "error_body_head_chars": 400,   # response body kept in the attempt log
}

# paths: keys the scripts index directly. load_config checks them once.
REQUIRED_PATHS = (
    "item_plan", "raw_generations", "raw_response_log", "style_exemplars",
    "place_aliases", "bluesky_dump", "dataset_csv", "finalise_report",
)

# Keys of the config's prompt: section that build_messages needs.
REQUIRED_PROMPT_KEYS = (
    "system", "task", "length_bands", "forbidden_toponym", "ambiguous_toponym",
    "exact_toponym", "entities_header", "clue_line", "noise_entity",
    "surface_ambiguous", "surface_exact", "exemplar_intro", "register_line",
    "length_line", "register_notes_line", "slots_footer", "output_intro",
    "output_fields",
)

# Status codes worth resending: request timeout, rate limit, server errors.
RETRYABLE_STATUS = frozenset({408, 429})

# Seeds are masked to 31 bits: providers that validate a signed int32 reject
# anything above 2**31 - 1.
SEED_MASK = 0x7FFFFFFF
SEED_HEX_DIGITS = 8


def gen_setting(gen: dict[str, Any], key: str) -> Any:
    """A generation: setting from the config, or its declared default."""
    return gen.get(key, GENERATION_DEFAULTS[key])

# The shape the model is asked to return. Sent as a JSON schema when the
# provider supports structured output, and restated in the prompt either way.
#
# The three fields beyond post_text are self-reports, not ground truth. They
# are cheap to collect and useful for triage in finalise.py: a mismatch between
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
    missing = [k for k in REQUIRED_PATHS if k not in config["paths"]]
    if missing:
        raise SystemExit(f"{path}: paths: is missing {missing}")
    missing = [k for k in REQUIRED_PROMPT_KEYS if k not in config["prompt"]]
    if missing:
        raise SystemExit(f"{path}: prompt: is missing {missing}")
    unwritten = [f for f in RESPONSE_SCHEMA["properties"]
                 if f not in config["prompt"]["output_fields"]]
    if unwritten:
        raise SystemExit(f"{path}: prompt.output_fields has no wording for "
                         f"{unwritten} (every RESPONSE_SCHEMA field needs one)")
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

    entity_type in the plan is the places pipeline's property family code (place_or_org).
    The prompt shows the wording from prompt.entity_type_labels instead; a
    family with no wording there gets no type note at all, rather than the raw
    code.
    """
    relation = clue.get("property_label") or clue.get("property")
    labels = prompt_cfg.get("entity_type_labels") or {}
    entity_type = labels.get(clue.get("entity_type"))
    suffix = f" ({entity_type})" if entity_type else ""
    return prompt_cfg["clue_line"].format(
        label=clue["clue_label"], type_note=suffix, relation=relation
    )


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
    surface = item["surface_form"]
    clues = item["clues"]
    noise = item.get("noise_entity")

    parts: list[str] = []

    # 1. task
    parts.append(prompt_cfg["task"].strip())

    # 2. entities the post must mention
    mentions: list[str] = []
    if clues:
        mentions.append(prompt_cfg["entities_header"].strip())
        mentions.extend(describe_clue(c, prompt_cfg) for c in clues)
    if noise:
        mentions.append(prompt_cfg["noise_entity"].strip().format(
            label=noise["clue_label"]))
    if surface.get("text"):
        key = ("surface_ambiguous" if surface["kind"] == "ambiguous_alias"
               else "surface_exact")
        mentions.append(prompt_cfg[key].strip().format(surface=surface["text"]))
    if mentions:
        parts.append("\n".join(mentions))

    # 3. hard constraint
    parts.append(build_constraint(item, prompt_cfg).strip())

    # 4. style references, labelled as tone only
    register = item["register"]
    pool = exemplars.get(register, [])
    n_exemplars = int(gen_setting(config["generation"], "exemplars_per_prompt"))
    if pool:
        # Deterministic per item, so the same item always gets the same
        # references and a cache hit means the same prompt.
        rng = random.Random(item["item_id"])
        chosen = rng.sample(pool, min(n_exemplars, len(pool)))
        block = [prompt_cfg["exemplar_intro"].strip()]
        block.extend(f"  {i}. {text}" for i, text in enumerate(chosen, start=1))
        parts.append("\n".join(block))

    # 5. slot values from the plan
    detail: list[str] = [
        prompt_cfg["register_line"].strip().format(register=register),
        prompt_cfg["length_line"].strip().format(
            length=prompt_cfg["length_bands"][item["length_band"]]),
    ]
    if item.get("register_description"):
        detail.append(prompt_cfg["register_notes_line"].strip().format(
            notes=item["register_description"]))
    for slot_name, slot_value in (item.get("slots") or {}).items():
        detail.append(f"{slot_name.replace('_', ' ').capitalize()}: {slot_value}")
    detail.append(prompt_cfg["slots_footer"].strip())
    parts.append("\n".join(detail))

    # 6. output shape, generated from RESPONSE_SCHEMA so the two cannot drift
    parts.append(build_output_shape(prompt_cfg))

    user_content = "\n\n".join(parts)
    system_content = prompt_cfg.get("system", "").strip()

    messages: list[dict[str, str]] = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    return messages


def build_output_shape(prompt_cfg: dict[str, Any]) -> str:
    """
    The "return one JSON object" block. Field names and their order come from
    RESPONSE_SCHEMA; the wording for each field is prompt.output_fields in the
    config, so the schema and the prompt cannot list different fields.
    """
    fields = list(RESPONSE_SCHEMA["properties"])
    width = max(len(name) for name in fields) + 2
    lines = [prompt_cfg["output_intro"].strip()]
    for name in fields:
        lines.append(f"  {name.ljust(width)}{prompt_cfg['output_fields'][name].strip()}")
    return "\n".join(lines)


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
    """
    A failed request. retryable says whether resending the same prompt could
    succeed; retry_after is the server's requested wait in seconds, if it gave
    one; attempt_log is filled in by call_with_backoff before it re-raises.
    """

    def __init__(self, message: str, status: int | None = None,
                 retryable: bool = False, body: str = "",
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.body = body
        self.retry_after = retry_after
        self.attempt_log: list[dict[str, Any]] = []


def is_retryable_status(code: int | None) -> bool:
    """408, 429 and every 5xx can succeed on a resend; other codes cannot."""
    return code is not None and (code in RETRYABLE_STATUS or code >= 500)


def parse_retry_after(value: str | None) -> float | None:
    """
    A Retry-After header is either a number of seconds or an HTTP date. Returns
    seconds to wait (never negative), or None if absent or unreadable.
    """
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


def _error_code(parsed: Any) -> int | None:
    """The numeric code inside a gateway's {"error": {"code": ...}} body."""
    error = parsed.get("error") if isinstance(parsed, dict) else None
    code = error.get("code") if isinstance(error, dict) else None
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


# Two OpenRouter clients exist in this repository (see the module docstring):
# this one has no third-party dependencies, src/llm_openrouter.py uses requests.
class ChatClient:
    """
    A minimal client for an OpenAI-compatible /chat/completions endpoint.

    Compatible providers include Mistral, OpenRouter, Together and most
    free-tier gateways, which is why this shape is used rather than a
    provider-specific SDK: switching provider is a config change.

    408, 429 and 5xx are retryable, as are network failures and a 200 response
    whose body is a gateway error carrying such a code. Other 4xx are not - a
    bad request does not become good by being resent.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        gen = config["generation"]
        self.base_url = gen["base_url"].rstrip("/")
        self.model = gen["model"]
        self.temperature = float(gen_setting(gen, "temperature"))
        self.top_p = float(gen_setting(gen, "top_p"))
        self.max_tokens = int(gen_setting(gen, "max_tokens"))
        self.timeout = float(gen_setting(gen, "timeout_seconds"))
        self.structured_output = gen_setting(gen, "structured_output")

        # Send a per-item seed when the model accepts one. Providers treat it
        # as best effort, not a guarantee, but it costs nothing and makes an
        # individual call closer to repeatable when you go back to inspect it.
        self.send_seed = bool(gen_setting(gen, "send_seed"))

        # Reasoning models put their thinking somewhere. Excluding it keeps the
        # content field clean and stops reasoning tokens eating max_tokens.
        self.exclude_reasoning = bool(gen_setting(gen, "exclude_reasoning"))

        # Excluding reasoning only hides it: the model still thinks, and the
        # thinking still counts against max_tokens. A reasoning model left at
        # its default effort can spend the whole budget thinking and return
        # empty content. The effort level is what actually shortens it.
        self.reasoning_effort = gen_setting(gen, "reasoning_effort") or None

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
                body = response.read().decode("utf-8", errors="replace")
                status = response.status
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except (OSError, http.client.HTTPException):
                body = ""
            retry_after = (parse_retry_after(exc.headers.get("Retry-After"))
                           if exc.headers else None)
            raise ApiError(
                f"HTTP {exc.code}", status=exc.code,
                retryable=is_retryable_status(exc.code), body=body,
                retry_after=retry_after,
            ) from exc
        except urllib.error.URLError as exc:
            raise ApiError(f"network error: {exc.reason}", retryable=True) from exc
        except TimeoutError as exc:
            raise ApiError("request timed out", retryable=True) from exc
        except (OSError, http.client.HTTPException) as exc:
            # A connection dropped while the response was being read
            # (RemoteDisconnected, IncompleteRead, ConnectionResetError...)
            # is not wrapped by urllib. It is a network blip, not a bad
            # request, so it is retried like one.
            raise ApiError(f"connection error: {type(exc).__name__}: {exc}",
                           retryable=True) from exc

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError("response body was not JSON", status=status,
                           retryable=False, body=body) from exc

        try:
            choice = parsed["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            # Gateways sometimes report an upstream failure inside a 200:
            # {"error": {"code": 429, ...}}. Retry when that code says so.
            code = _error_code(parsed)
            raise ApiError("response held no message content",
                           status=code if code is not None else status,
                           retryable=is_retryable_status(code), body=body) from exc

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
    Masked to 31 bits (SEED_MASK) because providers that validate a signed
    int32 reject anything above 2**31 - 1.
    """
    return int(item_id[:SEED_HEX_DIGITS], 16) & SEED_MASK


def call_with_backoff(
    client: ChatClient,
    messages: list[dict[str, str]],
    config: dict[str, Any],
    seed: int | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """
    Retry with exponential backoff and jitter. Returns
    (content, metadata, attempt_log) or raises the final ApiError, whose
    attempt_log attribute holds the failed attempts. A server-supplied
    Retry-After (seconds or HTTP date) lengthens the wait, up to the backoff
    maximum.

    Rate limits are the single largest cost in wall-clock time on a free tier,
    and a 429 storm is the reason the earlier run produced 178 empty results
    that were indistinguishable from "no such place". Backoff is not an
    optimisation here, it is what keeps the failure modes distinguishable.
    """
    gen = config["generation"]
    max_attempts = max(1, int(gen_setting(gen, "max_attempts")))
    base_delay = float(gen_setting(gen, "backoff_base_seconds"))
    max_delay = float(gen_setting(gen, "backoff_max_seconds"))
    jitter = float(gen_setting(gen, "jitter_fraction"))
    head_chars = int(gen_setting(gen, "error_body_head_chars"))

    attempts: list[dict[str, Any]] = []
    attempt = 0
    while True:
        attempt += 1
        try:
            content, metadata = client.call(messages, seed=seed)
            metadata["attempts"] = attempt
            return content, metadata, attempts
        except ApiError as exc:
            attempts.append({
                "attempt": attempt,
                "status": exc.status,
                "error": str(exc),
                "body_head": exc.body[:head_chars],
            })
            if not exc.retryable or attempt >= max_attempts:
                exc.attempt_log = attempts
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            if exc.retry_after is not None:
                delay = max(delay, min(exc.retry_after, max_delay))
            delay += random.uniform(0, delay * jitter)  # desynchronise
            print(f"    {exc} - retry {attempt}/{max_attempts - 1} "
                  f"in {delay:.1f}s", file=sys.stderr)
            time.sleep(delay)


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
    failure with the raw text preserved, and finalise.py decides what to do about
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
# resume, record building and the main loop
# --------------------------------------------------------------------------- #

def latest_rows(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    The output file is append-only, so an item that failed and was later
    retried has two rows. The last row for an item_id is the current state of
    that item, and that is what resume must read.

    finalise.py applies the same rule when it loads this file. Taking the
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


def config_fingerprint(config: dict[str, Any], use_exemplars: bool = True) -> str:
    """
    A short hash of everything that changes what the generator produces: the
    model, the whole prompt: section, the response schema, every sampling and
    reasoning setting and the exemplar switch. Stored on every row so a resume
    can tell when the config was edited part-way through a dataset.

    Not covered: the exemplar file's contents, and the plan itself (plan_version
    on the row tracks that).
    """
    gen = config["generation"]
    settings = {key: gen_setting(gen, key) for key in (
        "temperature", "top_p", "max_tokens", "structured_output", "send_seed",
        "exclude_reasoning", "reasoning_effort", "exemplars_per_prompt",
    )}
    material = {
        "generator_version": GENERATOR_VERSION,
        "model": gen["model"],
        "settings": settings,
        "prompt": config["prompt"],
        "schema": RESPONSE_SCHEMA,
        "use_exemplars": use_exemplars,
    }
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def foreign_config_rows(
    kept: Sequence[dict[str, Any]], current_hash: str
) -> list[dict[str, Any]]:
    """Rows whose config_hash is missing or differs from the current one."""
    return [r for r in kept if r.get("config_hash") != current_hash]


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
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
    parser.add_argument("--force-mixed", action="store_true",
                        help="resume even though existing rows were generated "
                             "with a different model, prompt or sampling "
                             "config (the dataset will mix generators)")
    return parser.parse_args(argv)


def show_prompts(
    items: list[dict[str, Any]],
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
) -> None:
    """The dry run: print every prompt that would be sent. No network, no key."""
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


def build_raw_entry(
    item: dict[str, Any],
    client: ChatClient,
    config_hash: str,
    messages: list[dict[str, str]],
    content: str | None,
    metadata: dict[str, Any],
    attempt_log: list[dict[str, Any]],
    api_error: str | None,
) -> dict[str, Any]:
    """One raw_responses.jsonl entry: the prompt and the unparsed reply."""
    return {
        "item_id": item["item_id"],
        "generator_version": GENERATOR_VERSION,
        "config_hash": config_hash,
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
    }


def build_record(
    item: dict[str, Any],
    model: str,
    config_hash: str,
    content: str | None,
    metadata: dict[str, Any],
    api_error: str | None,
) -> tuple[dict[str, Any], str]:
    """
    The raw_generations.jsonl row for one attempt, and its outcome: "ok",
    "parse_error" or "api_error". Rule 3: a failure is a row, not a gap.
    """
    record: dict[str, Any] = {
        "item_id": item["item_id"],
        "generator_version": GENERATOR_VERSION,
        "config_hash": config_hash,
        "plan_version": item.get("plan_version"),
        "generated_utc": _utc_now(),
        "model": model,
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
        record["error"] = api_error
        record["error_kind"] = "api"
        return record, "api_error"

    parsed, parse_error = parse_response(content or "")
    if parse_error is not None:
        record["error"] = parse_error
        record["error_kind"] = "parse"
        record["raw_content"] = content
        return record, "parse_error"

    record.update(parsed)
    return record, "ok"


def run_items(
    items: list[dict[str, Any]],
    client: ChatClient,
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
    config_hash: str,
    out_log: AppendLog,
    raw_log: AppendLog,
    tally: Counter[str],
) -> None:
    """
    Send each item, log the raw response, then log the parsed row, counting
    outcomes into tally. KeyboardInterrupt propagates; everything attempted so
    far is already on disk.
    """
    delay_between = float(gen_setting(config["generation"],
                                      "delay_between_calls_seconds"))

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
            attempt_log = exc.attempt_log
            metadata = {"http_status": exc.status}

        # Rule 1: the raw response is on disk before anything reads it.
        raw_log.write(build_raw_entry(item, client, config_hash, messages,
                                      content, metadata, attempt_log, api_error))

        record, outcome = build_record(item, client.model, config_hash,
                                       content, metadata, api_error)
        tally[outcome] += 1
        out_log.write(record)

        if delay_between and n < len(items):
            time.sleep(delay_between)


def main(argv: Sequence[str] | None = None) -> int:
    # Place names are not ASCII; the Windows console default codepage cannot
    # print them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    load_dotenv(args.env_file)
    config = load_config(args.config)
    paths = config["paths"]

    plan_path = args.plan or Path(paths["item_plan"])
    out_path = args.out or Path(paths["raw_generations"])
    raw_log_path = Path(paths["raw_response_log"])
    exemplar_path = Path(paths["style_exemplars"])

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

    # SYN-15: the rows being kept must come from the config about to be used.
    config_hash = config_fingerprint(config, use_exemplars=not args.no_exemplars)
    foreign = foreign_config_rows([existing[i] for i in done_ids], config_hash)
    if foreign:
        message = (f"{len(foreign)} existing rows in {out_path} were generated "
                   f"with a different config (or carry no config_hash); "
                   f"current hash is {config_hash}")
        if args.dry_run or args.force_mixed:
            print(f"warning    {message}", file=sys.stderr)
        else:
            raise SystemExit(
                f"{message}.\nResuming would mix generators in one dataset. "
                f"Move the old output aside and regenerate, or pass "
                f"--force-mixed to continue anyway."
            )

    if not items:
        print("\nnothing to do")
        return 0

    if args.dry_run:
        show_prompts(items, config, exemplars)
        return 0

    client = ChatClient(config)
    print(f"model      {client.model} at {client.base_url}")
    print(f"structured {client.structured_output}")
    print(f"config     {config_hash}\n")

    out_log = AppendLog(out_path)
    raw_log = AppendLog(raw_log_path)
    started = time.time()
    tally: Counter[str] = Counter()

    try:
        run_items(items, client, config, exemplars, config_hash,
                  out_log, raw_log, tally)
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
