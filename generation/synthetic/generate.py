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
import copy
import email.utils
import hashlib
import http.client
import json
import os
import random
import re
import ssl
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

import plan   # noqa: E402  (contains_word, so prompt and planner match names alike)


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
    "reasoning_enabled": True,
    "extra_body": {},
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
    "item_plan", "raw_generations", "raw_response_log", "style_exemplars", "length_dist",
    "place_inputs", "bluesky_dump", "dataset_csv", "finalise_report",
)

# Keys of the config's prompt: section that build_messages needs.
REQUIRED_PROMPT_KEYS = (
    "system", "task", "length_bands", "forbidden_toponym", "ambiguous_toponym",
    "place_clue_exception", "scale_lines", "valence_lines", "role_hint_intro", "role_hints",
    "claim_rules", "target_len_line",
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
        # First, so the model settles the one situation the post grows from
        # before it writes: with thinking off this is its only planning step.
        # A reason per name (smoke5-6) gave posts that listed the reasons.
        # Not dataset content.
        "situation": {
            "type": "string",
            "description": "One sentence: the single situation that ties every "
                           "name and the environmental point together.",
        },
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
        "angle": {
            "type": "string",
            "description": "The environmental angle the writer chose, in a short phrase.",
        },
    },
    "required": ["situation", "post_text", "entities_referenced", "naming_form_used",
                 "register", "angle"],
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

    The prompt gives each entity's full name, and the model writes it by name:
    a person by the shorter name people use, with the surname (finalise.py
    checks for it), anything else as given. An indirect reference ("the
    anatomy professor who lived here") gives a reader nothing to resolve, so
    the task would stop being geolocation and start being guesswork.

    entity_type in the plan is the places pipeline's property family code (place_or_org).
    The prompt shows the wording from prompt.entity_type_labels instead; a
    family with no wording there gets no type note at all, rather than the raw
    code.
    """
    phrases = prompt_cfg.get("relation_phrases") or {}
    relation = (phrases.get(clue.get("property"))
                or clue.get("property_label") or clue.get("property"))
    template = (prompt_cfg.get("clue_line_person") if clue.get("is_human") else None) \
        or prompt_cfg["clue_line"]
    note = type_note(clue, prompt_cfg)
    return template.format(
        label=clue["clue_label"], type_note=note, relation=relation,
        known_for=clue.get("description") or note.strip(" ()") or "a person",
    )


def type_note(clue: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    """' (a person)' and so on from prompt.entity_type_labels, or '' for an
    unlisted family or a property in prompt.untyped_properties."""
    labels = prompt_cfg.get("entity_type_labels") or {}
    entity_type = labels.get(clue.get("entity_type"))
    if clue.get("is_living") is not None:   # people: living or from history
        entity_type = labels.get("person_living" if clue["is_living"] else "person_historical",
                                 entity_type)
    if clue.get("property") in (prompt_cfg.get("untyped_properties") or []):
        entity_type = None
    return f" ({entity_type})" if entity_type else ""


def build_messages(
    item: dict[str, Any],
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
) -> list[dict[str, str]]:
    """
    Assemble the prompt in a fixed order:
        task -> entities -> hard constraint -> environmental theme -> style
        references -> slots -> schema

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
            label=noise["clue_label"], type_note=type_note(noise, prompt_cfg)))
    if surface.get("text"):
        key = ("surface_ambiguous" if surface["kind"] == "ambiguous_alias"
               else "surface_exact")
        mentions.append(prompt_cfg[key].strip().format(surface=surface["text"]))
    if mentions:
        parts.append("\n".join(mentions))

    # 3. hard constraint
    parts.append(build_constraint(item, prompt_cfg).strip())

    # 3b. what the post is about: an environmental claim on the planned theme,
    # with the planned valence and scale, a hint from the clues' roles and the
    # rules for claims (items planned before version 6 have no scale or role,
    # before version 8 no valence)
    if item.get("env_theme"):
        parts.append(prompt_cfg["environment_line"].strip().format(theme=item["env_theme"]))
        if item.get("valence"):
            parts.append(prompt_cfg["valence_lines"][item["valence"]].strip())
        if item.get("scale"):
            parts.append(prompt_cfg["scale_lines"][item["scale"]].strip())
        hints = prompt_cfg.get("role_hints") or {}
        if item["clues"]:
            parts.append("\n".join([prompt_cfg["role_hint_intro"].strip()] + [
                f'- "{c["clue_label"]}": {hints.get(c.get("role") or "other", hints["other"])}'
                for c in item["clues"]]))
        parts.append(prompt_cfg["claim_rules"].strip())

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
    # The plan's names are codes (personal_update, mood_marker); the model
    # sees words.
    slot_labels = prompt_cfg.get("slot_labels") or {}
    detail: list[str] = [
        prompt_cfg["register_line"].strip().format(register=register.replace("_", " ")),
        prompt_cfg["target_len_line"].strip().format(n=item["target_len"]) if "target_len" in item
        else prompt_cfg["length_line"].strip().format(
            length=prompt_cfg["length_bands"][item["length_band"]]),
    ]
    if item.get("register_description"):
        detail.append(prompt_cfg["register_notes_line"].strip().format(
            notes=item["register_description"]))
    # prompt.show_slots: false leaves the planned slots out of the prompt (they
    # stay in the plan). They competed with the clues for words.
    if prompt_cfg.get("show_slots", True) and item.get("slots"):
        for slot_name, slot_value in item["slots"].items():
            label = slot_labels.get(slot_name) or slot_name.replace("_", " ").capitalize()
            detail.append(f"{label}: {slot_value}")
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


def countries_of(item: dict[str, Any]) -> list[str]:
    """The place's countries. The place CSV joins several with "; " (Lake
    Geneva: "France; Switzerland")."""
    return [c.strip() for c in (item.get("place_country") or "").split(";") if c.strip()]


def quoted_list(names: Sequence[str]) -> str:
    """'"A"', '"A" or "B"', '"A", "B" or "C"'."""
    quoted = [f'"{name}"' for name in names]
    if len(quoted) <= 1:
        return "".join(quoted)
    return ", ".join(quoted[:-1]) + " or " + quoted[-1]


def build_constraint(item: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    """
    The constraint block, chosen by what the condition forbids.

    Templates live in the config so the wording can be revised without touching
    code. The banned-string list is built from the plan, not from the model's
    knowledge of the place.

    For an ambiguous form the place name is banned only when the form is not
    itself that name: a B item whose planned form is "Lake Baikal" must not
    also be told not to write "Lake Baikal".
    """
    surface = item["surface_form"]
    countries = countries_of(item)
    # Clues that are themselves place names (part, water and range families)
    # are allowed explicitly, and nothing else.
    toponyms = [c["clue_label"] for c in (item.get("clues") or []) + [item.get("noise_entity") or {}]
                if c.get("is_toponym")]
    river_line = (" " + prompt_cfg["place_clue_exception"].format(names=quoted_list(toponyms))
                  if toponyms else "")
    if surface["kind"] == "none":
        banned = prompt_cfg["forbidden_toponym"]
        return banned.format(
            place_name=item["place_name"],
            country=quoted_list(countries),
        ) + river_line
    if surface["kind"] == "ambiguous_alias":
        place = item["place_name"]
        others = [] if plan.contains_word(surface["text"], place) else [place]
        return prompt_cfg["ambiguous_toponym"].format(
            surface=surface["text"],
            other_names=quoted_list(others + countries),
        ) + river_line
    # EXPLICIT: the bare name, with nothing added. A country that is part of
    # the name itself ("Guatemala City") is not banned.
    others = [c for c in countries if not plan.contains_word(surface["text"], c)]
    return prompt_cfg["exact_toponym"].format(
        surface=surface["text"],
        place_name=item["place_name"],
        other_names=quoted_list(others) or "any country or region",
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


def tls_context() -> ssl.SSLContext:
    """
    Certificate checks against certifi's current list of authorities when it
    is installed (on its own, or the copy inside pip), else Python's default.
    The default store on this machine lacks Let's Encrypt's newer chain, so
    litellm.vse.cz (issuer YE1) failed with "certificate has expired" although
    its certificate is valid. Verification stays on either way.
    """
    try:
        import certifi
    except ImportError:
        try:
            from pip._vendor import certifi
        except ImportError:
            return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


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

        # Off altogether (OpenRouter's reasoning.enabled = false): no thinking
        # tokens at all, which effort and exclude cannot promise.
        self.reasoning_enabled = bool(gen_setting(gen, "reasoning_enabled"))

        # Provider-specific fields added to every request as they are, e.g.
        # reasoning_effort: none for Qwen behind LiteLLM.
        self.extra_body: dict[str, Any] = dict(gen_setting(gen, "extra_body") or {})

        # Optional attribution headers. OpenRouter recommends them and ignores
        # them if absent; other providers ignore them entirely.
        self.extra_headers: dict[str, str] = dict(gen.get("extra_headers") or {})

        self.ssl_context = tls_context()

        key_env = gen["api_key_env"]
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key:
            raise SystemExit(
                f"no API key: set {key_env} in .env or in the environment. "
                f"The key is never read from the config file, so the config can "
                f"live in the repository."
            )

    def build_payload(
        self, messages: list[dict[str, str]], seed: int | None = None,
        schema: tuple[dict[str, Any], str] | None = None,
    ) -> dict[str, Any]:
        schema_body, schema_name = schema or (RESPONSE_SCHEMA, "social_media_post")
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
        if not self.reasoning_enabled:
            reasoning["enabled"] = False
        else:
            if self.reasoning_effort:
                reasoning["effort"] = self.reasoning_effort
            if self.exclude_reasoning:
                reasoning["exclude"] = True
        if reasoning:
            payload["reasoning"] = reasoning
        payload.update(self.extra_body)
        if self.structured_output == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema_body,
                },
            }
        elif self.structured_output == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def call(
        self, messages: list[dict[str, str]], seed: int | None = None,
        schema: tuple[dict[str, Any], str] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        One request. Returns (raw_content, response_metadata).
        Raises ApiError; retrying is the caller's job.
        """
        payload = self.build_payload(messages, seed=seed, schema=schema)
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
            with urllib.request.urlopen(request, timeout=self.timeout,
                                        context=self.ssl_context) as response:
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
        # Some (Qwen behind LiteLLM/Ollama) put it inline as <think>...</think>
        # instead, where its braces would confuse the JSON extraction.
        if content and "<think>" in content:
            stripped = re.sub(r"<think>.*?(</think>|$)", "", content, flags=re.S)
            metadata["reasoning_chars"] = len(content) - len(stripped)
            content = stripped

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
    schema: tuple[dict[str, Any], str] | None = None,
) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    """
    Retry with exponential backoff and jitter. schema, (JSON schema, name),
    replaces RESPONSE_SCHEMA for this call (judge.py). Returns
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
            content, metadata = client.call(messages, seed=seed, schema=schema)
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
        "angle": parsed.get("angle"),
        "situation": parsed.get("situation"),
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

    # Reserve items (pool "reserve") replace posts that failed finalise.py;
    # they run only when asked for. Plans before version 5 have no pool.
    items = [i for i in items if i.get("pool", "plan") in args.pool]
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

    if getattr(args, "sample_per_condition", None):
        items = spread_sample(items, args.sample_per_condition)
    if args.limit:
        items = items[: args.limit]
    return items


def spread_sample(items: list[dict[str, Any]], per_condition: int) -> list[dict[str, Any]]:
    """
    per_condition items of each condition, taken in turn from each (continent,
    place type) group, so a small pilot covers the spread. Within a group the
    item_id order is kept (a hash, so arbitrary but stable).
    """
    chosen = []
    for condition in sorted({i["condition"] for i in items}):
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for i in items:
            if i["condition"] == condition:
                groups[i["place_continent"], i["place_type"]].append(i)
        queues = [groups[k] for k in sorted(groups)]
        picked: list[dict[str, Any]] = []
        while len(picked) < per_condition and any(queues):
            for q in queues:
                if q and len(picked) < per_condition:
                    picked.append(q.pop(0))
        chosen += picked
    return sorted(chosen, key=lambda i: i["item_id"])


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
        "exclude_reasoning", "reasoning_effort", "reasoning_enabled", "extra_body",
        "exemplars_per_prompt",
    )}
    material = {
        "generator_version": GENERATOR_VERSION,
        "model": gen["model"],
        "settings": settings,
        "prompt": config["prompt"],
        "schema": RESPONSE_SCHEMA,
        "use_exemplars": use_exemplars,
        # The judge decides which tries are kept, so a change to it is a
        # change to the dataset too.
        "judge": config.get("judge"),
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
    parser.add_argument("--pool", nargs="+", choices=["plan", "reserve"], default=["plan"],
                        help="which planned items to run: plan (default), reserve "
                             "(to replace failed posts), or both")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompts and send nothing")
    parser.add_argument("--no-exemplars", action="store_true",
                        help="generate without real-post style references "
                             "(normally built by build_exemplars.py)")
    parser.add_argument("--retry-errors", action="store_true",
                        help="re-attempt items whose existing row has an error")
    parser.add_argument("--sample-per-condition", type=int, default=None, metavar="N",
                        help="run only N items per condition, spread over continents and "
                             "place types (the pilot)")
    parser.add_argument("--no-judge", action="store_true",
                        help="skip the judge model; only the rule checks decide")
    parser.add_argument("--no-swap", action="store_true",
                        help="do not swap failed places for reserves")
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
            "reasoning_enabled": client.reasoning_enabled,
            "extra_body": client.extra_body,
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
        "length_band": item.get("length_band"),
        "target_len": item.get("target_len"),
        "scale": item.get("scale"),
        "valence": item.get("valence"),
        "role": item.get("role"),
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


def try_seed(item_id: str, n: int) -> int:
    """The seed for try n of an item: try 1 keeps the item's own seed, later
    tries hash the item_id with the try number."""
    if n == 1:
        return seed_for_item(item_id)
    return seed_for_item(hashlib.sha256(f"{item_id}#{n}".encode()).hexdigest())


def retry_feedback(last_try: dict[str, Any], prompt_cfg: dict[str, Any]) -> str:
    """The prompt.retry_feedback block for the next try: the rejected post, the
    plain-words explanation of its status and the checker's own reason
    (2026-11-02, user: retry with the rejection reason before failing)."""
    status = last_try["status"]
    problem = (prompt_cfg.get("rejection_explanations") or {}).get(status, status.replace("_", " "))
    return prompt_cfg["retry_feedback"].strip().format(
        post=(last_try.get("text") or "").strip().replace("\n", " "), problem=problem,
        detail=(last_try.get("reason") or "").strip() or status)


def swap_decorative(item: dict[str, Any], names: list[str]) -> list[dict[str, str]]:
    """
    Replace each clue the judge found only dropped in (by label) with the
    item's next spare clue, in place. Returns the swaps made; none when the
    item has no spare left (A3 mostly, B places with one clean clue).
    """
    swaps = []
    wanted = {n.strip().lower() for n in names}
    for k, clue in enumerate(item["clues"]):
        if clue["clue_label"].strip().lower() in wanted and item.get("spare_clues"):
            new = item["spare_clues"].pop(0)
            item["clues"][k] = new
            swaps.append({"from_qid": clue["clue_qid"], "from_label": clue["clue_label"],
                          "to_qid": new["clue_qid"], "to_label": new["clue_label"]})
    if swaps:
        item["role"] = ";".join(dict.fromkeys(c.get("role") or "other" for c in item["clues"]))
    return swaps


def run_item(
    item: dict[str, Any],
    client: ChatClient,
    judge: Any,
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
    config_hash: str,
    names: dict[str, Any],
    limits: dict[str, tuple[int, int]] | None,
    raw_log: AppendLog,
) -> dict[str, Any]:
    """
    Up to judge.max_tries tries of one item (Task 4). Each try is written,
    checked by the rules (finalise.check, without the keyword list) and, if it
    passes them, by the judge model (judge.py). A try the judge fails because
    a clue is only dropped in swaps that clue for a spare on the next try.
    Every try's raw reply goes to the raw log; the row returned is the last
    try, with the whole try_log, accepted True or False.
    """
    import finalise   # finalise imports this module

    max_tries = int((config.get("judge") or {}).get("max_tries", 3))
    work = copy.deepcopy(item)
    try_log: list[dict[str, Any]] = []
    clue_swaps: list[dict[str, str]] = []
    record: dict[str, Any] = {}
    for n in range(1, max_tries + 1):
        seed = try_seed(item["item_id"], n)
        messages = build_messages(work, config, exemplars)
        if try_log and try_log[-1]["status"] != "kept":
            messages[-1]["content"] += "\n\n" + retry_feedback(try_log[-1], config["prompt"])
        content, metadata, attempts, api_error = None, {}, [], None
        try:
            content, metadata, attempts = call_with_backoff(client, messages, config, seed=seed)
        except ApiError as exc:
            api_error, attempts, metadata = str(exc), exc.attempt_log, {"http_status": exc.status}
        # Rule 1: the raw response is on disk before anything reads it.
        raw_log.write({**build_raw_entry(work, client, config_hash, messages, content,
                                         metadata, attempts, api_error), "try": n, "seed": seed})
        record, outcome = build_record(work, client.model, config_hash, content, metadata, api_error)

        judgement = None
        if outcome != "ok":
            verdict = (outcome, str(record.get("error"))[:200])
        else:
            verdict = finalise.check(work, record["post_text"], names, None, limits)
            if verdict is None and judge is not None:
                judgement = judge.check(work, record["post_text"])
                verdict = tuple(judgement["verdict"]) if judgement["verdict"] else None
        try_log.append({"try": n, "seed": seed, "status": verdict[0] if verdict else "kept",
                        "reason": verdict[1] if verdict else "",
                        "clue_qids": [c["clue_qid"] for c in work["clues"]],
                        "text": record.get("post_text"),
                        # billed cost and tokens as the provider's usage block reports them
                        "writer_usage": (metadata or {}).get("usage"),
                        "judge_usage": (judgement or {}).get("usage")})
        record["judge"] = judgement
        if verdict is None:
            break
        if verdict[0] == "judge_decorative" and n < max_tries:
            decorative = (judgement or {}).get("answer", {}).get("decorative_names") or []
            for swap in swap_decorative(work, decorative):
                clue_swaps.append({**swap, "before_try": n + 1})

    record.update(
        tries=len(try_log),
        try_log=try_log,
        accepted=try_log[-1]["status"] == "kept",
        final_status=try_log[-1]["status"],
        final_reason=try_log[-1]["reason"],
        clue_swaps=clue_swaps,
        clues_used=work["clues"] if clue_swaps else None,
        judge_model=getattr(judge, "model", None),
        swapped_from=item.get("swapped_from"),
    )
    return record


def run_items(
    items: list[dict[str, Any]],
    client: ChatClient,
    config: dict[str, Any],
    exemplars: dict[str, list[str]],
    config_hash: str,
    out_log: AppendLog,
    raw_log: AppendLog,
    tally: Counter[str],
    judge: Any = None,
) -> list[dict[str, Any]]:
    """
    Run each item through run_item and log its row, counting outcomes into
    tally. Returns the rows. KeyboardInterrupt propagates; every finished
    item is already on disk.
    """
    import finalise   # finalise imports this module

    delay_between = float(gen_setting(config["generation"],
                                      "delay_between_calls_seconds"))
    aliases = finalise.load_aliases(Path(config["paths"]["place_inputs"]))
    limits = finalise.word_limits(config.get("finalise"))
    rows = []
    for n, item in enumerate(items, start=1):
        names = aliases.get(item["place_qid"], {"forms": [], "country": [], "regions": []})
        record = run_item(item, client, judge, config, exemplars, config_hash, names, limits, raw_log)
        tally["accepted" if record["accepted"] else "failed"] += 1
        tally[f"try_{record['tries']}" if record["accepted"] else "exhausted"] += 1
        out_log.write(record)
        rows.append({**record, "_item": item})
        steps = " > ".join(t["status"] for t in record["try_log"])
        print(f"[{n}/{len(items)}] {item['condition']:<9} {item['place_name'][:28]:<28} {steps}",
              flush=True)
        if delay_between and n < len(items):
            time.sleep(delay_between)
    return rows


A_SET, B_SET = ("A1", "A3", "NOISE_A1"), ("B", "NOISE_B")


def condition_set(condition: str) -> tuple[str, ...] | None:
    """The conditions a place is swapped in together (Task 4), or None (EXPLICIT)."""
    return A_SET if condition in A_SET else B_SET if condition in B_SET else None


def find_reserve(failed: dict[str, Any], plan: list[dict[str, Any]],
                 used: set[str]) -> str | None:
    """
    The reserve place to swap in for a failed item's place: a reserve of the
    same condition, continent, place type and bin (pbin for the A set, rbin
    for the B set) not used yet. Candidates are taken in a fixed hashed order
    of their QIDs, so a rerun picks the same one.
    """
    bin_key = "pbin" if failed["condition"] in A_SET else "rbin"
    candidates = {i["place_qid"] for i in plan
                  if i.get("pool") == "reserve" and i["condition"] == failed["condition"]
                  and i["place_continent"] == failed["place_continent"]
                  and i["place_type"] == failed["place_type"]
                  and i.get(bin_key) == failed.get(bin_key) and i["place_qid"] not in used}
    ordered = sorted(candidates, key=lambda q: hashlib.sha256(q.encode()).hexdigest())
    return ordered[0] if ordered else None


def plan_swaps(rows: list[dict[str, Any]], plan: list[dict[str, Any]], ran: list[dict[str, Any]],
               swapped: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    For every item that failed all its tries, swap its place out of the
    item's condition set and a matched reserve in (Task 4: an A place leaves
    A1, A3 and NOISE_A1 together, a B place B and NOISE_B). Returns the
    reserve items to run - the set's conditions that ran for the failed
    place, plus the reserve place's EXPLICIT - and the swap records.
    A failed place with no matching reserve left is recorded with
    replacement None.
    """
    by_place = defaultdict(list)
    for i in plan:
        by_place[i["place_qid"]].append(i)
    ran_by = {(i["place_qid"], i["condition"]) for i in ran}
    out_sets = {(s["place_qid"], s["set"]) for s in swapped}
    used = {s["replacement_qid"] for s in swapped if s.get("replacement_qid")}
    used |= {i["place_qid"] for i in plan if i.get("pool") == "plan"}
    todo, swaps = [], []
    for row in rows:
        item = row["_item"]
        cset = condition_set(item["condition"])
        if row["accepted"] or cset is None or (item["place_qid"], cset[0]) in out_sets:
            continue
        out_sets.add((item["place_qid"], cset[0]))
        new = find_reserve(item, plan, used)
        conditions = [c for c in cset if (item["place_qid"], c) in ran_by]
        swaps.append({"place_qid": item["place_qid"], "place_name": item["place_name"],
                      "set": cset[0], "failed_condition": item["condition"],
                      "failed_status": row["final_status"], "replacement_qid": new,
                      "conditions": conditions, "swapped_utc": _utc_now()})
        if new is None:
            continue
        used.add(new)
        for r in by_place[new]:
            if r["condition"] in conditions or (r["condition"] == "EXPLICIT" and r.get("pool") == "reserve"):
                todo.append({**r, "swapped_from": item["place_qid"]})
    return todo, swaps


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
    # --out (a pilot) keeps its raw log next to it, away from the dataset's.
    raw_log_path = out_path.parent / "raw_responses.jsonl" if args.out else Path(paths["raw_response_log"])
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
    judge = None
    if not args.no_judge:
        import judge as judge_module
        judge = judge_module.Judge(config)
    print(f"model      {client.model} at {client.base_url}")
    print(f"judge      {judge.model if judge else 'off'}")
    print(f"structured {client.structured_output}")
    print(f"config     {config_hash}\n")

    out_log = AppendLog(out_path)
    raw_log = AppendLog(raw_log_path)
    swaps_path = out_path.parent / "swaps.jsonl"
    swap_log = AppendLog(swaps_path)
    started = time.time()
    tally: Counter[str] = Counter()
    by_id = {i["item_id"]: i for i in plan}

    try:
        run_items(items, client, config, exemplars, config_hash,
                  out_log, raw_log, tally, judge)
        # Task 4: a place whose item failed every try leaves its condition set
        # and a matched reserve comes in; repeated while swapped-in places fail.
        while not args.no_swap:
            rows =[{**r, "_item": by_id[i]} for i, r in latest_rows(read_jsonl(out_path)).items()
                    if i in by_id and "accepted" in r]
            swapped = read_jsonl(swaps_path) if swaps_path.exists() else []
            todo, swaps = plan_swaps(rows, plan, [r["_item"] for r in rows], swapped)
            for s in swaps:
                swap_log.write(s)
                print(f"swap       {s['place_name']} out of the {s['set']} set "
                      f"({s['failed_condition']}: {s['failed_status']}) -> "
                      f"{s['replacement_qid'] or 'no matching reserve left'}")
            todo = [t for t in todo if t["item_id"] not in {r["item_id"] for r in rows}]
            if not todo:
                break
            run_items(todo, client, config, exemplars, config_hash,
                      out_log, raw_log, tally, judge)
    except KeyboardInterrupt:
        print("\ninterrupted - everything attempted so far is on disk; "
              "re-run to continue", file=sys.stderr)
    finally:
        out_log.close()
        raw_log.close()
        swap_log.close()

    elapsed = time.time() - started
    print(f"\nran {tally['accepted'] + tally['failed']} items in {elapsed / 60:.1f} min")
    for kind in ("try_1", "try_2", "try_3", "exhausted"):
        if tally[kind]:
            print(f"  {kind:<12} {tally[kind]:>5}")
    print(f"\nrows      {out_path}")
    print(f"raw log   {raw_log_path}")
    print(f"swaps     {swaps_path}")
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    sys.exit(main())
