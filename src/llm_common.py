"""Shared plumbing for the LLM providers, the response caches and response
parsing.

The provider is configuration (`llm.provider`); `make_client` dispatches on it.
Cache file names and entry keys are owned by the callers (llm_geo, llm_grounded);
this module adds the load/save mechanics, a fingerprint entry that notices
prompt or generation-parameter changes, and the JSON parsers.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
import re
import time
from typing import Any, Optional

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Real keys are sha256 hex digests, so this can never collide with one.
META_KEY = "__meta__"

# What a provider returns when the model blocked or emptied its answer. It
# parses as a refusal and is never written to the cache.
REFUSAL_SENTINEL = '{"refused": true}'

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{[^{}]*\"refused\"[^{}]*\}", re.DOTALL)


def parse_json_object(raw: str) -> dict | None:
    """The JSON object in a model response, or None.

    Tries the whole string, then fenced blocks, then a bare `{... "refused" ...}`
    object buried in chatter. Valid JSON that is not an object (a list, a
    number, a string) is rejected."""
    if not isinstance(raw, str):
        return None
    for candidate in (
        raw.strip(),
        *(m.group(1) for m in _JSON_FENCE.finditer(raw)),
        *(m.group(0) for m in _JSON_OBJ.finditer(raw)),
    ):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def fingerprint(prompt: str, params: dict[str, Any]) -> str:
    """Hash of the prompt template plus generation parameters."""
    payload = json.dumps({"prompt": prompt, "params": params},
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


class PickleCache:
    """A dict of responses persisted as `<name>.pkl` in CACHE_DIR.

    The dict carries a `__meta__` entry holding the fingerprint of the prompt
    and generation parameters that produced it. A cache with no fingerprint
    (written before this existed) is assumed current and stamped; one whose
    fingerprint differs is discarded for this run and a warning is logged.
    """

    def __init__(self, name: str, meta: str):
        self.name = name
        self.meta = meta
        self.path = os.path.join(CACHE_DIR, f"{name}.pkl")
        self.data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, dict):
                    data = loaded
                else:
                    log.warning("LLM cache %s is not a dict; starting empty", self.path)
            except Exception as e:
                log.warning("LLM cache %s unreadable (%s: %s); starting empty",
                            self.path, type(e).__name__, e)
        stored = data.get(META_KEY)
        if stored is not None and stored != self.meta:
            log.warning("LLM cache %r was built with a different prompt or generation "
                        "settings; ignoring its %d entries (kept as %s.stale)",
                        self.name, len(data) - 1, os.path.basename(self.path))
            os.replace(self.path, f"{self.path}.stale")
            data = {}
        data[META_KEY] = self.meta
        return data

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        """Store a response, refusing empty ones and the reserved key."""
        if key == META_KEY or not isinstance(value, str) or not value.strip():
            return
        self.data[key] = value

    def __len__(self) -> int:
        return sum(1 for k in self.data if k != META_KEY)

    def save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.data, f)
        os.replace(tmp, self.path)


# ------------------------------------------------------------------ providers
#
# One module for every provider, chosen by `llm.provider`. A provider client
# exposes `model_id`, `generation_params` (what changes the model's answers,
# which goes into the cache fingerprint) and `call(prompt) -> str`.

def generation_params(llm) -> dict[str, Any]:
    """The cache-fingerprint parameters of an LLMConfig. Fields left as None are
    not sent, so they are not recorded. Integral floats are stored as int, so a
    cache fingerprinted with `temperature: 0` before the sampling fields were
    configuration still matches `temperature: 0.0`."""
    params: dict[str, Any] = {"model": llm.model}
    for key in ("revision", "temperature", "top_p"):
        value = getattr(llm, key)
        if value is None:
            continue
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        params[key] = value
    return params


def _sampling(llm) -> dict[str, float]:
    return {k: v for k, v in (("temperature", llm.temperature),
                              ("top_p", llm.top_p)) if v is not None}


class GoogleClient:
    """Google AI Studio through `google.generativeai`.

    Note: `google.generativeai` is deprecated upstream in favour of
    `google.genai`; the SDK is deliberately not migrated here."""

    MAX_ATTEMPTS = 5
    INITIAL_DELAY_S = 2

    def __init__(self, llm):
        import google.generativeai as genai
        from google.api_core import exceptions as gax

        from config import ConfigError

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigError("GOOGLE_API_KEY is not set")
        if llm.revision is not None:
            raise ConfigError("provider 'google' has no model revision to pin")
        self.model_id = llm.model
        self.generation_params = generation_params(llm)
        self._genai = genai
        self._transient = (gax.InternalServerError, gax.ServiceUnavailable,
                           gax.ResourceExhausted, gax.DeadlineExceeded)
        self._sampling = _sampling(llm)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(llm.model)

    def call(self, prompt: str) -> str:
        """Raw text for `prompt`; a refusal sentinel if the model blocked or
        returned no text. Transient API errors are retried with backoff."""
        cfg = self._genai.types.GenerationConfig(**self._sampling)
        delay = self.INITIAL_DELAY_S
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                return self.model.generate_content(prompt, generation_config=cfg).text
            except ValueError as e:
                # `.text` raises when the candidate was blocked or has no parts.
                log.warning("Gemma returned no usable text (%s); recording a refusal", e)
                return REFUSAL_SENTINEL
            except self._transient as e:
                if attempt == self.MAX_ATTEMPTS:
                    raise
                log.warning("retry %d/%d after %s", attempt, self.MAX_ATTEMPTS - 1,
                            type(e).__name__)
                time.sleep(delay)
                delay *= 2


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_APP_TITLE = "GeoProject synthetic dataset generation"

DEFAULT_TIMEOUT_S = 600
DEFAULT_MAX_RETRIES = 5
INITIAL_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0
MAX_RETRY_AFTER_S = 120.0
ERROR_BODY_EXCERPT_CHARS = 200

# 408/429 and 5xx are worth retrying; other 4xx (bad key, bad model id,
# unsupported parameter) will fail the same way every time.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class OpenRouterError(RuntimeError):
    """Raised when a request fails permanently or retries are exhausted."""


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completions (the OpenAI-compatible
    endpoint) with retry/backoff.

    Every call returns (text, meta) where meta records the model that
    actually served the request, the generation id and token usage —
    needed for the generation metadata in DATASETS.md.

    The API key and, when `model` is not given, the model slug come from the
    environment (OPENROUTER_API_KEY, OPENROUTER_MODEL). This module does not
    load .env files; entry points do."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT_S,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 app_title: str = OPENROUTER_APP_TITLE) -> None:
        import requests

        self._requests = requests
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        self.model = model or os.environ.get("OPENROUTER_MODEL")
        if not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is not set (src/.env)")
        if not self.model:
            raise OpenRouterError("OPENROUTER_MODEL is not set (src/.env)")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": app_title,
        })

    def chat(self, messages: list[dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             json_mode: bool = False, **extra: Any) -> tuple[str, dict]:
        """Single chat completion. `extra` is merged into the request body
        (e.g. seed=42, top_p=0.9, provider={...}). Returns (text, meta)."""
        requests = self._requests
        body = {"model": self.model, "messages": messages, **extra}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        delay = INITIAL_BACKOFF_S
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            wait = None  # server-requested wait (Retry-After), if any
            try:
                resp = self.session.post(OPENROUTER_API_URL, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if last:
                    raise OpenRouterError(f"request failed: {e}") from e
                self._log_retry(attempt, type(e).__name__)
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        data = None
                    # OpenRouter can return 200 with an error payload when
                    # the upstream provider fails mid-request; a proxy can
                    # return 200 with a non-JSON body.
                    if not isinstance(data, dict) or "error" in data or not data.get("choices"):
                        if last:
                            detail = (data.get("error", data) if isinstance(data, dict)
                                      else resp.text)
                            raise OpenRouterError(
                                f"provider error: {str(detail)[:ERROR_BODY_EXCERPT_CHARS]}")
                        self._log_retry(attempt, "provider error")
                    else:
                        return self._unpack(data)
                elif resp.status_code in _RETRY_STATUS and not last:
                    wait = self._retry_after(resp)
                    self._log_retry(attempt, f"HTTP {resp.status_code}")
                else:
                    raise OpenRouterError(
                        f"HTTP {resp.status_code}: {resp.text[:ERROR_BODY_EXCERPT_CHARS]}")
            time.sleep(wait if wait is not None else random.uniform(delay / 2, delay))
            delay = min(delay * 2, MAX_BACKOFF_S)
        raise OpenRouterError("retries exhausted")

    def complete(self, user_message: str, system_prompt: Optional[str] = None,
                 **kwargs: Any) -> tuple[str, dict]:
        """Convenience: optional system prompt + one user message."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages, **kwargs)

    def _log_retry(self, attempt: int, reason: str) -> None:
        log.warning("retry %d/%d after %s", attempt + 1, self.max_retries - 1, reason)

    @staticmethod
    def _retry_after(resp) -> Optional[float]:
        """Seconds from a numeric Retry-After header, capped; None if absent/unparseable."""
        try:
            return min(max(float(resp.headers["Retry-After"]), 0.0), MAX_RETRY_AFTER_S)
        except (KeyError, ValueError):
            return None

    @staticmethod
    def _unpack(data: dict) -> tuple[str, dict]:
        choice = data["choices"][0]
        meta = {
            "id": data.get("id"),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }
        return choice["message"].get("content") or "", meta


def parse_json_reply(text: str) -> Any:
    """Parse a model reply as JSON, tolerating ```json fences and
    leading/trailing chatter around the outermost object/array."""
    cleaned = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1]
    if starts:
        start = min(starts)
        end = cleaned.rfind("}" if cleaned[start] == "{" else "]")
        if end > start:
            return json.loads(cleaned[start:end + 1])
    raise ValueError(f"reply is not valid JSON: {text[:300]!r}")


class OpenRouterProvider:
    """The pipeline's view of OpenRouter: one user message in, text out."""

    def __init__(self, llm):
        from config import ConfigError

        if llm.revision is not None:
            raise ConfigError("provider 'openrouter' has no model revision to pin")
        try:
            self.client = OpenRouterClient(model=llm.model,
                                           app_title="GeoProject evaluation")
        except OpenRouterError as e:
            raise ConfigError(str(e)) from None
        self.model_id = llm.model
        self.generation_params = generation_params(llm)
        self._sampling = _sampling(llm)

    def call(self, prompt: str) -> str:
        """Raw text for `prompt`; a refusal sentinel for an empty answer."""
        text, _ = self.client.complete(prompt, **self._sampling)
        return text if text.strip() else REFUSAL_SENTINEL


class LocalHFClient:
    """A Hugging Face causal LM run on this machine with transformers.

    The weights are pinned: `llm.revision` must be a commit hash of the model
    repository, so the row stays reproducible however the hosted model moves.
    Decoding is greedy (temperature 0, no top_p), the only setting that is
    deterministic here. Inference runs on the CPU in float32, because this
    machine has no GPU and torch's CPU bfloat16 matmul is slower than float32
    on it; dtype and the new-token cap change the answers, so both are part of
    the cache fingerprint."""

    DTYPE = "float32"
    MAX_NEW_TOKENS = 128

    def __init__(self, llm):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from config import ConfigError

        if not llm.revision:
            raise ConfigError("provider 'local_hf' requires llm.revision (a commit hash)")
        if llm.temperature not in (0, 0.0) or llm.top_p not in (None, 1, 1.0):
            raise ConfigError("provider 'local_hf' decodes greedily: set temperature 0 "
                              "and top_p null or 1.0")
        self._torch = torch
        self.model_id = llm.model
        self.generation_params = {**generation_params(llm), "dtype": self.DTYPE,
                                  "max_new_tokens": self.MAX_NEW_TOKENS}
        self.tokenizer = AutoTokenizer.from_pretrained(llm.model, revision=llm.revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm.model, revision=llm.revision, dtype=getattr(torch, self.DTYPE))
        self.model.eval()

    def call(self, prompt: str) -> str:
        """Raw text for `prompt`; a refusal sentinel for an empty answer."""
        ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            return_tensors="pt")
        with self._torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.MAX_NEW_TOKENS,
                                      do_sample=False, temperature=None, top_p=None,
                                      top_k=None,
                                      pad_token_id=self.tokenizer.eos_token_id)
        text = self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        return text if text.strip() else REFUSAL_SENTINEL


PROVIDERS = {"google": GoogleClient, "openrouter": OpenRouterProvider,
             "local_hf": LocalHFClient}


def make_client(llm):
    """The provider client for an LLMConfig, chosen by `llm.provider`."""
    from config import ConfigError

    cls = PROVIDERS.get(llm.provider)
    if cls is None:
        raise ConfigError(f"unknown llm.provider {llm.provider!r}")
    return cls(llm)
