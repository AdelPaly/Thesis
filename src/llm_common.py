"""Shared plumbing for the LLM providers, the response caches and response
parsing.

The provider is configuration (`llm.provider`); `make_client` dispatches on it.
Cache file names and entry keys are owned by the callers (llm_geo, llm_grounded);
this module adds the load/save mechanics, a fingerprint entry that notices
prompt or generation-parameter changes, and the JSON parser.
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

import meter

log = logging.getLogger(__name__)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Real keys are sha256 hex digests, so this can never collide with one.
META_KEY = "__meta__"

# What a provider returns when the model blocked or emptied its answer. It
# parses as a refusal and is never written to the cache.
REFUSAL_SENTINEL = '{"refused": true}'
# What a provider returns for an empty or blocked reply (revision 3). It still
# carries "refused": true, so rules v1-v3 and the grounded selector read it as
# a refusal, as they did when the providers returned REFUSAL_SENTINEL; rule v4
# (src/resolver.py) reads `empty_reply` and records the status `empty`. Until
# revision 3 the providers returned REFUSAL_SENTINEL itself, so in records
# written before it that exact string is an empty reply too. Neither is cached.
EMPTY_REPLY_SENTINEL = '{"refused": true, "empty_reply": true}'
SENTINELS = (REFUSAL_SENTINEL, EMPTY_REPLY_SENTINEL)

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
        """Raw text for `prompt`; EMPTY_REPLY_SENTINEL if the model blocked or
        returned no text. Transient API errors are retried with backoff."""
        cfg = self._genai.types.GenerationConfig(**self._sampling)
        delay = self.INITIAL_DELAY_S
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            try:
                response = self.model.generate_content(prompt, generation_config=cfg)
                meter.add("llm_calls")
                usage = getattr(response, "usage_metadata", None)
                meter.add("llm_prompt_tokens", getattr(usage, "prompt_token_count", 0) or 0)
                meter.add("llm_output_tokens", getattr(usage, "candidates_token_count", 0) or 0)
                return response.text
            except ValueError as e:
                # `.text` raises when the candidate was blocked or has no parts.
                log.warning("the model returned no usable text (%s); recording an empty reply", e)
                return EMPTY_REPLY_SENTINEL
            except self._transient as e:
                if attempt == self.MAX_ATTEMPTS:
                    raise
                log.warning("retry %d/%d after %s", attempt, self.MAX_ATTEMPTS - 1,
                            type(e).__name__)
                time.sleep(delay)
                delay *= 2


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

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
    actually served the request, the generation id and token usage; the
    providers below read the usage for the cost meter.

    The API key comes from the environment variable named by `key_env`. This
    module does not load .env files; entry points do."""

    def __init__(self, model: str,
                 timeout: float = DEFAULT_TIMEOUT_S,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 app_title: str = "GeoProject evaluation",
                 url: str = OPENROUTER_API_URL, key_env: str = "OPENROUTER_API_KEY") -> None:
        import requests

        self._requests = requests
        # Any endpoint that speaks the same chat-completions protocol (the VSE
        # gateway) is reached by giving its url and the name of its key.
        self.url = url
        self.api_key = os.environ.get(key_env)
        self.model = model
        if not self.api_key:
            raise OpenRouterError(f"{key_env} is not set (src/.env)")
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
             top_p: Optional[float] = None,
             max_tokens: Optional[int] = None) -> tuple[str, dict]:
        """Single chat completion. A parameter left as None is not sent.
        Returns (text, meta)."""
        requests = self._requests
        body = {"model": self.model, "messages": messages}
        if top_p is not None:
            body["top_p"] = top_p
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens

        delay = INITIAL_BACKOFF_S
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            wait = None  # server-requested wait (Retry-After), if any
            try:
                resp = self.session.post(self.url, json=body, timeout=self.timeout)
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

    def complete(self, user_message: str, **kwargs: Any) -> tuple[str, dict]:
        """Convenience: one user message; `kwargs` go to `chat`."""
        return self.chat([{"role": "user", "content": user_message}], **kwargs)

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


# An answer is one short JSON object (about 40 tokens). Without a limit a
# model that falls into a loop ("S.S.S.S...") writes until the request times
# out; with it the reply is cut, fails to parse and counts as no prediction.
MAX_OUTPUT_TOKENS = 512


class OpenRouterProvider:
    """The pipeline's view of OpenRouter: one user message in, text out.

    A subclass is another endpoint with the same protocol: it sets the name,
    the address, the environment variable holding the key and, optionally,
    one that overrides the address."""

    NAME = "openrouter"
    API_URL = OPENROUTER_API_URL
    URL_ENV: Optional[str] = None
    KEY_ENV = "OPENROUTER_API_KEY"

    def __init__(self, llm):
        from config import ConfigError

        if llm.revision is not None:
            raise ConfigError(f"provider {self.NAME!r} has no model revision to pin")
        url = (os.environ.get(self.URL_ENV) if self.URL_ENV else None) or self.API_URL
        try:
            self.client = OpenRouterClient(model=llm.model, url=url, key_env=self.KEY_ENV)
        except OpenRouterError as e:
            raise ConfigError(str(e)) from None
        self.model_id = llm.model
        self.generation_params = generation_params(llm)
        self._sampling = _sampling(llm)

    def call(self, prompt: str) -> str:
        """Raw text for `prompt`; EMPTY_REPLY_SENTINEL for an empty answer."""
        text, meta = self.client.complete(prompt, max_tokens=MAX_OUTPUT_TOKENS, **self._sampling)
        usage = (meta or {}).get("usage") or {}
        meter.add("llm_calls")
        meter.add("llm_prompt_tokens", usage.get("prompt_tokens") or 0)
        meter.add("llm_output_tokens", usage.get("completion_tokens") or 0)
        meter.add("llm_cost_usd", float(usage.get("cost") or 0))
        return text if text.strip() else EMPTY_REPLY_SENTINEL


VSE_API_URL = "https://litellm.vse.cz/chat/completions"


class VSEProvider(OpenRouterProvider):
    """The VSE gateway (a LiteLLM proxy run by the university): the same
    protocol as OpenRouter, another address and key (VSE_API_KEY). VSE_API_URL
    in the environment overrides the address."""

    NAME = "vse"
    API_URL = VSE_API_URL
    URL_ENV = "VSE_API_URL"
    KEY_ENV = "VSE_API_KEY"


PROVIDERS = {"google": GoogleClient, "openrouter": OpenRouterProvider, "vse": VSEProvider}


def make_client(llm):
    """The provider client for an LLMConfig, chosen by `llm.provider`."""
    from config import ConfigError

    cls = PROVIDERS.get(llm.provider)
    if cls is None:
        raise ConfigError(f"unknown llm.provider {llm.provider!r}")
    return cls(llm)
