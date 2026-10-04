"""OpenRouter API handler (OpenAI-compatible chat completions endpoint).

Used for programmatic generation of the synthetic datasets
(see scripts/generate_synthetic.py). Not wired into the geolocation
pipeline modes.

Configuration is read from the environment. This module does not load .env
files itself; entry points do (scripts/generate_synthetic.py, and the
__main__ smoke test below, which reads src/.env and a repo-root .env):

  OPENROUTER_API_KEY  — required
  OPENROUTER_MODEL    — required, OpenRouter model slug (e.g. "provider/model-name")

Quick smoke test (makes one short, billed request):
    python src/llm_openrouter.py
"""
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

API_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_TITLE = "GeoProject synthetic dataset generation"

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
    """Thin wrapper around OpenRouter chat completions with retry/backoff.

    Every call returns (text, meta) where meta records the model that
    actually served the request, the generation id and token usage —
    needed for the generation metadata in DATASETS.md."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT_S,
                 max_retries: int = DEFAULT_MAX_RETRIES) -> None:
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
            "X-Title": APP_TITLE,
        })

    def chat(self, messages: list[dict[str, str]],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None,
             json_mode: bool = False, **extra: Any) -> tuple[str, dict]:
        """Single chat completion. `extra` is merged into the request body
        (e.g. seed=42, top_p=0.9, provider={...}). Returns (text, meta)."""
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
                resp = self.session.post(API_URL, json=body, timeout=self.timeout)
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
        logger.warning("retry %d/%d after %s", attempt + 1, self.max_retries - 1, reason)

    @staticmethod
    def _retry_after(resp: requests.Response) -> Optional[float]:
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


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    src_dir = Path(__file__).resolve().parent
    load_dotenv(src_dir / ".env")
    load_dotenv(src_dir.parent / ".env")
    client = OpenRouterClient()
    # Generous max_tokens: reasoning models spend tokens thinking before replying.
    text, meta = client.complete("Reply with the single word: pong", max_tokens=1000)
    print(f"model={client.model}\nreply={text!r}\nmeta={json.dumps(meta, indent=2)}")
