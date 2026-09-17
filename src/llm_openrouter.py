"""OpenRouter API handler (OpenAI-compatible chat completions endpoint).

Used for programmatic generation of the synthetic datasets
(see scripts/generate_synthetic.py). Not wired into the geolocation
pipeline modes.

Configuration is read from environment, loaded from src/.env (and a
repo-root .env, if present):

  OPENROUTER_API_KEY  — required
  OPENROUTER_MODEL    — required, OpenRouter model slug (e.g. "provider/model-name")

Quick smoke test (makes one short, billed request):
    python src/llm_openrouter.py
"""
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

SRC_DIR = Path(__file__).resolve().parent
load_dotenv(SRC_DIR / ".env")
load_dotenv(SRC_DIR.parent / ".env")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
APP_TITLE = "GeoProject synthetic dataset generation"

# 408/429 and 5xx are worth retrying; other 4xx (bad key, bad model id,
# unsupported parameter) will fail the same way every time.
_RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class OpenRouterError(RuntimeError):
    pass


class OpenRouterClient:
    """Thin wrapper around OpenRouter chat completions with retry/backoff.

    Every call returns (text, meta) where meta records the model that
    actually served the request, the generation id and token usage —
    needed for the generation metadata in DATASETS.md."""

    def __init__(self, model=None, api_key=None, timeout=600, max_retries=5):
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

    def chat(self, messages, temperature=None, max_tokens=None,
             json_mode=False, **extra):
        """Single chat completion. `extra` is merged into the request body
        (e.g. seed=42, top_p=0.9, provider={...})."""
        body = {"model": self.model, "messages": messages, **extra}
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        delay = 2
        for attempt in range(self.max_retries):
            last = attempt == self.max_retries - 1
            try:
                resp = self.session.post(API_URL, json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                if last:
                    raise OpenRouterError(f"request failed: {e}") from e
                self._log_retry(attempt, type(e).__name__)
            else:
                if resp.status_code == 200:
                    data = resp.json()
                    # OpenRouter can return 200 with an error payload when
                    # the upstream provider fails mid-request.
                    if "error" in data or not data.get("choices"):
                        if last:
                            raise OpenRouterError(f"provider error: {data.get('error', data)}")
                        self._log_retry(attempt, "provider error")
                    else:
                        return self._unpack(data)
                elif resp.status_code in _RETRY_STATUS and not last:
                    self._log_retry(attempt, f"HTTP {resp.status_code}")
                else:
                    raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:500]}")
            time.sleep(delay)
            delay *= 2

    def complete(self, user_message, system_prompt=None, **kwargs):
        """Convenience: optional system prompt + one user message."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return self.chat(messages, **kwargs)

    def complete_json(self, user_message, system_prompt=None, **kwargs):
        """Like complete(), but parses the reply as JSON. Returns (obj, text, meta)."""
        text, meta = self.complete(user_message, system_prompt, **kwargs)
        return parse_json_reply(text), text, meta

    def _log_retry(self, attempt, reason):
        print(f" [retry {attempt + 1}/{self.max_retries - 1} after {reason}]",
              end="", flush=True)

    @staticmethod
    def _unpack(data):
        choice = data["choices"][0]
        meta = {
            "id": data.get("id"),
            "model": data.get("model"),
            "provider": data.get("provider"),
            "finish_reason": choice.get("finish_reason"),
            "usage": data.get("usage"),
        }
        return choice["message"].get("content") or "", meta


def parse_json_reply(text):
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
    client = OpenRouterClient()
    # Generous max_tokens: reasoning models spend tokens thinking before replying.
    text, meta = client.complete("Reply with the single word: pong", max_tokens=1000)
    print(f"model={client.model}\nreply={text!r}\nmeta={json.dumps(meta, indent=2)}")
