"""Google Gemma handler.

The model id and the cache name come from the configuration (`llm.model`,
`llm.cache_name`); nothing here names a model. `build()` is the factory the
pipeline calls, and instances are memoised per (model, cache) pair so a run
loads one client.

Note: `google.generativeai` is deprecated upstream in favour of `google.genai`;
the SDK is deliberately not migrated here.
"""
from __future__ import annotations

import logging
import os
import time

import google.generativeai as genai
from google.api_core import exceptions as gax

from config import ConfigError
from llm_common import REFUSAL_SENTINEL
from llm_geo import LLMGeoPipeline

log = logging.getLogger(__name__)

_TRANSIENT = (gax.InternalServerError, gax.ServiceUnavailable,
              gax.ResourceExhausted, gax.DeadlineExceeded)

_MAX_ATTEMPTS = 5
_INITIAL_DELAY_S = 2
_TEMPERATURE = 0

_instances: dict[tuple[str, str], GemmaPipeline] = {}


class GemmaPipeline(LLMGeoPipeline):
    def __init__(self, model_id: str, cache_name: str):
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigError("GOOGLE_API_KEY is not set")
        self.model_id = model_id
        self.generation_params = {"model": model_id, "temperature": _TEMPERATURE}
        super().__init__(cache_name)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_id)

    def call_llm(self, prompt: str) -> str:
        """Raw text for `prompt`; a refusal sentinel if the model blocked or
        returned no text. Transient API errors are retried with backoff."""
        cfg = genai.types.GenerationConfig(temperature=_TEMPERATURE)
        delay = _INITIAL_DELAY_S
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return self.model.generate_content(prompt, generation_config=cfg).text
            except ValueError as e:
                # `.text` raises when the candidate was blocked or has no parts.
                log.warning("Gemma returned no usable text (%s); recording a refusal", e)
                return REFUSAL_SENTINEL
            except _TRANSIENT as e:
                if attempt == _MAX_ATTEMPTS:
                    raise
                log.warning("retry %d/%d after %s", attempt, _MAX_ATTEMPTS - 1,
                            type(e).__name__)
                time.sleep(delay)
                delay *= 2


def build(model_id: str, cache_name: str) -> GemmaPipeline:
    key = (model_id, cache_name)
    if key not in _instances:
        _instances[key] = GemmaPipeline(model_id, cache_name)
    return _instances[key]


def clear_instances() -> None:
    """Forget memoised pipelines (tests, multi-config in-process runs)."""
    _instances.clear()
