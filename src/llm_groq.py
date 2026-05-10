"""Groq API handler. Two pipeline instances are exposed, one per
production model:

  geolocate_groq_oss   -> openai/gpt-oss-120b
  geolocate_groq_llama -> meta-llama/llama-4-scout-17b-16e-instruct

Both share the same SYSTEM_PROMPT (edit below). The per-call task spec
lives in llm_geo.PROMPT and is sent as the user message — keep this
system slot for behaviour shaping (format strictness, abstention policy).

GROQ_API_KEY is read from environment (loaded from src/.env by the
worker entrypoints).
"""
import os
import time

from groq import (Groq, APIConnectionError, APIStatusError,
                  InternalServerError, RateLimitError)

from llm_geo import LLMGeoPipeline

# ============================================================
# SYSTEM PROMPT — edit freely. Sent as the `system` role on every call.
# ============================================================
SYSTEM_PROMPT = """You are a precise geolocation assistant. Always respond with valid JSON exactly matching the schema in the user's message — no commentary, no markdown fences, no extra fields."""
# ============================================================

MODEL_OSS = "openai/gpt-oss-120b"
MODEL_LLAMA = "meta-llama/llama-4-scout-17b-16e-instruct"

_TRANSIENT = (APIConnectionError, RateLimitError, InternalServerError)


class GroqPipeline(LLMGeoPipeline):
    """LLMGeoPipeline backed by a Groq-hosted chat model. cache_name is
    per-model so swapping models doesn't reuse another model's answers."""

    def __init__(self, model_id, cache_name):
        super().__init__(cache_name)
        self.model_id = model_id
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"])

    def send_message(self, user_message):
        """Single chat completion: SYSTEM_PROMPT + user_message.
        Returns the assistant's text reply with retry/backoff on
        transient failures."""
        delay = 2
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    temperature=0,
                )
                return resp.choices[0].message.content
            except _TRANSIENT as e:
                if attempt == 4:
                    raise
                print(f" [retry {attempt + 1}/4 after {type(e).__name__}]",
                      end="", flush=True)
                time.sleep(delay)
                delay *= 2

    def _call_llm(self, prompt):
        return self.send_message(prompt)


_oss = GroqPipeline(MODEL_OSS, "groq_gpt_oss_120b")
_llama = GroqPipeline(MODEL_LLAMA, "groq_llama4_scout_17b")

geolocate_groq_oss = _oss.geolocate
save_cache_oss = _oss.save_cache

geolocate_groq_llama = _llama.geolocate
save_cache_llama = _llama.save_cache
