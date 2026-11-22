"""Whole-text LLM geolocation (tier 4): one prompt, one answer, resolved
against Wikidata.

The provider, model, sampling parameters and prompt template all come from the
configuration (`llm.*`); `build()` is the factory the pipeline calls, and
instances are memoised per configuration so a run loads one client.
"""
from __future__ import annotations

import hashlib

import meter

from llm_common import REFUSAL_SENTINEL, PickleCache, fingerprint, make_client

from resolver import RESOLVERS, clean_name

PROMPT = """You are a geolocation assistant. Given a text, identify the single most \
relevant real-world location the text refers to.

Respond with JSON only, no explanation:
{{"city": "<city name or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "refused": false}}

Set "refused" to true ONLY if the text contains no geographic signal at all.
Do not guess if you have no confidence — refuse instead.

Text: {text}"""

# v2 is v1 plus the coordinates, which the v2 resolver uses to choose between
# places of the same name (src/resolver.py).
PROMPT_V2 = """You are a geolocation assistant. Given a text, identify the single most relevant real-world location the text refers to.

Respond with JSON only, no explanation:
{{"city": "<city name or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "latitude": <approximate latitude of the location in decimal degrees, or null if unknown>, "longitude": <approximate longitude in decimal degrees, or null if unknown>, "refused": false}}

Set "refused" to true ONLY if the text contains no geographic signal at all.
Do not guess if you have no confidence — refuse instead.

Text: {text}"""

# `llm.prompt_template` names the prompt and, with it, the rule that resolves
# the answer: v1 takes the first search result for the city, v2 chooses by
# country and coordinates.
PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2}

_clean_name = clean_name          # the name other modules import


class LLMGeoPipeline:
    """One provider client, one prompt template, one response cache.

    `client` is a provider client from llm_common.make_client: it supplies
    `call(prompt)`, `model_id` and `generation_params` (anything that changes
    the model's answers, which goes into the cache fingerprint)."""

    def __init__(self, client, cache_name: str, prompt_template: str = "v1"):
        self.client = client
        self.model_id = client.model_id
        self.generation_params = client.generation_params
        self.prompt = PROMPTS[prompt_template]
        self.resolve = RESOLVERS[prompt_template]
        self._cache = PickleCache(cache_name, fingerprint(self.prompt, self.generation_params))

    def save_cache(self) -> None:
        self._cache.save()

    def call_llm(self, prompt: str) -> str:
        """Return raw text response from the LLM."""
        return self.client.call(prompt)

    def _get_response(self, text: str) -> str:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached:
            meter.add("llm_cache_hits")
            return cached
        prompt = self.prompt.format(text=text)
        try:
            raw = self.call_llm(prompt)
        except BaseException:
            # Keep what this run already paid for before the error propagates.
            self.save_cache()
            raise
        if raw != REFUSAL_SENTINEL:
            self._cache.put(key, raw)
        return raw

    def geolocate_verbose(self, text: str) -> tuple[dict | None, str, dict]:
        """Return (candidate_or_None, raw_response_text, resolution).

        The raw response is what the model actually said, before parsing --
        the runner stores it in `llm_raw_response` so a parsing change can be
        re-scored without another API call. `resolution` says how the answer
        became a Wikidata item, or why it did not (src/resolver.py)."""
        raw = self._get_response(text)
        cand, resolution = self.resolve(raw)
        return cand, raw, resolution


_instances: dict[tuple, LLMGeoPipeline] = {}


def build(llm) -> LLMGeoPipeline:
    """The pipeline for an LLMConfig, memoised on everything that defines it."""
    key = (llm.provider, llm.model, llm.revision, llm.temperature, llm.top_p,
           llm.prompt_template, llm.cache_name)
    if key not in _instances:
        _instances[key] = LLMGeoPipeline(make_client(llm), llm.cache_name,
                                         llm.prompt_template)
    return _instances[key]
