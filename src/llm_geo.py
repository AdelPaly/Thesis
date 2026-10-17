"""Whole-text LLM geolocation (tier 4): one prompt, one answer, resolved
against Wikidata.

The provider, model, sampling parameters and prompt template all come from the
configuration (`llm.*`); `build()` is the factory the pipeline calls, and
instances are memoised per configuration so a run loads one client.
"""
from __future__ import annotations

import hashlib
import logging

from llm_common import (CACHE_DIR, REFUSAL_SENTINEL, PickleCache,  # noqa: F401
                        fingerprint, make_client, parse_json_object)

from wikidata import (search_entities, fetch_entities, get_coords_of,
                      resolve_country_continent)

log = logging.getLogger(__name__)

PROMPT = """You are a geolocation assistant. Given a text, identify the single most \
relevant real-world location the text refers to.

Respond with JSON only, no explanation:
{{"city": "<city name or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "refused": false}}

Set "refused" to true ONLY if the text contains no geographic signal at all.
Do not guess if you have no confidence — refuse instead.

Text: {text}"""

PROMPT_V2 = """Read the post below and work out where in the world it is about.

Post:
\"\"\"
{text}
\"\"\"

Reply with one JSON object and nothing else, with exactly these keys:
- "city": the city, or null if you cannot tell
- "country": the country, or null if you cannot tell
- "continent": the continent, or null if you cannot tell
- "refused": true only if the post gives no geographic clue at all, otherwise false

If you are not confident, refuse rather than guess."""

PROMPT_V3 = """Task: geolocate a text.

<text>
{text}
</text>

Question: which single real-world location does this text refer to?

Output: JSON only, in this form:
{{"city": string or null, "country": string or null, "continent": string or null, "refused": boolean}}

"refused" is true only when the text contains no geographic signal. When unsure, refuse instead of guessing."""

# Three wordings and layouts of one question, with one answer format, so a
# difference between templates is prompt sensitivity and nothing else.
PROMPTS = {"v1": PROMPT, "v2": PROMPT_V2, "v3": PROMPT_V3}


# Models are asked for JSON null when a field is unknown, and mostly comply --
# but a few responses put the *string* "null" (or "none", "unknown") in the
# field instead. A truthy string sails through `if not name: continue`, so the
# old code searched Wikidata for the literal word "null" and, for continent,
# could emit a prediction whose continent is "null": never correct, yet counted
# as a prediction, which suppresses the no-result rate. Measured across the
# three response caches (1646 responses): 5 city, 3 country, 1 continent.
_PLACEHOLDER_NAMES = {"", "null", "none", "nil", "n/a", "na", "unknown",
                      "not specified", "unspecified", "no city", "no country"}


def _clean_name(value) -> str | None:
    """A place name, or None for anything that is not one."""
    if not isinstance(value, str):
        return None
    name = value.strip()
    return None if name.lower() in _PLACEHOLDER_NAMES else name


class LLMGeoPipeline:
    """One provider client, one prompt template, one response cache.

    `client` is a provider client from llm_common.make_client: it supplies
    `call(prompt)`, `model_id` and `generation_params` (anything that changes
    the model's answers, which goes into the cache fingerprint)."""

    def __init__(self, client, cache_name: str, prompt_template: str = "v1"):
        self.client = client
        self.model_id = client.model_id
        self.generation_params = client.generation_params
        self.model_name = cache_name
        self.prompt_template = prompt_template
        self.prompt = PROMPTS[prompt_template]
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

    def _parse_json(self, raw: str) -> dict | None:
        return parse_json_object(raw)

    def _resolve_to_candidate(self, parsed: dict) -> dict | None:
        """Cascading QID resolution: city -> country -> continent string."""
        for level in ("city", "country"):
            name = _clean_name(parsed.get(level))
            if not name:
                continue
            qids = search_entities(name, limit=1)
            if not qids:
                continue
            ents = fetch_entities(qids)
            ent = ents.get(qids[0], {})
            coords = get_coords_of(ent)
            if not coords:
                continue
            geo = resolve_country_continent({qids[0]: ent})
            country, continent = geo.get(qids[0], ("", ""))
            return {
                "entity": name,
                "entity_type": "LOC",
                "lat": coords[0], "lon": coords[1],
                "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
                "qid": qids[0],
                "country": country, "continent": continent,
            }

        continent = _clean_name(parsed.get("continent"))
        if continent:
            # Continent-only: the model named a continent but nothing that
            # resolves to a gazetteer entry. There is no coordinate here, so
            # lat/lon stay None. Returning 0.0/0.0 would put the prediction on
            # Null Island and hand the scorer a ~6000 km "error distance" for
            # an item that produced no coordinate at all.
            return {
                "entity": continent,
                "entity_type": "LOC",
                "lat": None, "lon": None,
                "description": continent,
                "qid": None,
                "country": None, "continent": continent,
            }

        return None

    def geolocate_verbose(self, text: str) -> tuple[dict | None, str]:
        """Return (candidate_or_None, raw_response_text).

        The raw response is what the model actually said, before parsing --
        the runner stores it in `llm_raw_response` so a parsing change can be
        re-scored without another API call."""
        raw = self._get_response(text)
        parsed = self._parse_json(raw)
        if parsed is None or parsed.get("refused"):
            return None, raw
        return self._resolve_to_candidate(parsed), raw


_instances: dict[tuple, LLMGeoPipeline] = {}


def build(llm) -> LLMGeoPipeline:
    """The pipeline for an LLMConfig, memoised on everything that defines it."""
    key = (llm.provider, llm.model, llm.revision, llm.temperature, llm.top_p,
           llm.prompt_template, llm.cache_name)
    if key not in _instances:
        _instances[key] = LLMGeoPipeline(make_client(llm), llm.cache_name,
                                         llm.prompt_template)
    return _instances[key]


def clear_instances() -> None:
    """Forget memoised pipelines (tests, multi-config in-process runs)."""
    _instances.clear()
