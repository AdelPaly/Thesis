from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod

from llm_common import (CACHE_DIR, REFUSAL_SENTINEL, PickleCache,  # noqa: F401
                        fingerprint, parse_json_object)

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


class LLMGeoPipeline(ABC):
    """Base class — subclasses only implement call_llm(prompt) -> str and may
    set `generation_params` (anything that changes the model's answers)."""

    generation_params: dict = {}

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._cache = PickleCache(model_name, fingerprint(PROMPT, self.generation_params))

    def save_cache(self) -> None:
        self._cache.save()

    @abstractmethod
    def call_llm(self, prompt: str) -> str:
        """Return raw text response from the LLM."""

    def _get_response(self, text: str) -> str:
        key = hashlib.sha256(text.encode()).hexdigest()
        cached = self._cache.get(key)
        if cached:
            return cached
        prompt = PROMPT.format(text=text)
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
