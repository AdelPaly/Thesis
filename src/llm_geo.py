import hashlib
import json
import os
import pickle
import re
from abc import ABC, abstractmethod
from wikidata import _search_entities, _fetch_entities, _get_coords, _resolve_country_continent

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJ = re.compile(r"\{[^{}]*\"refused\"[^{}]*\}", re.DOTALL)

CACHE_DIR = os.path.join(os.path.dirname(__file__), ".llm_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

PROMPT = """You are a geolocation assistant. Given a text, identify the single most \
relevant real-world location the text refers to.

Respond with JSON only, no explanation:
{{"city": "<city name or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "refused": false}}

Set "refused" to true ONLY if the text contains no geographic signal at all.
Do not guess if you have no confidence — refuse instead.

Text: {text}"""


class LLMGeoPipeline(ABC):
    """Base class — subclasses only implement _call_llm(prompt) -> str."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._cache = self._load_cache()

    def _cache_path(self):
        return os.path.join(CACHE_DIR, f"{self.model_name}.pkl")

    def _load_cache(self):
        path = self._cache_path()
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        tmp = self._cache_path() + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self._cache, f)
        os.replace(tmp, self._cache_path())

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Return raw text response from the LLM."""

    def _get_response(self, text):
        key = hashlib.sha256(text.encode()).hexdigest()
        if key not in self._cache:
            prompt = PROMPT.format(text=text)
            self._cache[key] = self._call_llm(prompt)
        return self._cache[key]

    def _parse_json(self, raw):
        for candidate in (
            raw.strip(),
            *(m.group(1) for m in _JSON_FENCE.finditer(raw)),
            *(m.group(0) for m in _JSON_OBJ.finditer(raw)),
        ):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _resolve_to_candidate(self, parsed):
        """Cascading QID resolution: city -> country -> continent string."""
        for level in ("city", "country"):
            name = parsed.get(level)
            if not name:
                continue
            qids = _search_entities(name, limit=1)
            if not qids:
                continue
            ents = _fetch_entities(qids)
            ent = ents.get(qids[0], {})
            coords = _get_coords(ent)
            if not coords:
                continue
            geo = _resolve_country_continent({qids[0]: ent})
            country, continent = geo.get(qids[0], ("", ""))
            return {
                "entity": name,
                "entity_type": "LOC",
                "lat": coords[0], "lon": coords[1],
                "description": ent.get("descriptions", {}).get("en", {}).get("value", ""),
                "qid": qids[0],
                "country": country, "continent": continent,
            }

        if parsed.get("continent"):
            return {
                "entity": parsed["continent"],
                "entity_type": "LOC",
                "lat": 0.0, "lon": 0.0,
                "description": parsed["continent"],
                "qid": "",
                "country": "", "continent": parsed["continent"],
            }

        return None

    def geolocate(self, text):
        raw = self._get_response(text)
        parsed = self._parse_json(raw)
        if parsed is None or parsed.get("refused"):
            return None
        return self._resolve_to_candidate(parsed)
