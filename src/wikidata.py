from __future__ import annotations

import atexit
import logging
import os
import pickle
import threading
import time
from collections.abc import MutableMapping

import requests

logger = logging.getLogger(__name__)

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
# Wikimedia's User-Agent policy asks for a real contact. EDIT THIS to your own
# URL or email before running at scale.
CONTACT = "https://github.com/BrookeYada/geo"
HEADERS = {"User-Agent": f"GeoThesisBot/1.0 (student research; {CONTACT})"}

MIN_REQUEST_INTERVAL = 0.5     # seconds between API calls
REQUEST_TIMEOUT = 15           # seconds
MAX_ATTEMPTS = 6
MAX_BACKOFF = 32               # cap on the exponential wait, seconds
MAX_RETRY_AFTER = 120          # cap on a server-supplied Retry-After, seconds
RETRYABLE_API_ERRORS = ("maxlag", "ratelimited")
BATCH_SIZE = 50                # wbgetentities id limit

SEARCH_LIMIT = 10              # default wbsearchentities limit
MAX_SEARCH_LIMIT = 50
LOC_CAP = 10                   # per-type candidate caps at the default top_n
PER_CAP = 5
ORG_CAP = 10
ENTITY_TYPES = ("LOC", "PER", "ORG")

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".wikidata_cache")


class WikidataError(Exception):
    """A Wikidata API call failed (network, HTTP or API-level error)."""


class Cache(MutableMapping):
    """Lazily loaded dict backed by a pickle file.

    Nothing is read at construction. Saving only rewrites the file if an entry
    was added or changed, and merges in entries another process wrote since we
    loaded, so concurrent runs do not discard each other's work.
    """

    def __init__(self, name: str, filename: str):
        self.name = name
        self.path = os.path.join(_CACHE_DIR, filename)
        self._data: dict | None = None
        self._dirty = False
        self._load_failed = False
        self._stamp: tuple | None = None
        self._lock = threading.RLock()

    def _stat(self) -> tuple | None:
        try:
            st = os.stat(self.path)
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _ensure(self) -> dict:
        if self._data is None:
            with self._lock:
                if self._data is None:
                    self._stamp = self._stat()
                    self._data = {}
                    if self._stamp is not None:
                        try:
                            with open(self.path, "rb") as f:
                                self._data = pickle.load(f)
                        except Exception as e:                  # noqa: BLE001
                            self._load_failed = True
                            logger.warning("Cache load failed for %s: %s", self.path, e)
                    logger.info("Wikidata %s cache loaded: %d entries",
                                self.name, len(self._data))
        return self._data

    def __getitem__(self, key):
        return self._ensure()[key]

    def __setitem__(self, key, value) -> None:
        data = self._ensure()
        if key not in data or data[key] != value:
            data[key] = value
            self._dirty = True

    def __delitem__(self, key) -> None:
        del self._ensure()[key]
        self._dirty = True

    def __iter__(self):
        return iter(self._ensure())

    def __len__(self) -> int:
        return len(self._ensure())

    def __contains__(self, key) -> bool:
        return key in self._ensure()

    @property
    def loaded(self) -> bool:
        return self._data is not None

    def save(self) -> bool:
        """Write to disk if changed. Returns True if the file was rewritten."""
        with self._lock:
            if self._data is None or not self._dirty:
                return False
            if self._load_failed:
                logger.warning("Not saving %s cache: the existing file failed to load "
                               "and would be overwritten", self.name)
                return False
            tmp = f"{self.path}.{os.getpid()}.tmp"
            try:
                os.makedirs(_CACHE_DIR, exist_ok=True)
                if self._stat() != self._stamp:
                    with open(self.path, "rb") as f:
                        disk = pickle.load(f)
                    for k, v in disk.items():
                        if k not in self._data:
                            self._data[k] = v
                with open(tmp, "wb") as f:
                    pickle.dump(self._data, f)
                os.replace(tmp, self.path)
            except Exception as e:                              # noqa: BLE001
                logger.warning("Cache save failed for %s: %s", self.name, e)
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                return False
            self._stamp = self._stat()
            self._dirty = False
            return True


_entity_cache = Cache("entity", "entity.pkl")   # QID -> entity data
_label_cache = Cache("label", "label.pkl")      # QID -> English label
_coords_cache = Cache("coords", "coords.pkl")   # (text, type, top_n) -> candidates
_search_cache = Cache("search", "search.pkl")   # (text, limit) -> [qid]
_alias_cache = Cache("alias", "alias.pkl")      # QID -> [alias]
_CACHES = (_entity_cache, _label_cache, _coords_cache, _search_cache, _alias_cache)


def save_caches() -> None:
    """Write any changed caches to disk; also called at exit, and by long-running
    drivers to checkpoint mid-run."""
    saved = [c for c in _CACHES if c.save()]
    if saved:
        logger.info("Wikidata cache saved: %s",
                    ", ".join(f"{c.name}={len(c)}" for c in _CACHES))


_save_caches = save_caches
atexit.register(save_caches)


class RateLimiter:
    """Thread-safe minimum spacing between calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            delay = self.min_interval - (time.time() - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.time()


_rate_limiter = RateLimiter(MIN_REQUEST_INTERVAL)


def _retry_wait(resp: requests.Response | None, attempt: int) -> float:
    wait = min(2 ** attempt, MAX_BACKOFF)
    if resp is not None:
        try:
            wait = max(wait, min(float(resp.headers.get("Retry-After", 0)), MAX_RETRY_AFTER))
        except ValueError:
            pass
    return wait


def _api_get(params: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
    """Single entry point for all Wikidata API calls.

    Rate limited; retries on 429, 5xx, connection errors, timeouts and
    retryable API errors (honouring Retry-After). Raises WikidataError on
    failure, including a 200 response carrying an {"error": ...} body.
    """
    last_error = "no attempt made"
    for attempt in range(MAX_ATTEMPTS):
        _rate_limiter.wait()
        resp = None
        try:
            resp = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_error = str(e)
        else:
            if resp.status_code == 429 or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code}"
            elif resp.status_code >= 400:
                raise WikidataError(f"HTTP {resp.status_code} from Wikidata")
            else:
                try:
                    data = resp.json()
                except ValueError as e:
                    raise WikidataError(f"Invalid JSON from Wikidata: {e}") from e
                err = data.get("error") if isinstance(data, dict) else None
                if not err:
                    return data
                code = err.get("code", "") if isinstance(err, dict) else str(err)
                last_error = f"API error: {code}"
                if code not in RETRYABLE_API_ERRORS:
                    raise WikidataError(last_error)
        wait = _retry_wait(resp, attempt)
        logger.warning("Wikidata request failed (%s), retrying in %.0fs", last_error, wait)
        time.sleep(wait)
    raise WikidataError(f"Wikidata request failed after {MAX_ATTEMPTS} attempts: {last_error}")


def _search_checked(text: str, limit: int = SEARCH_LIMIT) -> list[str]:
    """wbsearchentities, disk-cached. Raises WikidataError on failure.

    A failed search is never cached, so a transient network error does not
    become a permanent empty answer. Caching successful searches is what makes
    a second run over the same datasets close to free.
    """
    key = (text, limit)
    if key in _search_cache:
        return _search_cache[key]
    data = _api_get({
        "action": "wbsearchentities",
        "search": text,
        "language": "en",
        "limit": limit,
        "format": "json",
    })
    hits = [item["id"] for item in data.get("search", [])]
    _search_cache[key] = hits
    return hits


def search_entities(text: str, limit: int = SEARCH_LIMIT) -> list[str]:
    """Search Wikidata for QIDs matching text; [] on failure (not cached)."""
    try:
        return _search_checked(text, limit)
    except WikidataError as e:
        logger.warning("Search error for %r: %s", text, e)
        return []


def _fetch_entities_status(qids: list[str]) -> tuple[dict[str, dict], bool]:
    """Batch-fetch entity data via the cache. Returns ({QID: entity}, complete),
    where complete is False if any wanted QID could not be fetched."""
    missing = list(dict.fromkeys(q for q in qids if q not in _entity_cache))
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        try:
            data = _api_get({
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "claims|descriptions|sitelinks/urls|labels",
                "languages": "en",
                "format": "json",
            })
        except WikidataError as e:
            logger.warning("Entity fetch error: %s", e)
            continue
        for qid, ent in data.get("entities", {}).items():
            _entity_cache[qid] = ent
            lbl = ent.get("labels", {}).get("en", {}).get("value", "")
            if lbl:
                _label_cache[qid] = lbl
    return ({q: _entity_cache.get(q, {}) for q in qids},
            all(q in _entity_cache for q in qids))


def fetch_entities(qids: list[str]) -> dict[str, dict]:
    """Batch-fetch entity data, using cache. Returns dict QID -> entity data
    ({} for QIDs that could not be fetched; those are not cached)."""
    return _fetch_entities_status(qids)[0]


def _fetch_labels_status(qids: list[str]) -> tuple[dict[str, str], bool]:
    """Batch-fetch English labels via the cache. Returns ({QID: label}, complete)."""
    missing = list(dict.fromkeys(q for q in qids if q not in _label_cache))
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        try:
            data = _api_get({
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": "en",
                "format": "json",
            })
        except WikidataError as e:
            logger.warning("Label fetch error: %s", e)
            continue
        for qid, ent in data.get("entities", {}).items():
            _label_cache[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
    return ({q: _label_cache.get(q, "") for q in qids},
            all(q in _label_cache for q in qids))


def _fetch_labels(qids: list[str]) -> dict[str, str]:
    """Batch-fetch English labels, using cache. Returns dict QID -> label."""
    return _fetch_labels_status(qids)[0]


def fetch_aliases(qids: list[str]) -> dict[str, list[str]]:
    """Batch-fetch English aliases, using a cache. {QID: [alias, ...]}.

    Aliases are a separate call rather than an extra prop on fetch_entities,
    because the entity cache already holds thousands of entries fetched without
    them and widening the props there would silently return alias-less entries
    for everything already cached.
    """
    missing = [q for q in qids if q not in _alias_cache]
    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        try:
            data = _api_get({
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "aliases",
                "languages": "en",
                "format": "json",
            })
        except WikidataError as e:
            logger.warning("Alias fetch error: %s", e)
            continue
        for qid, ent in data.get("entities", {}).items():
            _alias_cache[qid] = [a.get("value", "")
                                 for a in ent.get("aliases", {}).get("en", [])
                                 if a.get("value")]
        for q in batch:
            _alias_cache.setdefault(q, [])
    return {q: _alias_cache.get(q, []) for q in qids}


def _claim_id(entity: dict, prop: str) -> str | None:
    """Get QID of the most appropriate single value for a property claim.

    Skips deprecated-rank claims and end-dated (P582) historical claims.
    Prefers preferred-rank over normal-rank when both exist. Falls back to
    the raw first-in-storage value if filtering removes every claim, so
    entities with only historical values still resolve to something."""
    claims = entity.get("claims", {})
    if prop not in claims:
        return None

    preferred = []
    normal = []
    for claim in claims[prop]:
        if claim.get("rank") == "deprecated":
            continue
        if "P582" in claim.get("qualifiers", {}):
            continue
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not (isinstance(val, dict) and "id" in val):
            continue
        if claim.get("rank") == "preferred":
            preferred.append(val["id"])
        else:
            normal.append(val["id"])

    if preferred:
        return preferred[0]
    if normal:
        return normal[0]

    val = claims[prop][0].get("mainsnak", {}).get("datavalue", {}).get("value")
    if isinstance(val, dict) and "id" in val:
        return val["id"]
    return None


def _claim_ids(entity: dict, prop: str) -> list[str]:
    """Get all QIDs from a property claim."""
    claims = entity.get("claims", {})
    out = []
    for claim in claims.get(prop, []):
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(val, dict) and "id" in val:
            out.append(val["id"])
    return out


def get_coords_of(entity: dict) -> tuple[float, float] | None:
    """Coordinate location (P625) as (lat, lon), or None.

    Rank handling matches SPARQL `wdt:P625` truthy semantics -- preferred rank
    wins, deprecated claims are skipped, normal rank is the fallback. The gold
    coordinates in the datasets were sourced through wdt:P625, so the pipeline
    reading a different claim off the same item would put a systematic offset
    between gold and predicted coordinates on multi-coordinate entities.
    """
    claims = entity.get("claims", {})
    if "P625" not in claims:
        return None
    normal = None
    for claim in claims["P625"]:
        if claim.get("rank") == "deprecated":
            continue
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if not (val and "latitude" in val):
            continue
        if claim.get("rank") == "preferred":
            return val["latitude"], val["longitude"]
        if normal is None:
            normal = (val["latitude"], val["longitude"])
    if normal is not None:
        return normal
    # Every claim was deprecated or malformed: fall back to the raw first
    # value rather than reporting no coordinate at all.
    val = claims["P625"][0].get("mainsnak", {}).get("datavalue", {}).get("value")
    if val and "latitude" in val:
        return val["latitude"], val["longitude"]
    return None


def coords_for_qids(qids) -> dict[str, tuple[float, float] | None]:
    """Coordinates for a batch of QIDs: {qid: (lat, lon) | None}.

    The one coordinate client. The dataset tooling used to reach Wikidata
    through a second transport (a SPARQL endpoint) with a second on-disk cache
    (src/.qid_coords_cache.pkl) for exactly this fact; both are gone. This goes
    through the same rate-limited wbgetentities path and the same
    src/.wikidata_cache the pipeline uses, so a QID fetched by either side is
    cached for both. QIDs that could not be fetched map to None for this call
    and are not cached.
    """
    wanted = sorted({q for q in qids if isinstance(q, str) and q.startswith("Q")})
    if not wanted:
        return {}
    entities = fetch_entities(wanted)
    return {q: get_coords_of(entities.get(q, {})) for q in wanted}


def _resolve_cc(entities_dict: dict[str, dict]) -> tuple[dict[str, tuple[str, str]], bool]:
    """As resolve_country_continent, plus a flag that is False if any
    entity or label lookup failed (the result may then have blank labels)."""
    country_of = {qid: _claim_id(ent, "P17") for qid, ent in entities_dict.items()}
    continent_of = {qid: _claim_id(ent, "P30") for qid, ent in entities_dict.items()}
    complete = True

    # Places without a direct continent inherit their country's continent.
    need_country = {c for qid, c in country_of.items() if c and not continent_of[qid]}
    if need_country:
        country_ents, complete = _fetch_entities_status(list(need_country))
        country_continent = {cq: _claim_id(ent, "P30") for cq, ent in country_ents.items()}
        for qid, c in country_of.items():
            if c and not continent_of[qid]:
                continent_of[qid] = country_continent.get(c)

    label_qids = {q for q in (*country_of.values(), *continent_of.values()) if q}
    labels: dict[str, str] = {}
    if label_qids:
        labels, labels_ok = _fetch_labels_status(list(label_qids))
        complete = complete and labels_ok

    result = {qid: (labels.get(country_of[qid] or "", ""),
                    labels.get(continent_of[qid] or "", ""))
              for qid in entities_dict}
    return result, complete


def resolve_country_continent(entities_dict: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """For a dict of QID->entity, resolve country/continent labels in bulk.
    Returns dict of QID -> (country_label, continent_label)."""
    return _resolve_cc(entities_dict)[0]


def _description(ent: dict, default: str) -> str:
    return ent.get("descriptions", {}).get("en", {}).get("value", default)


def _candidate(place_qid: str, ent: dict, coords: tuple[float, float],
               geo: dict[str, tuple[str, str]], entity_text: str) -> dict:
    country, continent = geo.get(place_qid, ("", ""))
    return {
        "lat": coords[0], "lon": coords[1],
        "description": _description(ent, entity_text),
        "qid": place_qid, "country": country, "continent": continent,
    }


def _loc_candidates(entity_text: str, qids: list[str], cap: int) -> tuple[list[dict], bool]:
    entities, ok = _fetch_entities_status(qids)
    place_ents = {q: e for q, e in entities.items() if get_coords_of(e)}
    geo, geo_ok = _resolve_cc(place_ents)
    ranked = sorted(place_ents, key=lambda q: len(place_ents[q].get("sitelinks", {})),
                    reverse=True)[:cap]
    return ([_candidate(q, place_ents[q], get_coords_of(place_ents[q]), geo, entity_text)
             for q in ranked], ok and geo_ok)


def _per_candidates(entity_text: str, qids: list[str], cap: int) -> tuple[list[dict], bool]:
    entities, ok = _fetch_entities_status(qids)
    person_to_bp = {}
    for qid in qids:
        ent = entities.get(qid, {})
        if "Q5" not in _claim_ids(ent, "P31"):
            continue
        bp = _claim_id(ent, "P19")
        if bp:
            person_to_bp[qid] = bp
    if not person_to_bp:
        return [], ok

    bp_ents, bp_ok = _fetch_entities_status(list(person_to_bp.values()))
    place_ents = {bp: e for bp, e in bp_ents.items() if get_coords_of(e)}
    geo, geo_ok = _resolve_cc(place_ents)
    out = []
    for bp_qid in list(person_to_bp.values())[:cap]:
        if bp_qid in place_ents:
            out.append(_candidate(bp_qid, place_ents[bp_qid], get_coords_of(place_ents[bp_qid]),
                                  geo, entity_text))
    return out, ok and bp_ok and geo_ok


def _org_candidates(entity_text: str, qids: list[str], cap: int) -> tuple[list[dict], bool]:
    entities, ok = _fetch_entities_status(qids)
    hq_map = {}
    for qid in qids:
        hq = _claim_id(entities.get(qid, {}), "P159")
        if hq:
            hq_map[qid] = hq
    if not hq_map:
        return [], ok

    hq_ents, hq_ok = _fetch_entities_status(list(dict.fromkeys(hq_map.values())))
    ok = ok and hq_ok

    # For HQs without coords, fall back to the P131 admin territory.
    hq_to_admin = {}
    for hq_qid, hq_ent in hq_ents.items():
        if not get_coords_of(hq_ent):
            admin = _claim_id(hq_ent, "P131")
            if admin:
                hq_to_admin[hq_qid] = admin
    admin_ents: dict[str, dict] = {}
    if hq_to_admin:
        admin_ents, admin_ok = _fetch_entities_status(list(hq_to_admin.values()))
        ok = ok and admin_ok

    place_ents = {}
    for hq_qid in hq_map.values():
        hq_ent = hq_ents.get(hq_qid, {})
        if get_coords_of(hq_ent):
            place_ents[hq_qid] = hq_ent
        elif hq_qid in hq_to_admin:
            admin_ent = admin_ents.get(hq_to_admin[hq_qid], {})
            if get_coords_of(admin_ent):
                place_ents[hq_to_admin[hq_qid]] = admin_ent

    # One candidate per place, in first-seen org order, then by sitelinks.
    ordered = []
    for hq_qid in hq_map.values():
        place_qid = hq_qid if hq_qid in place_ents else hq_to_admin.get(hq_qid)
        if place_qid in place_ents and place_qid not in ordered:
            ordered.append(place_qid)
    ordered.sort(key=lambda q: len(place_ents[q].get("sitelinks", {})), reverse=True)

    geo, geo_ok = _resolve_cc(place_ents)
    return ([_candidate(q, place_ents[q], get_coords_of(place_ents[q]), geo, entity_text)
             for q in ordered[:cap]], ok and geo_ok)


_TYPE_HANDLERS = {
    "LOC": (_loc_candidates, LOC_CAP),
    "PER": (_per_candidates, PER_CAP),
    "ORG": (_org_candidates, ORG_CAP),
}


def _search_with_fallback(entity_text: str, limit: int) -> list[str]:
    """Search, progressively dropping trailing words if nothing is found
    (e.g. "Kaizer Chiefs Foundation" -> "Kaizer Chiefs"). Raises WikidataError."""
    qids = _search_checked(entity_text, limit)
    words = entity_text.split()
    while not qids and len(words) > 1:
        words.pop()
        qids = _search_checked(" ".join(words), limit)
    return qids


def get_coords(entity_text: str, entity_type: str, top_n: int = 10) -> list[dict]:
    """Candidate places for an entity mention, as dicts with lat/lon/description/
    qid/country/continent.

    Considers the top_n search hits. The per-type result caps (LOC 10, PER 5,
    ORG 10) apply for top_n up to SEARCH_LIMIT and scale to top_n beyond it.
    If any network lookup fails the partial result is returned but not cached.
    """
    cache_key = (entity_text, entity_type, top_n)
    if cache_key in _coords_cache:
        return _coords_cache[cache_key]
    if entity_type not in ENTITY_TYPES:
        return []

    try:
        qids = _search_with_fallback(entity_text, min(max(top_n, SEARCH_LIMIT), MAX_SEARCH_LIMIT))
    except WikidataError as e:
        logger.warning("Search error for %r: %s", entity_text, e)
        return []
    if not qids:
        _coords_cache[cache_key] = []
        return []

    handler, default_cap = _TYPE_HANDLERS[entity_type]
    cap = default_cap if top_n <= SEARCH_LIMIT else top_n
    coords, complete = handler(entity_text, qids[:top_n], cap)
    if complete:
        _coords_cache[cache_key] = coords
    return coords


GEO_PROP_CHAIN = ["P159", "P131", "P19", "P937", "P276"]


def resolve_qid_to_candidate(qid: str, entity_text: str, entity_type: str) -> dict | None:
    """Resolve a known QID (e.g. from BLINK) to a geographic candidate dict via
    the property chain P625 (coords) → P159 → P131 → P19 → P937 → P276, with one
    extra P131 hop if the chained target also lacks coords. Returns None if no
    coordinates can be reached."""
    if not qid:
        return None
    ent = fetch_entities([qid]).get(qid, {})
    if not ent:
        return None

    place_qid = None
    place_ent = None

    if get_coords_of(ent):
        place_qid, place_ent = qid, ent
    else:
        for prop in GEO_PROP_CHAIN:
            target = _claim_id(ent, prop)
            if not target:
                continue
            target_ent = fetch_entities([target]).get(target, {})
            if get_coords_of(target_ent):
                place_qid, place_ent = target, target_ent
                break
            admin = _claim_id(target_ent, "P131")
            if admin:
                admin_ent = fetch_entities([admin]).get(admin, {})
                if get_coords_of(admin_ent):
                    place_qid, place_ent = admin, admin_ent
                    break

    if not place_ent:
        return None

    c = get_coords_of(place_ent)
    geo = resolve_country_continent({place_qid: place_ent})
    country, continent = geo.get(place_qid, ("", ""))
    return {
        "entity": entity_text,
        "entity_type": entity_type,
        "lat": c[0], "lon": c[1],
        "description": _description(place_ent, entity_text),
        "qid": place_qid,
        "country": country,
        "continent": continent,
    }


def get_all_coords(entities: list[dict], top_n: int = 10) -> list[dict]:
    """Take the entity list from ner.py and return candidate coordinates."""
    all_candidates = []
    for entity in entities:
        coords = get_coords(entity["text"], entity["type"], top_n=top_n)
        for coord in coords:
            all_candidates.append({
                "entity": entity["text"],
                "entity_type": entity["type"],
                "lat": coord["lat"],
                "lon": coord["lon"],
                "description": coord["description"],
                "qid": coord["qid"],
                "country": coord["country"],
                "continent": coord["continent"],
            })
    return all_candidates


# Backward-compatible aliases for the former private names.
_search_entities = search_entities
_fetch_entities = fetch_entities
_get_coords = get_coords_of
_resolve_country_continent = resolve_country_continent
