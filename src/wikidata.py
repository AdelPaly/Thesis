import atexit
import os
import pickle
import requests
import time

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "GeoThesisBot/1.0 (student research; contact via GitHub)"}

_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".wikidata_cache")
_CACHE_FILES = {
    "entity": os.path.join(_CACHE_DIR, "entity.pkl"),
    "label": os.path.join(_CACHE_DIR, "label.pkl"),
    "coords": os.path.join(_CACHE_DIR, "coords.pkl"),
}


def _load_pickle(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Cache load failed for {path}: {e}")
        return {}


os.makedirs(_CACHE_DIR, exist_ok=True)
_entity_cache = _load_pickle(_CACHE_FILES["entity"])   # QID -> entity data
_label_cache = _load_pickle(_CACHE_FILES["label"])     # QID -> English label
_coords_cache = _load_pickle(_CACHE_FILES["coords"])   # (text, type) -> candidates
_initial_sizes = (len(_entity_cache), len(_label_cache), len(_coords_cache))
print(f"  Wikidata cache loaded: {_initial_sizes[0]} entities, "
      f"{_initial_sizes[1]} labels, {_initial_sizes[2]} coord lookups")


def _save_caches():
    # Only write if something changed, to avoid clobbering with empty dicts on error.
    sizes = (len(_entity_cache), len(_label_cache), len(_coords_cache))
    if sizes == _initial_sizes:
        return
    try:
        for key, path in _CACHE_FILES.items():
            tmp = path + ".tmp"
            data = {"entity": _entity_cache, "label": _label_cache, "coords": _coords_cache}[key]
            with open(tmp, "wb") as f:
                pickle.dump(data, f)
            os.replace(tmp, path)
        print(f"  Wikidata cache saved: {sizes[0]} entities, {sizes[1]} labels, {sizes[2]} coord lookups")
    except Exception as e:
        print(f"  Cache save failed: {e}")


atexit.register(_save_caches)


def save_caches():
    """Public helper so long-running drivers can checkpoint mid-run."""
    _save_caches()

_last_request_time = 0.0
_MIN_REQUEST_INTERVAL = 0.5


def _api_get(params, timeout=15):
    """Single entry point for all Wikidata API calls, with rate limiting and retry on 429."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)

    for attempt in range(6):
        _last_request_time = time.time()
        resp = requests.get(WIKIDATA_API, params=params, headers=HEADERS, timeout=timeout)
        if resp.status_code == 429:
            wait = 2 ** attempt
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise requests.exceptions.HTTPError("429 Too Many Requests after retries")


def _search_entities(text, limit=10):
    try:
        data = _api_get({
            "action": "wbsearchentities",
            "search": text,
            "language": "en",
            "limit": limit,
            "format": "json",
        })
        return [item["id"] for item in data.get("search", [])]
    except Exception as e:
        print(f"  Search error: {e}")
        return []


def _fetch_entities(qids):
    """Batch-fetch entity data, using cache. Returns dict QID -> entity data."""
    missing = [q for q in qids if q not in _entity_cache]
    if missing:
        for i in range(0, len(missing), 50):
            batch = missing[i:i+50]
            try:
                data = _api_get({
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "claims|descriptions|sitelinks/urls|labels",
                    "languages": "en",
                    "format": "json",
                })
                for qid, ent in data.get("entities", {}).items():
                    _entity_cache[qid] = ent
                    lbl = ent.get("labels", {}).get("en", {}).get("value", "")
                    if lbl:
                        _label_cache[qid] = lbl
            except Exception as e:
                print(f"  API error: {e}")
    return {q: _entity_cache.get(q, {}) for q in qids}


def _fetch_labels(qids):
    """Batch-fetch English labels, using cache. Returns dict QID -> label."""
    missing = [q for q in qids if q not in _label_cache]
    if missing:
        for i in range(0, len(missing), 50):
            batch = missing[i:i+50]
            try:
                data = _api_get({
                    "action": "wbgetentities",
                    "ids": "|".join(batch),
                    "props": "labels",
                    "languages": "en",
                    "format": "json",
                })
                for qid, ent in data.get("entities", {}).items():
                    _label_cache[qid] = ent.get("labels", {}).get("en", {}).get("value", "")
            except Exception:
                pass
    return {q: _label_cache.get(q, "") for q in qids}


def _claim_id(entity, prop):
    """Get QID of first value of a property claim."""
    claims = entity.get("claims", {})
    if prop not in claims:
        return None
    val = claims[prop][0].get("mainsnak", {}).get("datavalue", {}).get("value")
    if isinstance(val, dict) and "id" in val:
        return val["id"]
    return None


def _claim_ids(entity, prop):
    """Get all QIDs from a property claim."""
    claims = entity.get("claims", {})
    out = []
    for claim in claims.get(prop, []):
        val = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(val, dict) and "id" in val:
            out.append(val["id"])
    return out


def _get_coords(entity):
    claims = entity.get("claims", {})
    if "P625" not in claims:
        return None
    val = claims["P625"][0].get("mainsnak", {}).get("datavalue", {}).get("value")
    if val and "latitude" in val:
        return val["latitude"], val["longitude"]
    return None


def _resolve_country_continent(entities_dict):
    """For a dict of QID->entity, resolve country/continent labels in bulk.
    Returns dict of QID -> (country_label, continent_label)."""
    # Collect all country and continent QIDs
    country_qids = {}   # place_qid -> country_qid
    continent_qids = {}  # place_qid -> continent_qid

    for qid, ent in entities_dict.items():
        country_qids[qid] = _claim_id(ent, "P17")
        continent_qids[qid] = _claim_id(ent, "P30")

    # Fetch all country entities (to get their P30 continent) in one batch
    all_country_qids = set(q for q in country_qids.values() if q)
    # Also need labels for country QIDs and continent QIDs
    all_label_qids = set()
    all_label_qids.update(all_country_qids)
    all_label_qids.update(q for q in continent_qids.values() if q)

    # For places without direct continent, we need the country entity to find its continent
    need_country_entity = set()
    for qid in entities_dict:
        if not continent_qids[qid] and country_qids[qid]:
            need_country_entity.add(country_qids[qid])

    # Batch fetch country entities if needed
    if need_country_entity:
        country_ents = _fetch_entities(list(need_country_entity))
        for cq, cent in country_ents.items():
            cont_qid = _claim_id(cent, "P30")
            if cont_qid:
                all_label_qids.add(cont_qid)
                # Store this for later lookup
                for qid in entities_dict:
                    if country_qids[qid] == cq and not continent_qids[qid]:
                        continent_qids[qid] = cont_qid

    # Single batch label fetch for all country + continent QIDs
    if all_label_qids:
        _fetch_labels(list(all_label_qids))

    result = {}
    for qid in entities_dict:
        country_label = _label_cache.get(country_qids.get(qid, ""), "")
        continent_label = _label_cache.get(continent_qids.get(qid, ""), "")
        result[qid] = (country_label, continent_label)
    return result


def get_coords(entity_text, entity_type, top_n=10):
    cache_key = (entity_text, entity_type, top_n)
    if cache_key in _coords_cache:
        return _coords_cache[cache_key]

    if entity_type not in ["LOC", "PER", "ORG"]:
        return []

    # Step 1: search, with progressive fallback for long entity names
    qids = _search_entities(entity_text)
    if not qids:
        # Try removing trailing words one at a time (e.g. "Kaizer Chiefs Foundation" -> "Kaizer Chiefs")
        words = entity_text.split()
        while not qids and len(words) > 1:
            words.pop()
            qids = _search_entities(" ".join(words))
    if not qids:
        _coords_cache[cache_key] = []
        return []

    # Restrict to top-N search results (original mode: top_n=1 = strict popularity-biased pick)
    qids = qids[:top_n]

    # Step 2: batch-fetch all candidate entities (1 API call)
    entities = _fetch_entities(qids)

    coords = []

    if entity_type == "LOC":
        # Filter to entities with coordinates
        place_ents = {}
        sitelink_counts = {}
        for qid in qids:
            ent = entities.get(qid, {})
            c = _get_coords(ent)
            if c:
                place_ents[qid] = ent
                sitelink_counts[qid] = len(ent.get("sitelinks", {}))

        # Batch resolve country/continent (1-2 API calls)
        geo = _resolve_country_continent(place_ents)

        for qid in sorted(place_ents, key=lambda q: sitelink_counts.get(q, 0), reverse=True)[:10]:
            ent = place_ents[qid]
            c = _get_coords(ent)
            country, continent = geo[qid]
            coords.append({
                "lat": c[0], "lon": c[1],
                "description": ent.get("descriptions", {}).get("en", {}).get("value", entity_text),
                "qid": qid, "country": country, "continent": continent,
            })

    elif entity_type == "PER":
        # Find humans and collect birthplace QIDs
        bp_qids = []
        person_to_bp = {}
        for qid in qids:
            ent = entities.get(qid, {})
            types = _claim_ids(ent, "P31")
            if "Q5" not in types:
                continue
            bp = _claim_id(ent, "P19")
            if bp:
                person_to_bp[qid] = bp
                bp_qids.append(bp)

        # Batch fetch all birthplace entities (1 API call)
        if bp_qids:
            bp_ents = _fetch_entities(bp_qids)
            place_ents = {}
            for person_qid, bp_qid in person_to_bp.items():
                bp_ent = bp_ents.get(bp_qid, {})
                c = _get_coords(bp_ent)
                if c:
                    place_ents[bp_qid] = bp_ent

            geo = _resolve_country_continent(place_ents)

            for person_qid, bp_qid in list(person_to_bp.items())[:5]:
                bp_ent = bp_ents.get(bp_qid, {})
                c = _get_coords(bp_ent)
                if c:
                    country, continent = geo.get(bp_qid, ("", ""))
                    coords.append({
                        "lat": c[0], "lon": c[1],
                        "description": bp_ent.get("descriptions", {}).get("en", {}).get("value", entity_text),
                        "qid": bp_qid, "country": country, "continent": continent,
                    })

    elif entity_type == "ORG":
        # Collect all HQ QIDs
        hq_map = {}
        for qid in qids:
            ent = entities.get(qid, {})
            hq = _claim_id(ent, "P159")
            if hq:
                hq_map[qid] = hq

        # Batch fetch all HQ entities (1 API call)
        if hq_map:
            hq_qids = list(set(hq_map.values()))
            hq_ents = _fetch_entities(hq_qids)

            # For HQs without coords, try P131 admin territory
            admin_qids = []
            hq_to_admin = {}
            for hq_qid, hq_ent in hq_ents.items():
                if not _get_coords(hq_ent):
                    admin = _claim_id(hq_ent, "P131")
                    if admin:
                        hq_to_admin[hq_qid] = admin
                        admin_qids.append(admin)

            if admin_qids:
                _fetch_entities(admin_qids)

            # Collect all place entities we'll use
            place_ents = {}
            org_to_place = {}
            for org_qid, hq_qid in hq_map.items():
                hq_ent = hq_ents.get(hq_qid, {})
                if _get_coords(hq_ent):
                    place_ents[hq_qid] = hq_ent
                    org_to_place[org_qid] = hq_qid
                elif hq_qid in hq_to_admin:
                    admin_qid = hq_to_admin[hq_qid]
                    admin_ent = _entity_cache.get(admin_qid, {})
                    if _get_coords(admin_ent):
                        place_ents[admin_qid] = admin_ent
                        org_to_place[org_qid] = admin_qid

            geo = _resolve_country_continent(place_ents)

            for org_qid, place_qid in org_to_place.items():
                ent = place_ents[place_qid]
                c = _get_coords(ent)
                country, continent = geo.get(place_qid, ("", ""))
                sitelinks = len(ent.get("sitelinks", {}))
                coords.append({
                    "lat": c[0], "lon": c[1],
                    "description": ent.get("descriptions", {}).get("en", {}).get("value", entity_text),
                    "qid": place_qid, "country": country, "continent": continent,
                    "_sitelinks": sitelinks,
                })
            coords.sort(key=lambda x: x.pop("_sitelinks", 0), reverse=True)
            coords = coords[:10]

    _coords_cache[cache_key] = coords
    return coords


GEO_PROP_CHAIN = ["P159", "P131", "P19", "P937", "P276"]


def resolve_qid_to_candidate(qid, entity_text, entity_type):
    """Resolve a known QID (e.g. from BLINK) to a geographic candidate dict via
    the property chain P625 (coords) → P159 → P131 → P19 → P937 → P276, with one
    extra P131 hop if the chained target also lacks coords. Returns None if no
    coordinates can be reached."""
    if not qid:
        return None
    ents = _fetch_entities([qid])
    ent = ents.get(qid, {})
    if not ent:
        return None

    place_qid = None
    place_ent = None

    if _get_coords(ent):
        place_qid, place_ent = qid, ent
    else:
        for prop in GEO_PROP_CHAIN:
            target = _claim_id(ent, prop)
            if not target:
                continue
            target_ent = _fetch_entities([target]).get(target, {})
            if _get_coords(target_ent):
                place_qid, place_ent = target, target_ent
                break
            admin = _claim_id(target_ent, "P131")
            if admin:
                admin_ent = _fetch_entities([admin]).get(admin, {})
                if _get_coords(admin_ent):
                    place_qid, place_ent = admin, admin_ent
                    break

    if not place_ent:
        return None

    c = _get_coords(place_ent)
    geo = _resolve_country_continent({place_qid: place_ent})
    country, continent = geo.get(place_qid, ("", ""))
    return {
        "entity": entity_text,
        "entity_type": entity_type,
        "lat": c[0], "lon": c[1],
        "description": place_ent.get("descriptions", {}).get("en", {}).get("value", entity_text),
        "qid": place_qid,
        "country": country,
        "continent": continent,
    }


def get_labels(qids):
    """Public wrapper: batch-fetch English labels for QIDs."""
    return _fetch_labels(qids)


def get_all_coords(entities, top_n=10):
    """Vezme seznam entit z ner.py a vrátí kandidátní souřadnice"""
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
        time.sleep(0.5)
    return all_candidates
