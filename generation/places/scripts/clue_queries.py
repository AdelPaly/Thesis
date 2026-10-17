"""Clue query definitions and fetchers shared by 02_clues.py and its consumers.

Holds the property/pattern specs, the sampling constants, the SPARQL builders
(`sampled_query`, `*_where`) and the per-query fetch functions. Importable as a
normal module; 02_clues.py is a thin CLI over it and re-exports these names.
"""

from __future__ import annotations

import hashlib
import time

from wdqs import QueryError, qid_from_uri, sparql, value_of

# Flat inbound properties -> human-readable name.
PROPERTIES = {
    "P19": "place of birth",
    "P20": "place of death",
    "P276": "location",
    "P159": "headquarters location",
    "P937": "work location",
    "P138": "named after",
    "P1071": "location of creation",
    "P495": "country of origin",
    "P1435": "heritage designation",
    "P131": "located in administrative entity",
}

OPTIONAL_PROPERTIES = {
    "P2848": "Wi-Fi access",  # off by default; see 02_clues.py docstring
}

PATTERN_LABELS = {
    "SPORTS": "sports club or team based here",
    "EVENTS": "event held here",
    "WORKS": "work set or filmed here",
}

EVENT_CLASS = "Q1656682"          # event
SPORTS_ORG_CLASS = "Q4438121"     # sports organization
ENTITY_URI_PREFIX = "http://www.wikidata.org/entity/"

DEFAULT_SEED = "wanderer-2026"
PER_PROPERTY = 25     # random sample size per place per property
EVENT_POOL = 400      # candidates drawn before the local event-class filter
VALUES_CHUNK = 200    # max QIDs bound into a single VALUES block
SLEEP_BETWEEN = 0.25  # seconds between requests
EVENT_CLASS_TIMEOUT = 300  # seconds: the one big event-subclass query


def query_keys(include_p2848: bool = False) -> dict[str, str]:
    """Every query run per place, in report order: flat properties then patterns."""
    properties = dict(PROPERTIES)
    if include_p2848:
        properties.update(OPTIONAL_PROPERTIES)
    return {**properties, **PATTERN_LABELS}


def sample_key(clue_qid: str, seed: str) -> str:
    """The same ordering the endpoint applies, recomputed locally.

    MD5(CONCAT(STR(?clue), seed)) over the full entity URI, so a record's rank
    here matches its rank in the SPARQL sample.
    """
    payload = f"{ENTITY_URI_PREFIX}{clue_qid}{seed}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Query construction
# --------------------------------------------------------------------------

def sampled_query(where: str, place_qid: str, limit: int, seed: str) -> str:
    """A random sample of `limit` entities matching `where`.

    The LIMIT lives in a subquery so the sampling is applied to the raw hits;
    the English label is attached outside it and is optional, since a clue with
    no en label is still a real link and should not silently vanish.

    Ordering by MD5 of the entity URI plus a fixed seed gives a sample that is
    deterministic across runs but uncorrelated with notability, so a famous
    place no longer yields uniformly famous clues.
    """
    return f"""
SELECT ?clue ?sitelinks ?label WHERE {{
  {{
    SELECT DISTINCT ?clue ?sitelinks WHERE {{
      {where}
      ?clue wikibase:sitelinks ?sitelinks .
      FILTER(?clue != wd:{place_qid})
    }}
    ORDER BY MD5(CONCAT(STR(?clue),"{seed}"))
    LIMIT {limit}
  }}
  OPTIONAL {{ ?clue rdfs:label ?label FILTER(lang(?label) = "en") }}
}}
"""


def flat_where(place_qid: str, prop: str) -> str:
    """WHERE fragment for a flat inbound property."""
    return f"?clue wdt:{prop} wd:{place_qid} ."


def sports_where(place_qid: str) -> str:
    """Clubs and teams tied to the place by venue, or sports bodies sited there.

    The venue hop (P115 -> P131/P276) is what the flat property list misses: a
    club records its stadium, and only the stadium records the town.
    """
    return f"""{{
    {{ ?clue wdt:P115 ?venue . ?venue (wdt:P131|wdt:P276) wd:{place_qid} . }}
    UNION {{ ?clue wdt:P115 wd:{place_qid} . }}
    UNION {{
      ?clue (wdt:P131|wdt:P159|wdt:P276) wd:{place_qid} .
      ?clue wdt:P31/wdt:P279* wd:{SPORTS_ORG_CLASS} .
    }}
  }}"""


def works_where(place_qid: str) -> str:
    """Works set in (narrative location) or filmed in the place."""
    return f"""{{
    {{ ?clue wdt:P840 wd:{place_qid} . }}
    UNION {{ ?clue wdt:P915 wd:{place_qid} . }}
  }}"""


def event_pool_where(place_qid: str) -> str:
    """Dated things located in the place -- the cheap candidate net for EVENTS.

    A date (P585/P580) is the one signal that separates events from buildings
    and streets without touching the event subclass tree, which is what makes
    the server-side form time out on large places.
    """
    return f"""?clue (wdt:P276|wdt:P131) wd:{place_qid} .
    ?clue (wdt:P585|wdt:P580) ?when ."""


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------

def _collect(rows: list[dict], prop: str, label: str) -> dict[str, dict]:
    """Fold result rows into one record per clue entity (first label wins)."""
    out: dict[str, dict] = {}
    for row in rows:
        clue = value_of(row, "clue")
        if not clue:
            continue
        clue_qid = qid_from_uri(clue)
        sitelinks = value_of(row, "sitelinks")
        record = out.setdefault(
            clue_qid,
            {
                "clue_qid": clue_qid,
                "clue_label": None,
                "property": prop,
                "property_label": label,
                "sitelinks": int(sitelinks) if sitelinks is not None else 0,
            },
        )
        row_label = value_of(row, "label")
        if row_label and not record["clue_label"]:
            record["clue_label"] = row_label
    return out


def fetch_sampled(place_qid: str, prop: str, label: str, where: str,
                  limit: int, seed: str) -> list[dict]:
    """One query for one place. Raises QueryError if the query fails."""
    rows = sparql(sampled_query(where, place_qid, limit, seed))
    found = _collect(rows, prop, label)
    return sorted(found.values(), key=lambda r: sample_key(r["clue_qid"], seed))


def fetch_event_classes() -> set[str]:
    """Every subclass of `event`, in one query. ~55k QIDs, ~10s."""
    rows = sparql(f"SELECT ?c WHERE {{ ?c wdt:P279* wd:{EVENT_CLASS} . }}",
                  timeout=EVENT_CLASS_TIMEOUT)
    return {qid_from_uri(value_of(row, "c") or "") for row in rows} - {""}


def fetch_instance_classes(qids: list[str]) -> dict[str, set[str]]:
    """P31 values for a bounded list of entities, chunked to keep VALUES small."""
    classes: dict[str, set[str]] = {}
    for start in range(0, len(qids), VALUES_CHUNK):
        chunk = qids[start:start + VALUES_CHUNK]
        values = " ".join(f"wd:{q}" for q in chunk)
        rows = sparql(
            f"SELECT ?clue ?c WHERE {{ VALUES ?clue {{ {values} }} ?clue wdt:P31 ?c . }}"
        )
        for row in rows:
            clue = value_of(row, "clue")
            cls = value_of(row, "c")
            if clue and cls:
                classes.setdefault(qid_from_uri(clue), set()).add(qid_from_uri(cls))
        time.sleep(SLEEP_BETWEEN)
    return classes


def fetch_events(place_qid: str, limit: int, seed: str, pool: int,
                 event_classes: set[str]) -> list[dict]:
    """Sample dated things in the place, then keep the ones that are events.

    The pool is already in MD5 order, so taking the first `limit` survivors
    keeps the sample random rather than re-ranking what came back.
    """
    rows = sparql(sampled_query(event_pool_where(place_qid), place_qid, pool, seed))
    found = _collect(rows, "EVENTS", PATTERN_LABELS["EVENTS"])
    if not found:
        return []
    ordered = sorted(found.values(), key=lambda r: sample_key(r["clue_qid"], seed))
    classes = fetch_instance_classes([r["clue_qid"] for r in ordered])
    kept = [r for r in ordered if classes.get(r["clue_qid"], set()) & event_classes]
    return kept[:limit]


def fetch_query(key: str, qid: str, properties: dict[str, str], per_property: int,
                seed: str, event_pool: int, event_classes: set[str]) -> list[dict]:
    """Run the query named `key` (a P-number, SPORTS, EVENTS or WORKS) for one place.

    Raises QueryError on failure.
    """
    if key == "EVENTS":
        return fetch_events(qid, per_property, seed, event_pool, event_classes)
    if key == "SPORTS":
        return fetch_sampled(qid, key, PATTERN_LABELS[key], sports_where(qid),
                             per_property, seed)
    if key == "WORKS":
        return fetch_sampled(qid, key, PATTERN_LABELS[key], works_where(qid),
                             per_property, seed)
    return fetch_sampled(qid, key, properties[key], flat_where(qid, key),
                         per_property, seed)


__all__ = [
    "QueryError", "PROPERTIES", "OPTIONAL_PROPERTIES", "PATTERN_LABELS", "EVENT_CLASS",
    "SPORTS_ORG_CLASS", "DEFAULT_SEED", "PER_PROPERTY", "EVENT_POOL", "VALUES_CHUNK",
    "SLEEP_BETWEEN", "query_keys", "sample_key", "sampled_query", "flat_where",
    "sports_where", "works_where", "event_pool_where", "fetch_sampled",
    "fetch_event_classes", "fetch_instance_classes", "fetch_events", "fetch_query",
]
