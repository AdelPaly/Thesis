#!/usr/bin/env python3
"""Collect Wikidata "clue" entities for every place in a place CSV (default data/01_places_final.csv).

A clue is any entity that points AT the place. Two kinds of query find them.

FLAT PROPERTIES -- one hop, `?clue wdt:PROP wd:PLACE`:

    P19    place of birth          people born there
    P20    place of death          people who died there
    P276   location                things located there
    P159   headquarters location   organisations based there
    P937   work location           people who worked there
    P138   named after             things named after it
    P1071  location of creation    works/objects made there
    P495   country of origin       works originating there
    P1435  heritage designation    listed/protected things there
    P131   located in admin entity  things inside the place

    P2848 (Wi-Fi access) is NOT queried by default. It is an outbound facility
    property; probed inbound over a 15-place sample and over all 150 places it
    returned zero rows every time. --include-p2848 turns it back on.

PATTERNS -- multi-hop shapes that catch associations no single property does:

    SPORTS  clubs/teams whose home venue (P115) is the place or sits in it,
            plus sports organisations (P279* of Q4438121) sited there
    EVENTS  festivals, competitions and other instances of an event class
            (P279* of Q1656682) whose location is the place
    WORKS   books, films and songs set in (P840) or filmed in (P915) the place
            (works *named after* the place already arrive via P138)

EVENTS runs in two steps. A server-side `P31/P279* Q1656682` walk times out on
large places, so instead a cheap query samples dated things located in the
place, and their P31 classes are intersected locally against the event subclass
set -- fetched once per run in a single ~10s query.

SAMPLING. Each query returns a RANDOM sample of up to --per-property items,
ordered by MD5(CONCAT(STR(?clue), seed)) with a seed fixed by --seed. The old
ORDER BY DESC(?sitelinks) biased every clue set toward the most famous
associated entities, which contaminated difficulty control: easy places got the
easiest clues. Sitelink counts are still recorded, just no longer used to rank.

Output: data/02_clues.jsonl, one JSON object per clue:
    place_qid, place_name, clue_qid, clue_label, property, property_label,
    sitelinks

`property` is a P-number for a flat property, or SPORTS / EVENTS / WORKS for a
pattern. A place whose queries all fail is reported as failed, not as "0 clues".

Usage:
    py -3 scripts/02_clues.py [--limit N] [--per-property N] [--seed S]
                              [--include-p2848] [--no-cache]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT / "data" / "01_places_final.csv"
OUTPUT_JSONL = ROOT / "data" / "02_clues.jsonl"
CACHE_PATH = ROOT / "data" / "02_clues_cache.json"

ENDPOINT = "https://query.wikidata.org/sparql"
# WDQS rejects requests without a descriptive User-Agent.
USER_AGENT = (
    "places-clues/2.0 (https://github.com/; clue collection for a 150-row place "
    "dataset; contact: shay@shaypunter.co.uk) python-urllib"
)

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
    "P2848": "Wi-Fi access",  # off by default; see module docstring
}

PATTERN_LABELS = {
    "SPORTS": "sports club or team based here",
    "EVENTS": "event held here",
    "WORKS": "work set or filmed here",
}

EVENT_CLASS = "Q1656682"          # event
SPORTS_ORG_CLASS = "Q4438121"     # sports organization

DEFAULT_SEED = "wanderer-2026"
PER_PROPERTY = 25    # random sample size per place per property
EVENT_POOL = 400     # candidates drawn before the local event-class filter
VALUES_CHUNK = 200   # max QIDs bound into a single VALUES block
SLEEP_BETWEEN = 0.25  # seconds between requests
HTTP_TIMEOUT = 180   # seconds
ATTEMPTS = 3
CACHE_VERSION = 2    # bumped: sampling changed, v1 entries are ordered by fame


class QueryError(RuntimeError):
    """A SPARQL request failed. Not an empty result."""


# --------------------------------------------------------------------------
# TLS
# --------------------------------------------------------------------------
# Some Windows trust stores carry expired cross-signed intermediates. OpenSSL
# chains through one of those and fails with "certificate has expired" even
# though a valid path exists. Verification stays fully on; we just drop expired
# anchors before handing them to OpenSSL.

_ssl_context: ssl.SSLContext | None = None


def _filtered_ssl_context() -> ssl.SSLContext | None:
    if not hasattr(ssl, "enum_certificates"):
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    pems = []
    for store in ("ROOT", "CA"):
        try:
            certs = ssl.enum_certificates(store)
        except Exception:
            continue
        for der, encoding, trust in certs:
            if encoding != "x509_asn":
                continue
            if isinstance(trust, set) and "1.3.6.1.5.5.7.3.1" not in trust:
                continue  # not trusted for server auth
            probe = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            try:
                probe.load_verify_locations(cadata=der)
                not_after = probe.get_ca_certs()[0]["notAfter"]
                expires = datetime.datetime.strptime(
                    not_after, "%b %d %H:%M:%S %Y %Z"
                ).replace(tzinfo=datetime.timezone.utc)
            except Exception:
                continue
            if expires > now:
                pems.append(ssl.DER_cert_to_PEM_cert(der))
    if not pems:
        return None
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cadata="".join(pems))
    return context


def ssl_context() -> ssl.SSLContext:
    """Verifying context: truststore > certifi > OS store minus expired anchors."""
    global _ssl_context
    if _ssl_context is not None:
        return _ssl_context
    try:
        import truststore  # type: ignore

        _ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        return _ssl_context
    except Exception:
        pass
    try:
        import certifi  # type: ignore

        _ssl_context = ssl.create_default_context(cafile=certifi.where())
        return _ssl_context
    except Exception:
        pass
    _ssl_context = ssl.create_default_context()
    return _ssl_context


def _retry_without_expired_anchors() -> bool:
    """Swap in the filtered context after a verification failure. True if changed."""
    global _ssl_context
    filtered = _filtered_ssl_context()
    if filtered is None:
        return False
    _ssl_context = filtered
    return True


# --------------------------------------------------------------------------
# SPARQL plumbing
# --------------------------------------------------------------------------

def _summarise_error(body: str) -> str:
    body = re.sub(r"\s+", " ", body).strip()
    if "TimeoutException" in body or "QueryTimeout" in body:
        return "query timeout"
    if "MalformedQuery" in body:
        return "malformed query: " + body[:200]
    return body[:200] if body else "no response body"


def sparql(query: str, *, attempts: int = ATTEMPTS, timeout: int = HTTP_TIMEOUT) -> list[dict]:
    """POST a query and return the result bindings. Raises QueryError on failure."""
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=data,
        method="POST",
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    last = "unknown error"
    tls_retried = False
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=ssl_context()) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload["results"]["bindings"]
        except urllib.error.HTTPError as exc:
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # pragma: no cover - defensive
                body = ""
            last = f"HTTP {exc.code}: {_summarise_error(body)}"
            if exc.code == 429:
                wait = float(exc.headers.get("Retry-After") or 5)
                time.sleep(min(wait, 60))
                continue
            if exc.code < 500:
                break  # 4xx other than rate limiting will not fix itself
        except Exception as exc:  # URLError, timeout, malformed JSON, ...
            last = f"{type(exc).__name__}: {exc}"
            if "CERTIFICATE_VERIFY_FAILED" in str(exc) and not tls_retried:
                tls_retried = True
                if _retry_without_expired_anchors():
                    print("      ! TLS verify failed; retrying without expired trust anchors")
                    continue
        if attempt < attempts:
            time.sleep(2 * attempt)
    raise QueryError(last)


def value_of(binding: dict, key: str) -> str | None:
    cell = binding.get(key)
    return cell.get("value") if cell else None


def qid_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def sample_key(clue_qid: str, seed: str) -> str:
    """The same ordering the endpoint applies, recomputed locally.

    MD5(CONCAT(STR(?clue), seed)) over the full entity URI, so a record's rank
    here matches its rank in the SPARQL sample.
    """
    payload = f"http://www.wikidata.org/entity/{clue_qid}{seed}"
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
    rows = sparql(f"SELECT ?c WHERE {{ ?c wdt:P279* wd:{EVENT_CLASS} . }}", timeout=300)
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


# --------------------------------------------------------------------------
# Input / cache
# --------------------------------------------------------------------------

def read_places(path: Path, limit: int | None) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    places = []
    for row in rows:
        qid = (row.get("wikidata_id") or "").strip()
        if not qid:
            continue
        places.append({"qid": qid, "name": (row.get("place_name") or "").strip()})
    if limit is not None:
        places = places[:limit]
    return places


def load_cache(use_cache: bool) -> dict:
    if not use_cache or not CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("version") != CACHE_VERSION:
        return {}
    return payload.get("entries", {})


def save_cache(entries: dict) -> None:
    CACHE_PATH.write_text(
        json.dumps({"version": CACHE_VERSION, "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_table(places: list[dict], counts: dict[str, int], failures: dict[str, str]) -> None:
    name_width = max([len(p["name"]) for p in places] + [len("place")])
    qid_width = max([len(p["qid"]) for p in places] + [len("qid")])
    header = f"{'place':<{name_width}}  {'qid':<{qid_width}}  {'clues':>5}"
    print()
    print(header)
    print("-" * len(header))
    for place in places:
        note = ""
        if place["qid"] in failures:
            note = f"  ! {failures[place['qid']]}"
        print(
            f"{place['name']:<{name_width}}  {place['qid']:<{qid_width}}  "
            f"{counts.get(place['qid'], 0):>5}{note}"
        )


def print_buckets(places: list[dict], counts: dict[str, int]) -> None:
    zero = sum(1 for p in places if counts.get(p["qid"], 0) == 0)
    few = sum(1 for p in places if 1 <= counts.get(p["qid"], 0) <= 2)
    many = sum(1 for p in places if counts.get(p["qid"], 0) >= 3)
    print()
    print(f"places with 0 clues   : {zero}")
    print(f"places with 1-2 clues : {few}")
    print(f"places with 3+ clues  : {many}")
    print(f"total places          : {len(places)}")


def print_per_property(records: list[dict], keys: dict[str, str]) -> None:
    """Records and distinct places touched, per query. Zero rows are the point."""
    print()
    print(f"{'query':<8} {'meaning':<34} {'records':>8} {'places':>7}")
    print("-" * 60)
    for key, label in keys.items():
        hits = [r for r in records if r["property"] == key]
        places = {r["place_qid"] for r in hits}
        print(f"{key:<8} {label:<34} {len(hits):>8} {len(places):>7}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only process the first N places")
    parser.add_argument(
        "--per-property",
        type=int,
        default=PER_PROPERTY,
        help=f"max clues sampled per place per query (default {PER_PROPERTY})",
    )
    parser.add_argument(
        "--seed",
        default=DEFAULT_SEED,
        help=f"sampling seed mixed into the MD5 ordering (default {DEFAULT_SEED!r})",
    )
    parser.add_argument(
        "--event-pool",
        type=int,
        default=EVENT_POOL,
        help=f"candidates drawn before the event-class filter (default {EVENT_POOL})",
    )
    parser.add_argument(
        "--include-p2848",
        action="store_true",
        help="also query P2848 (Wi-Fi access); returns nothing, off by default",
    )
    parser.add_argument("--no-cache", action="store_true", help="ignore and overwrite the cache")
    parser.add_argument("--input", type=Path, default=INPUT_CSV)
    parser.add_argument("--out", type=Path, default=OUTPUT_JSONL)
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2

    places = read_places(args.input, args.limit)
    if not places:
        print("no places with a wikidata_id in the input", file=sys.stderr)
        return 2

    properties = dict(PROPERTIES)
    if args.include_p2848:
        properties.update(OPTIONAL_PROPERTIES)

    # Every query run per place, in report order: flat properties then patterns.
    query_keys = {**properties, **PATTERN_LABELS}

    use_cache = not args.no_cache
    cache = load_cache(use_cache)
    counts: dict[str, int] = {}
    failures: dict[str, str] = {}
    records: list[dict] = []

    print(f"{len(places)} places x {len(query_keys)} queries "
          f"(<= {args.per_property} clues each, seed {args.seed!r})")

    print("fetching event subclass tree ...", flush=True)
    try:
        event_classes = fetch_event_classes()
        print(f"  {len(event_classes)} event classes", flush=True)
    except QueryError as exc:
        print(f"  ! could not fetch event classes ({exc}); EVENTS disabled", file=sys.stderr)
        event_classes = set()

    for index, place in enumerate(places, start=1):
        qid = place["qid"]
        place_records: list[dict] = []
        errors: list[str] = []
        print(f"[{index}/{len(places)}] {place['name']} ({qid})", flush=True)

        for key in query_keys:
            cache_key = f"{qid}|{key}|{args.per_property}|{args.seed}"
            if key == "EVENTS":
                cache_key += f"|{args.event_pool}"
            cached = cache.get(cache_key) if use_cache else None
            if cached is not None:
                clues = cached
            elif key == "EVENTS" and not event_classes:
                continue
            else:
                try:
                    if key == "EVENTS":
                        clues = fetch_events(qid, args.per_property, args.seed,
                                             args.event_pool, event_classes)
                    elif key == "SPORTS":
                        clues = fetch_sampled(qid, key, PATTERN_LABELS[key],
                                              sports_where(qid), args.per_property, args.seed)
                    elif key == "WORKS":
                        clues = fetch_sampled(qid, key, PATTERN_LABELS[key],
                                              works_where(qid), args.per_property, args.seed)
                    else:
                        clues = fetch_sampled(qid, key, properties[key],
                                              flat_where(qid, key), args.per_property, args.seed)
                except QueryError as exc:
                    errors.append(f"{key}: {exc}")
                    print(f"      ! {key} failed: {exc}", flush=True)
                    continue
                cache[cache_key] = clues
                time.sleep(SLEEP_BETWEEN)

            for clue in clues:
                place_records.append(
                    {
                        "place_qid": qid,
                        "place_name": place["name"],
                        "clue_qid": clue["clue_qid"],
                        "clue_label": clue["clue_label"],
                        "property": clue["property"],
                        "property_label": clue["property_label"],
                        "sitelinks": clue["sitelinks"],
                    }
                )

        # Deterministic but notability-neutral order, matching the sampling.
        place_records.sort(key=lambda r: (sample_key(r["clue_qid"], args.seed), r["property"]))
        records.extend(place_records)
        counts[qid] = len(place_records)
        if errors and not place_records:
            failures[qid] = f"{len(errors)}/{len(query_keys)} queries failed"

        if use_cache and index % 5 == 0:
            save_cache(cache)

    if use_cache:
        save_cache(cache)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print_table(places, counts, failures)
    print_buckets(places, counts)
    print_per_property(records, query_keys)
    unlabelled = sum(1 for r in records if not r["clue_label"])
    print(f"\nwrote {len(records)} clue records to {args.out}")
    if unlabelled:
        print(f"  ({unlabelled} have no English label)")
    if failures:
        print(f"  ({len(failures)} places returned nothing because every query failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
