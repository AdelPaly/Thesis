#!/usr/bin/env python3
"""Re-derive every queryable value in data/01_places_final.csv from Wikidata.

Read-only verification: the input CSV is never written to. Disagreements go to
data/verify_report.csv, P31 labels for manual review go to data/verify_types.csv.

Checks (all rows):
  resolves        every wikidata_id returns an entity
  sitelinks       wikibase:sitelinks vs the file's sitelink column
  inbound_links   distinct Q-entity subjects pointing at the item vs wikidata_links
  name_sharing    items with rdfs:label "<place_name>"@en vs places_sharing_name
  country         P17 vs the country column
  coordinates     flags items with no P625
  area            normalised P2046 (m2 -> km2) vs area_km2, 1% tolerance
  duplicate_*     repeated wikidata_id / place_name within the file
  type sanity     P31 labels dumped for human scanning, never auto-validated

A failed query is reported as "<check>:query_failed", never as a mismatch, and
does not affect the exit code. Exit 0 if no disagreements, 1 otherwise.

Usage:
    py -3 scripts/verify_places.py [--limit N] [--seed N] [--strict-sitelinks]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import random
import re
import ssl
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ENDPOINT = "https://query.wikidata.org/sparql"
# WDQS rejects requests without a descriptive User-Agent.
USER_AGENT = (
    "places-verify/1.0 (https://github.com/; verification of a 150-row place "
    "dataset; contact: shay@shaypunter.co.uk) python-urllib"
)

BATCH_SIZE = 40          # QIDs per VALUES clause
LABEL_BATCH_SIZE = 20    # labels per VALUES clause
SITELINK_BATCH_SIZE = 20 # smaller: this one aggregates over every sitelink
SLEEP_BETWEEN = 0.2      # seconds between requests
HTTP_TIMEOUT = 180       # seconds
ATTEMPTS = 3

ENTITY_PREFIX = "http://www.wikidata.org/entity/Q"

# The spec calls this column wikimedia_sitelinks; the shipped file names it
# wikipedia_languages. Accept either.
SITELINK_COLUMNS = ("wikimedia_sitelinks", "wikipedia_languages", "sitelinks")

AREA_TOLERANCE = 0.01  # 1% relative


class QueryError(RuntimeError):
    """A SPARQL request failed. Not a mismatch."""


# --------------------------------------------------------------------------
# TLS
# --------------------------------------------------------------------------
# Some Windows trust stores carry expired cross-signed intermediates (R3, the
# ISRG Root X2 cross-sign, ...). OpenSSL chains through one of those and fails
# with "certificate has expired" even though a valid path exists. Verification
# stays fully on; we just drop expired anchors before handing them to OpenSSL.

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
            if exc.code < 500 and exc.code != 429:
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


def escape_literal(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )


def chunks(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def query_core(qids: list[str]) -> dict[str, dict]:
    """sitelinks, P17, P625 presence and normalised P2046 for a batch of items."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?version ?redirect ?sitelinks ?country ?coord ?area WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item schema:version ?version }}
  OPTIONAL {{ ?item owl:sameAs ?redirect }}
  OPTIONAL {{ ?item wikibase:sitelinks ?sitelinks }}
  OPTIONAL {{ ?item wdt:P17 ?country }}
  OPTIONAL {{ ?item wdt:P625 ?coord }}
  OPTIONAL {{
    ?item p:P2046 ?areaStatement .
    ?areaStatement psn:P2046/wikibase:quantityAmount ?area .
    FILTER NOT EXISTS {{ ?areaStatement wikibase:rank wikibase:DeprecatedRank }}
  }}
}}
"""
    out: dict[str, dict] = {
        q: {
            "exists": False,
            "redirect": None,
            "sitelinks": None,
            "countries": set(),
            "has_coord": False,
            "areas": set(),
        }
        for q in qids
    }
    for row in sparql(query):
        item = value_of(row, "item")
        if not item:
            continue
        record = out.get(qid_from_uri(item))
        if record is None:
            continue
        if value_of(row, "version"):
            record["exists"] = True
        redirect = value_of(row, "redirect")
        if redirect and redirect.startswith(ENTITY_PREFIX):
            record["redirect"] = qid_from_uri(redirect)
        sitelinks = value_of(row, "sitelinks")
        if sitelinks is not None:
            record["exists"] = True
            record["sitelinks"] = int(sitelinks)
        country = value_of(row, "country")
        if country:
            record["countries"].add(qid_from_uri(country))
        if value_of(row, "coord"):
            record["has_coord"] = True
        area = value_of(row, "area")
        if area is not None:
            record["areas"].add(float(area))
    return out


def query_wikipedia_sitelinks(qids: list[str]) -> dict[str, int]:
    """Sitelinks restricted to Wikipedia editions.

    Anchored on ?item via schema:about, so it stays bounded. Used only to
    explain a systematic offset against the file's column; failure is tolerated.
    """
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item (COUNT(DISTINCT ?site) AS ?count) WHERE {{
  VALUES ?item {{ {values} }}
  ?article schema:about ?item ; schema:isPartOf ?site .
  FILTER(STRENDS(STR(?site), ".wikipedia.org/"))
}}
GROUP BY ?item
"""
    out: dict[str, int] = {}
    for row in sparql(query):
        item = value_of(row, "item")
        count = value_of(row, "count")
        if item and count is not None:
            out[qid_from_uri(item)] = int(count)
    return out


def query_label_counts(labels: list[str]) -> dict[str, int]:
    """How many Wikidata items carry each exact English label."""
    values = " ".join(f'"{escape_literal(label)}"@en' for label in labels)
    query = f"""
SELECT ?label (COUNT(DISTINCT ?entity) AS ?count) WHERE {{
  VALUES ?label {{ {values} }}
  ?entity rdfs:label ?label .
  FILTER(STRSTARTS(STR(?entity), "{ENTITY_PREFIX}"))
}}
GROUP BY ?label
"""
    counts = {label: 0 for label in labels}
    for row in sparql(query):
        label = value_of(row, "label")
        count = value_of(row, "count")
        if label is not None and count is not None:
            counts[label] = int(count)
    return counts


def query_p31_labels(qids: list[str]) -> dict[str, list[str]]:
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?classLabel WHERE {{
  VALUES ?item {{ {values} }}
  ?item wdt:P31 ?class .
  ?class rdfs:label ?classLabel .
  FILTER(LANG(?classLabel) = "en")
}}
"""
    out: dict[str, list[str]] = defaultdict(list)
    for row in sparql(query):
        item = value_of(row, "item")
        label = value_of(row, "classLabel")
        if item and label:
            qid = qid_from_uri(item)
            if label not in out[qid]:
                out[qid].append(label)
    return out


def query_country_names(qids: list[str]) -> dict[str, dict]:
    """English label plus aliases for the country items we saw."""
    values = " ".join(f"wd:{q}" for q in qids)
    query = f"""
SELECT ?item ?label ?alias WHERE {{
  VALUES ?item {{ {values} }}
  OPTIONAL {{ ?item rdfs:label ?label FILTER(LANG(?label) = "en") }}
  OPTIONAL {{ ?item skos:altLabel ?alias FILTER(LANG(?alias) = "en") }}
}}
"""
    out: dict[str, dict] = {q: {"label": None, "aliases": set()} for q in qids}
    for row in sparql(query):
        item = value_of(row, "item")
        if not item:
            continue
        record = out.setdefault(qid_from_uri(item), {"label": None, "aliases": set()})
        label = value_of(row, "label")
        if label:
            record["label"] = label
        alias = value_of(row, "alias")
        if alias:
            record["aliases"].add(alias)
    return out


def query_inbound_links(qid: str) -> int:
    """Distinct Q-entity subjects with a statement pointing at this item.

    Slow: unbound predicate against one object. Cached by the caller.
    """
    query = f"""
SELECT (COUNT(DISTINCT ?subject) AS ?count) WHERE {{
  ?subject ?predicate wd:{qid} .
  FILTER(STRSTARTS(STR(?subject), "{ENTITY_PREFIX}"))
}}
"""
    rows = sparql(query)
    if not rows:
        raise QueryError("no rows returned")
    count = value_of(rows[0], "count")
    if count is None:
        raise QueryError("no count in result")
    return int(count)


# --------------------------------------------------------------------------
# Comparison helpers
# --------------------------------------------------------------------------

def normalise_name(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("'", "'").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("the "):
        text = text[4:]
    return text


def parse_int(text: str) -> int | None:
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_float(text: str) -> float | None:
    text = (text or "").strip().replace(",", "")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def within_tolerance(file_value: float, live_value: float) -> bool:
    if file_value == 0:
        return abs(live_value) < 1e-9
    return abs(live_value - file_value) / abs(file_value) <= AREA_TOLERANCE


def format_number(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.6g}"


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------

class Cache:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.data: dict[str, dict] = {"inbound_links": {}}
        self._dirty = False
        if enabled and path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data["inbound_links"].update(loaded.get("inbound_links", {}))
            except (OSError, ValueError) as exc:
                print(f"  ! ignoring unreadable cache {path}: {exc}")

    def get_inbound(self, qid: str) -> int | None:
        return self.data["inbound_links"].get(qid)

    def set_inbound(self, qid: str, count: int) -> None:
        self.data["inbound_links"][qid] = count
        self._dirty = True

    def save(self, force: bool = False) -> None:
        if not self.enabled or (not self._dirty and not force):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError as exc:
            print(f"  ! could not write cache {self.path}: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    try:
        # utf-8: place and class labels are not ASCII and the Windows console
        # codepage is not either. line_buffering: this run takes minutes, so
        # progress has to stream even when stdout is redirected to a file.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:  # pragma: no cover - older/odd stdout objects
        pass

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Verify data/01_places_final.csv against Wikidata.")
    parser.add_argument("--csv", default=str(root / "data" / "01_places_final.csv"), help="input CSV (read only)")
    parser.add_argument("--out-dir", default=str(root / "data"), help="where reports and cache go")
    parser.add_argument("--limit", type=int, default=0, help="check only the first N rows (smoke test)")
    parser.add_argument("--seed", type=int, default=None, help="seed for the random spot-check sample")
    parser.add_argument("--no-cache", action="store_true", help="ignore and do not write the inbound-link cache")
    parser.add_argument(
        "--strict-sitelinks",
        action="store_true",
        help="report every difference from wikibase:sitelinks, even when the file matches the Wikipedia-only count",
    )
    parser.add_argument("--skip-inbound", action="store_true", help="skip the slow inbound-link check")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    report_path = out_dir / "verify_report.csv"
    types_path = out_dir / "verify_types.csv"
    cache_path = out_dir / "verify_cache.json"

    # Hard guard: this script must never write over its own input.
    for output in (report_path, types_path, cache_path):
        if output == csv_path:
            print(f"refusing to write output over the input CSV: {csv_path}")
            return 2

    if not csv_path.exists():
        print(f"input not found: {csv_path}")
        return 2

    with csv_path.open(encoding="utf-8-sig", newline="") as handle:  # BOM
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"no rows in {csv_path}")
        return 2

    columns = list(rows[0].keys())
    sitelink_column = next((c for c in SITELINK_COLUMNS if c in columns), None)
    if sitelink_column is None:
        print(f"no sitelink column found; looked for {SITELINK_COLUMNS} in {columns}")
        return 2

    all_rows = rows
    if args.limit:
        rows = rows[: args.limit]

    print(f"input      : {csv_path}")
    print(f"rows       : {len(rows)}" + (f" (of {len(all_rows)}, --limit)" if args.limit else ""))
    print(f"sitelinks  : comparing wikibase:sitelinks against column '{sitelink_column}'")
    print(f"outputs    : {report_path}, {types_path}")
    print()

    qids = [row["wikidata_id"].strip() for row in rows]
    unique_qids = sorted({q for q in qids if q})

    disagreements: list[dict] = []
    failures: list[dict] = []

    def disagree(row: dict, check: str, file_value, live_value) -> None:
        disagreements.append(
            {
                "wikidata_id": row.get("wikidata_id", ""),
                "place_name": row.get("place_name", ""),
                "check": check,
                "file_value": "" if file_value is None else str(file_value),
                "live_value": "" if live_value is None else str(live_value),
            }
        )

    def fail(row: dict, check: str, file_value, error) -> None:
        failures.append(
            {
                "wikidata_id": row.get("wikidata_id", ""),
                "place_name": row.get("place_name", ""),
                "check": f"{check}:query_failed",
                "file_value": "" if file_value is None else str(file_value),
                "live_value": str(error),
            }
        )

    # ---- duplicates (local, no queries) ---------------------------------
    print("[1/6] duplicates")
    for column, check in (("wikidata_id", "duplicate_wikidata_id"), ("place_name", "duplicate_place_name")):
        seen = defaultdict(list)
        for index, row in enumerate(rows, start=2):  # +2: header is line 1
            seen[row.get(column, "").strip()].append(index)
        for value, lines in seen.items():
            if value and len(lines) > 1:
                first = next(r for r in rows if r.get(column, "").strip() == value)
                disagree(
                    first,
                    check,
                    value,
                    f"{len(lines)} rows (csv lines {', '.join(map(str, lines))})",
                )
    print(f"      {sum(1 for d in disagreements if d['check'].startswith('duplicate'))} duplicate(s)")

    # ---- core batch: resolves / sitelinks / country / coords / area ------
    print(f"[2/6] core batch ({len(unique_qids)} items, {BATCH_SIZE} per query)")
    core: dict[str, dict] = {}
    core_errors: dict[str, str] = {}
    for index, batch in enumerate(chunks(unique_qids, BATCH_SIZE), start=1):
        try:
            core.update(query_core(batch))
        except QueryError as exc:
            for qid in batch:
                core_errors[qid] = str(exc)
            print(f"      batch {index}: FAILED ({exc})")
        else:
            print(f"      batch {index}: ok ({len(batch)} items)")
        time.sleep(SLEEP_BETWEEN)

    # Wikipedia-only sitelink counts, purely as context for rows where the total
    # disagrees. Queried lazily so a clean dataset costs nothing extra.
    mismatched_sitelinks = sorted(
        {
            row["wikidata_id"].strip()
            for row in rows
            if (record := core.get(row["wikidata_id"].strip()))
            and record["sitelinks"] is not None
            and parse_int(row.get(sitelink_column, "")) != record["sitelinks"]
        }
    )
    wikipedia_counts: dict[str, int] = {}
    wikipedia_available = bool(mismatched_sitelinks)
    if mismatched_sitelinks:
        print(f"[3/6] wikipedia-only sitelink counts ({len(mismatched_sitelinks)} mismatching items)")
        for index, batch in enumerate(chunks(mismatched_sitelinks, SITELINK_BATCH_SIZE), start=1):
            try:
                wikipedia_counts.update(query_wikipedia_sitelinks(batch))
            except QueryError as exc:
                wikipedia_available = False
                print(f"      batch {index}: FAILED ({exc}) - falling back to total sitelinks only")
            time.sleep(SLEEP_BETWEEN)
        # Items with no Wikipedia article return no group, which means zero.
        if wikipedia_available:
            for qid in mismatched_sitelinks:
                wikipedia_counts.setdefault(qid, 0)
    else:
        print("[3/6] wikipedia-only sitelink counts: not needed (all sitelink counts match)")

    # ---- country labels --------------------------------------------------
    country_qids = sorted({c for record in core.values() for c in record["countries"]})
    country_names: dict[str, dict] = {}
    country_name_error: str | None = None
    if country_qids:
        for batch in chunks(country_qids, BATCH_SIZE):
            try:
                country_names.update(query_country_names(batch))
            except QueryError as exc:
                country_name_error = str(exc)
            time.sleep(SLEEP_BETWEEN)

    # ---- name sharing ----------------------------------------------------
    names = sorted({row["place_name"] for row in rows if row.get("place_name")})
    print(f"[4/6] name sharing ({len(names)} labels, {LABEL_BATCH_SIZE} per query)")
    label_counts: dict[str, int] = {}
    label_errors: dict[str, str] = {}
    for index, batch in enumerate(chunks(names, LABEL_BATCH_SIZE), start=1):
        try:
            label_counts.update(query_label_counts(batch))
        except QueryError as exc:
            # Retry one at a time: one pathological label should not blind the rest.
            print(f"      batch {index}: FAILED ({exc}) - retrying individually")
            for name in batch:
                try:
                    label_counts.update(query_label_counts([name]))
                except QueryError as inner:
                    label_errors[name] = str(inner)
                time.sleep(SLEEP_BETWEEN)
        time.sleep(SLEEP_BETWEEN)

    # ---- P31 labels ------------------------------------------------------
    print(f"[5/6] P31 labels ({BATCH_SIZE} per query)")
    p31: dict[str, list[str]] = {}
    p31_errors: dict[str, str] = {}
    for index, batch in enumerate(chunks(unique_qids, BATCH_SIZE), start=1):
        try:
            p31.update(query_p31_labels(batch))
        except QueryError as exc:
            for qid in batch:
                p31_errors[qid] = str(exc)
            print(f"      batch {index}: FAILED ({exc})")
        time.sleep(SLEEP_BETWEEN)

    # ---- inbound links (slow, cached, per item) --------------------------
    cache = Cache(cache_path, enabled=not args.no_cache)
    inbound: dict[str, int] = {}
    inbound_errors: dict[str, str] = {}
    if args.skip_inbound:
        print("[6/6] inbound links: skipped (--skip-inbound)")
    else:
        todo = [q for q in unique_qids if cache.get_inbound(q) is None]
        cached = len(unique_qids) - len(todo)
        print(f"[6/6] inbound links ({len(unique_qids)} items, {cached} from cache)")
        interactive = sys.stdout.isatty()
        try:
            for position, qid in enumerate(unique_qids, start=1):
                hit = cache.get_inbound(qid)
                if hit is not None:
                    inbound[qid] = hit
                    continue
                record = core.get(qid)
                if record is not None and not record["exists"]:
                    continue  # nothing to count for an item that does not resolve
                label = f"      [{position:>3}/{len(unique_qids)}] {qid}"
                if interactive:
                    print(f"{label} ...", end="\r", flush=True)
                try:
                    count = query_inbound_links(qid)
                except QueryError as exc:
                    inbound_errors[qid] = str(exc)  # not cached: retry on re-run
                    print(f"{label} FAILED ({exc})")
                else:
                    inbound[qid] = count
                    cache.set_inbound(qid, count)
                    if interactive:
                        print(f"{label} {count}      ", end="\r", flush=True)
                    elif position % 10 == 0:
                        print(f"{label} {count}", flush=True)
                if position % 5 == 0:
                    cache.save()
                time.sleep(SLEEP_BETWEEN)
        except KeyboardInterrupt:
            cache.save(force=True)
            print("\n      interrupted; cache saved, re-run to resume")
            raise
        finally:
            cache.save(force=True)
        if interactive:
            print(" " * 60, end="\r")
        print(f"      done ({len(inbound)} counted, {len(inbound_errors)} failed)")

    # ---- compare ---------------------------------------------------------
    type_rows: list[dict] = []
    sitelinks_wikipedia_only = 0  # file matches the Wikipedia-only count, not the total
    unresolved: set[str] = set()

    for row in rows:
        qid = row["wikidata_id"].strip()
        name = row.get("place_name", "")

        # resolves
        if qid in core_errors:
            fail(row, "resolves", qid, core_errors[qid])
            unresolved.add(qid)
        else:
            record = core.get(qid)
            if record is None or not record["exists"]:
                disagree(row, "resolves", qid, "no entity found")
                unresolved.add(qid)
            elif record["redirect"]:
                disagree(row, "resolves", qid, f"redirects to {record['redirect']}")
                unresolved.add(qid)

        # name sharing (independent of whether the QID resolves)
        file_shared = parse_int(row.get("places_sharing_name", ""))
        if name in label_errors:
            fail(row, "name_sharing", row.get("places_sharing_name", ""), label_errors[name])
        elif name in label_counts:
            live_shared = label_counts[name]
            if file_shared is None:
                disagree(row, "name_sharing", row.get("places_sharing_name", ""), live_shared)
            elif file_shared != live_shared:
                disagree(row, "name_sharing", file_shared, live_shared)

        # P31 labels for manual review
        classes = p31.get(qid, [])
        type_rows.append(
            {
                "place_name": name,
                "assigned_type": row.get("type", ""),
                "p31_labels": " | ".join(classes)
                if classes
                else ("QUERY FAILED" if qid in p31_errors else "(none)"),
            }
        )
        if qid in p31_errors:
            fail(row, "p31_labels", row.get("type", ""), p31_errors[qid])

        if qid in unresolved:
            continue  # everything below would just echo the unresolved item

        record = core[qid]

        # sitelinks
        file_sitelinks = parse_int(row.get(sitelink_column, ""))
        live_sitelinks = record["sitelinks"]
        if live_sitelinks is None:
            fail(row, "sitelinks", row.get(sitelink_column, ""), "wikibase:sitelinks missing")
        elif file_sitelinks != live_sitelinks:
            wikipedia_only = wikipedia_counts.get(qid) if wikipedia_available else None
            if (
                not args.strict_sitelinks
                and wikipedia_only is not None
                and file_sitelinks == wikipedia_only
            ):
                # Definitional, not a data error: the column counts Wikipedia
                # editions, wikibase:sitelinks counts every project. Summarised
                # once instead of once per row.
                sitelinks_wikipedia_only += 1
            else:
                live = str(live_sitelinks)
                if wikipedia_only is not None:
                    live += f" (wikipedia-only: {wikipedia_only})"
                disagree(row, "sitelinks", row.get(sitelink_column, ""), live)

        # inbound links
        file_links = parse_int(row.get("wikidata_links", ""))
        if qid in inbound_errors:
            fail(row, "inbound_links", row.get("wikidata_links", ""), inbound_errors[qid])
        elif qid in inbound:
            if file_links != inbound[qid]:
                disagree(row, "inbound_links", row.get("wikidata_links", ""), inbound[qid])

        # country
        file_country = (row.get("country", "") or "").strip()
        live_countries = sorted(record["countries"])
        if not live_countries:
            disagree(row, "country", file_country, "no P17")
        else:
            display, accepted = [], set()
            for country_qid in live_countries:
                info = country_names.get(country_qid)
                if not info or not info["label"]:
                    display.append(country_qid)
                    if country_name_error:
                        continue
                else:
                    display.append(f"{info['label']} ({country_qid})")
                    accepted.add(normalise_name(info["label"]))
                    accepted.update(normalise_name(a) for a in info["aliases"])
            if not accepted and country_name_error:
                fail(row, "country", file_country, country_name_error)
            elif normalise_name(file_country) not in accepted:
                disagree(row, "country", file_country, ", ".join(display))

        # coordinates
        if not record["has_coord"]:
            disagree(row, "coordinates", "expected P625", "no P625")

        # area
        file_area = parse_float(row.get("area_km2", ""))
        live_areas_km2 = sorted(a / 1_000_000 for a in record["areas"])
        if file_area is None:
            if live_areas_km2:
                disagree(
                    row,
                    "area",
                    "(blank)",
                    " / ".join(format_number(a) for a in live_areas_km2) + " km2",
                )
        elif not live_areas_km2:
            disagree(row, "area", row.get("area_km2", ""), "no P2046")
        elif not any(within_tolerance(file_area, live) for live in live_areas_km2):
            disagree(
                row,
                "area",
                row.get("area_km2", ""),
                " / ".join(format_number(a) for a in live_areas_km2) + " km2",
            )

    # ---- write reports ---------------------------------------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    report_rows = disagreements + failures
    with report_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["wikidata_id", "place_name", "check", "file_value", "live_value"]
        )
        writer.writeheader()
        writer.writerows(report_rows)

    with types_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["place_name", "assigned_type", "p31_labels"])
        writer.writeheader()
        writer.writerows(type_rows)

    # ---- type sanity, printed for scanning -------------------------------
    print()
    print("=" * 78)
    print("TYPE SANITY - assigned type vs P31 (human judgement, not auto-validated)")
    print("=" * 78)
    for entry in sorted(type_rows, key=lambda e: (e["assigned_type"], e["place_name"])):
        print(f"{entry['place_name'][:28]:<28} {entry['assigned_type']:<18} | {entry['p31_labels']}")

    # ---- summary ---------------------------------------------------------
    by_check = Counter(d["check"] for d in disagreements)
    by_failure = Counter(f["check"] for f in failures)

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"rows checked        : {len(rows)}")
    print(f"distinct QIDs       : {len(unique_qids)}")
    print(f"disagreements       : {len(disagreements)}")
    print(f"query failures      : {len(failures)}")
    print()
    if by_check:
        print("disagreements by check:")
        for check, count in sorted(by_check.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {check:<24} {count}")
    else:
        print("disagreements by check: none")
    if by_failure:
        print()
        print("query failures by check (not counted as mismatches):")
        for check, count in sorted(by_failure.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {check:<24} {count}")
    if sitelinks_wikipedia_only:
        print()
        print(
            f"note: {sitelinks_wikipedia_only} row(s) differ from wikibase:sitelinks but match the\n"
            f"      Wikipedia-only sitelink count exactly, i.e. column '{sitelink_column}' counts\n"
            f"      Wikipedia editions while wikibase:sitelinks counts every project. Not reported\n"
            f"      as disagreements; use --strict-sitelinks to report them anyway."
        )
    print()
    print(f"report              : {report_path} ({len(report_rows)} rows)")
    print(f"types               : {types_path} ({len(type_rows)} rows)")
    if not args.no_cache and not args.skip_inbound:
        print(f"cache               : {cache_path}")
    print()
    result = "FAIL" if disagreements else "PASS"
    if failures:
        # A failed query verified nothing, so a clean PASS would be a lie.
        result += f" (inconclusive: {len(failures)} check(s) could not be queried)"
    print(f"RESULT: {result}")

    # ---- random spot check ----------------------------------------------
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(10, len(rows)))
    print()
    print("=" * 78)
    print("SPOT CHECK - open these and confirm label, country and type match")
    print("=" * 78)
    for entry in sample:
        print(
            f"{entry.get('place_name', ''):<28} {entry.get('type', ''):<18} "
            f"{entry.get('country', ''):<24} "
            f"https://www.wikidata.org/wiki/{entry.get('wikidata_id', '').strip()}"
        )

    return 1 if disagreements else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted")
        sys.exit(130)
