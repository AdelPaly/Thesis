"""The single Wikidata Query Service client for the places pipeline.

Everything that talks to WDQS goes through `sparql()`. It owns the TLS context
(verification stays on), retry/backoff with jitter, `Retry-After` handling and
the 4xx fail-fast rule, so fixes land in one place.

The User-Agent is a parameter of `sparql()` with a module default, never a value
assigned onto another module.
"""

from __future__ import annotations

import datetime
import json
import os
import random
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

ENDPOINT = "https://query.wikidata.org/sparql"

# The one place the operator contact lives. WDQS rejects requests without a
# descriptive User-Agent and asks for a way to reach the operator. Set
# WDQS_CONTACT (an email or URL) in your environment to be reachable; it is read
# at run time so no personal address has to be committed. The default is the
# project's public repository.
CONTACT = os.environ.get("WDQS_CONTACT") or "https://github.com/AdelPaly/Thesis"
USER_AGENT = (
    f"places-wdqs/1.0 (place dataset clue collection and verification; "
    f"contact: {CONTACT}) python-urllib"
)

HTTP_TIMEOUT = 180          # seconds per request
ATTEMPTS = 3                # tries per query
BACKOFF_BASE = 2.0          # linear backoff: BACKOFF_BASE * attempt seconds
BACKOFF_JITTER = 1.0        # up to this many extra seconds, random
DEFAULT_RETRY_AFTER = 5.0   # used when Retry-After is absent or not a number
MAX_RETRY_AFTER = 60.0      # never sleep longer than this for a 429
HTTP_RATE_LIMITED = 429
HTTP_SERVER_ERROR = 500
TLS_FAILURE_MARKER = "CERTIFICATE_VERIFY_FAILED"
SERVER_AUTH_OID = "1.3.6.1.5.5.7.3.1"
ERROR_SNIPPET = 200         # characters of an error body kept in messages


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
    """A verifying context built from the OS store minus expired anchors, or None."""
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
            if isinstance(trust, set) and SERVER_AUTH_OID not in trust:
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

def summarise_error(body: str) -> str:
    """Shorten an HTTP error body to something fit for a one-line message."""
    body = re.sub(r"\s+", " ", body).strip()
    if "TimeoutException" in body or "QueryTimeout" in body:
        return "query timeout"
    if "MalformedQuery" in body:
        return "malformed query: " + body[:ERROR_SNIPPET]
    return body[:ERROR_SNIPPET] if body else "no response body"


_summarise_error = summarise_error  # historical name


def _retry_after_seconds(exc: urllib.error.HTTPError) -> float:
    """Seconds to wait for a 429. An HTTP-date or junk value falls back to 5s."""
    try:
        raw = exc.headers.get("Retry-After") if exc.headers is not None else None
        wait = float(raw or DEFAULT_RETRY_AFTER)
    except (ValueError, TypeError):
        wait = DEFAULT_RETRY_AFTER
    return max(0.0, min(wait, MAX_RETRY_AFTER))


def sparql(query: str, *, user_agent: str | None = None, attempts: int = ATTEMPTS,
           timeout: int = HTTP_TIMEOUT) -> list[dict]:
    """POST a query and return the result bindings. Raises QueryError on failure.

    4xx other than 429 fails fast. 429 honours Retry-After (capped). Anything
    else retries with linear backoff plus jitter.
    """
    data = urllib.parse.urlencode({"query": query}).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT,
        data=data,
        method="POST",
        headers={
            "User-Agent": user_agent or USER_AGENT,
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
            last = f"HTTP {exc.code}: {summarise_error(body)}"
            if exc.code == HTTP_RATE_LIMITED:
                time.sleep(_retry_after_seconds(exc))
                continue
            if exc.code < HTTP_SERVER_ERROR:
                break  # 4xx other than rate limiting will not fix itself
        except Exception as exc:  # URLError, timeout, malformed JSON, ...
            last = f"{type(exc).__name__}: {exc}"
            if TLS_FAILURE_MARKER in str(exc) and not tls_retried:
                tls_retried = True
                if _retry_without_expired_anchors():
                    print("      ! TLS verify failed; retrying without expired trust anchors")
                    continue
        if attempt < attempts:
            time.sleep(BACKOFF_BASE * attempt + random.uniform(0, BACKOFF_JITTER))
    raise QueryError(last)


def value_of(binding: dict, key: str) -> str | None:
    """The string value of one cell in a SPARQL result row, or None if unbound."""
    cell = binding.get(key)
    return cell.get("value") if cell else None


def qid_from_uri(uri: str) -> str:
    """'http://www.wikidata.org/entity/Q64' -> 'Q64'."""
    return uri.rsplit("/", 1)[-1]


def escape_literal(text: str) -> str:
    """Escape a string for use inside a double-quoted SPARQL literal."""
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
