"""Join keys shared by the runner, the pipeline and the dev/test split."""
import hashlib


def item_id(text: str) -> str:
    """sha256 of the text, first 16 chars. No dataset carries an id column and
    row order has been in doubt, so a positional index is not a safe key. The
    text hash is free and matches what the LLM response caches key on."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


def place_key(qid, location_name) -> str:
    """The split unit. A gold QID when there is one, else the place's name."""
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    name = str(location_name or "").strip().casefold()
    return f"name:{name}" if name else ""
