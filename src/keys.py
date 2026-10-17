"""Join keys shared by the runner, the pipeline and the dev/test split."""
import hashlib


def item_id(text: str) -> str:
    """sha256 of the text, first 16 chars. No dataset carries an id column and
    row order has been in doubt, so a positional index is not a safe key. The
    text hash is free and matches what the LLM response caches key on."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:16]


# Configurations renamed after prediction files were written under the old
# name. Those files keep their old names and their old `config_name` field;
# every reader maps through current_config_name, so a renamed configuration
# and its old files are one row in every table.
RENAMED_CONFIGS = {
    "baseline_majority": "majority_class",
    "baseline_population": "population_wikidata",
    "two_stage": "retrieve_rerank",
}


def current_config_name(name):
    """The configuration's current name, given an old or a current one."""
    return RENAMED_CONFIGS.get(name, name)


def legacy_config_name(name):
    """The pre-rename name of a renamed configuration, else None."""
    for old, new in RENAMED_CONFIGS.items():
        if new == name:
            return old
    return None


def place_key(qid, location_name) -> str:
    """The split unit. A gold QID when there is one, else the place's name."""
    if isinstance(qid, str) and qid.strip():
        return qid.strip()
    name = str(location_name or "").strip().casefold()
    return f"name:{name}" if name else ""
