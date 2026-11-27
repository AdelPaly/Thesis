"""The LLM-only prompt v4 (src/llm_geo.py), revision 3: its exact text as the
model sees it, and that it writes to caches of its own.

    python -m pytest tests/test_prompt.py

No network and no model.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402
import llm_geo  # noqa: E402
from llm_common import fingerprint  # noqa: E402

EXPECTED_V4 = """You are a geolocation assistant. Given a text, identify the single most relevant real-world location the text refers to.

The location can be a city, town or village, or any other named place. Name the location itself, not a person, organisation or event mentioned in the text.

Respond with JSON only, no explanation:
{"place": "<name of the location, or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "latitude": <approximate latitude of the location in decimal degrees, or null if unknown>, "longitude": <approximate longitude in decimal degrees, or null if unknown>, "refused": false}

Set "refused" to true only if the text contains no geographic signal at all.

Text: {text}"""


def test_v4_reads_exactly_as_specified():
    # format() turns the file's doubled braces into single ones; {text} stays a
    # placeholder by being substituted with itself.
    assert llm_geo.PROMPT_V4.format(text="{text}") == EXPECTED_V4


def test_v4_names_no_dataset_place_type():
    for word in ("island", "lake", "mountain", "park", "dam", "nearby town"):
        assert word not in llm_geo.PROMPT_V4


def test_v4_has_its_own_fingerprint_and_rule():
    assert llm_geo.PROMPTS["v4"] is llm_geo.PROMPT_V4
    assert fingerprint(llm_geo.PROMPTS["v4"], {}) != fingerprint(llm_geo.PROMPTS["v3"], {})
    assert llm_geo.RESOLVERS["v4"].__name__ == "resolve_place_wide"


def test_llm_only_configs_write_to_new_caches():
    """The reported runs used the *_v2 caches; v4 has caches of its own, so
    PickleCache never sees (and never sets aside as .stale) a reported one."""
    for name, reported in (("gemma4", "gemma4_12b_v2"), ("qwen3_vl", "qwen3_vl_235b_v2")):
        cfg = config.load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.llm.prompt_template == "v4"
        assert cfg.llm.cache_name != reported and cfg.llm.cache_name.endswith("_v4")
