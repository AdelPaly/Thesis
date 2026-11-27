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


V4_PARAGRAPH = ("The location can be a city, town or village, or any other named place. "
                "Name the location itself, not a person, organisation or event mentioned "
                "in the text.")
V5_PARAGRAPH = ("The location can be a city, town or village, or any other named place. "
                "If the text is about a building, street or venue, name the city, town or "
                "village it is in. If it is about a person, organisation or event, name the "
                "place where they are.")


def test_v5_is_v4_with_the_second_paragraph_replaced():
    """The first v5 (results/dev_rev3b/) had a "settlement" field; the v5 that
    replaced it keeps v4's JSON fields and changes only the second paragraph."""
    assert V5_PARAGRAPH in llm_geo.PROMPT_V5
    assert llm_geo.PROMPT_V5.split("\n\n")[1] == V5_PARAGRAPH
    assert llm_geo.PROMPT_V5.replace(V5_PARAGRAPH, V4_PARAGRAPH) == llm_geo.PROMPT_V4


def test_v5_has_no_settlement_field():
    assert "settlement" not in llm_geo.PROMPT_V5
    reply_format = next(line for line in llm_geo.PROMPT_V5.format(text="").split("\n")
                        if line.startswith("{"))
    assert [k for k in ("place", "country", "continent", "latitude", "longitude", "refused")
            if f'"{k}"' in reply_format] == ["place", "country", "continent", "latitude",
                                             "longitude", "refused"]


def test_v5_names_no_dataset_place_type():
    for word in ("island", "lake", "mountain", "park", "dam", "nearby town", "river",
                 "airport", "station", "museum", "church", "bridge", "beach", "forest",
                 "valley", "volcano", "glacier", "region"):
        assert word not in llm_geo.PROMPT_V5.lower()


def test_v5_has_its_own_fingerprint():
    assert llm_geo.PROMPTS["v5"] is llm_geo.PROMPT_V5
    assert fingerprint(llm_geo.PROMPTS["v5"], {}) != fingerprint(llm_geo.PROMPTS["v4"], {})
    assert "v5" in config.PROMPT_TEMPLATES


def test_rule_v5_is_kept():
    # Rule v5 stays in the code (tests/test_resolver.py tests it); the
    # configurations resolve prompt v5 with rule v4.
    assert llm_geo.RESOLVERS["v5"].__name__ == "resolve_place_settlement"


def test_earlier_prompts_are_unchanged():
    assert llm_geo.PROMPTS["v3"] is llm_geo.PROMPT_V2
    assert llm_geo.RESOLVERS["v4"].__name__ == "resolve_place_wide"
    assert "settlement" not in llm_geo.PROMPT_V4


def test_llm_only_configs_write_to_new_caches():
    """The reported runs used the *_v2 caches and the dev run of revision 3
    the *_v4 ones, the first v5 the *_v5 ones; the v5 that replaced it has
    caches of its own (*_v5b), so PickleCache never sees (and never sets aside
    as .stale) an earlier one. It is resolved with rule v4: rule v5 needs the
    settlement field."""
    for name, reported, v4 in (("gemma4", "gemma4_12b_v2", "gemma4_12b_v4"),
                               ("qwen3_vl", "qwen3_vl_235b_v2", "qwen3_vl_235b_v4"),
                               ("qwen3_vl_32b", None, "qwen3_vl_32b_v4")):
        cfg = config.load_config(ROOT / "configs" / f"{name}.yaml")
        assert cfg.llm.prompt_template == "v5"
        assert cfg.llm.resolver_rule == "v4"
        assert cfg.llm.cache_name not in (reported, v4, v4[:-1] + "5")
        assert cfg.llm.cache_name == v4[:-1] + "5b"


def test_resolver_rule_picks_the_rule_and_defaults_to_the_prompt(tmp_path, monkeypatch):
    import llm_common
    monkeypatch.setattr(llm_common, "CACHE_DIR", str(tmp_path))

    class Client:
        model_id, generation_params = "m", {}

    for template, rule, expected in (("v5", "v4", "resolve_place_wide"),
                                     ("v5", None, "resolve_place_settlement"),
                                     ("v4", None, "resolve_place_wide")):
        p = llm_geo.LLMGeoPipeline(Client(), "test_only", template, rule)
        assert p.prompt is llm_geo.PROMPTS[template]
        assert p.resolve.__name__ == expected


def test_resolver_rule_is_a_late_field():
    """Unset, it leaves every configuration's hash as it was."""
    import dataclasses
    cfg = config.load_config(ROOT / "configs" / "gemma4.yaml")
    unset = dataclasses.replace(cfg, llm=dataclasses.replace(cfg.llm, resolver_rule=None))
    assert "resolver_rule" not in unset.behavioural_dict()["llm"]
    assert cfg.behavioural_dict()["llm"]["resolver_rule"] == "v4"
