#!/usr/bin/env python3
"""
The model check on a generated post (Task 5).

generate.py's rule checks (finalise.check) catch what a string match can:
the place or country named, a clue or the planned name missing, the length.
What they cannot judge is whether the post reads like a person wrote it, so a
second model reads every post that passes them and answers a fixed set of
yes/no questions (JUDGE_SCHEMA). A different model family from the writer
(config judge:), so the checker does not share the writer's blind spots.

The judge gets the post and the names the post was asked to contain, nothing
about the place it is set in: it judges the text, not the geography.

verdict() turns the answers into a drop reason in a fixed order, the same way
finalise.check does, so a post gets one reason even when several apply.
"judge_decorative" is the one generate.py can act on: it swaps the named clue
for a spare one on the next try.
"""
from __future__ import annotations

import copy
import json
from typing import Any

import generate as gen

JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decorative_names": {
            "type": "array", "items": {"type": "string"},
            "description": "Names from the list that are only dropped in and take no part in "
                           "what happens or in the point of the post. Empty if all take part.",
        },
        "unanchored_names": {
            "type": "array", "items": {"type": "string"},
            "description": "Names from the list that the post does not tie to where it is set: the "
                           "writer is not at it or right by it, and it is not shown acting on its own "
                           "home area or community. A name whose part could happen anywhere in its "
                           "country, the writer's own neighbourhood included, is not tied. Empty if "
                           "all are tied.",
        },
        "invented_setting": {
            "type": "array", "items": {"type": "string"},
            "description": "Details of the setting that could help locate it and that come neither "
                           "from the names nor from the topic: a landscape (wetlands, valley, coast, "
                           "mountains), a facility (concrete plant, port, mine), a climate or season "
                           "detail. Ordinary things (a road, a street, traffic, dust, a building site) "
                           "do not count. Empty if none.",
        },
        "imagined_reaction": {
            "type": "boolean",
            "description": "A named person is used as an imagined judge of the situation "
                           "(would hate this, would not recognise it, what would they say).",
        },
        "unlisted_people": {
            "type": "array", "items": {"type": "string"},
            "description": "Names of people in the post that are not on the list.",
        },
        "unlisted_places": {
            "type": "array", "items": {"type": "string"},
            "description": "Place names in the post that are not on the list, including streets, "
                           "squares or buildings named after someone.",
        },
        "environmental_claim": {
            "type": "boolean",
            "description": "The post makes a claim about an environmental problem, improvement or change.",
        },
        "coherent": {
            "type": "boolean",
            "description": "Each sentence follows from the one before; no unexplained jump.",
        },
        "reason": {"type": "string", "description": "One short sentence on the main problem, or ok."},
    },
    "required": ["decorative_names", "unanchored_names", "invented_setting", "imagined_reaction", "unlisted_people", "unlisted_places",
                 "environmental_claim", "coherent", "reason"],
    "additionalProperties": False,
}

# Drop reasons in the order they are checked: the first that applies is the one
# recorded. Names outside the list come first: they can leak the location.
# judge_unanchored (2026-10-30, user): a clue only mentioned in passing points
# to its country, not to the place ("thanks Party of the Revolution" could be
# anywhere in Tanzania). Only planned clues count: the place name needs no
# anchor, and the distractor belongs to another place.
# judge_setting (same day, user): an invented landscape or facility ("that
# concrete plant", "the wetlands") is a clue nobody planned; a system can use
# it, or be misled by it, and the gold label assumes the planned clues are the
# only locating signal. It ranks with the leaks.
ORDER = ("judge_place", "judge_person", "judge_setting", "judge_imagined", "judge_decorative",
         "judge_unanchored", "judge_no_environment", "judge_incoherent")


def allowed_names(item: dict[str, Any]) -> list[str]:
    """Every name the post was asked or allowed to contain."""
    names = [c["clue_label"] for c in item.get("clues") or []]
    if item.get("noise_entity"):
        names.append(item["noise_entity"]["clue_label"])
    surface = (item.get("surface_form") or {}).get("text")
    if surface:
        names.append(surface)
    return names


def judge_messages(item: dict[str, Any], text: str, config: dict[str, Any]) -> list[dict[str, str]]:
    cfg = config["judge"]
    names = allowed_names(item)
    listed = "\n".join(f'- "{n}"' for n in names) or "- (none: the post should name no person or place)"
    fields = "\n".join(f"  {k}: {v['description']}" for k, v in JUDGE_SCHEMA["properties"].items())
    topic = f'\n\nTopic the writer was given: {item["env_theme"]}' if item.get("env_theme") else ""
    user = (cfg["task"].strip() + topic + "\n\nNames the post was asked to contain:\n" + listed
            + "\n\nThe post:\n\"\"\"\n" + text.strip() + "\n\"\"\"\n\n"
            + "Reply with one JSON object with these fields:\n" + fields)
    return [{"role": "system", "content": cfg["system"].strip()},
            {"role": "user", "content": user}]


def parse(raw: str) -> tuple[dict[str, Any] | None, str | None]:
    candidate = gen.extract_json_object(raw or "")
    if candidate is None:
        return None, "no JSON object in the judge's reply"
    try:
        answer = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return None, f"judge reply not valid JSON: {exc}"
    missing = [k for k in JUDGE_SCHEMA["required"] if k not in answer]
    if missing:
        return None, f"judge reply missing {missing}"
    return answer, None


def _norm(name: str) -> str:
    return " ".join(name.lower().replace('"', "").split())


def verdict(item: dict[str, Any], answer: dict[str, Any]) -> tuple[str, str] | None:
    """(status, reason) when the judge's answers fail the post, None when it passes.
    Names the judge lists that are in fact allowed (it sometimes lists the planned
    name) are ignored."""
    allowed = {_norm(n) for n in allowed_names(item)}
    # A proper name has a capital letter: the judge sometimes lists "community
    # field" or "the market" as a place name.
    extra = lambda key: [n for n in answer.get(key) or []  # noqa: E731
                         if _norm(n) not in allowed and n.strip() and any(ch.isupper() for ch in n)]
    decorative = [n for n in answer.get("decorative_names") or [] if _norm(n) in allowed]
    clue_names = {_norm(c["clue_label"]) for c in item.get("clues") or []}
    unanchored = [n for n in answer.get("unanchored_names") or [] if _norm(n) in clue_names]
    checks = {
        "judge_place": extra("unlisted_places"),
        "judge_person": extra("unlisted_people"),
        "judge_setting": [n for n in answer.get("invented_setting") or [] if n.strip()],
        "judge_imagined": answer.get("imagined_reaction") is True,
        "judge_decorative": decorative,
        "judge_unanchored": unanchored,
        "judge_no_environment": answer.get("environmental_claim") is False,
        "judge_incoherent": answer.get("coherent") is False,
    }
    for status in ORDER:
        hit = checks[status]
        if hit:
            detail = ", ".join(f'"{n}"' for n in hit) if isinstance(hit, list) else ""
            return status, (detail + " - " if detail else "") + str(answer.get("reason") or "")[:200]
    return None


class Judge:
    """One client for the judge model (config judge:, an OpenAI-compatible endpoint)."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = copy.deepcopy(config)
        self.config["generation"] = {**config["generation"], **config["judge"]["model_settings"]}
        self.client = gen.ChatClient(self.config)
        self.client.structured_output = "json_schema"
        self.model = self.client.model

    def check(self, item: dict[str, Any], text: str) -> dict[str, Any]:
        """{"answer", "verdict", "error", "usage"}; a failed call is recorded, not raised."""
        messages = judge_messages(item, text, self.config)
        try:
            content, metadata, _ = gen.call_with_backoff(
                self.client, messages, self.config, seed=None,
                schema=(JUDGE_SCHEMA, "post_check"))
        except gen.ApiError as exc:
            return {"answer": None, "verdict": ["judge_error", str(exc)[:200]], "error": str(exc)}
        answer, error = parse(content)
        if error:
            return {"answer": None, "verdict": ["judge_error", error], "error": error,
                    "raw": (content or "")[:500]}
        v = verdict(item, answer)
        return {"answer": answer, "verdict": list(v) if v else None, "error": None,
                "usage": (metadata or {}).get("usage")}
