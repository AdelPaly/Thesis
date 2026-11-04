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
import re
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
        "evidence": {
            "type": "array",
            "description": "One entry per planned name from the list (not the distractor, not the place "
                           "name): the exact words copied from the post that say where the name belongs "
                           "(its link: 'in his home town', 'outside their offices', 'at their ground'), and "
                           "the exact words that show the environmental point coming out of it ('the crowd "
                           "left rubbish everywhere after the gig'). An empty string when the post has no "
                           "such words. Empty array when there are no planned names.",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "link_words": {"type": "string"},
                               "connection_words": {"type": "string"}},
                "required": ["name", "link_words", "connection_words"],
                "additionalProperties": False,
            },
        },
        "claim_follows_from_names": {
            "type": "boolean",
            "description": "The environmental point comes out of what the planned names (not the distractor) "
                           "are doing where they belong, said through their link (their home town, their "
                           "offices, their ground): caused by their event, site or work, or happening there "
                           "because of it. False when a name and the point are only set side by side, or a "
                           "name is tied by 'here' alone. True when there are no planned names.",
        },
        "claim_tied_to_place": {
            "type": "boolean",
            "description": "The environmental claim is clearly about the place the post is set in, the place "
                           "the listed names belong to (seen there or reported from there). False when the "
                           "claim is general, about somewhere else, or only sits next to the names.",
        },
        "listed_fragments": {
            "type": "boolean",
            "description": "The post sets unrelated pieces side by side (a name, then a separate claim, then "
                           "another detail) instead of making one connected statement.",
        },
        "media_untied": {
            "type": "boolean",
            "description": "Something is brought in through a screen, song, article, broadcast or match, and "
                           "the environmental claim is not tied to the place because of it.",
        },
        "misused_names": {
            "type": "array", "items": {"type": "string"},
            "description": "Listed names used as something they are not: a past event written as happening "
                           "now, as a present venue or as the cause of today's problem; an organisation, "
                           "building or thing doing what it could not do. Empty if none.",
        },
        "invented_person_claims": {
            "type": "array", "items": {"type": "string"},
            "description": "Listed real people to whom the post gives something that does not fit who they "
                           "are and their link to the place (as given in the list): a role they do not have, a "
                           "quote, a specific wrongdoing, or, for a person from history, anything they do, back "
                           "or notice today. Empty if none.",
        },
        "on_theme": {
            "type": "boolean",
            "description": "The post is about the topic the writer was given (not a different environmental "
                           "subject). True when no topic was given.",
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
                 "evidence", "claim_follows_from_names", "claim_tied_to_place", "listed_fragments", "media_untied", "misused_names", "invented_person_claims",
                 "on_theme", "environmental_claim", "coherent", "reason"],
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
# 2026-11-02 (user, after smoke13): invented claims about real people, names
# used against their type, a claim not tied to the place, listed fragments and
# untied media framing fail as well. A post off the planned topic fails as
# judge_off_theme, reported apart from judge_no_environment (no claim at all).
ORDER = ("judge_place", "judge_person", "judge_setting", "judge_invented_person", "judge_misused",
         "judge_untied_claim", "judge_unconnected", "judge_fragments", "judge_media", "judge_imagined", "judge_decorative",
         "judge_unanchored", "judge_no_environment", "judge_off_theme", "judge_incoherent")


def allowed_names(item: dict[str, Any]) -> list[str]:
    """Every name the post was asked or allowed to contain."""
    names = [c["clue_label"] for c in item.get("clues") or []]
    if item.get("noise_entity"):
        names.append(item["noise_entity"]["clue_label"])
    surface = (item.get("surface_form") or {}).get("text")
    if surface:
        names.append(surface)
    return names


def relation(clue: dict[str, Any], config: dict[str, Any] | None) -> str:
    """The clue's link to the place in the words the writer was given."""
    phrases = ((config or {}).get("prompt") or {}).get("relation_phrases") or {}
    return phrases.get(clue.get("property")) or clue.get("property_label") or "linked to it"


def describe(item: dict[str, Any], name: str, config: dict[str, Any] | None = None) -> str:
    """'- "name"' with what it is: for a person what they are known for, for
    anything else its type; the distractor and the place name get a note."""
    for c in item.get("clues") or []:
        if c["clue_label"] == name:
            if c.get("is_human"):
                era = {True: "a living person", False: "a person from history, not alive today"}.get(
                    c.get("is_living"), "a real person")
                return (f'- "{name}" ({era}, known as: {c.get("description") or "a person"}; '
                        f'link to the place: {relation(c, config)})')
            kind = {"place_or_org": "a place or organisation", "work_or_event": "a work or event",
                    "species": "a plant or animal species",
                    "media": "a newspaper or broadcaster"}.get(c.get("entity_type"), "")
            return f'- "{name}"' + (f" ({kind}; " if kind else " (") + f"link to the place: {relation(c, config)})"
    if (item.get("noise_entity") or {}).get("clue_label") == name:
        return f'- "{name}" (belongs to a different place; it need not be tied to where the post is set)'
    return f'- "{name}" (the place name)'


def judge_messages(item: dict[str, Any], text: str, config: dict[str, Any]) -> list[dict[str, str]]:
    cfg = config["judge"]
    names = allowed_names(item)
    listed = "\n".join(describe(item, n, config) for n in names) or "- (none: the post should name no person or place)"
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


BARE_LINK = {"here", "there", "around here", "near here", "nearby", "in town", "back", "locally", "local"}


def missing_evidence(item: dict[str, Any], answer: dict[str, Any], text: str) -> list[str]:
    """Planned names whose link or connection the judge could not quote from the
    post: no entry, empty words, words that are not in the post, or a link that
    is only "here" (2026-11-02, user: "say gig in his hometown"). The quotes are
    checked against the text, so the judge cannot pass a post on a tie it
    imagined."""
    folded = " ".join(_norm(text).split())
    given = {_norm(e.get("name") or ""): e for e in answer.get("evidence") or []}
    out = []
    for c in item.get("clues") or []:
        e = given.get(_norm(c["clue_label"]))
        if e is None:
            out.append(f'{c["clue_label"]}: no evidence')
            continue
        for key, what in (("link_words", "link"), ("connection_words", "connection")):
            words = " ".join(_norm(e.get(key) or "").strip(" .,!?").split())
            if not words or words in BARE_LINK:
                out.append(f'{c["clue_label"]}: no {what} stated')
                break
            # the judge shortens long quotes with "..."; each piece must be in the post, in order
            pieces = [w.strip(" .,") for w in re.split(r"\.\.\.|…", words) if w.strip(" .,")]
            at = 0
            for piece in pieces:
                at = folded.find(piece, at)
                if at < 0:
                    break
            if not pieces or at < 0:
                out.append(f'{c["clue_label"]}: {what} "{e.get(key)}" not in the post')
                break
    return out


def verdict(item: dict[str, Any], answer: dict[str, Any], text: str | None = None) -> tuple[str, str] | None:
    """(status, reason) when the judge's answers fail the post, None when it passes.
    Names the judge lists that are in fact allowed (it sometimes lists the planned
    name) are ignored."""
    allowed = {_norm(n) for n in allowed_names(item)}
    # A proper name has a capital letter: the judge sometimes lists "community
    # field" or "the market" as a place name.
    extra = lambda key: [n for n in answer.get(key) or []  # noqa: E731
                         if _norm(n) not in allowed and n.strip() and any(ch.isupper() for ch in n)]
    # The distractor is exempt: its job is to be there without belonging (2026-11-02: Qwen failed "a kid
    # holding a Paul the Apostle banner in the crowd" as decorative). Listed fragments still fail.
    noise_name = _norm((item.get("noise_entity") or {}).get("clue_label") or "")
    decorative = [n for n in answer.get("decorative_names") or [] if _norm(n) in allowed and _norm(n) != noise_name]
    clue_names = {_norm(c["clue_label"]) for c in item.get("clues") or []}
    unanchored = [n for n in answer.get("unanchored_names") or [] if _norm(n) in clue_names]
    checks = {
        "judge_place": extra("unlisted_places"),
        "judge_person": extra("unlisted_people"),
        "judge_setting": [n for n in answer.get("invented_setting") or [] if n.strip()],
        "judge_invented_person": [n for n in answer.get("invented_person_claims") or [] if _norm(n) in clue_names],
        "judge_misused": [n for n in answer.get("misused_names") or [] if _norm(n) in clue_names],
        "judge_untied_claim": answer.get("claim_tied_to_place") is False,
        "judge_unconnected": (bool(item.get("clues")) and answer.get("claim_follows_from_names") is False)
                             or (missing_evidence(item, answer, text) if text is not None else []),
        "judge_fragments": answer.get("listed_fragments") is True,
        "judge_media": answer.get("media_untied") is True,
        "judge_off_theme": bool(item.get("env_theme")) and answer.get("on_theme") is False,
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
        v = verdict(item, answer, text)
        return {"answer": answer, "verdict": list(v) if v else None, "error": None,
                "usage": (metadata or {}).get("usage")}
