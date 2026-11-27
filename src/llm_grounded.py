"""Grounded LLM selection over retrieved candidates (tier 5).

Retrieval-augmented, not routed: candidates are retrieved first, handed to the
model as a numbered list, and the model's job is to SELECT from that list or to
decline. It may not name a place that is not in the list.

Retrieval-augmented generation is the direction adjacent tasks have
taken (Img2Loc for image geolocation, Spatial-RAG for spatial QA); the
text-geolocation equivalent is unexplored. Combining the paradigms is a cleaner
contribution than choosing between them.

Two properties are deliberate:

* **Abstention is a first-class answer.** A system that declines when the right
  place is absent from the candidates is behaving correctly, not failing. Tier
  2's high no-result rate makes abstention a real axis of this comparison, so
  refusal is recorded as refusal rather than folded into error.
* **Selection is validated.** If the model answers with an index out of range,
  or a QID that is not in the list, that is recorded as a violation in the
  record's `error` field. A grounded system that ignores its grounding has to
  be visible, not silently scored as a wrong guess.

The response cache is keyed on the text AND the candidate list, because the
answer depends on both -- changing the retriever must invalidate the cached
answer, not reuse it.
"""
from __future__ import annotations

import hashlib
import json
import re

import meter

from llm_common import (SENTINELS, PickleCache, fingerprint,
                        parse_json_object)

# The prompt of the runs until revision 3, kept for the record. Its last rule
# asked the model to decline when unsure, which the LLM-only prompt v4 no
# longer does; with both the matched comparison (grounded_gemma against
# gemma4) would compare two different rules for declining.
PROMPT_V1 = """You are a geolocation assistant. Identify the single real-world location the text refers to, choosing from the numbered candidate list.

Respond with JSON only, no explanation:
{{"selection": <the number of the chosen candidate, or null>, "refused": <true or false>}}

Rules:
- You may ONLY choose a number that appears in the candidate list.
- Do not name a place that is not in the list.
- If none of the candidates is the location the text refers to, set "selection" to null and "refused" to true.
- Choosing nothing is better than choosing a candidate you do not believe in.

Text: {text}

Candidates:
{candidates}"""

# Revision 3 (2026-11-26): PROMPT_V1 without its last rule. The model declines
# when no candidate is the location, as the LLM-only prompt v4 declines when
# the text has no geographic signal; neither is told to decline when unsure.
PROMPT_VERSION = "v2"
PROMPT = """You are a geolocation assistant. Identify the single real-world location the text refers to, choosing from the numbered candidate list.

Respond with JSON only, no explanation:
{{"selection": <the number of the chosen candidate, or null>, "refused": <true or false>}}

Rules:
- You may ONLY choose a number that appears in the candidate list.
- Do not name a place that is not in the list.
- If none of the candidates is the location the text refers to, set "selection" to null and "refused" to true.

Text: {text}

Candidates:
{candidates}"""


def format_candidates(candidates: list[dict]) -> str:
    """The numbered list the model sees: index, QID, place name, country,
    description.

    The place name is the candidate's `label`. Until revision 3 the list showed
    `entity`, which for a linked candidate is the mention it came from: a
    place reached through a person showed the person's name and never its own
    ("Nick Willis -- city in the Wellington Region of New Zealand"). Dense
    candidates carry the place name in `entity` and have no `label`."""
    lines = []
    for i, cand in enumerate(candidates, start=1):
        label = cand.get("label") or cand.get("entity") or cand.get("qid") or "?"
        qid = cand.get("qid") or "-"
        country = cand.get("country") or "unknown country"
        desc = (cand.get("description") or "").strip()
        line = f"{i}. {qid} — {label} — {country}"
        if desc and desc.lower() != label.lower():
            line += f" — {desc}"
        lines.append(line)
    return "\n".join(lines) if lines else "(no candidates were retrieved)"


class GroundedSelector:
    """Wraps a provider pipeline (llm_geo.LLMGeoPipeline) for the raw call,
    with its own prompt and its own response cache."""

    def __init__(self, provider, cache_name: str):
        self.provider = provider
        params = getattr(provider, "generation_params", {})
        self._cache = PickleCache(cache_name, fingerprint(PROMPT, params))

    def save_cache(self) -> None:
        self._cache.save()

    def _key(self, text: str, candidates: list[dict]) -> str:
        payload = json.dumps({
            "text": text,
            "candidates": [c.get("qid") for c in candidates],
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _get_response(self, text: str, candidates: list[dict]) -> str:
        key = self._key(text, candidates)
        cached = self._cache.get(key)
        if cached:
            meter.add("llm_cache_hits")
            return cached
        prompt = PROMPT.format(text=text, candidates=format_candidates(candidates))
        try:
            raw = self.provider.call_llm(prompt)
        except BaseException:
            self.save_cache()
            raise
        if raw not in SENTINELS:
            self._cache.put(key, raw)
        return raw

    def select(self, text: str, candidates: list[dict]) -> tuple[dict | None, dict, str]:
        """Return (chosen_candidate | None, selection_record, raw_response).

        selection_record is stored on the prediction record so that a wrong
        answer can be attributed to bad retrieval versus bad selection:

            {"index": int | None, "qid": str | None,
             "abstained": bool, "violation": str | None}
        """
        raw = self._get_response(text, candidates)
        parsed = parse_json_object(raw)
        record = {"index": None, "qid": None, "abstained": False, "violation": None}

        if parsed is None:
            record["violation"] = "unparseable response"
            return None, record, raw

        if parsed.get("refused") or parsed.get("selection") is None:
            record["abstained"] = True
            return None, record, raw

        selection = parsed.get("selection")
        if isinstance(selection, str):
            # Some models answer with the QID, or with "3." -- accept a bare
            # integer inside the string, and accept a QID that is in the list.
            stripped = selection.strip()
            by_qid = {c.get("qid"): i for i, c in enumerate(candidates, start=1)}
            if stripped in by_qid:
                selection = by_qid[stripped]
            else:
                digits = re.match(r"\s*(\d+)", stripped)
                selection = int(digits.group(1)) if digits else None

        if isinstance(selection, float) and selection.is_integer():
            selection = int(selection)
        if isinstance(selection, bool) or not isinstance(selection, int):
            record["violation"] = f"selection is not an index: {parsed.get('selection')!r}"
            return None, record, raw
        if not (1 <= selection <= len(candidates)):
            record["violation"] = (f"selection {selection} outside the candidate list "
                                   f"of {len(candidates)}")
            return None, record, raw

        chosen = candidates[selection - 1]
        record["index"] = selection
        record["qid"] = chosen.get("qid")
        return chosen, record, raw


_selectors: dict[tuple, GroundedSelector] = {}


def build(provider, cache_name: str) -> GroundedSelector:
    key = (type(provider).__qualname__, getattr(provider, "model_id", None), cache_name)
    if key not in _selectors:
        _selectors[key] = GroundedSelector(provider, cache_name)
    return _selectors[key]
