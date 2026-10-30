#!/usr/bin/env python3
"""
plan.py - item planning (step 1 of 3: plan.py, generate.py, finalise.py).

WHAT THIS SCRIPT DOES
    Decides every item that will be generated, and writes one JSON object per
    item to output/item_plan.jsonl. Nothing here touches the network and
    nothing here calls a model.

    Every choice that could otherwise be made by the generating model is made
    here instead, or upstream in ../reform_places.py:
        - which places the items are about, and which condition and pool
          (plan or reserve) each belongs to: the A, B and EXPLICIT columns of
          out/places.csv
        - which clue entities appear (see NOTE ON CLUE SELECTION below)
        - which register the text should follow, and its target length
          (drawn from the real Bluesky post lengths)
        - the environmental topic, the scale of the claim and its valence
          (a problem or good news)
        - the value of every style slot (time reference, activity, and so on)
        - which distractor entity is inserted for the noise conditions

    The plan is a reviewable artefact. Read it, diff it, edit it by hand if you
    want, and only then run generate.py against it.

    plan.py also publishes the name-matching helpers (fold, contains_word)
    that finalise.py uses for its checks, so planner and validator agree.

WHAT THIS SCRIPT MUST NEVER DO
    - make network calls of any kind
    - call a language model
    - compute metrics or validate generated text (that is finalise.py)
    - silently substitute a different place, clue or condition when the one it
      wanted is unavailable. Unavailable means infeasible: the pair is skipped
      and the reason is written to the rejection log.

INPUTS
    ../out/places.csv              the place set, its clues, namesakes, split
                                   and condition pools (reform_places.py)
    ../out/place_inputs.jsonl      names, country and region forms, clue
                                   properties, B clues and themes per place
                                   (place_inputs.py)
    sampler/geo_random.txt         one pre-generated integer per line (the
                                   source of each batch is logged in
                                   sampler/geo_random_batches.csv)
    config.yaml         everything configurable

OUTPUTS
    output/item_plan.jsonl        one item per line
    output/plan_rejections.jsonl  every (place, condition) pair places.csv
                                   asked for that was skipped, with the reason
    state/plan_stream.json      random stream cursor, so a later run
                                   continues the stream instead of reusing it

RANDOMNESS
    All randomness comes from sampler/geo_random.txt, a file of pre-generated
    integers (sampler/geo_random_batches.csv records how each batch was
    produced). The file is consumed as a stream: a
    cursor is persisted between runs so no integer is ever used twice for two
    different decisions. Each item records the stream positions it consumed, so
    any single decision can be traced back to the exact integer behind it.

    Consequence worth understanding: re-running this script does NOT reproduce
    the previous plan. It produces a new plan from the next stretch of the
    stream. Reproducibility comes from keeping the plan file and the logs, not
    from re-running. If you want the old plan, use the old plan file.

NOTE ON CLUE SELECTION
    reform_places.py picks each place's clues (clean of the place's names,
    country and region; from at least two families for A) and orders them by
    sitelinks. No draw is made here: A1 and NOISE_A1 use the first clue, A3
    the first three, B and NOISE_B the first clue with no clue relation to
    the rival namesake (place_inputs.py checks that, so the clue cannot point
    at both places). Only the NOISE distractor is drawn from the stream.

    Clues of the part, water and range families are themselves place names
    (a municipality, a river, a mountain range). They carry is_toponym, and
    the prompt allows exactly those names.

NOTE ON SURFACE FORMS
    Entity names (people, organisations, works) are always written in full,
    recognisable form. B, NOISE_B and EXPLICIT write the place's English
    label; for B that label is shared with the rival namesake, and the
    surrounding context is what disambiguates it.

USAGE
    python plan.py
    python plan.py --config config.yaml --out output/item_plan.jsonl
    python plan.py --dry-run          # print a summary, write nothing
    python plan.py --seed-audit       # print the stream positions consumed
    python plan.py --replace          # archive the existing plan, write a new one

    An existing plan is never overwritten without --replace. Generated items
    are keyed by item_id, and a new plan has new ids, so replacing the plan
    after generation has started orphans every row already generated. With
    --replace the old plan and rejection log are moved to output/archive/ with a
    timestamp, so they remain available to finalise.py.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required: pip install pyyaml")


HERE = Path(__file__).resolve().parent

PLAN_VERSION = "8"   # 2: items carry env_theme; 3: B/NOISE_B carry a clue, EXPLICIT is the bare label; 4: fixed core, type groups; 5: reform_places.py set, plan/reserve pools; 6: scale, role, target_len from the Bluesky lengths; 7: spare clues, per-condition lengths, living/occupation on people; 8: valence (negative/positive claim), registers limited by valence, EXPLICIT paired with B/A1 and not planned for infrastructure-only places

# The clue fields a plan item carries, for its clues and its spare clues.
CLUE_KEYS = ("clue_qid", "clue_label", "property", "property_label", "family", "entity_type",
             "sitelinks", "role", "is_toponym", "is_human", "is_living", "description", "aliases",
             "linked_to_rival", "label_leak")

# The conditions, and what each one requires to be feasible for a given place.
# n_clues        how many entity clues appear in the text
# needs_toponym  "ambiguous" -> the label, shared with a namesake 161+ km away
#                "label"     -> the place's bare label, whether or not other
#                               places share it (EXPLICIT; the row records which)
#                None        -> no place name may appear at all
# noise          whether a distractor entity from another place is added
# B and NOISE_B carry one A1-style clue: a name shared by several places has no
# single answer without something to pick the namesake.
CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "A1":       {"n_clues": 1, "needs_toponym": None,        "noise": False},
    "A3":       {"n_clues": 3, "needs_toponym": None,        "noise": False},
    "B":        {"n_clues": 1, "needs_toponym": "ambiguous", "noise": False},
    "EXPLICIT": {"n_clues": 0, "needs_toponym": "label",     "noise": False},
    "NOISE_A1": {"n_clues": 1, "needs_toponym": None,        "noise": True},
    "NOISE_B":  {"n_clues": 1, "needs_toponym": "ambiguous", "noise": True},
}


# --------------------------------------------------------------------------- #
# small shared helpers (finalise.py and build_exemplars.py import these)
# --------------------------------------------------------------------------- #

def utc_now() -> str:
    """The current UTC time as an ISO 8601 string, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure_stdout() -> None:
    """Make stdout and stderr UTF-8: place names are not ASCII and the Windows
    console default codepage cannot print them."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def atomic_write_text(path: Path, text: str, newline: str | None = None) -> None:
    """
    Write text to path through a temporary file and os.replace, so a crash
    leaves either the old file or the complete new one, never a partial file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline=newline) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# random stream
# --------------------------------------------------------------------------- #

class RandomStream:
    """
    An append-only cursor over a file of pre-generated integers.

    Every draw consumes one integer, or occasionally more (see below()), and
    advances the cursor past everything it consumed. The cursor is persisted
    so two runs never spend the same integer on two different decisions.
    Nothing here derives extra values from one integer - each integer serves
    at most one decision, so the audit trail stays traceable.

    low and high are the inclusive range the integers were generated over.
    They are declared rather than inferred from the file, because
    the observed minimum and maximum are only estimates of the true range and
    the uniformity of every draw depends on it being exact.
    """

    def __init__(self, path: Path, state_path: Path, low: int, high: int) -> None:
        if high < low:
            raise SystemExit(f"random stream range is empty: {low}..{high}")
        self.path = path
        self.state_path = state_path
        self.low = low
        self.high = high
        self.values = self._load_values(path)
        out_of_range = [v for v in self.values if not low <= v <= high]
        if out_of_range:
            raise SystemExit(
                f"{path} holds {len(out_of_range)} integers outside the declared "
                f"range {low}..{high} (first: {out_of_range[0]}). Check "
                f"random_min and random_max in the config against the range "
                f"the integers were requested with."
            )
        self.start_position = self._load_cursor(state_path)
        self.position = self.start_position

        if self.position >= len(self.values):
            raise SystemExit(
                f"random stream exhausted before this run started: cursor is at "
                f"{self.position} and {path} holds {len(self.values)} integers. "
                f"Generate more integers in the declared range, append them to "
                f"the file and log the batch in geo_random_batches.csv."
            )

    @staticmethod
    def _load_values(path: Path) -> list[int]:
        if not path.exists():
            raise SystemExit(f"random stream not found: {path}")
        values: list[int] = []
        with path.open(encoding="utf-8-sig") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    values.append(int(line))
                except ValueError as exc:
                    raise SystemExit(
                        f"{path}:{line_no} is not an integer: {line!r}"
                    ) from exc
        if not values:
            raise SystemExit(f"{path} holds no integers")
        return values

    @staticmethod
    def _load_cursor(state_path: Path) -> int:
        if not state_path.exists():
            return 0
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("position", 0))

    def save_cursor(self) -> None:
        """Persist the cursor atomically (a crash cannot leave a torn file)."""
        atomic_write_text(
            self.state_path,
            json.dumps(
                {
                    "position": self.position,
                    "stream_file": str(self.path),
                    "stream_length": len(self.values),
                    "updated_utc": utc_now(),
                },
                indent=2,
            ),
        )

    def _next(self) -> int:
        if self.position >= len(self.values):
            raise SystemExit(
                f"random stream exhausted at position {self.position}. "
                f"Append more integers to {self.path} (log the batch in "
                f"geo_random_batches.csv) and re-run; "
                f"the cursor in {self.state_path} means nothing already used "
                f"will be reused."
            )
        value = self.values[self.position]
        self.position += 1
        return value

    def below(self, n: int) -> int:
        """
        An index in [0, n), exactly uniform, by rejection sampling.

        Plain value % n favours the low indices whenever the range size is not
        a multiple of n, and when n exceeds the range size the high indices
        can never be drawn at all. Instead, integers from the incomplete top
        slice of the range are discarded and the next one is drawn. Usually
        one integer is consumed; the chance of needing another is below
        n / range size.
        """
        if n <= 0:
            raise ValueError("n must be positive")
        span = self.high - self.low + 1
        if n > span:
            raise SystemExit(
                f"cannot choose uniformly among {n} options with integers from "
                f"{self.low}..{self.high} ({span} possible values). Generate "
                f"integers over a wider range."
            )
        limit = span - span % n
        while True:
            value = self._next() - self.low
            if value < limit:
                return value % n

    def pick(self, options: Sequence[Any]) -> Any:
        """One element of options. One below() draw."""
        if not options:
            raise ValueError("cannot pick from an empty sequence")
        return options[self.below(len(options))]

    def sample(self, options: Sequence[Any], k: int) -> list[Any]:
        """
        k distinct elements, drawn without replacement by partial
        Fisher-Yates. One below() draw per element.
        """
        if k > len(options):
            raise ValueError(f"cannot draw {k} from {len(options)} options")
        pool = list(options)
        drawn: list[Any] = []
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
            drawn.append(pool[i])
        return drawn


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #

@dataclass
class Place:
    qid: str
    name: str
    # places.csv type (settlement, natural, island, manmade, infra) and the
    # finer kind (lake, mountain, hydro plant...).
    place_type: str
    sub: str
    country: str
    country_qid: str
    continent: str
    # dev or test, fixed per place by reform_places.py before generation.
    split: str = "test"
    sitelinks: int = 0
    lat: float | None = None
    lon: float | None = None
    # places.csv A and B columns: "plan", "reserve" or "" (not in that set).
    # Reserve places replace posts that fail finalise.py.
    a_pool: str = ""
    b_pool: str = ""
    explicit: bool = False
    # reform_places.py bins, carried into every item for analysis.
    pbin: str = ""
    rbin: str = ""
    cbin: str = ""
    # The best-known namesake more than 161 km away (qid, label, sitelinks,
    # km) and how many places share the name, this one included. {} and 1
    # when the name is unique.
    rival: dict[str, Any] = field(default_factory=dict)
    places_sharing_name: int = 1
    env_themes: list[str] = field(default_factory=list)
    # Good-news themes, for items planned 'positive' (plan.valences).
    env_themes_positive: list[str] = field(default_factory=list)
    # The clues reform_places.py picked, most documented first.
    clues: list[dict[str, Any]] = field(default_factory=list)
    # The clues B and NOISE_B may use: those with no clue relation to the rival.
    b_clues: list[dict[str, Any]] = field(default_factory=list)
    # The place's names, its country's names and demonyms, and the regions it
    # sits in. A distractor from elsewhere is checked against these too.
    name_forms: list[str] = field(default_factory=list)
    country_forms: list[str] = field(default_factory=list)
    region_forms: list[str] = field(default_factory=list)
    # distractor_clues() memoised per donor qid; see reset_caches().
    _distractors: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False, compare=False)
    # noise_donors() memoised.
    _donors: list[str] | None = field(default=None, repr=False, compare=False)

    def reset_caches(self) -> None:
        """Forget memoised distractors and donors. Call after name_forms or any
        donor's clues change."""
        self._distractors.clear()
        self._donors = None

    def distractor_clues(self, donor: "Place") -> list[dict[str, Any]]:
        """
        The donor's clues whose labels name nothing of this place.
        Memoised per donor; do not mutate the returned list.
        """
        cached = self._distractors.get(donor.qid)
        if cached is None:
            # Not an entity that also points at this place: the distractor must
            # differ from what is to be guessed.
            own = {c.get("clue_qid") for c in self.clues}
            cached = [
                c for c in donor.clues
                if c.get("clue_qid") not in own
                and label_leak(c["clue_label"], self.name_forms, self.country_forms,
                               self.region_forms) is None
            ]
            self._distractors[donor.qid] = cached
        return cached


def _read_inputs(path: Path) -> dict[str, dict[str, Any]]:
    """place_qid -> its place_inputs.jsonl record (written by ../place_inputs.py)."""
    if not path.exists():
        raise SystemExit(f"place inputs not found: {path}. Run ../place_inputs.py.")
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record["place_qid"] in records:
                raise SystemExit(f"{path}:{line_no} is a duplicate entry for {record['place_qid']}")
            records[record["place_qid"]] = record
    return records


def _clue(raw: dict[str, Any], name_forms: list[str], country_forms: list[str],
          region_forms: list[str]) -> dict[str, Any]:
    """A place_inputs.jsonl clue in the shape the plan items carry."""
    return {
        "clue_qid": raw["clue_qid"],
        "clue_label": raw["clue_label"],
        "property": raw.get("property") or "",
        "property_label": raw.get("property_label") or "",
        "family": raw.get("family"),
        "entity_type": raw.get("entity_type"),
        "sitelinks": raw.get("sitelinks"),
        "role": raw.get("role") or "",
        # Place-name clues (part, water, range families): the prompt allows
        # exactly these names.
        "is_toponym": bool(raw.get("is_toponym")),
        # A person may be written by a shorter name in the post (finalise.py
        # accepts the surname); anything else needs its label or an alias.
        "is_human": bool(raw.get("is_human")),
        # True / False for people (no date of death and born after 1920), None
        # otherwise: the prompt says "a living person" or "a person from history".
        "is_living": raw.get("is_living"),
        # What a person is known for, shown instead of their link to the place.
        "description": raw.get("description") or "",
        "aliases": list(raw.get("aliases") or []),
        "linked_to_rival": raw.get("linked_to_rival"),
        "label_leak": label_leak(raw["clue_label"], name_forms, country_forms, region_forms),
    }


def load_places(path: Path, inputs_path: Path) -> dict[str, Place]:
    """
    Read out/places.csv (reform_places.py) and the matching
    out/place_inputs.jsonl. Every place in the CSV must have an inputs record.
    """
    if not path.exists():
        raise SystemExit(f"place file not found: {path}")
    inputs = _read_inputs(inputs_path)

    places: dict[str, Place] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"i", "label", "type", "sub", "c", "continent", "split", "A", "B", "EXPLICIT"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            qid = row["i"].strip()
            if qid not in inputs:
                raise SystemExit(f"{inputs_path} has no record for {qid} ({row['label']}). "
                                 f"Re-run ../place_inputs.py.")
            rec = inputs[qid]
            names, countries, regions = (rec.get("name_forms") or [row["label"]],
                                         rec.get("country_forms") or [],
                                         rec.get("region_forms") or [])
            clues = [_clue(c, names, countries, regions) for c in rec.get("clues") or []]
            b_ok = set(rec.get("b_clues") or [])
            rival = ({"qid": row["rival"], "label": rec.get("rival_label") or "",
                      "sitelinks": _as_int(row.get("rival_sl"), 0),
                      "km": _as_int(row.get("rival_km"), 0)} if row.get("rival") else {})
            places[qid] = Place(
                qid=qid,
                name=row["label"].strip(),
                place_type=row["type"].strip(),
                sub=row["sub"].strip(),
                country=rec.get("country") or row["c"],
                country_qid=row["c"].strip(),
                continent=row["continent"].strip(),
                split=row["split"].strip(),
                sitelinks=_as_int(row.get("sl"), 0),
                lat=float(row["lat"]) if row.get("lat") else None,
                lon=float(row["lon"]) if row.get("lon") else None,
                a_pool=row["A"].strip(),
                b_pool=row["B"].strip(),
                explicit=row["EXPLICIT"].strip() == "yes",
                pbin=row.get("pbin", ""),
                rbin=row.get("rbin", ""),
                cbin=row.get("cbin", ""),
                rival=rival,
                places_sharing_name=_as_int(row.get("n_names"), 1),
                env_themes=list(rec.get("env_themes") or []),
                env_themes_positive=list(rec.get("env_themes_positive") or []),
                clues=clues,
                b_clues=[c for c in clues if c["clue_qid"] in b_ok],
                name_forms=names,
                country_forms=countries,
                region_forms=regions,
            )
    if not places:
        raise SystemExit(f"{path} yielded no places")
    return places


@functools.lru_cache(maxsize=None)
def fold(text: str) -> str:
    """Case- and accent-insensitive form, so "Panamá" matches "Panama"."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


@functools.lru_cache(maxsize=None)
def _word_pattern(folded_needle: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(folded_needle)}(?!\w)")


def contains_word(haystack: str, needle: str) -> bool:
    """needle occurs in haystack as a whole word, ignoring case and accents."""
    needle = fold(needle).strip()
    return bool(needle) and _word_pattern(needle).search(fold(haystack)) is not None


# Shorter place forms are left to the whole-word test: "Ely" or "Rye" would
# otherwise catch every "Elysian" and "Ryerson". Four letters is the floor that
# still catches "Capetonians" (Cape) and "yellowstonii" (YELL); on the
# 2026-10-20 place set it excluded those and four more, and nothing else.
MIN_WORD_START_FORM = 4


@functools.lru_cache(maxsize=None)
def _word_start_pattern(folded_needle: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(folded_needle)}")


def starts_word(haystack: str, needle: str) -> bool:
    """
    needle begins a word of haystack, ignoring case and accents: a derived
    word such as "Huronian" for Huron or "lémanique" for Léman.
    """
    needle = fold(needle).strip()
    return bool(needle) and _word_start_pattern(needle).search(fold(haystack)) is not None


def _is_qualified(text: str, country_forms: list[str]) -> bool:
    """
    A form that already narrows itself: it carries a comma qualifier ("Las
    Cruces, NM") or contains the country's name ("Guatemala City", the bare
    "Guatemala", the Spanish "Panamá"). The B constraint forbids exactly
    those additions and forbids writing the country, so such a form cannot
    serve as an ambiguous toponym without the prompt contradicting itself.
    """
    if "," in text:
        return True
    return any(contains_word(text, form) for form in country_forms)


def label_leak(label: str, place_forms: Sequence[str], country_forms: Sequence[str],
               region_forms: Sequence[str] = ()) -> str | None:
    """
    Why a clue's own label gives the answer away, or None. It checks every
    surface form of the place ("Qaw el-Kebir" for Tjebu, "Mount McKinley" for
    Denali) and the country's name and demonyms ("Tuvalu Association...",
    "MIAT Mongolian Airlines"). Whole words, ignoring case and accents; a place
    form of MIN_WORD_START_FORM letters or more also counts at the start of a
    longer word ("Huronian glaciation" for Lake Huron, "Capetonians" for Cape
    Town). Country and region forms stay whole-word: demonyms are listed.
    """
    for form in place_forms:
        if contains_word(label, form):
            return f'label contains the place name or alias "{form}"'
        if len(fold(form).strip()) >= MIN_WORD_START_FORM and starts_word(label, form):
            return f'label has a word starting with the place name or alias "{form}"'
    for form in country_forms:
        if contains_word(label, form):
            return f'label contains the country or its demonym "{form}"'
    for form in region_forms:
        if contains_word(label, form):
            return f'label contains a region the place is in "{form}"'
    return None


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("paths", "plan", "registers", "slots"):
        if section not in config:
            raise SystemExit(f"{path} is missing the {section!r} section")
    # Paths in the config are relative to the config file's folder, so the
    # scripts behave the same whatever directory they are started from.
    base = path.resolve().parent
    config["paths"] = {key: str((base / value).resolve())
                      for key, value in config["paths"].items()}
    return config


# --------------------------------------------------------------------------- #
# feasibility
# --------------------------------------------------------------------------- #

def noise_donors(place: Place, places: dict[str, Place]) -> list[str]:
    """
    Places that can supply a distractor for this place: any other place with
    at least one clue to draw from. It may be on the same continent or on
    another, but not in the same country as the post's place. Memoised on the
    place (reset_caches clears it).
    """
    # A distractor whose label names THIS place ("2013 South American
    # Championships" carries "American" into a United States item) would put a
    # forbidden name into the post, so donors are counted only for the clues
    # that survive that check.
    if place._donors is None:
        place._donors = [
            qid
            for qid, other in places.items()
            if qid != place.qid
            and donor_relation(place, other) != "same_country"
            and place.distractor_clues(other)
        ]
    return place._donors


def donor_relation(place: Place, donor: Place) -> str:
    """How far the distractor's place is from the item's place: same_country,
    same_continent or other_continent (what is known from the place table)."""
    if place.country_qid == donor.country_qid:
        return "same_country"
    if place.continent and place.continent == donor.continent:
        return "same_continent"
    return "other_continent"


def feasible(
    place: Place, condition: str, places: dict[str, Place]
) -> tuple[bool, str]:
    """
    Can this place support this condition? Returns (ok, reason_if_not).
    Only called for the conditions places.csv puts the place in.

    This is the only place where an item is allowed to disappear. Nothing
    downstream may quietly repair an infeasible pair by swapping in a
    different place, clue or condition.
    """
    spec = CONDITION_SPECS[condition]

    if spec["noise"] and not noise_donors(place, places):
        return False, "no other place has an eligible clue to donate"

    toponym = spec["needs_toponym"]
    if toponym == "ambiguous":
        if not place.rival:
            return False, "no namesake recorded"
        if not place.b_clues:
            return False, "no clue without a link to the rival (see out/b_need_clues.csv)"
        if _is_qualified(place.name, place.country_forms):
            return False, "name already carries a qualifier or the country"
        return True, ""

    n_needed = spec["n_clues"]
    if len(place.clues) < n_needed:
        return False, f"{len(place.clues)} clues, {condition} needs {n_needed}"
    return True, ""


# --------------------------------------------------------------------------- #
# per-item decisions
# --------------------------------------------------------------------------- #

def choose_clues(place: Place, condition: str) -> list[dict[str, Any]]:
    """
    The clues are fixed by reform_places.py, most documented first, so no draw
    is made: A1 and NOISE_A1 use the first clue (the most documented, its A1
    rule), A3 the first three (always from at least two families), B and
    NOISE_B the first clue with no link to the rival.
    """
    n = CONDITION_SPECS[condition]["n_clues"]
    if n == 0:
        return []
    if CONDITION_SPECS[condition]["needs_toponym"] == "ambiguous":
        return place.b_clues[:1]
    return place.clues[:n]


def spare_clues(place: Place, condition: str) -> list[dict[str, Any]]:
    """
    The clues after the chosen ones, in the same order: what generate.py
    swaps in, one per try, when the judge finds a chosen clue only dropped
    into the post (judge_decorative). Each swap is recorded on the row, so an
    item's clue prominence is read from the clue it actually used.
    """
    n = CONDITION_SPECS[condition]["n_clues"]
    if n == 0:
        return []
    if CONDITION_SPECS[condition]["needs_toponym"] == "ambiguous":
        return place.b_clues[1:]
    return place.clues[n:]


def choose_surface_form(place: Place, condition: str) -> dict[str, Any]:
    """
    No draw: B, NOISE_B and EXPLICIT all write the place's English label.
    top_namesake is the best-known other place 161+ km away with the same
    name, so results can be split by whether the intended place is the obvious
    reading (compare its sitelinks with the item's place_sitelinks).
    """
    kind = CONDITION_SPECS[condition]["needs_toponym"]
    if kind is None:
        return {"kind": "none", "text": None, "shared_by": None}
    return {
        "kind": "ambiguous_alias" if kind == "ambiguous" else "exact",
        "text": place.name,
        "shared_by": place.places_sharing_name if place.rival else None,
        "name_ambiguous": bool(place.rival),
        "top_namesake": dict(place.rival) or None,
    }


def choose_noise_entity(
    place: Place,
    condition: str,
    places: dict[str, Place],
    stream: RandomStream,
) -> dict[str, Any] | None:
    """
    For the noise conditions, add one entity associated with another place.
    The other place can be anywhere except the post's own country
    (donor_relation records whether it is on the same or another continent).

    The distractor is a real clue for a real other place, which is what makes
    it a useful distractor: a system that follows the wrong association lands
    somewhere specific and wrong, rather than nowhere.
    """
    if not CONDITION_SPECS[condition]["noise"]:
        return None

    candidate_qids = noise_donors(place, places)
    if not candidate_qids:
        # feasible() rejects this pair before it reaches the slate, so this
        # means the two have drifted apart. Never plan a noise item without
        # its noise.
        raise SystemExit(
            f"no distractor donor for {place.qid} under {condition}, which "
            f"feasible() should have rejected"
        )

    donor_qid = stream.pick(candidate_qids)
    donor = places[donor_qid]
    donor_clue = stream.pick(place.distractor_clues(donor))
    return {
        "donor_relation": donor_relation(place, donor),
        "clue_qid": donor_clue["clue_qid"],
        "clue_label": donor_clue["clue_label"],
        "property": donor_clue["property"],
        "property_label": donor_clue.get("property_label"),
        "family": donor_clue.get("family"),
        "entity_type": donor_clue.get("entity_type"),
        "sitelinks": donor_clue.get("sitelinks"),
        "is_toponym": donor_clue.get("is_toponym"),
        "label_leak": donor_clue.get("label_leak"),
        "is_human": donor_clue.get("is_human"),
        "aliases": donor_clue.get("aliases") or [],
        "donor_place_qid": donor_qid,
        "donor_place_name": donor.name,
        "donor_continent": donor.continent,
    }


def choose_register(config: dict[str, Any], stream: RandomStream,
                    valence: str | None = None) -> dict[str, Any]:
    """A register that fits the valence: a register may list the valences it
    suits (a complaint cannot carry good news); one that lists none suits all."""
    fits = [r for r in config["registers"]
            if valence is None or valence in r.get("valences", [valence])]
    if not fits:
        raise SystemExit(f"no register suits valence {valence!r}")
    return stream.pick(fits)


def themes_for_valence(place: "Place", valence: str) -> list[str]:
    """The place's themes for a valence: its problems, or its good news."""
    if valence == "positive":
        if not place.env_themes_positive:
            raise SystemExit(f"{place.qid} ({place.name}) has no env_themes_positive. "
                             f"Re-run ../place_inputs.py.")
        return place.env_themes_positive
    return place.env_themes


def choose_slots(
    register: dict[str, Any], config: dict[str, Any], stream: RandomStream
) -> dict[str, str]:
    """
    Fill the slots this register declares. Filling them here rather than in the
    prompt is what stops the model from reaching for the same three openers and
    the same weather for every item.
    """
    slots: dict[str, str] = {}
    for slot_name in register.get("slots", []):
        options = config["slots"].get(slot_name)
        if not options:
            raise SystemExit(
                f"register {register.get('name')!r} asks for slot {slot_name!r}, "
                f"which is not defined in the config's slots section"
            )
        slots[slot_name] = stream.pick(options)
    return slots


def make_item_id(payload: dict[str, Any], replicate: int) -> str:
    """
    A hash of the decisions, not a counter.

    A counter shifts every id downstream the moment you insert, remove or
    reorder an item, which invalidates the generation cache and makes two runs
    impossible to compare. A hash is stable: an item keeps its id no matter
    what happens to its neighbours.

    The replicate index is part of the hash so two items that happen to make
    identical decisions still get distinct ids.
    """
    canonical = json.dumps(
        {
            "place_qid": payload["place_qid"],
            "condition": payload["condition"],
            "clue_qids": sorted(c["clue_qid"] for c in payload["clues"]),
            "surface_text": payload["surface_form"]["text"],
            "noise_qid": (payload["noise_entity"] or {}).get("clue_qid"),
            "register": payload["register"],
            "target_len": payload["target_len"],
            "slots": payload["slots"],
            "env_theme": payload.get("env_theme"),
            "scale": payload.get("scale"),
            "valence": payload.get("valence"),
            "replicate": replicate,
            "plan_version": PLAN_VERSION,
        },
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# allocation
# --------------------------------------------------------------------------- #

def allocate_even(available: dict[str, int], n: int) -> dict[str, int]:
    """
    Split n picks across groups as evenly as the supply allows: every group gets
    the same share until one runs out, and its unused share goes to the rest.
    The remainder after an even split goes to groups in sorted-name order.
    Shared with finalise.py, which cuts the conditions the same way.
    """
    take = dict.fromkeys(available, 0)
    remaining = min(n, sum(available.values()))
    open_groups = sorted(g for g, k in available.items() if k > 0)
    while remaining > 0 and open_groups:
        share, extra = divmod(remaining, len(open_groups))
        progressed = 0
        for index, group in enumerate(open_groups):
            want = share + (1 if index < extra else 0)
            got = min(want, available[group] - take[group])
            take[group] += got
            progressed += got
        remaining -= progressed
        open_groups = [g for g in open_groups if take[g] < available[g]]
        if not progressed:
            break
    return take


def conditions_for(place: Place, conditions: Iterable[str]) -> list[tuple[str, str]]:
    """
    (condition, pool) for every condition places.csv puts this place in: the A
    set (A column) gets the A conditions, the B set the B conditions. EXPLICIT
    is the reference the other conditions are read against (the name stated
    outright), so it is planned only for places in the A or B plan; a reserve
    place gets it as a reserve item, generated only if the place is swapped in.
    The EXPLICIT-only infrastructure places in places.csv get nothing: they
    have no A or B item to compare with (2026-10-30, user).
    pool is "plan" or "reserve".
    """
    out = []
    in_plan = "plan" in (place.a_pool, place.b_pool)
    for condition in conditions:
        spec = CONDITION_SPECS[condition]
        if condition == "EXPLICIT":
            pool = ("" if not place.explicit else "plan" if in_plan
                    else "reserve" if place.a_pool or place.b_pool else "")
        elif spec["needs_toponym"] == "ambiguous":
            pool = place.b_pool
        else:
            pool = place.a_pool
        if pool:
            out.append((condition, pool))
    return out


def build_slate(
    places: dict[str, Place], config: dict[str, Any]
) -> tuple[list[tuple[str, str, str]], list[dict[str, Any]]]:
    """
    Decide which (place, condition, pool) triples to plan: one item for every
    condition places.csv puts each place in (conditions_for), when the place
    can support it (feasible). reform_places.py has already balanced the set
    by continent, type, country and prominence, and sized each condition's
    plan and reserve, so nothing is drawn or topped up here.
    """
    conditions = list(config["plan"]["conditions"])
    unknown = set(conditions) - set(CONDITION_SPECS)
    if unknown:
        raise SystemExit(f"config names unknown conditions: {sorted(unknown)}")

    slate: list[tuple[str, str, str]] = []
    rejections: list[dict[str, Any]] = []
    for qid in sorted(places):
        place = places[qid]
        for condition, pool in conditions_for(place, sorted(conditions)):
            ok, reason = feasible(place, condition, places)
            if not ok:
                rejections.append({
                    "record": "pair_rejected",
                    "place_qid": qid,
                    "place_name": place.name,
                    "condition": condition,
                    "pool": pool,
                    "reason": reason,
                    "clue_count": len(place.clues),
                    "b_clue_count": len(place.b_clues),
                })
                continue
            slate.append((qid, condition, pool))
    slate.sort(key=lambda t: (t[1], t[2], t[0]))
    return slate, rejections


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def load_lengths(path: Path) -> list[tuple[int, int]]:
    """(word count, posts) for every length of the real Bluesky posts, from
    length_dist.py."""
    if not path.exists():
        raise SystemExit(f"{path} not found: run length_dist.py first")
    counts = json.loads(path.read_text(encoding="utf-8"))["counts"]
    return sorted((int(n), int(k)) for n, k in counts.items())


def length_range(config: dict[str, Any], condition: str) -> tuple[int, int]:
    """The words a post of this condition may have: finalise.words_by_condition,
    else finalise.min_words to max_words. The same range finalise.py keeps."""
    fin = config.get("finalise") or {}
    lo, hi = (fin.get("words_by_condition") or {}).get(
        condition, (fin.get("min_words"), fin.get("max_words")))
    if lo is None or hi is None:
        raise SystemExit("config must set finalise.min_words and max_words: "
                         "the target lengths are drawn within them")
    return int(lo), int(hi)


def choose_target_len(lengths: list[tuple[int, int]], stream: RandomStream,
                      lo: int = 0, hi: int = 10 ** 9) -> int:
    """A word count from lo to hi drawn in proportion to how many real posts
    have it: one draw over those posts, mapped onto the cumulative counts."""
    lengths = [(n, k) for n, k in lengths if lo <= n <= hi]
    draw = stream.below(sum(k for _, k in lengths))
    for n, k in lengths:
        if draw < k:
            return n
        draw -= k
    raise AssertionError("draw outside the distribution")


def plan_items(
    places: dict[str, Place],
    config: dict[str, Any],
    stream: RandomStream,
    lengths: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slate, rejections = build_slate(places, config)
    scales = list(config["plan"]["scales"])
    valences = list(config["plan"]["valences"])

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    replicate = 0   # one item per (place, condition); kept in the id hash
    # EXPLICIT is read against the place's B item (else its A1 item): it takes
    # that item's style and topic draws, so the two posts differ by the clue,
    # not by the writing. The slate is sorted by condition, so A1 and B come first.
    siblings: dict[str, dict[str, Any]] = {}

    for qid, condition, pool in slate:
        place = places[qid]
        position_before = stream.position

        clues = choose_clues(place, condition)
        surface_form = choose_surface_form(place, condition)
        noise_entity = choose_noise_entity(place, condition, places, stream)
        twin = siblings.get(qid) if condition == "EXPLICIT" else None
        if twin:
            register = next(r for r in config["registers"] if r["name"] == twin["register"])
            valence, target_len, slots = twin["valence"], twin["target_len"], dict(twin["slots"])
            env_theme, scale = twin["env_theme"], twin["scale"]
        else:
            valence = stream.pick(valences)
            register = choose_register(config, stream, valence)
            target_len = choose_target_len(lengths, stream, *length_range(config, condition))
            slots = choose_slots(register, config, stream)
            themes = themes_for_valence(place, valence)
            env_theme = stream.pick(themes) if themes else None
            scale = stream.pick(scales)

        payload: dict[str, Any] = {
            "plan_version": PLAN_VERSION,
            "place_qid": qid,
            "place_name": place.name,
            "place_type": place.place_type,
            "place_sub": place.sub,
            "place_country": place.country,
            "place_country_qid": place.country_qid,
            "place_continent": place.continent,
            "place_split": place.split,
            "place_lat": place.lat,
            "place_lon": place.lon,
            "place_sitelinks": place.sitelinks,
            "places_sharing_name": place.places_sharing_name,
            "pbin": place.pbin,
            "rbin": place.rbin,
            "cbin": place.cbin,
            "condition": condition,
            "pool": pool,
            "signal": _signal_label(condition),
            "clues": [{key: c.get(key) for key in CLUE_KEYS} for c in clues],
            "spare_clues": [
                {key: c.get(key) for key in CLUE_KEYS} for c in spare_clues(place, condition)
            ],
            "surface_form": surface_form,
            "noise_entity": noise_entity,
            "register": register["name"],
            "register_description": register.get("description"),
            "target_len": target_len,
            "slots": slots,
            "env_theme": env_theme,
            "scale": scale,
            "valence": valence,
            # The planned clues' roles, a hint to the writer (prompt.role_hints).
            "role": ";".join(dict.fromkeys(c.get("role") or "other" for c in clues)),
            # The B or A1 item this EXPLICIT item copies its draws from.
            "paired_with": twin["item_id"] if twin else None,
            "replicate": replicate,
        }

        item_id = make_item_id(payload, replicate)
        if item_id in seen_ids:
            # Should be impossible: a (place, condition) pair is planned once.
            raise SystemExit(f"duplicate item_id {item_id} for {qid}/{condition}")
        seen_ids.add(item_id)

        payload["item_id"] = item_id
        payload["stream_positions"] = {
            "start": position_before,
            "end": stream.position,
        }
        payload["planned_utc"] = utc_now()
        items.append(payload)
        # same pool only: a reserve EXPLICIT item pairs with the reserve B item
        if condition == "B" or (condition == "A1" and qid not in siblings):
            if pool == ("plan" if "plan" in (place.a_pool, place.b_pool) else "reserve"):
                siblings[qid] = payload

    return items, rejections


def _signal_label(condition: str) -> str:
    """
    Map a condition onto the Gritta signal taxonomy the thesis uses.
    Signal A is an entity clue with no toponym. Signal B is an ambiguous
    toponym disambiguated by context.
    """
    spec = CONDITION_SPECS[condition]
    if condition == "EXPLICIT":
        return "explicit"
    has_a = spec["n_clues"] > 0
    has_b = spec["needs_toponym"] == "ambiguous"
    if has_a and has_b:
        return "A+B"
    if has_a:
        return "A"
    if has_b:
        return "B"
    return "none"


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    """Write records as JSONL atomically (tmp file + os.replace). Returns the count."""
    lines = [json.dumps(record, ensure_ascii=False) + "\n" for record in records]
    atomic_write_text(path, "".join(lines))
    return len(lines)


def archive_existing(paths: Iterable[Path]) -> list[Path]:
    """
    Move each existing file to an archive/ folder beside it, stamped with the
    current UTC time. Returns the archive paths written. One stamp is shared by
    the whole call, so a plan and its rejection log stay paired.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        target = path.parent / "archive" / f"{path.stem}.{stamp}{path.suffix}"
        if target.exists():
            raise SystemExit(f"archive target already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
        archived.append(target)
    return archived


def summarise(items: list[dict[str, Any]], rejections: list[dict[str, Any]]) -> None:
    print(f"\nplanned {len(items)} items over "
          f"{len({i['place_qid'] for i in items})} places")

    by_condition = Counter(i["condition"] for i in items)
    print("\n  by condition")
    for condition in CONDITION_SPECS:
        if condition in by_condition:
            print(f"    {condition:<10} {by_condition[condition]:>5}")

    by_continent = Counter(i["place_continent"] for i in items)
    print("\n  by continent")
    for continent, n in sorted(by_continent.items()):
        print(f"    {continent:<16} {n:>5}")

    print("\n  plan / reserve per condition")
    for condition in CONDITION_SPECS:
        mine = Counter(i["pool"] for i in items if i["condition"] == condition)
        if mine:
            print(f"    {condition:<10} {mine['plan']:>5} {mine['reserve']:>5}")

    print("\n  place types per condition (%)")
    groups = sorted({i["place_type"] for i in items})
    print("    " + " " * 10 + "".join(f"{g[:9]:>10}" for g in groups))
    for condition in sorted({i["condition"] for i in items}):
        mine = Counter(i["place_type"] for i in items if i["condition"] == condition)
        total = sum(mine.values())
        print(f"    {condition:<10}" + "".join(f"{100 * mine[g] / total:>10.1f}" for g in groups))

    print("\n  countries per condition (most used)")
    for condition in sorted({i["condition"] for i in items}):
        mine = Counter(i["place_country"]
                       for i in items if i["condition"] == condition)
        top = ", ".join(f"{c} {n}" for c, n in mine.most_common(5))
        print(f"    {condition:<10} {len(mine):>3} countries; {top}")

    by_register = Counter(i["register"] for i in items)
    print("\n  by register")
    for register, n in sorted(by_register.items()):
        print(f"    {register:<22} {n:>5}")

    print("\n  scale per condition")
    for condition in CONDITION_SPECS:
        mine = Counter(i["scale"] for i in items if i["condition"] == condition)
        if mine:
            print(f"    {condition:<10} " + "  ".join(f"{s} {n}" for s, n in sorted(mine.items())))

    print("\n  valence per condition")
    for condition in CONDITION_SPECS:
        mine = Counter(i.get("valence") for i in items if i["condition"] == condition)
        if mine:
            print(f"    {condition:<10} " + "  ".join(f"{v} {n}" for v, n in sorted(mine.items(), key=str)))

    print("\n  target length per condition (words)")
    for condition in CONDITION_SPECS:
        lens = sorted(i["target_len"] for i in items if i["condition"] == condition)
        if lens:
            print(f"    {condition:<10} median {lens[len(lens) // 2]}, range {lens[0]}..{lens[-1]}")

    themed = sum(1 for i in items if i.get("env_theme"))
    if themed:
        print(f"\n  {themed} of {len(items)} items have an environmental theme")

    pairs = [r for r in rejections if r["record"] == "pair_rejected"]
    if pairs:
        print(f"\n  {len(pairs)} (place, condition) pairs skipped")
        by_reason = Counter((r["condition"], r["pool"], r["reason"]) for r in pairs)
        for (condition, pool, reason), n in sorted(by_reason.items()):
            print(f"    {n:>5}  {condition:<9} {pool:<8} {reason}")
        places_fully_excluded = {
            r["place_qid"] for r in pairs
        } - {i["place_qid"] for i in items}
        if places_fully_excluded:
            print(f"\n  {len(places_fully_excluded)} places support no condition "
                  f"at all and appear in no item")


def main(argv: Sequence[str] | None = None) -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Plan the generation items.")
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--out", type=Path, default=None,
                        help="override the plan path from the config")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the summary, write no files, do not advance "
                             "the random stream cursor")
    parser.add_argument("--seed-audit", action="store_true",
                        help="print the stream range each item consumed")
    parser.add_argument("--replace", action="store_true",
                        help="archive the existing plan and rejection log to "
                             "output/archive/ and write a new plan")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = config["paths"]

    places_path = Path(paths["places_csv"])
    inputs_path = Path(paths["place_inputs"])
    stream_path = Path(paths["random_stream"])
    state_path = Path(paths["plan_stream_state"])
    plan_path = args.out or Path(paths["item_plan"])
    reject_path = Path(paths["plan_rejections"])

    # Checked before any work is done, so a refused run spends nothing.
    if plan_path.exists() and not args.dry_run and not args.replace:
        generations = Path(paths["raw_generations"])
        warning = (f"\n{generations} exists, so generation has already started "
                   f"against the current plan. Replacing the plan orphans those "
                   f"rows." if generations.exists() else "")
        raise SystemExit(
            f"{plan_path} already exists. A new plan draws the next stretch of "
            f"the random stream, so it has different item_ids and cannot be "
            f"reproduced from the old one.{warning}\n"
            f"Re-run with --replace to archive the existing plan and write a "
            f"new one, or --dry-run to preview."
        )

    print(f"places   {places_path}")
    print(f"inputs   {inputs_path}")
    print(f"random   {stream_path}")

    plan_cfg = config["plan"]
    if plan_cfg.get("random_min") is None or plan_cfg.get("random_max") is None:
        raise SystemExit(
            f"{args.config} must set plan.random_min and plan.random_max to the "
            f"range the integers in {stream_path} were generated over. "
            f"Every draw's uniformity depends on it."
        )

    places = load_places(places_path, inputs_path)
    stream = RandomStream(
        stream_path, state_path,
        low=int(plan_cfg["random_min"]), high=int(plan_cfg["random_max"]),
    )

    print(f"loaded {len(places)} places, "
          f"{sum(1 for p in places.values() if p.clues)} with at least one clue, "
          f"{sum(1 for p in places.values() if len(p.clues) >= 3)} with three, "
          f"{sum(1 for p in places.values() if p.b_pool and p.b_clues)} of "
          f"{sum(1 for p in places.values() if p.b_pool)} B places with a clue "
          f"not linked to the rival")
    print(f"random stream starts at position {stream.start_position} "
          f"of {len(stream.values)}")

    items, rejections = plan_items(places, config, stream,
                                   load_lengths(Path(paths["length_dist"])))
    summarise(items, rejections)

    print(f"\nrandom stream consumed positions "
          f"{stream.start_position}..{stream.position - 1} "
          f"({stream.position - stream.start_position} integers)")

    if args.seed_audit:
        print("\n  stream audit")
        for item in items:
            pos = item["stream_positions"]
            print(f"    {item['item_id']}  {item['condition']:<9} "
                  f"{pos['start']:>6}..{pos['end'] - 1:<6} {item['place_name']}")

    if args.dry_run:
        print("\ndry run: nothing written, stream cursor not advanced")
        return 0

    # The cursor goes first. The plan records the stream positions it used, so
    # if the run dies after this line the integers are burned, never reused:
    # the invariant that no integer serves two decisions survives a crash. The
    # cost is a gap in the stream, which is harmless.
    stream.save_cursor()

    if args.replace:
        for archived in archive_existing([plan_path, reject_path]):
            print(f"archived {archived}")

    n_items = write_jsonl(plan_path, items)
    n_rejects = write_jsonl(reject_path, rejections)

    print(f"\nwrote {n_items} items to {plan_path}")
    print(f"wrote {n_rejects} rejections to {reject_path}")
    print(f"stream cursor saved to {state_path} at position {stream.position}")
    return 0


def _as_int(value: Any, default: int) -> int:
    try:
        text = str(value).strip()
        return int(float(text)) if text else default
    except (TypeError, ValueError):
        return default


_utc_now = utc_now  # underscore alias kept for older callers


if __name__ == "__main__":
    sys.exit(main())
