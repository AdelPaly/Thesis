#!/usr/bin/env python3
"""
plan.py - item planning (step 1 of 3: plan.py, generate.py, finalise.py).

WHAT THIS SCRIPT DOES
    Decides every item that will be generated, and writes one JSON object per
    item to output/item_plan.jsonl. Nothing here touches the network and
    nothing here calls a model.

    Every choice that could otherwise be made by the generating model is made
    here instead:
        - which place the item is about
        - which condition the item belongs to (A1, A3, B, EXPLICIT, NOISE_*)
        - which clue entities appear (see NOTE ON CLUE SELECTION below)
        - which surface form of the place name is used, if any
        - which register and length band the text should follow
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
    ../places/data/01_places_final.csv the 102-place set (17 per continent)
    ../places/data/05_clue_sets.jsonl stage 5 best_sets per place
    ../places/data/04b_place_aliases.jsonl surface forms per place
    ../places/data/05b_clue_types.jsonl which clue entities are themselves places
    sampler/geo_random.txt         one pre-generated integer per line (the
                                   source of each batch is logged in
                                   sampler/geo_random_batches.csv)
    config.yaml         everything configurable

OUTPUTS
    output/item_plan.jsonl        one item per line
    output/plan_rejections.jsonl  every (place, condition) pair that was
                                   skipped, and every clue or surface form
                                   held back, each with the reason. The
                                   "record" field says which:
                                   pair_rejected, clue_excluded, form_excluded
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
    Stage 5 exports only its best-ranked clue sets per place, so the clue pool
    here is the union of those sets, not every clue that passed the filter.
    Within that pool the draw is uniform, but the pool itself is a ranked
    selection, and the methodology should say so.

    Single-clue conditions (A1, NOISE_A1, and the NOISE distractors) draw
    only clues whose place_ambiguity is within the configured ceiling: a lone
    clue that several places satisfy has no single correct answer. A3 draws a
    complete size-3 set as stage 5 built it, because that set's intersection
    is known to resolve and three loose clues carry no such guarantee.

    No discriminativeness score exists upstream. The field is kept in the plan
    as null so finalise.py and the analysis can see the absence rather than mistake it for a value.
    place_ambiguity, surface_ambiguity and source_set_sizes travel with each
    clue for analysis.

NOTE ON SURFACE FORMS
    Entity names (people, organisations, works) are always written in full,
    recognisable form. Only place names carry ambiguity. That asymmetry is the
    point of the B conditions: the toponym is ambiguous and the
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

PLAN_VERSION = "4"   # 2: items carry env_theme; 3: B/NOISE_B carry a clue, EXPLICIT is the bare label; 4: fixed core, type groups

# Defaults for the plan: section of config.yaml. config.yaml is the source of
# truth; these apply only when a key is absent from it.
DEFAULT_MAX_PLACE_AMBIGUITY = 1
DEFAULT_SURPLUS_FACTOR = 1.0
DEFAULT_MAX_REPLICATES_PER_PAIR = 6
DEFAULT_LENGTH_BANDS = ["7_to_30_words"]

# The conditions, and what each one requires to be feasible for a given place.
# n_clues        how many entity clues appear in the text
# needs_toponym  "ambiguous" -> an alias shared with other places must exist
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
    place_type: str
    country: str
    continent: str
    places_sharing_name: int
    sitelinks: int
    wikidata_links: int
    # Added for its shared name (b_extra_places.csv). Since 2026-10-27 it is
    # planned like any other place wherever its clues allow; the flag is kept
    # so results can still be split by it.
    b_only: bool = False
    # Date the place joined the set (ISO, from the place file's added column;
    # blank for the older places). plan.min_sitelinks_from compares against it.
    added: str = ""
    # The environmental topics a post about this place may take, from the
    # place file's env_themes column (";"-separated). Empty when it has none.
    env_themes: list[str] = field(default_factory=list)
    clues: list[dict[str, Any]] = field(default_factory=list)
    surface_forms: list[dict[str, Any]] = field(default_factory=list)
    # The place's own label as 04b recorded it (source "label"), kept before any
    # filtering: kind says whether other places share it, and an ambiguous one
    # carries shared_by and top_namesake. {} when 04b has no label form.
    label_form: dict[str, Any] = field(default_factory=dict)
    # The clues a single-clue condition may use: place_ambiguity within the
    # configured ceiling and, when so configured, not leaking the place name.
    single_clues: list[dict[str, Any]] = field(default_factory=list)
    # Complete size-3 best_sets entries, for A3. Each is
    # {"source_set_index": i, "clues": [...]}.
    triple_sets: list[dict[str, Any]] = field(default_factory=list)
    # The place's name and surface forms, its country forms and its region
    # forms. A distractor from elsewhere is checked against these too.
    name_forms: list[str] = field(default_factory=list)
    country_forms: list[str] = field(default_factory=list)
    region_forms: list[str] = field(default_factory=list)
    # distractor_clues() memoised per donor qid; see reset_caches().
    _distractors: dict[str, list[dict[str, Any]]] = field(
        default_factory=dict, repr=False, compare=False)

    def reset_caches(self) -> None:
        """Forget memoised distractors. Call after name_forms or any donor's
        single_clues change (load_clue_sets does)."""
        self._distractors.clear()

    def distractor_clues(self, donor: "Place") -> list[dict[str, Any]]:
        """
        The donor's eligible clues whose labels name nothing of this place.
        Memoised per donor; do not mutate the returned list.
        """
        cached = self._distractors.get(donor.qid)
        if cached is None:
            # Not an entity that also points at this place: the distractor must
            # differ from what is to be guessed.
            own = {c.get("clue_qid") for c in self.clues}
            cached = [
                c for c in donor.single_clues
                if c.get("clue_qid") not in own
                and label_leak(c["clue_label"], self.name_forms, self.country_forms,
                               self.region_forms) is None
            ]
            self._distractors[donor.qid] = cached
        return cached

    def ambiguous_forms(self) -> list[dict[str, Any]]:
        return [f for f in self.surface_forms if f.get("kind") == "ambiguous_alias"]


def load_places(path: Path) -> dict[str, Place]:
    """
    Read the place CSV (01_places_final.csv). The file is written with a UTF-8 BOM by Excel,
    hence utf-8-sig; reading it as plain utf-8 corrupts the first column name
    and every lookup on it fails.
    """
    if not path.exists():
        raise SystemExit(f"place file not found: {path}")

    places: dict[str, Place] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"place_name", "type", "country", "continent", "wikidata_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")

        for row in reader:
            qid = (row.get("wikidata_id") or "").strip()
            if not qid:
                continue
            places[qid] = Place(
                qid=qid,
                name=(row.get("place_name") or "").strip(),
                place_type=(row.get("type") or "").strip(),
                country=(row.get("country") or "").strip(),
                continent=(row.get("continent") or "").strip(),
                places_sharing_name=_as_int(row.get("places_sharing_name"), 1),
                sitelinks=_as_int(row.get("wikimedia_sitelinks"), 0),
                wikidata_links=_as_int(row.get("wikidata_links"), 0),
                b_only=(row.get("b_only") or "").strip().lower() == "true",
                added=(row.get("added") or "").strip(),
                env_themes=[t.strip() for t in (row.get("env_themes") or "").split(";")
                            if t.strip()],
            )
    if not places:
        raise SystemExit(f"{path} yielded no places")
    return places


def _read_jsonl_by_place(path: Path, what: str) -> dict[str, tuple[int, dict]]:
    """place_qid -> (line number, record). Duplicates and bad lines are fatal."""
    if not path.exists():
        raise SystemExit(f"{what} file not found: {path}")
    by_place: dict[str, tuple[int, dict]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no} is not valid JSON: {exc}") from exc
            qid = (record.get("place_qid") or "").strip()
            if not qid:
                raise SystemExit(f"{path}:{line_no} has no place_qid")
            if qid in by_place:
                raise SystemExit(f"{path}:{line_no} is a duplicate entry for {qid}")
            by_place[qid] = (line_no, record)
    return by_place


def _load_clue_types(path: Path) -> dict[str, dict[str, Any]]:
    """clue_qid -> its 05b_clue_types.jsonl row. toponym must be decided."""
    if not path.exists():
        raise SystemExit(f"clue type file not found: {path}. "
                         f"Run scripts/05b_clue_types.py.")
    types: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("toponym") is None:
                raise SystemExit(f"{path}:{line_no} {row.get('clue_qid')} has toponym "
                                 f"null (its lookup failed). Re-run "
                                 f"scripts/05b_clue_types.py.")
            types[row["clue_qid"]] = row
    return types


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


@dataclass(frozen=True)
class ClueRules:
    """The plan: flags that decide which clues and forms are held back."""
    max_place_ambiguity: int = DEFAULT_MAX_PLACE_AMBIGUITY
    exclude_qualified: bool = True
    exclude_toponyms: bool = True
    exclude_label_leaks: bool = True
    # Toponym categories that may still be clues (rivers: naming one does not
    # name the place). body_of_water rides along, since 05b gives rivers both.
    toponym_clue_categories: frozenset[str] = frozenset()

    @classmethod
    def from_config(cls, plan_cfg: dict[str, Any]) -> "ClueRules":
        return cls(
            max_place_ambiguity=int(plan_cfg.get(
                "max_place_ambiguity_for_single_clue", DEFAULT_MAX_PLACE_AMBIGUITY)),
            exclude_qualified=bool(plan_cfg.get("exclude_qualified_ambiguous_forms", True)),
            exclude_toponyms=bool(plan_cfg.get("exclude_toponym_clues", True)),
            exclude_label_leaks=bool(plan_cfg.get("exclude_label_leaks", True)),
            toponym_clue_categories=frozenset(plan_cfg.get("toponym_clue_categories") or ()),
        )

    def allows_toponym(self, categories: Sequence[str]) -> bool:
        """A toponym clue whose categories are all allowed (body_of_water aside)."""
        cats = set(categories)
        return bool(cats & self.toponym_clue_categories) and \
            cats <= self.toponym_clue_categories | {"body_of_water"}


def check_alias_record(alias_path: Path, alias_line: int,
                       alias_record: dict[str, Any]) -> None:
    """Abort when a 04b_place_aliases.jsonl record lacks a field the planner needs."""
    for key in ("surface_forms", "country_forms", "region_forms"):
        if key not in alias_record:
            raise SystemExit(f"{alias_path}:{alias_line} has no {key} field. "
                             f"Re-run scripts/04b_place_aliases.py.")


def name_forms_of(place: Place, alias_record: dict[str, Any]
                  ) -> tuple[list[str], list[str], list[str]]:
    """
    (place forms, country forms, region forms): the strings that name this
    place and its country. The derived "<name>, <country>" is left out of the
    place forms: it is the name plus the country, and each of those is checked
    on its own.
    """
    place_forms = [place.name] + [
        f["text"] for f in alias_record["surface_forms"] or []
        if f.get("source") != "derived"
    ]
    country_forms = [f["text"] for f in alias_record["country_forms"] or []]
    region_forms = [f["text"] for f in alias_record["region_forms"] or []]
    return place_forms, country_forms, region_forms


def best_sets_of(clue_path: Path, clue_sets: dict[str, tuple[int, dict]],
                 qid: str) -> list[dict[str, Any]]:
    """The place's best_sets entries, [] when stage 5 exported none."""
    if qid not in clue_sets:
        return []
    line_no, record = clue_sets[qid]
    if "best_sets" not in record:
        raise SystemExit(
            f"{clue_path}:{line_no} has no best_sets field; it holds "
            f"{sorted(record)}. Check the stage 5 export."
        )
    return record["best_sets"] or []


def collect_clues(
    qid: str,
    best_sets: list[dict[str, Any]],
    clue_path: Path,
    types_path: Path,
    clue_types: dict[str, dict[str, Any]],
    place_forms: list[str],
    country_forms: list[str],
    region_forms: list[str],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    The union of best_sets[].clues[], de-duplicated on (clue_qid, property),
    keyed by that pair. Each clue records the sizes of the sets it appeared in.
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for set_index, entry in enumerate(best_sets):
        for raw in entry.get("clues") or []:
            for key in ("clue_qid", "clue_label", "property"):
                if not raw.get(key):
                    raise SystemExit(
                        f"{clue_path}: {qid} best_sets[{set_index}] has a clue "
                        f"missing {key!r}: {raw}"
                    )
            if not raw.get("property_label"):
                raise SystemExit(
                    f"{clue_path}: {qid} has a clue with no property_label. "
                    f"Re-run scripts/05_score_sets.py, which now carries it "
                    f"through from 04_clues_scored.jsonl."
                )
            key = (raw["clue_qid"], raw["property"])
            if raw["clue_qid"] not in clue_types:
                raise SystemExit(
                    f"{types_path} has no row for clue {raw['clue_qid']} "
                    f"({raw['clue_label']}). Re-run scripts/05b_clue_types.py."
                )
            if key not in by_key:
                by_key[key] = {
                    "clue_qid": raw["clue_qid"],
                    "clue_label": raw["clue_label"],
                    "property": raw["property"],
                    "property_label": raw["property_label"],
                    "entity_type": raw.get("property_family"),
                    "place_ambiguity": raw.get("place_ambiguity"),
                    "surface_ambiguity": raw.get("surface_ambiguity"),
                    "discriminativeness": None,
                    "source_set_sizes": [],
                    "is_toponym": clue_types[raw["clue_qid"]]["toponym"],
                    # A person may be written by a shorter name in the post
                    # (finalise.py accepts the surname); anything else needs
                    # its label or a Wikidata alias.
                    "is_human": "human" in (clue_types[raw["clue_qid"]].get("instance_of") or []),
                    "aliases": list(raw.get("aliases") or []),
                    "label_leak": label_leak(raw["clue_label"], place_forms,
                                              country_forms, region_forms),
                }
            by_key[key]["source_set_sizes"].append(entry.get("size", len(entry["clues"])))
    return by_key


def clue_excluded_because(
    clue: dict[str, Any], clue_types: dict[str, dict[str, Any]], rules: ClueRules
) -> str | None:
    """Why this clue is held back from every condition, or None."""
    if rules.exclude_toponyms and clue["is_toponym"]:
        cats = clue_types[clue["clue_qid"]].get("categories") or []
        if not rules.allows_toponym(cats):
            return f"is itself a place name ({', '.join(cats)})"
    if rules.exclude_label_leaks and clue["label_leak"]:
        return clue["label_leak"]
    return None


def select_clues(
    qid: str,
    place: Place,
    best_sets: list[dict[str, Any]],
    by_key: dict[tuple[str, str], dict[str, Any]],
    clue_types: dict[str, dict[str, Any]],
    rules: ClueRules,
) -> list[dict[str, Any]]:
    """
    Fill place.clues, place.single_clues and place.triple_sets from the
    collected clues and return one clue_excluded record per held-back clue.
    A best_sets entry containing an excluded clue is excluded whole, and so is
    one naming the same clue entity twice.
    """
    place.clues = list(by_key.values())
    reasons = {id(c): clue_excluded_because(c, clue_types, rules) for c in place.clues}

    held_back = [
        {
            "record": "clue_excluded",
            "place_qid": qid,
            "place_name": place.name,
            "clue_qid": clue["clue_qid"],
            "clue_label": clue["clue_label"],
            "property": clue["property"],
            "reason": reasons[id(clue)],
        }
        for clue in place.clues if reasons[id(clue)]
    ]
    usable = [c for c in place.clues if reasons[id(c)] is None]
    place.single_clues = [
        c for c in usable
        if c["place_ambiguity"] is not None
        and c["place_ambiguity"] <= rules.max_place_ambiguity
    ]
    place.triple_sets = []
    for set_index, entry in enumerate(best_sets):
        if len(entry.get("clues") or []) != 3 or entry.get("resolves") is not True:
            continue
        members = [by_key[(c["clue_qid"], c["property"])] for c in entry["clues"]]
        if any(reasons[id(c)] for c in members):
            continue
        # One entity reached twice (born and died there: P19 and P20) makes
        # a "three-clue" set two clues; finalise.py drops it as repeated_clue.
        if len({c["clue_qid"] for c in members}) < len(members):
            continue
        place.triple_sets.append({"source_set_index": set_index, "clues": members})
    return held_back


def label_form_of(source_forms: list[dict[str, Any]]) -> dict[str, Any]:
    """
    The place's own label as 04b recorded it (source "label"), or {} when it
    has none. kind is "exact" when no other place shares the label and
    "ambiguous_alias" when others do.
    """
    return next((f for f in source_forms if f.get("source") == "label"), {})


def form_excluded_because(
    form: dict[str, Any], country_forms: list[str], rules: ClueRules,
) -> str | None:
    """Why this surface form is held back, or None."""
    if (rules.exclude_qualified and form.get("kind") == "ambiguous_alias"
            and _is_qualified(form["text"], country_forms)):
        return "ambiguous form already qualified or contains the country"
    return None


def filter_forms(
    qid: str,
    place: Place,
    source_forms: list[dict[str, Any]],
    country_forms: list[str],
    rules: ClueRules,
) -> list[dict[str, Any]]:
    """
    Fill place.surface_forms with the forms that survive and return one
    form_excluded record per form held back.

    Only ambiguous forms are filtered (B and NOISE_B draw from them). Exact
    forms are not drawn at all: EXPLICIT uses the bare label for every place,
    which is kept whole in place.label_form before any filtering.
    """
    place.label_form = label_form_of(source_forms)
    held_back: list[dict[str, Any]] = []
    forms = []
    for form in source_forms:
        reason = form_excluded_because(form, country_forms, rules)
        if reason:
            held_back.append({
                "record": "form_excluded",
                "place_qid": qid,
                "place_name": place.name,
                "text": form["text"],
                "kind": form.get("kind"),
                "source": form.get("source"),
                "shared_by": form.get("shared_by"),
                "reason": reason,
            })
            continue
        forms.append(form)
    place.surface_forms = forms
    return held_back


def load_clue_sets(
    clue_path: Path,
    alias_path: Path,
    types_path: Path,
    places: dict[str, Place],
    plan_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Adapter from the stage 4/5 outputs to the planner's per-place shape.
    Fills place.clues, place.single_clues, place.triple_sets and
    place.surface_forms in place, and returns one log record per clue or form
    that was held back, for the rejection log.

    Reads
        05_clue_sets.jsonl       best_sets[].clues[] per place
        04b_place_aliases.jsonl  surface_forms per place
        05b_clue_types.jsonl     which clue entities are themselves places
    The place attributes come from the place CSV via load_places().

    THE CLUE LIST is the union of best_sets[].clues[], de-duplicated on
    (clue_qid, property) (collect_clues). Stage 5 exports only its best sets,
    so this is the pool the planner draws from, and it is a ranked selection,
    not every clue that passed the filter. Each clue carries:
        entity_type         <- property_family
        discriminativeness  null: no such score exists upstream, and
                            finalise.py must be able to see that
        source_set_sizes    the sizes of the best_sets containing it

    SINGLE-CLUE CONDITIONS (A1, NOISE_A1) draw only from clues with
    place_ambiguity <= max_place_ambiguity_for_single_clue. One clue that
    several places satisfy gives the item no single correct answer.

    A3 draws a complete size-3 best_sets entry, never three loose clues: stage
    5 kept those sets because their intersection resolves to the place, and a
    random three-clue group has no such guarantee.

    A set containing an excluded clue is excluded whole (select_clues).

    TOPONYM CLUES - a village, island, river, street, time zone - are excluded
    the same way when exclude_toponym_clues is set. Such a clue puts a real
    place name into a post whose prompt forbids place names (A1, A3,
    NOISE_A1), and the smoke run showed the model
    inventing around the contradiction. Each clue carries is_toponym either
    way.

    LABEL LEAKS are excluded when exclude_label_leaks is set: a clue whose own
    label contains any surface form of its place ("Rock-cut tombs of Qaw
    el-Kebir" for Tjebu) or the country's name or demonym ("Tuvalu Association
    of Sports", "MIAT Mongolian Airlines"). The prompt would require the post
    to carry the name it forbids. Each clue carries label_leak, the reason or
    null, either way.

    AMBIGUOUS FORMS that are already qualified (see _is_qualified) are held
    back when exclude_qualified_ambiguous_forms is set.

    EXPLICIT names the place by its bare label, unique or shared, so the
    plan can say how many places share it (filter_forms keeps label_form).
    """
    rules = ClueRules.from_config(plan_cfg)

    clue_sets = _read_jsonl_by_place(clue_path, "clue set")
    aliases = _read_jsonl_by_place(alias_path, "place alias")
    clue_types = _load_clue_types(types_path)
    held_back: list[dict[str, Any]] = []

    for source, path in ((clue_sets, clue_path), (aliases, alias_path)):
        # The place file may be a subset of what these files cover (the
        # environmental set leaves out stadiums, palaces and the like); only
        # the places in it are planned.
        extra = sorted(set(source) - set(places))
        if extra:
            print(f"note: {path.name} has {len(extra)} places that are not in the "
                  f"place file; they are not planned")

    for qid, place in places.items():
        if qid not in aliases:
            raise SystemExit(f"{alias_path} has no record for {qid} ({place.name}). "
                             f"Re-run scripts/04b_place_aliases.py.")
        alias_line, alias_record = aliases[qid]
        check_alias_record(alias_path, alias_line, alias_record)
        # 04b's live count, fetched with the namesakes' counts so the two
        # compare like for like; the CSV column is blank for some places.
        if alias_record.get("sitelinks") is not None:
            place.sitelinks = int(alias_record["sitelinks"])
        place_forms, country_forms, region_forms = name_forms_of(place, alias_record)
        place.name_forms = place_forms
        place.country_forms = country_forms
        place.region_forms = region_forms

        best_sets = best_sets_of(clue_path, clue_sets, qid)
        by_key = collect_clues(qid, best_sets, clue_path, types_path, clue_types,
                               place_forms, country_forms, region_forms)
        held_back += select_clues(qid, place, best_sets, by_key, clue_types, rules)
        held_back += filter_forms(qid, place, alias_record["surface_forms"] or [],
                                  country_forms, rules)

    # The name forms and donor clues just changed, so drop memoised distractors.
    for place in places.values():
        place.reset_caches()
    return held_back


def drop_little_known(places: dict[str, Place], plan_cfg: dict[str, Any]
                      ) -> list[dict[str, Any]]:
    """
    Remove places under plan.min_sitelinks (the live count from the alias
    file) from `places`, for every condition and as noise donors, and return
    one rejection record each. With plan.min_sitelinks_from set, only places
    whose added date is on or after it are held to the floor; older places
    stay whatever their count. No floor when min_sitelinks is unset.
    plan.max_sitelinks likewise leaves out the very best-known places (every
    place, whatever its date), whose posts would be too easy to place.
    """
    floor, ceiling = plan_cfg.get("min_sitelinks"), plan_cfg.get("max_sitelinks")
    since = str(plan_cfg.get("min_sitelinks_from") or "")
    records = []
    for q, p in list(places.items()):
        if floor is not None and p.sitelinks < int(floor) and (not since or p.added >= since):
            reason = f"{p.sitelinks} sitelinks, under min_sitelinks {floor}"
        elif ceiling is not None and p.sitelinks > int(ceiling):
            reason = f"{p.sitelinks} sitelinks, over max_sitelinks {ceiling}"
        else:
            continue
        records.append({"record": "place_rejected", "place_qid": q, "place_name": p.name,
                        "reason": reason})
        del places[q]
    return records


def load_clue_articles(path: Path) -> dict[str, int]:
    """clue_qid -> Wikipedia articles about it (external/clues.jsonl), max over rows."""
    articles: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                n = int(row.get("clue_wikipedia_articles") or 0)
                articles[row["clue_qid"]] = max(articles.get(row["clue_qid"], 0), n)
    return articles


def drop_unknown_clues(places: dict[str, Place], articles_path: Path,
                       plan_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Hold back clues nobody could know: with plan.min_clue_wikipedia_articles
    set, a single clue (A1, B, NOISE and distractors) needs at least that many
    Wikipedia articles about it, and a three-clue set (A3) needs at least one
    member that has. A post built only on entities with no article in any
    language is a guess, not a test. One record per place affected.
    """
    floor = plan_cfg.get("min_clue_wikipedia_articles")
    if not floor:
        return []
    articles = load_clue_articles(articles_path)
    known = lambda c: articles.get(c["clue_qid"], 0) >= int(floor)  # noqa: E731
    records = []
    for qid, place in places.items():
        singles = [c for c in place.single_clues if known(c)]
        triples = [s for s in place.triple_sets if any(known(c) for c in s["clues"])]
        lost = (len(place.single_clues) - len(singles), len(place.triple_sets) - len(triples))
        if any(lost):
            records.append({"record": "clue_excluded", "place_qid": qid,
                            "place_name": place.name, "clue_qid": None, "clue_label": None,
                            "property": None,
                            "reason": "clues with no Wikipedia article held back",
                            "single_clues_held_back": lost[0],
                            "three_clue_sets_held_back": lost[1]})
        place.single_clues, place.triple_sets = singles, triples
        place.reset_caches()
    return records


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
    at least one single-clue-eligible clue to draw from. It may be on the same
    continent or on another, but not in the same country as the post's place.
    The distractor has to
    point somewhere specific to be wrong in a useful way, so it meets the same
    place_ambiguity and label-leak rules as an A1 clue.
    """
    # A distractor whose label names THIS place ("2013 South American
    # Championships" carries "American" into a United States item) would put a
    # forbidden name into the post, so donors are counted only for the clues
    # that survive that check.
    return [
        qid
        for qid, other in places.items()
        if qid != place.qid
        and donor_relation(place, other) != "same_country"
        and place.distractor_clues(other)
    ]


def donor_relation(place: Place, donor: Place) -> str:
    """How far the distractor's place is from the item's place: same_country,
    same_continent or other_continent (what is known from the place table)."""
    countries = lambda p: {c.strip() for c in p.country.split(";") if c.strip()}  # noqa: E731
    if countries(place) & countries(donor):
        return "same_country"
    if place.continent and place.continent == donor.continent:
        return "same_continent"
    return "other_continent"


def feasible(
    place: Place, condition: str, places: dict[str, Place]
) -> tuple[bool, str]:
    """
    Can this place support this condition? Returns (ok, reason_if_not).

    This is the only place where an item is allowed to disappear. Nothing
    downstream may quietly repair an infeasible pair by swapping in a
    different place, clue or condition.
    """
    spec = CONDITION_SPECS[condition]

    if spec["noise"] and not noise_donors(place, places):
        return False, "no other place has an eligible clue to donate"

    n_needed = spec["n_clues"]
    if n_needed == 1 and not place.single_clues:
        return False, "no single-clue-eligible clue (place_ambiguity ceiling, label leak or toponym)"
    if n_needed == 3 and not place.triple_sets:
        return False, "no complete size-3 best_sets entry that resolves and holds no excluded clue"
    if n_needed not in (0, 1, 3):
        raise SystemExit(f"{condition} needs {n_needed} clues; the adapter only "
                         f"supplies single clues and size-3 sets")

    toponym = spec["needs_toponym"]
    if toponym == "ambiguous" and not place.ambiguous_forms():
        return False, "no usable ambiguous surface form"
    if toponym == "label" and not place.label_form:
        return False, "no label form recorded in the place aliases"

    return True, ""


# --------------------------------------------------------------------------- #
# per-item decisions
# --------------------------------------------------------------------------- #

def choose_clues(
    place: Place, n: int, stream: RandomStream
) -> tuple[list[dict[str, Any]], int | None]:
    """
    Returns (clues, source_set_index).

    One clue: a uniform draw over the place's single-clue-eligible clues.
    Three clues: a uniform draw over its complete size-3 best_sets entries,
    taken whole; source_set_index is that entry's index in best_sets.
    """
    if n == 0:
        return [], None
    if n == 1:
        return [stream.pick(place.single_clues)], None
    if n == 3:
        chosen = stream.pick(place.triple_sets)
        return list(chosen["clues"]), chosen["source_set_index"]
    raise SystemExit(f"no clue selection rule for {n} clues")


def choose_surface_form(
    place: Place, condition: str, stream: RandomStream
) -> dict[str, Any]:
    kind = CONDITION_SPECS[condition]["needs_toponym"]
    if kind is None:
        return {"kind": "none", "text": None, "shared_by": None}
    if kind == "label":
        # No draw: the bare place name, whatever else shares it. name_ambiguous
        # and top_namesake carry the same facts B records, so EXPLICIT rows can
        # be binned like B rows.
        label = place.label_form
        return {
            "kind": "exact",
            "text": place.name,
            "shared_by": label.get("shared_by"),
            "name_ambiguous": label.get("kind") == "ambiguous_alias",
            "geo_shared_by": label.get("geo_shared_by"),
            "top_namesake": label.get("top_namesake"),
        }
    chosen = stream.pick(place.ambiguous_forms())
    form = {
        "kind": "ambiguous_alias",
        "text": chosen["text"],
        "shared_by": chosen.get("shared_by"),
        "name_ambiguous": True,
    }
    if kind == "ambiguous":
        # The best-known other place 161+ km away with this name, so results
        # can be split by whether the intended place is the obvious reading
        # (compare its sitelinks with the item's place_sitelinks).
        form["geo_shared_by"] = chosen.get("geo_shared_by")
        form["top_namesake"] = chosen.get("top_namesake")
    return form


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
        "entity_type": donor_clue.get("entity_type"),
        "place_ambiguity": donor_clue.get("place_ambiguity"),
        "is_toponym": donor_clue.get("is_toponym"),
        "label_leak": donor_clue.get("label_leak"),
        "is_human": donor_clue.get("is_human"),
        "aliases": donor_clue.get("aliases") or [],
        "donor_place_qid": donor_qid,
        "donor_place_name": donor.name,
        "donor_continent": donor.continent,
    }


def choose_register(config: dict[str, Any], stream: RandomStream) -> dict[str, Any]:
    return stream.pick(config["registers"])


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
            "length_band": payload["length_band"],
            "slots": payload["slots"],
            "env_theme": payload.get("env_theme"),
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


def per_condition_target(config: dict[str, Any]) -> int | None:
    """
    Items to plan per condition: finalise.per_condition (the final size) times
    plan.surplus_factor, rounded up. None when finalise.per_condition is unset,
    which keeps the older replicate-based sizing.
    """
    final = (config.get("finalise") or {}).get("per_condition")
    if final is None:
        return None
    factor = float(config["plan"].get("surplus_factor", DEFAULT_SURPLUS_FACTOR))
    return math.ceil(int(final) * factor)


def type_group(place: Place, plan_cfg: dict[str, Any]) -> str:
    """The place's type group (plan.type_groups), or its own type when unlisted."""
    return (plan_cfg.get("type_groups") or {}).get(place.place_type, place.place_type)


def group_quotas(supply: dict[str, int], targets: dict[str, float], n: int,
                 label: str = "") -> dict[str, int]:
    """
    Split n items over type groups by their target shares (largest remainder),
    each capped by its supply. A group short of places hands its shortfall to
    the groups that still have places, in proportion to their targets (an
    untargeted group counts as a tiny share), so the total is n whenever the
    supply allows. Every shortfall is reported.
    """
    groups = sorted(set(supply) | set(targets))
    weight = {g: targets.get(g, 0.0) or 1e-6 for g in groups}
    quotas = dict.fromkeys(groups, 0)
    remaining = min(n, sum(supply.values()))
    while remaining > 0:
        open_groups = [g for g in groups if quotas[g] < supply.get(g, 0)]
        if not open_groups:
            break
        total = sum(weight[g] for g in open_groups)
        raw = {g: remaining * weight[g] / total for g in open_groups}
        share = {g: math.floor(v) for g, v in raw.items()}
        for g in sorted(open_groups, key=lambda g: (share[g] - raw[g], g))[:remaining - sum(share.values())]:
            share[g] += 1
        progressed = 0
        for g in open_groups:
            got = min(share[g], supply[g] - quotas[g])
            quotas[g] += got
            progressed += got
        remaining -= progressed
        if not progressed:
            break
    ideal = {g: round(n * targets.get(g, 0.0)) for g in groups}
    short = {g: ideal[g] - quotas[g] for g in groups if quotas[g] < ideal[g]}
    if short:
        print(f"  note: {label} type groups short of their target share: "
              + ", ".join(f"{g} {quotas[g]}/{ideal[g]}" for g in sorted(short)),
              file=sys.stderr)
    return {g: q for g, q in quotas.items() if q}


def country_key(place: Place) -> str:
    """The country a place counts against for balancing (its first listed one)."""
    return place.country.split(";")[0].strip()


def pick_country_balanced(qids: list[str], n: int, places: dict[str, Place],
                          country_uses: dict[str, int], stream: RandomStream,
                          cap: int | None = None) -> list[str]:
    """
    Up to n of qids, each drawn from the countries used least so far in this
    condition (country_uses, plus the picks made here), so no country gets
    another place while one with fewer still has a place to offer, and none
    goes past `cap`. Ties are drawn from the stream. country_uses itself is
    not changed.
    """
    uses = dict(country_uses)
    pool = sorted(qids)
    chosen: list[str] = []
    for _ in range(n):
        if cap is not None:
            pool = [q for q in pool if uses.get(country_key(places[q]), 0) < cap]
        if not pool:
            break
        least = min(uses.get(country_key(places[q]), 0) for q in pool)
        qid = stream.pick([q for q in pool if uses.get(country_key(places[q]), 0) == least])
        chosen.append(qid)
        pool.remove(qid)
        key = country_key(places[qid])
        uses[key] = uses.get(key, 0) + 1
    return chosen


def target_slate(
    places: dict[str, Place],
    feasible_by_condition: dict[str, list[str]],
    conditions: list[str],
    target: int,
    max_per_pair: int,
    stream: RandomStream,
    plan_cfg: dict[str, Any] | None = None,
    prefer: set[str] | None = None,
) -> list[tuple[str, str, int]]:
    """
    `target` items per condition, spread as widely as the supply allows.

    With plan.type_targets set, the target is first split into type groups
    (group_quotas) and each group is filled separately as below. With `prefer`
    (the core places), each group is filled from preferred places first, so
    conditions outside the core share as many places with it as they can.

    In rounds: while the open places (feasible, under max_per_pair) number no
    more than what is still needed, each gets one more item; otherwise the
    remaining items go to places drawn without replacement, split evenly over
    continents (allocate_even). So no place gets a second item while another
    feasible place has none, and a condition short of places repeats them
    evenly. A condition that cannot reach the target is reported, not padded.
    """
    plan_cfg = plan_cfg or {}
    targets = plan_cfg.get("type_targets") or {}
    slate: list[tuple[str, str, int]] = []
    for condition in conditions:
        feasible_qids = feasible_by_condition.get(condition, [])
        groups: dict[str, list[str]] = defaultdict(list)
        for qid in feasible_qids:
            groups[type_group(places[qid], plan_cfg) if targets else ""].append(qid)
        quotas = (group_quotas({g: len(v) * max_per_pair for g, v in groups.items()},
                               targets, target, condition)
                  if targets else {"": target})
        quotas.pop("*", None)
        counts: dict[str, int] = defaultdict(int)
        # Places per country so far in this condition, across type groups.
        country_uses: dict[str, int] = defaultdict(int)
        cap = plan_cfg.get("max_places_per_country")
        under_cap = lambda q: cap is None or country_uses[country_key(places[q])] < cap  # noqa: E731
        balance = plan_cfg.get("balance_countries") or cap is not None
        planned = 0
        # A group that cannot fill its quota within the country cap leaves the
        # rest to a last pass over every group ("*"), still within the cap.
        for group in sorted(quotas) + ["*"]:
            remaining = quotas[group] if group != "*" else target - planned
            members = groups[group] if group != "*" else feasible_qids
            while remaining > 0:
                open_places = [q for q in members if counts[q] < max_per_pair and under_cap(q)]
                if not open_places:
                    break
                if prefer:
                    open_places = [q for q in open_places if q in prefer] or open_places
                if remaining >= len(open_places) and not balance:
                    chosen = open_places
                else:
                    by_continent: dict[str, list[str]] = defaultdict(list)
                    for qid in open_places:
                        by_continent[places[qid].continent].append(qid)
                    take = allocate_even({c: len(v) for c, v in by_continent.items()},
                                         remaining)
                    chosen = []
                    for continent in sorted(by_continent):
                        if not take[continent]:
                            continue
                        if balance:
                            chosen += pick_country_balanced(by_continent[continent],
                                                            take[continent], places,
                                                            country_uses, stream, cap)
                        else:
                            chosen += stream.sample(by_continent[continent], take[continent])
                if not chosen:
                    break
                for qid in chosen:
                    slate.append((qid, condition, counts[qid]))
                    counts[qid] += 1
                    country_uses[country_key(places[qid])] += 1
                remaining -= len(chosen)
                planned += len(chosen)
        if planned < target:
            print(f"  note: {condition} planned {planned} of {target}: every feasible "
                  f"place is used ({max_per_pair} per place) or its country is at "
                  f"max_places_per_country {cap}", file=sys.stderr)
    return slate


def build_slate(
    places: dict[str, Place], config: dict[str, Any], stream: RandomStream
) -> tuple[list[tuple[str, str, int]], list[dict[str, Any]]]:
    """
    Decide which (place, condition, replicate) triples to plan.

    With finalise.per_condition set (the size of each condition in the final
    dataset), every condition is planned at per_condition_target() items: that
    size times surplus_factor. The surplus exists because the model will
    sometimes name the place outright in a condition that forbids it; those
    items are dropped in finalise.py, so the plan must be larger than the
    final set, but only by the expected loss. See target_slate() for how the
    items are spread over places.

    Without it, the older sizing applies. Base pass: every feasible pair gets
    the replicate count from the config. Surplus pass: extra replicates are
    drawn at random until the total is surplus_factor times the base.
    """
    plan_cfg = config["plan"]
    replicates: dict[str, int] = plan_cfg["replicates"]
    surplus_factor = float(plan_cfg.get("surplus_factor", DEFAULT_SURPLUS_FACTOR))
    max_per_pair = int(plan_cfg.get("max_replicates_per_pair",
                                    DEFAULT_MAX_REPLICATES_PER_PAIR))

    unknown = set(replicates) - set(CONDITION_SPECS)
    if unknown:
        raise SystemExit(f"config names unknown conditions: {sorted(unknown)}")

    slate: list[tuple[str, str, int]] = []
    rejections: list[dict[str, Any]] = []
    counts: dict[tuple[str, str], int] = defaultdict(int)
    feasible_by_condition: dict[str, list[str]] = defaultdict(list)

    for qid in sorted(places):
        place = places[qid]
        for condition in sorted(replicates):
            ok, reason = feasible(place, condition, places)
            if not ok:
                rejections.append(
                    {
                        "record": "pair_rejected",
                        "place_qid": qid,
                        "place_name": place.name,
                        "condition": condition,
                        "reason": reason,
                        "clue_count": len(place.clues),
                        "single_clue_eligible": len(place.single_clues),
                        "size3_sets": len(place.triple_sets),
                        "ambiguous_forms": len(place.ambiguous_forms()),
                    }
                )
                continue
            feasible_by_condition[condition].append(qid)

    # cover_all_places conditions get replicates[condition] items on every
    # place that can support them, not an evenly spread share of a target.
    cover_all = sorted(set(plan_cfg.get("cover_all_places") or []) & set(replicates))
    covered = [(qid, condition, replicate)
               for condition in cover_all
               for qid in sorted(feasible_by_condition[condition])
               for replicate in range(replicates[condition])]

    target = per_condition_target(config)
    if target is not None:
        spread = [c for c in sorted(replicates) if c not in cover_all]
        # The core conditions share one set of places, chosen once from the
        # places feasible for all of them, so their results compare like for
        # like. Each core condition then has one item per core place.
        core_conditions = [c for c in plan_cfg.get("core_conditions") or [] if c in spread]
        slate = []
        core: list[str] = []
        if core_conditions:
            pool = sorted(set.intersection(*(set(feasible_by_condition[c])
                                             for c in core_conditions)))
            core = [q for q, _, _ in target_slate(places, {"CORE": pool}, ["CORE"], target,
                                                  1, stream, plan_cfg)]
            slate += [(q, c, 0) for c in core_conditions for q in core]
        # The other conditions take core places first wherever they can (B:
        # core places with a shared name), so they share most places with it.
        slate += target_slate(places, feasible_by_condition,
                              [c for c in spread if c not in core_conditions],
                              target, max_per_pair, stream, plan_cfg, prefer=set(core))
        if plan_cfg.get("cover_used_places_only"):
            used = {q for q, _, _ in slate}
            covered = [(q, c, r) for q, c, r in covered if q in used]
        return slate + covered, rejections

    for condition in sorted(replicates):
        for qid in feasible_by_condition[condition]:
            for replicate in range(replicates[condition]):
                slate.append((qid, condition, replicate))
                counts[(qid, condition)] += 1
    slate.sort()

    base_total = len(slate)
    target_total = round(base_total * surplus_factor)
    extra_needed = max(0, target_total - base_total)

    if extra_needed:
        # Only pairs that are feasible and not already at the per-pair cap.
        pool = [
            (qid, condition)
            for (qid, condition), n in counts.items()
            if n < max_per_pair and condition not in cover_all
        ]
        pool.sort()
        added = 0
        while added < extra_needed and pool:
            qid, condition = stream.pick(pool)
            if counts[(qid, condition)] >= max_per_pair:
                pool = [p for p in pool if counts[p] < max_per_pair]
                continue
            slate.append((qid, condition, counts[(qid, condition)]))
            counts[(qid, condition)] += 1
            added += 1
        if added < extra_needed:
            print(
                f"  note: surplus target was {extra_needed} extra items, "
                f"added {added} before every pair hit the cap of {max_per_pair}",
                file=sys.stderr,
            )

    return slate, rejections


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def plan_items(
    places: dict[str, Place],
    config: dict[str, Any],
    stream: RandomStream,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    slate, rejections = build_slate(places, config, stream)

    items: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for qid, condition, replicate in slate:
        place = places[qid]
        position_before = stream.position

        clues, source_set_index = choose_clues(
            place, CONDITION_SPECS[condition]["n_clues"], stream
        )
        surface_form = choose_surface_form(place, condition, stream)
        noise_entity = choose_noise_entity(place, condition, places, stream)
        register = choose_register(config, stream)
        bands = register.get("length_bands", DEFAULT_LENGTH_BANDS)
        length_band = bands[0] if len(bands) == 1 else stream.pick(bands)
        slots = choose_slots(register, config, stream)
        env_theme = stream.pick(place.env_themes) if place.env_themes else None

        payload: dict[str, Any] = {
            "plan_version": PLAN_VERSION,
            "place_qid": qid,
            "place_name": place.name,
            "place_type": place.place_type,
            "place_type_group": type_group(place, config["plan"]),
            "place_country": place.country,
            "place_continent": place.continent,
            "place_sitelinks": place.sitelinks,
            "place_wikidata_links": place.wikidata_links,
            "places_sharing_name": place.places_sharing_name,
            "condition": condition,
            "signal": _signal_label(condition),
            "clues": [
                {
                    "clue_qid": c["clue_qid"],
                    "clue_label": c["clue_label"],
                    "property": c["property"],
                    "property_label": c.get("property_label"),
                    "entity_type": c.get("entity_type"),
                    "discriminativeness": c.get("discriminativeness"),
                    "place_ambiguity": c.get("place_ambiguity"),
                    "surface_ambiguity": c.get("surface_ambiguity"),
                    "source_set_sizes": c.get("source_set_sizes"),
                    "is_toponym": c.get("is_toponym"),
                    "label_leak": c.get("label_leak"),
                    "is_human": c.get("is_human"),
                    "aliases": c.get("aliases") or [],
                }
                for c in clues
            ],
            "source_set_index": source_set_index,
            "surface_form": surface_form,
            "noise_entity": noise_entity,
            "register": register["name"],
            "register_description": register.get("description"),
            "length_band": length_band,
            "slots": slots,
            "env_theme": env_theme,
            "replicate": replicate,
        }

        item_id = make_item_id(payload, replicate)
        if item_id in seen_ids:
            # Should be impossible, because replicate is in the hash. If it
            # ever fires, something upstream is reusing a replicate index.
            raise SystemExit(
                f"duplicate item_id {item_id} for {qid}/{condition}/{replicate}"
            )
        seen_ids.add(item_id)

        payload["item_id"] = item_id
        payload["stream_positions"] = {
            "start": position_before,
            "end": stream.position,
        }
        payload["planned_utc"] = utc_now()
        items.append(payload)

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

    print("\n  type groups per condition (%)")
    groups = sorted({i["place_type_group"] for i in items})
    print("    " + " " * 10 + "".join(f"{g[:9]:>10}" for g in groups))
    for condition in sorted({i["condition"] for i in items}):
        mine = Counter(i["place_type_group"] for i in items if i["condition"] == condition)
        total = sum(mine.values())
        print(f"    {condition:<10}" + "".join(f"{100 * mine[g] / total:>10.1f}" for g in groups))

    print("\n  countries per condition (most used)")
    for condition in sorted({i["condition"] for i in items}):
        mine = Counter(i["place_country"].split(";")[0].strip()
                       for i in items if i["condition"] == condition)
        top = ", ".join(f"{c} {n}" for c, n in mine.most_common(5))
        print(f"    {condition:<10} {len(mine):>3} countries; {top}")

    by_register = Counter(i["register"] for i in items)
    print("\n  by register")
    for register, n in sorted(by_register.items()):
        print(f"    {register:<22} {n:>5}")

    themed = sum(1 for i in items if i.get("env_theme"))
    if themed:
        print(f"\n  {themed} of {len(items)} items have an environmental theme")

    held_back = [r for r in rejections if r["record"] != "pair_rejected"]
    for kind, label in (("clue_excluded", "clues excluded"),
                        ("form_excluded", "surface forms excluded")):
        group = [r for r in held_back if r["record"] == kind]
        print(f"\n  {len(group)} {label}")
        for reason, n in Counter(r["reason"] for r in group).most_common():
            print(f"    {n:>5}  {reason}")

    pairs = [r for r in rejections if r["record"] == "pair_rejected"]
    if pairs:
        print(f"\n  {len(pairs)} (place, condition) pairs skipped")
        by_reason = Counter((r["condition"], r["reason"]) for r in pairs)
        for (condition, reason), n in sorted(by_reason.items()):
            print(f"    {n:>5}  {condition:<9} {reason}")
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
    clues_path = Path(paths["clue_sets"])
    aliases_path = Path(paths["place_aliases"])
    types_path = Path(paths["clue_types"])
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
    print(f"clues    {clues_path}")
    print(f"aliases  {aliases_path}")
    print(f"types    {types_path}")
    print(f"random   {stream_path}")

    plan_cfg = config["plan"]
    if plan_cfg.get("random_min") is None or plan_cfg.get("random_max") is None:
        raise SystemExit(
            f"{args.config} must set plan.random_min and plan.random_max to the "
            f"range the integers in {stream_path} were generated over. "
            f"Every draw's uniformity depends on it."
        )

    places = load_places(places_path)
    held_back = load_clue_sets(
        clues_path, aliases_path, types_path, places, plan_cfg
    )
    little_known = drop_little_known(places, plan_cfg)
    if little_known:
        print(f"left out {len(little_known)} places outside the sitelink band "
              f"{plan_cfg.get('min_sitelinks')}..{plan_cfg.get('max_sitelinks')}")
    held_back += little_known
    unknown = drop_unknown_clues(places, Path(paths["clue_prominence"]), plan_cfg)
    if unknown:
        print(f"held back {len(unknown)} clues and clue sets no reader could know "
              f"(min_clue_wikipedia_articles {plan_cfg.get('min_clue_wikipedia_articles')})")
    held_back += unknown
    stream = RandomStream(
        stream_path, state_path,
        low=int(plan_cfg["random_min"]), high=int(plan_cfg["random_max"]),
    )

    print(f"loaded {len(places)} places, "
          f"{sum(1 for p in places.values() if p.clues)} with at least one clue, "
          f"{sum(1 for p in places.values() if p.single_clues)} with a "
          f"single-clue-eligible clue, "
          f"{sum(1 for p in places.values() if p.triple_sets)} with a size-3 set")
    print(f"random stream starts at position {stream.start_position} "
          f"of {len(stream.values)}")

    items, pair_rejections = plan_items(places, config, stream)
    rejections = held_back + pair_rejections
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
