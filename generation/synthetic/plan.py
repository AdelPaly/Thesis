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
        - which condition the item belongs to (A1, A3, B, AB, CEILING, NOISE_*)
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
    ../places/data/05_name_leakage.csv which clues contain their place's name
    ../places/data/05b_clue_types.jsonl which clue entities are themselves places
    sampler/geo_random.txt         one random.org integer per line
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
    All randomness comes from sampler/geo_random.txt, which holds integers
    fetched from the random.org HTTP API. The file is consumed as a stream: a
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

    Single-clue conditions (A1, AB, NOISE_A1, and the NOISE distractors) draw
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
    point of the B and AB conditions: the toponym is ambiguous and the
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

PLAN_VERSION = "1"

# Defaults for the plan: section of config.yaml. config.yaml is the source of
# truth; these apply only when a key is absent from it.
DEFAULT_MAX_PLACE_AMBIGUITY = 1
DEFAULT_SURPLUS_FACTOR = 1.0
DEFAULT_MAX_REPLICATES_PER_PAIR = 6
DEFAULT_LENGTH_BANDS = ["medium"]

# The conditions, and what each one requires to be feasible for a given place.
# n_clues        how many entity clues appear in the text
# needs_toponym  "ambiguous" -> an alias shared with other places must exist
#                "exact"     -> the plain, unambiguous name is used
#                None        -> no place name may appear at all
# noise          whether a distractor entity from another continent is added
CONDITION_SPECS: dict[str, dict[str, Any]] = {
    "A1":       {"n_clues": 1, "needs_toponym": None,        "noise": False},
    "A3":       {"n_clues": 3, "needs_toponym": None,        "noise": False},
    "B":        {"n_clues": 0, "needs_toponym": "ambiguous", "noise": False},
    "AB":       {"n_clues": 1, "needs_toponym": "ambiguous", "noise": False},
    "CEILING":  {"n_clues": 0, "needs_toponym": "exact",     "noise": False},
    "NOISE_A1": {"n_clues": 1, "needs_toponym": None,        "noise": True},
    "NOISE_B":  {"n_clues": 0, "needs_toponym": "ambiguous", "noise": True},
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
    An append-only cursor over a file of integers fetched from random.org.

    Every draw consumes one integer, or occasionally more (see below()), and
    advances the cursor past everything it consumed. The cursor is persisted
    so two runs never spend the same integer on two different decisions.
    Nothing here derives extra values from one integer - each integer serves
    at most one decision, so the audit trail stays traceable.

    low and high are the inclusive range the integers were requested with from
    random.org. They are declared rather than inferred from the file, because
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
                f"Fetch more integers from random.org and append them to the file."
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
                f"Append more random.org integers to {self.path} and re-run; "
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
                f"{self.low}..{self.high} ({span} possible values). Fetch "
                f"integers over a wider range from random.org."
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
    clues: list[dict[str, Any]] = field(default_factory=list)
    surface_forms: list[dict[str, Any]] = field(default_factory=list)
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
            cached = [
                c for c in donor.single_clues
                if label_leak(c["clue_label"], self.name_forms, self.country_forms,
                              self.region_forms) is None
            ]
            self._distractors[donor.qid] = cached
        return cached

    def ambiguous_forms(self) -> list[dict[str, Any]]:
        return [f for f in self.surface_forms if f.get("kind") == "ambiguous_alias"]

    def exact_forms(self) -> list[dict[str, Any]]:
        return [f for f in self.surface_forms if f.get("kind") == "exact"]


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


def _load_leaks(path: Path) -> set[tuple[str, str, str]]:
    """
    (place_qid, clue_qid, property) for every row of 05_name_leakage.csv with
    leak_strict == true. The CSV lists only rows that leak by the strict or
    relaxed test, so a clue absent from it leaks by neither.
    """
    if not path.exists():
        raise SystemExit(f"name leakage file not found: {path}")
    leaks: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"place_qid", "clue_qid", "property", "leak_strict"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            value = (row["leak_strict"] or "").strip().lower()
            if value not in ("true", "false"):
                raise SystemExit(f"{path}: leak_strict is {row['leak_strict']!r}, "
                                 f"expected true or false")
            if value == "true":
                leaks.add((row["place_qid"].strip(), row["clue_qid"].strip(),
                           row["property"].strip()))
    return leaks


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


# Underscore aliases kept for older callers.
_fold = fold
_contains_word = contains_word


def _is_qualified(text: str, country_forms: list[str]) -> bool:
    """
    A form that already narrows itself: it carries a comma qualifier ("Las
    Cruces, NM") or contains the country's name ("Guatemala City", the bare
    "Guatemala", the Spanish "Panamá"). The B and AB constraint forbids exactly
    those additions and forbids writing the country, so such a form cannot
    serve as an ambiguous toponym without the prompt contradicting itself.
    """
    if "," in text:
        return True
    return any(contains_word(text, form) for form in country_forms)


def label_leak(label: str, place_forms: Sequence[str], country_forms: Sequence[str],
               region_forms: Sequence[str] = ()) -> str | None:
    """
    Why a clue's own label gives the answer away, or None. The strict test in
    05_name_leakage.csv checks only the bare place name; this also checks every
    surface form of the place ("Qaw el-Kebir" for Tjebu, "Mount McKinley" for
    Denali) and the country's name and demonyms ("Tuvalu Association...",
    "MIAT Mongolian Airlines"). Whole words only, ignoring case and accents.
    """
    for form in place_forms:
        if contains_word(label, form):
            return f'label contains the place name or alias "{form}"'
    for form in country_forms:
        if contains_word(label, form):
            return f'label contains the country or its demonym "{form}"'
    for form in region_forms:
        if contains_word(label, form):
            return f'label contains a region the place is in "{form}"'
    return None


_label_leak = label_leak  # underscore alias kept for older callers


@dataclass(frozen=True)
class ClueRules:
    """The plan: flags that decide which clues and forms are held back."""
    max_place_ambiguity: int = DEFAULT_MAX_PLACE_AMBIGUITY
    exclude_leaks: bool = True
    exclude_qualified: bool = True
    exclude_toponyms: bool = True
    exclude_label_leaks: bool = True
    ceiling_plain_name: bool = True

    @classmethod
    def from_config(cls, plan_cfg: dict[str, Any]) -> "ClueRules":
        return cls(
            max_place_ambiguity=int(plan_cfg.get(
                "max_place_ambiguity_for_single_clue", DEFAULT_MAX_PLACE_AMBIGUITY)),
            exclude_leaks=bool(plan_cfg.get("exclude_leaking_clues", True)),
            exclude_qualified=bool(plan_cfg.get("exclude_qualified_ambiguous_forms", True)),
            exclude_toponyms=bool(plan_cfg.get("exclude_toponym_clues", True)),
            exclude_label_leaks=bool(plan_cfg.get("exclude_label_leaks", True)),
            ceiling_plain_name=bool(plan_cfg.get("ceiling_plain_name", True)),
        )


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
    leaks: set[tuple[str, str, str]],
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
                    "leaks_place_name": (qid, raw["clue_qid"], raw["property"]) in leaks,
                    "is_toponym": clue_types[raw["clue_qid"]]["toponym"],
                    "label_leak": label_leak(raw["clue_label"], place_forms,
                                              country_forms, region_forms),
                }
            by_key[key]["source_set_sizes"].append(entry.get("size", len(entry["clues"])))
    return by_key


def clue_excluded_because(
    clue: dict[str, Any], clue_types: dict[str, dict[str, Any]], rules: ClueRules
) -> str | None:
    """Why this clue is held back from every condition, or None."""
    if rules.exclude_leaks and clue["leaks_place_name"]:
        return "leaks the place name (leak_strict)"
    if rules.exclude_toponyms and clue["is_toponym"]:
        cats = clue_types[clue["clue_qid"]].get("categories") or []
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
    A best_sets entry containing an excluded clue is excluded whole.
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
        place.triple_sets.append({"source_set_index": set_index, "clues": members})
    return held_back


def ceiling_source_of(source_forms: list[dict[str, Any]], rules: ClueRules) -> str | None:
    """
    The one source of exact form CEILING may use: the plain label when it is
    unique, otherwise the derived "<name>, <country>". None when
    ceiling_plain_name is off and every exact form is allowed.
    """
    if not rules.ceiling_plain_name:
        return None
    label_exact = any(f.get("source") == "label" and f.get("kind") == "exact"
                      for f in source_forms)
    return "label" if label_exact else "derived"


def form_excluded_because(
    form: dict[str, Any], country_forms: list[str], ceiling_source: str | None,
    rules: ClueRules,
) -> str | None:
    """Why this surface form is held back, or None."""
    kind = form.get("kind")
    if (rules.exclude_qualified and kind == "ambiguous_alias"
            and _is_qualified(form["text"], country_forms)):
        return "ambiguous form already qualified or contains the country"
    if ceiling_source and kind == "exact" and form.get("source") != ceiling_source:
        return (f"exact form other than the plain name; CEILING uses the "
                f"{'unique label' if ceiling_source == 'label' else 'derived name, country'}")
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

    CEILING is the only consumer of exact forms. With ceiling_plain_name set it
    gets one: the plain label when that is unique, otherwise the derived
    "<name>, <country>". Any other exact form - an altLabel, a nickname, a
    native-language name - is held back, because an obscure form ("Whangas",
    "Ō-hine-puhiawe") makes the upper bound harder than the conditions it
    bounds.
    """
    ceiling_source = ceiling_source_of(source_forms, rules)
    held_back: list[dict[str, Any]] = []
    forms = []
    for form in source_forms:
        reason = form_excluded_because(form, country_forms, ceiling_source, rules)
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
    leak_path: Path,
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
        05_name_leakage.csv      which (place, clue, property) leak the name
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
        leaks_place_name    from 05_name_leakage.csv (strict test)

    SINGLE-CLUE CONDITIONS (A1, AB, NOISE_A1) draw only from clues with
    place_ambiguity <= max_place_ambiguity_for_single_clue. One clue that
    several places satisfy gives the item no single correct answer.

    A3 draws a complete size-3 best_sets entry, never three loose clues: stage
    5 kept those sets because their intersection resolves to the place, and a
    random three-clue group has no such guarantee.

    LEAKING CLUES are excluded from both when exclude_leaking_clues is set. A
    set containing a leaking clue is excluded whole (select_clues).

    TOPONYM CLUES - a village, island, river, street, time zone - are excluded
    the same way when exclude_toponym_clues is set. Such a clue puts a real
    place name into a post whose prompt forbids place names (A1, A3,
    NOISE_A1) or allows exactly one (AB), and the smoke run showed the model
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

    EXACT FORMS, which only CEILING uses, are cut to one when
    ceiling_plain_name is set (filter_forms): the plain label if it is unique,
    otherwise the derived "<name>, <country>". CEILING is meant to name the
    place plainly.
    """
    rules = ClueRules.from_config(plan_cfg)

    clue_sets = _read_jsonl_by_place(clue_path, "clue set")
    aliases = _read_jsonl_by_place(alias_path, "place alias")
    leaks = _load_leaks(leak_path)
    clue_types = _load_clue_types(types_path)
    held_back: list[dict[str, Any]] = []

    for source, path in ((clue_sets, clue_path), (aliases, alias_path)):
        extra = sorted(set(source) - set(places))
        if extra:
            raise SystemExit(f"{path} has records for {len(extra)} places not in "
                             f"the place set, e.g. {extra[:3]}")

    for qid, place in places.items():
        if qid not in aliases:
            raise SystemExit(f"{alias_path} has no record for {qid} ({place.name}). "
                             f"Re-run scripts/04b_place_aliases.py.")
        alias_line, alias_record = aliases[qid]
        check_alias_record(alias_path, alias_line, alias_record)
        place_forms, country_forms, region_forms = name_forms_of(place, alias_record)
        place.name_forms = place_forms
        place.country_forms = country_forms
        place.region_forms = region_forms

        best_sets = best_sets_of(clue_path, clue_sets, qid)
        by_key = collect_clues(qid, best_sets, clue_path, types_path, clue_types,
                               leaks, place_forms, country_forms, region_forms)
        held_back += select_clues(qid, place, best_sets, by_key, clue_types, rules)
        held_back += filter_forms(qid, place, alias_record["surface_forms"] or [],
                                  country_forms, rules)

    # The name forms and donor clues just changed, so drop memoised distractors.
    for place in places.values():
        place.reset_caches()
    return held_back


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
    Places that can supply a distractor for this place: on a different
    continent, and with at least one single-clue-eligible clue to draw from.
    The distractor has to point somewhere specific to be wrong in a useful way,
    so it meets the same place_ambiguity and leakage rules as an A1 clue. A
    place with no continent recorded has no donors, because "different
    continent" cannot be established for it.
    """
    if not place.continent:
        return []
    # A distractor whose label names THIS place ("2013 South American
    # Championships" carries "American" into a United States item) would put a
    # forbidden name into the post, so donors are counted only for the clues
    # that survive that check.
    return [
        qid
        for qid, other in places.items()
        if other.continent and other.continent != place.continent
        and place.distractor_clues(other)
    ]


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

    if spec["noise"]:
        if not place.continent:
            return False, "no continent recorded, so no distractor can be chosen"
        if not noise_donors(place, places):
            return False, "no place on another continent has an eligible clue to donate"

    n_needed = spec["n_clues"]
    if n_needed == 1 and not place.single_clues:
        return False, "no single-clue-eligible clue (place_ambiguity ceiling, leakage or toponym)"
    if n_needed == 3 and not place.triple_sets:
        return False, "no complete size-3 best_sets entry that resolves and holds no excluded clue"
    if n_needed not in (0, 1, 3):
        raise SystemExit(f"{condition} needs {n_needed} clues; the adapter only "
                         f"supplies single clues and size-3 sets")

    toponym = spec["needs_toponym"]
    if toponym == "ambiguous" and not place.ambiguous_forms():
        return False, "no usable ambiguous surface form"
    if toponym == "exact" and not place.exact_forms():
        return False, "no exact surface form recorded"

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
    options = place.ambiguous_forms() if kind == "ambiguous" else place.exact_forms()
    chosen = stream.pick(options)
    return {
        "kind": kind if kind == "exact" else "ambiguous_alias",
        "text": chosen["text"],
        "shared_by": chosen.get("shared_by"),
    }


def choose_noise_entity(
    place: Place,
    condition: str,
    places: dict[str, Place],
    stream: RandomStream,
) -> dict[str, Any] | None:
    """
    For the noise conditions, add one entity associated with a place on a
    different continent. Continent is used as the separation criterion rather
    than a distance threshold because the place table carries no coordinates;
    a cross-continent distractor is unambiguously far enough to be wrong.

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
        "clue_qid": donor_clue["clue_qid"],
        "clue_label": donor_clue["clue_label"],
        "property": donor_clue["property"],
        "property_label": donor_clue.get("property_label"),
        "entity_type": donor_clue.get("entity_type"),
        "place_ambiguity": donor_clue.get("place_ambiguity"),
        "leaks_place_name": donor_clue.get("leaks_place_name"),
        "is_toponym": donor_clue.get("is_toponym"),
        "label_leak": donor_clue.get("label_leak"),
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

def build_slate(
    places: dict[str, Place], config: dict[str, Any], stream: RandomStream
) -> tuple[list[tuple[str, str, int]], list[dict[str, Any]]]:
    """
    Decide which (place, condition, replicate) triples to plan.

    Base pass: every feasible pair gets the replicate count from the config.
    Surplus pass: extra replicates are drawn at random until the surplus target
    is met. The surplus exists because the model will sometimes name the place
    outright in a condition that forbids it. Those items are dropped at
    validation in finalise.py, so the plan has to be larger than the target set.
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
            for replicate in range(replicates[condition]):
                slate.append((qid, condition, replicate))
                counts[(qid, condition)] += 1

    base_total = len(slate)
    target_total = round(base_total * surplus_factor)
    extra_needed = max(0, target_total - base_total)

    if extra_needed:
        # Only pairs that are feasible and not already at the per-pair cap.
        pool = [
            (qid, condition)
            for (qid, condition), n in counts.items()
            if n < max_per_pair
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
        length_band = stream.pick(register.get("length_bands", DEFAULT_LENGTH_BANDS))
        slots = choose_slots(register, config, stream)

        payload: dict[str, Any] = {
            "plan_version": PLAN_VERSION,
            "place_qid": qid,
            "place_name": place.name,
            "place_type": place.place_type,
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
                    "leaks_place_name": c.get("leaks_place_name"),
                    "is_toponym": c.get("is_toponym"),
                    "label_leak": c.get("label_leak"),
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
    if condition == "CEILING":
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

    by_register = Counter(i["register"] for i in items)
    print("\n  by register")
    for register, n in sorted(by_register.items()):
        print(f"    {register:<22} {n:>5}")

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
    leakage_path = Path(paths["name_leakage"])
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
    print(f"leakage  {leakage_path}")
    print(f"types    {types_path}")
    print(f"random   {stream_path}")

    plan_cfg = config["plan"]
    if plan_cfg.get("random_min") is None or plan_cfg.get("random_max") is None:
        raise SystemExit(
            f"{args.config} must set plan.random_min and plan.random_max to the "
            f"range the integers in {stream_path} were requested with from "
            f"random.org. Every draw's uniformity depends on it."
        )

    places = load_places(places_path)
    held_back = load_clue_sets(
        clues_path, aliases_path, leakage_path, types_path, places, plan_cfg
    )
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
