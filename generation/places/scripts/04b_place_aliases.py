#!/usr/bin/env python3
"""Collect the surface forms of each PLACE: the strings a post could use to name
it, and how many things each string could mean.

04 asked this of clue entities. This script asks it of the places themselves,
with 04's own machinery, imported rather than restated:

  aliases       04's alias_query / collect_aliases -- P1449 nickname, P1705
                native label, and English skos:altLabel.
  shared_by     04's surface-ambiguity count: distinct Wikidata entities that
                carry the string as an English label or alias. Entities, not
                only places -- the same question 04 put to clue labels, and
                what an entity linker sees when the string shows up in a post.

The bare place name is the exception. Its shared_by is taken from
places_sharing_name in the place CSV, as specified, and that column counts
items with the string as an English rdfs:label only (PROVENANCE.md). The two
counts are therefore not the same measurement: an alias count includes
altLabel matches, the bare-name count does not. `source` on each form says
which one applies.

KIND
    ambiguous_alias  shared_by >= 2
    exact            shared_by == 1, and always the derived "<name>, <country>"
    unmatched        shared_by == 0: the string is not an English label or
                     alias of anything, the place included. Typically a P1705
                     native label in another language or script, or a P1449
                     nickname nobody entered as an alias. Kept on record, never
                     selected by the planner, which only reads the two kinds
                     above.

A string reached by more than one source is written once, under the first
source in this order: label, P1449, P1705, altLabel, derived. One exception:
the derived "<name>, <country>" always wins its own string. Six places carry
it as an altLabel too, and one of those (Santo Domingo) is shared by two
entities -- letting the alias win would classify a string that names the
country as ambiguous, and hand it to a B item as the ambiguous toponym.

COUNTRY FORMS are written alongside, never as surface forms: the CSV country,
the English label of the place's P17 country, and that country's English
demonyms (P1549). The planner uses them to exclude clues whose label gives the
country away.

REGION FORMS likewise: the English labels and aliases of every region the
place sits inside (P131+), plus each with a trailing type word dropped
("Övörkhangai Province" -> "Övörkhangai"). Forms under four letters and
ISO-style codes are left out, because they would match ordinary words.

Output:
    data/04b_place_aliases.jsonl   one record per place
    data/04b_alias_cache.json      query cache, same format as 04's

Usage:
    python3 scripts/04b_place_aliases.py [--places data/01_places_final.csv]
                                         [--no-altlabel] [--no-cache] [--dry-run]

--no-cache ignores the cache and does not write it.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from importlib import import_module
from pathlib import Path

from io_utils import utf8_stdout, write_jsonl_atomic
from jsoncache import JsonCache, cache_key
from wdqs import qid_from_uri, value_of

# Query builders and collectors shared with 04; they live in a numeric-prefix
# module, so this is the one place that needs import_module.
_score = import_module("04_score_clues")
alias_query = _score.alias_query
collect_aliases = _score.collect_aliases
surface_count_query = _score.surface_count_query
collect_surface_counts = _score.collect_surface_counts
run_chunked = _score.run_chunked
ALIAS_CHUNK = _score.ALIAS_CHUNK
LABEL_CHUNK = _score.LABEL_CHUNK

ROOT = Path(__file__).resolve().parent.parent
PLACES = ROOT / "data" / "01_places_final.csv"
OUT_JSONL = ROOT / "data" / "04b_place_aliases.jsonl"
CACHE_PATH = ROOT / "data" / "04b_alias_cache.json"

CACHE_VERSION = 1

SOURCE_ORDER = ["label", "P1449", "P1705", "altLabel", "derived"]
KINDS_REPORTED = ("ambiguous_alias", "exact", "unmatched")

REGION_CHUNK = 25         # P131+ is expensive; a 102-place block timed out on WDQS
MIN_REGION_FORM_LETTERS = 4   # shorter forms ("ON" for Ontario) match ordinary words
AMBIGUOUS_MIN_SHARED = 2  # shared_by at or above this is an ambiguous alias
BAD_PLACES_SHOWN = 5
FAILURES_SHOWN = 10
ERROR_SNIPPET = 110
EXIT_OK = 0
EXIT_FAILED_LOOKUPS = 1
EXIT_BAD_INPUT = 2

# Trailing words that only say what kind of region it is. "Övörkhangai
# Province" appears in clue labels as plain "Övörkhangai", so the bare name is
# added alongside the full one.
REGION_TYPE_WORDS = {
    "province", "district", "region", "county", "municipality", "department",
    "state", "governorate", "oblast", "prefecture", "commune", "aimag",
    "parish", "canton", "borough", "division", "territory", "city", "town",
    "subdistrict", "regency", "emirate", "voivodeship", "kreis", "landkreis",
}


# --------------------------------------------------------------------------
# Queries and collectors
# --------------------------------------------------------------------------

def country_query(qids: list[str]) -> str:
    """English label and demonyms (P1549) of each place's P17 country."""
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT DISTINCT ?place ?text ?src WHERE {{
  VALUES ?place {{ {values} }}
  ?place wdt:P17 ?country .
  {{ ?country rdfs:label ?text FILTER(lang(?text) = "en") BIND("P17 label" AS ?src) }}
  UNION
  {{ ?country wdt:P1549 ?text FILTER(lang(?text) = "en") BIND("P1549" AS ?src) }}
}}
"""


def collect_text_forms(batch: list[str], rows: list[dict], out: dict) -> None:
    """Fill `out[qid] = [[text, source], ...]` (deduplicated) from ?place/?text/?src rows.

    Used for country forms and for region names alike: both queries return the
    same three columns.
    """
    found: dict[str, list] = {q: [] for q in batch}
    for row in rows:
        place = value_of(row, "place")
        text = value_of(row, "text")
        if not place or not text:
            continue
        qid = qid_from_uri(place)
        entry = [text, value_of(row, "src") or ""]
        if qid in found and entry not in found[qid]:
            found[qid].append(entry)
    out.update(found)


def region_query(qids: list[str]) -> str:
    """Every region each place sits inside (P131+), as QIDs."""
    values = " ".join(f"wd:{q}" for q in qids)
    return f"""
SELECT DISTINCT ?place ?region WHERE {{
  VALUES ?place {{ {values} }}
  ?place wdt:P131+ ?region .
}}
"""


def collect_regions(batch: list[str], rows: list[dict], out: dict) -> None:
    """Fill `out[qid] = sorted region QIDs` for every place in the batch."""
    found: dict[str, list[str]] = {q: [] for q in batch}
    for row in rows:
        place = value_of(row, "place")
        region = value_of(row, "region")
        if not place or not region:
            continue
        qid, rid = qid_from_uri(place), qid_from_uri(region)
        if qid in found and rid not in found[qid]:
            found[qid].append(rid)
    out.update({q: sorted(v) for q, v in found.items()})


def region_name_query(rids: list[str]) -> str:
    """English labels and aliases of the given regions. Bound by VALUES, so
    the label lookups start from a handful of entities, as alias_query does."""
    values = " ".join(f"wd:{q}" for q in rids)
    return f"""
SELECT DISTINCT ?place ?text ?src WHERE {{
  VALUES ?place {{ {values} }}
  {{ ?place rdfs:label ?text FILTER(lang(?text) = "en") BIND("P131 label" AS ?src) }}
  UNION
  {{ ?place skos:altLabel ?text FILTER(lang(?text) = "en") BIND("P131 altLabel" AS ?src) }}
}}
"""


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def usable_region_form(text: str) -> bool:
    """
    Drop the forms that would match ordinary words or codes rather than the
    region: anything under four letters ("ON" for Ontario would match "on"),
    and ISO-style codes such as "TH-90".
    """
    letters = sum(ch.isalpha() for ch in text)
    if letters < MIN_REGION_FORM_LETTERS:
        return False
    if re.fullmatch(r"[A-Z0-9][A-Z0-9 .\-]*", text):
        return False
    return True


def region_variants(text: str) -> list[str]:
    """The region name plus the same with a leading/trailing type word dropped, usable only."""
    out = [text]
    words = text.split()
    if len(words) >= 2 and words[-1].casefold() in REGION_TYPE_WORDS:
        out.append(" ".join(words[:-1]))
    if len(words) >= 2 and words[0].casefold() in REGION_TYPE_WORDS:
        out.append(" ".join(words[1:]))
    return [v for v in out if usable_region_form(v)]


def kind_of(shared_by: int) -> str:
    """ambiguous_alias, exact or unmatched, by how many entities share the string."""
    if shared_by >= AMBIGUOUS_MIN_SHARED:
        return "ambiguous_alias"
    if shared_by == 1:
        return "exact"
    return "unmatched"


def as_int(value: str | None) -> int | None:
    """int(float(value)), or None for blank / non-numeric input."""
    try:
        return int(float(value)) if value not in (None, "") else None
    except ValueError:
        return None


def source_rank(source: str) -> int:
    """Position of an alias source in SOURCE_ORDER (unknown sources sort last)."""
    return SOURCE_ORDER.index(source) if source in SOURCE_ORDER else len(SOURCE_ORDER)


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_places(path: Path) -> list[dict]:
    """Rows of the places CSV that have a wikidata_id."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if (row.get("wikidata_id") or "").strip()]


# --------------------------------------------------------------------------
# Cached, chunked fetching
# --------------------------------------------------------------------------

def split_cached(items: list[str], prefix: str, cache: JsonCache) -> tuple[dict, list[str]]:
    """Partition `items` into (cached values by item, items still to query)."""
    found: dict = {}
    want: list[str] = []
    for item in items:
        key = cache_key(prefix, item)
        if key in cache:
            found[item] = cache.get(key)
        else:
            want.append(item)
    return found, want


def fill_missing(found: dict, want: list[str], prefix: str, cache: JsonCache, chunk: int,
                 build, collect, label: str, failures: list[tuple[str, str]]) -> dict:
    """Query `want` in chunks, store results in `found` and the cache, save the cache."""
    if want:
        got = run_chunked(want, chunk, build, collect, label, failures)
        for item, value in got.items():
            found[item] = value
            cache.set(cache_key(prefix, item), value)
        cache.save()
    return found


# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------

def build_forms_by_place(places: list[dict], aliases: dict[str, list]) -> dict[str, list[dict]]:
    """Candidate surface forms per place, first source wins ({'text', 'source'} dicts)."""
    forms_by_place: dict[str, list[dict]] = {}
    for place in places:
        qid = place["wikidata_id"].strip()
        name = place["place_name"].strip()
        country = place["country"].strip()
        derived = f"{name}, {country}"
        candidates = [(name, "label")]
        found = sorted(aliases.get(qid, []), key=lambda pair: source_rank(pair[1]))
        candidates += [(text.strip(), src) for text, src in found
                       if text.strip() and text.strip() != derived]
        candidates.append((derived, "derived"))

        seen: set[str] = set()
        forms = []
        for text, src in candidates:
            if text in seen:
                continue
            seen.add(text)
            forms.append({"text": text, "source": src})
        forms_by_place[qid] = forms
    return forms_by_place


def measure_forms(place: dict, forms: list[dict], surface: dict[str, int]) -> tuple[list[dict], int]:
    """Attach kind/shared_by to each form. Returns (forms, number left unmeasured)."""
    out = []
    unmeasured = 0
    for form in forms:
        src = form["source"]
        if src == "label":
            shared_by = as_int(place["places_sharing_name"])
        elif src == "derived":
            shared_by = 1
        elif form["text"] in surface:
            shared_by = surface[form["text"]]
        else:
            # The count query failed after splitting. No number, so no kind:
            # written as unmeasured rather than guessed.
            unmeasured += 1
            out.append({"text": form["text"], "kind": "unmeasured",
                        "shared_by": None, "source": src})
            continue
        kind = "exact" if src == "derived" else kind_of(shared_by)
        out.append({"text": form["text"], "kind": kind,
                    "shared_by": shared_by, "source": src})
    return out, unmeasured


def country_forms_for(csv_country: str, forms: list) -> tuple[list[dict], set[str]]:
    """The CSV country first, then the P17 label and demonyms, deduplicated."""
    seen_country = {csv_country}
    country = [{"text": csv_country, "source": "csv"}]
    for text, src in forms:
        if text not in seen_country:
            seen_country.add(text)
            country.append({"text": text, "source": src})
    return country, seen_country


def region_forms_for(raw: list, taken: set[str]) -> list[dict]:
    """Usable region forms, skipping any already covered by the place or country forms."""
    regions, seen_region = [], set()
    for text, src in raw:
        for variant in region_variants(text.strip()):
            folded = variant.casefold()
            if folded in taken or folded in seen_region:
                continue
            seen_region.add(folded)
            regions.append({"text": variant,
                            "source": src if variant == text.strip() else f"{src}, type word dropped"})
    return regions


def build_records(places: list[dict], forms_by_place: dict[str, list[dict]],
                  surface: dict[str, int], country_forms: dict[str, list],
                  region_raw: dict[str, list]) -> tuple[list[dict], int]:
    """One output record per place. Returns (records, number of unmeasured forms)."""
    out_records = []
    unmeasured_total = 0
    for place in places:
        qid = place["wikidata_id"].strip()
        forms, unmeasured = measure_forms(place, forms_by_place[qid], surface)
        unmeasured_total += unmeasured
        csv_country = place["country"].strip()
        country, seen_country = country_forms_for(csv_country, country_forms.get(qid, []))
        # A region form that is also the place's own name or a country form is
        # already checked under those, and would be misreported here.
        taken = {f["text"].casefold() for f in forms} | {t.casefold() for t in seen_country}
        out_records.append({
            "place_qid": qid,
            "place_name": place["place_name"].strip(),
            "country": csv_country,
            "surface_forms": forms,
            "country_forms": country,
            "region_forms": region_forms_for(region_raw.get(qid, []), taken),
        })
    return out_records, unmeasured_total


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_report(out_records: list[dict]) -> None:
    """Counts of surface forms by kind and by (source, kind)."""
    all_forms = [f for r in out_records for f in r["surface_forms"]]
    by_kind = Counter(f["kind"] for f in all_forms)
    by_source_kind = Counter((f["source"], f["kind"]) for f in all_forms)
    no_ambiguous = [r for r in out_records
                    if not any(f["kind"] == "ambiguous_alias" for f in r["surface_forms"])]

    print("\n" + "=" * 70)
    print(f"places                          : {len(out_records)}")
    print(f"surface forms                   : {len(all_forms)}")
    for kind in ("ambiguous_alias", "exact", "unmatched", "unmeasured"):
        if by_kind[kind]:
            print(f"  {kind:<30}: {by_kind[kind]}")
    print(f"places with no ambiguous form   : {len(no_ambiguous)}")
    print(f"places with >= 1 ambiguous form : {len(out_records) - len(no_ambiguous)}")

    print(f"\n{'source':<10} " + " ".join(f"{k:>16}" for k in KINDS_REPORTED))
    print("-" * 62)
    for src in SOURCE_ORDER:
        row = [by_source_kind[(src, k)] for k in KINDS_REPORTED]
        if any(row):
            print(f"{src:<10} " + " ".join(f"{n:>16}" for n in row))


def print_failures(failures: list[tuple[str, str]], unmeasured: int) -> None:
    """Warn about lookups that failed even after splitting."""
    if failures:
        print(f"\n!! {len(failures)} lookups failed after splitting; {unmeasured} forms "
              f"written as unmeasured. Re-run to retry just those.")
        for item, err in failures[:FAILURES_SHOWN]:
            print(f"   {item}: {err[:ERROR_SNIPPET]}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Command-line options."""
    parser = argparse.ArgumentParser(description="Collect place surface forms.")
    parser.add_argument("--places", type=Path, default=PLACES)
    parser.add_argument("--out", type=Path, default=OUT_JSONL)
    parser.add_argument("--no-altlabel", action="store_true",
                        help="aliases from P1449/P1705 only, ignoring skos:altLabel")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore the cache and do not write it")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the query plan and exit without querying")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load -> query (cached, chunked) -> build records -> write -> report."""
    args = parse_args(argv)

    if not args.places.exists():
        print(f"place file not found: {args.places}", file=sys.stderr)
        return EXIT_BAD_INPUT

    places = load_places(args.places)
    if not places:
        print("no places", file=sys.stderr)
        return EXIT_BAD_INPUT

    bad = [p["wikidata_id"] for p in places if as_int(p.get("places_sharing_name")) is None]
    if bad:
        print(f"places_sharing_name is blank or not a number for {len(bad)} places: "
              f"{bad[:BAD_PLACES_SHOWN]}", file=sys.stderr)
        return EXIT_BAD_INPUT

    qids = [p["wikidata_id"].strip() for p in places]
    cache = JsonCache(CACHE_PATH, CACHE_VERSION, enabled=not args.no_cache)
    failures: list[tuple[str, str]] = []
    alias_prefix = "alP" if args.no_altlabel else "alA"

    print(f"places             : {len(places)}")
    print(f"aliases            : "
          f"{'P1449/P1705 only' if args.no_altlabel else 'altLabel + P1449/P1705'}")

    # ---- pass 1: aliases ----
    aliases, want = split_cached(qids, alias_prefix, cache)
    print(f"\npass 1: aliases     {len(qids)} places, {len(want)} to query")
    if args.dry_run:
        print("\n--dry-run: nothing queried, nothing written")
        return EXIT_OK
    fill_missing(aliases, want, alias_prefix, cache, ALIAS_CHUNK,
                 lambda batch: alias_query(batch, not args.no_altlabel),
                 collect_aliases, "aliases", failures)

    forms_by_place = build_forms_by_place(places, aliases)

    # ---- pass 2: shared_by for alias strings ----
    alias_texts = sorted({f["text"] for forms in forms_by_place.values()
                          for f in forms if f["source"] not in ("label", "derived")})
    surface, want_texts = split_cached(alias_texts, "sa", cache)
    print(f"\npass 2: shared_by   {len(alias_texts)} alias strings, "
          f"{len(want_texts)} to query")
    fill_missing(surface, want_texts, "sa", cache, LABEL_CHUNK, surface_count_query,
                 collect_surface_counts, "surface", failures)

    # ---- pass 3: country forms ----
    # The strings that name the place's country: the CSV country, the P17
    # country's English label, and its English demonyms (P1549). Not surface
    # forms of the place -- never offered as a toponym -- but the planner needs
    # them to spot a clue label that gives the country away ("New Zealand
    # Cricket", "MIAT Mongolian Airlines"), which the A prompts forbid.
    country_forms, want = split_cached(qids, "cf", cache)
    print(f"\npass 3: country     {len(qids)} places, {len(want)} to query")
    fill_missing(country_forms, want, "cf", cache, ALIAS_CHUNK, country_query,
                 collect_text_forms, "country", failures)

    # ---- pass 4: region forms ----
    # Names of the regions the place sits inside, for the same leak check as
    # country_forms: "Museum of Övörkhangai Province" in an Arvaikheer item
    # writes the province name into a post that may not carry one.
    # Two plain queries rather than one: joining the names onto the P131+ path
    # in a single query timed out on WDQS even for one place, while each half
    # on its own returns in under a second.
    region_ids, want = split_cached(qids, "rg", cache)
    print(f"\npass 4a: regions    {len(qids)} places, {len(want)} to query")
    fill_missing(region_ids, want, "rg", cache, REGION_CHUNK, region_query,
                 collect_regions, "regions", failures)

    all_regions = sorted({r for ids in region_ids.values() for r in ids})
    region_names, want = split_cached(all_regions, "rn", cache)
    print(f"pass 4b: names      {len(all_regions)} regions, {len(want)} to query")
    fill_missing(region_names, want, "rn", cache, ALIAS_CHUNK, region_name_query,
                 collect_text_forms, "region names", failures)

    region_raw: dict[str, list] = {
        qid: [pair for rid in region_ids.get(qid, []) for pair in region_names.get(rid, [])]
        for qid in qids
    }

    out_records, unmeasured = build_records(places, forms_by_place, surface,
                                            country_forms, region_raw)
    write_jsonl_atomic(args.out, out_records)

    print_report(out_records)
    print_failures(failures, unmeasured)
    print(f"\nwrote {args.out}   {len(out_records)} places")
    return EXIT_FAILED_LOOKUPS if failures else EXIT_OK


if __name__ == "__main__":
    utf8_stdout()
    raise SystemExit(main())
