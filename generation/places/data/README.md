# Place data

The current place set and everything derived from it. Column-level provenance
is in `PROVENANCE.md`.

## Place counts, and why they differ between files

| file(s) | places | what that set is |
|---|---|---|
| `01_places_final.csv`, `02_clues_filtered.jsonl`, `04_clues_scored.jsonl`, `05_clue_sets.jsonl`, `04b_place_aliases.jsonl` | 102 | The final set, 17 per continent. |
| `02_clues.jsonl` | 155 | The 147-place candidate list plus 8 places added later. All 8 are in the final 102. |
| `01_places_status.csv`, `archive/01_places_usable.bak-areas.csv` | 147 | The candidate list. 94 of its places are in the final 102 (the other 8 final places are the ones added later). 99 of the 147 reached stage `sets`. |

So the final 102 are 94 places from the 147 candidates plus the 8 added ones.
`05_name_leakage.csv` has no rows for those 8 places; re-running
`05_name_leakage.py` on `02_clues.jsonl` adds 16 rows and leaves the other 653
unchanged.

The 147-row candidate CSV is **not shipped as an input**. Its only copy is
`archive/01_places_usable.bak-areas.csv` (15 columns, with `origin`, `note`,
`selection_method`). `06_status.py` therefore defaults to the 102-row final CSV
and reports the clue files' extra places as a warning, not an error. The
shipped `01_places_status.csv` is the 147-row table; re-running `06_status.py`
with its defaults writes a 102-row one over it, so pass `--out` to keep both.

How the 102 are chosen is now defined in code: `../build_inputs.py`,
`choose_survivors` keeps a place if at least one of its `best_sets` in
`05_clue_sets.jsonl` resolves to that place alone, and with `--per-continent N`
cuts each continent to the N places with the most resolving best sets (ties by
place name). The files do not record that the shipped 102 came from exactly that
call.

## Read by the planner (`../../synthetic/plan.py`)

| file | what it is |
|---|---|
| `01_places_final.csv` | The 102 places posts are generated about, 17 per continent. Columns: `place_name, type, country, continent, wikidata_id, clue_count, clue_total, capped, wikimedia_sitelinks, wikidata_links, places_sharing_name, area_km2`. |
| `05_clue_sets.jsonl` | Per place: the best sets of clues that together point to it alone. |
| `04b_place_aliases.jsonl` | Per place: its name variants and how many things share each, plus its country names and demonyms and the regions it sits in (used for leak checks). |
| `05_name_leakage.csv` | Which clues contain their place's name. |
| `05b_clue_types.jsonl` | Which clue entities are themselves places (excluded by the planner). |

## Intermediate outputs, in stage order

| file | produced by | what it is |
|---|---|---|
| `02_clues.jsonl` | `02_clues.py` | Raw clues for the 155 places listed above. |
| `02_clues_filtered.jsonl` | `03_filter_clues.py` | Clues left after dropping name leaks and unlabelled clues (102 places). |
| `03_removed_clues.jsonl` | `03_filter_clues.py` | The dropped clues, with reasons. |
| `04_clues_scored.jsonl`, `04_score_summary.csv` | `04_score_clues.py` | Each clue with how many places it could point to. |
| `01_places_status.csv` | `06_status.py` | The 147 candidate places with how far each got (`stage_passed`). Not read by anything. |

## Hand-maintained input

| file | what it is |
|---|---|
| `05b_clue_type_overrides.csv` | Input to `05b_clue_types.py`, not produced by it. Hand-reviewed corrections to the toponym classification, each with a reason. Edit this, not the code. |

## Caches, reports and snapshots

`*_cache.json` files are Wikidata query caches shared by every run. Deleting one
only makes the next run slower.

`verify_*` files are written by the check scripts and read by people, not
scripts. `verify_places.py` writes `verify_report.csv`, `verify_types.csv` and
`verify_cache.json`; `verify_usable.py` writes `verify_usable_report.csv`,
`verify_usable_types.csv` and `verify_usable_cache.json`, so neither overwrites
the other. The `verify_report.csv` and `verify_types.csv` currently in this
folder are older `verify_usable.py` output, from when both scripts shared those
names (their report has an `origin` column that is blank for the final file).

The `.bak` files are hand-made snapshots, not versioned data, and nothing in the
code reads them:

| file | what it is |
|---|---|
| `02_clues.bak-20260828.jsonl` | Raw clues from 2026-08-28 for 90 places (the 90 `retained` candidates). |
| `02_clues_cache.bak-20260828.json` | The clue query cache from the same date. |
| `archive/01_places_usable.bak-20260828.csv`, `.bak2-20260828.csv` | Earlier 13-column versions of the 147-row candidate list (no `clue_total`/`capped`). |
| `archive/01_places_usable.bak-areas.csv` | The 147-row candidate list with `clue_total`/`capped` (15 columns), as backed up by `04_fix_areas.py` just before it re-derived the areas. The latest of the three. |

`../scripts/04_fix_areas.py` writes a fresh `*.bak-areas.csv` next to the CSV it
rewrites, so more of these will appear.

A new place list is built into its own subfolder by `../build_inputs.py`
(`--out-dir` is relative to `generation/places/`), so it never overwrites these
files. Each build also writes a `run_manifest.json` (date, seed, arguments, git
hash) into that folder.
