# Place data

The current place set and everything derived from it. Column-level provenance
is in `PROVENANCE.md`.

## Read by the planner (`../../synthetic/plan.py`)

| file | what it is |
|---|---|
| `01_places_final.csv` | The 102 places posts are generated about, 17 per continent. |
| `05_clue_sets.jsonl` | Per place: the best sets of clues that together point to it alone. |
| `04b_place_aliases.jsonl` | Per place: its name variants and how many things share each, plus its country names and demonyms and the regions it sits in (used for leak checks). |
| `05_name_leakage.csv` | Which clues contain their place's name. |
| `05b_clue_types.jsonl` | Which clue entities are themselves places (excluded by the planner). |

## Intermediate outputs, in stage order

| file | produced by | what it is |
|---|---|---|
| `02_clues.jsonl` | `02_clues.py` | Raw clues for all 147 candidate places. |
| `02_clues_filtered.jsonl` | `03_filter_clues.py` | Clues left after dropping name leaks and unlabelled clues. |
| `03_removed_clues.jsonl` | `03_filter_clues.py` | The dropped clues, with reasons. |
| `04_clues_scored.jsonl`, `04_score_summary.csv` | `04_score_clues.py` | Each clue with how many places it could point to. |
| `05b_clue_type_overrides.csv` | hand-reviewed | Corrections to the toponym classification. Edit this, not the code. |
| `01_places_status.csv` | `06_status.py` | All 147 candidate places with how far each got (`stage_passed`). Not read by anything. |

`*_cache.json` files are Wikidata query caches shared by every run. Deleting one
only makes the next run slower. `verify_*` files are written by the check
scripts and read by people, not scripts.

`02_clues.bak-20260828.jsonl`, `02_clues_cache.bak-20260828.json` and
`archive/01_places_usable.*` are snapshots of earlier versions of the
candidate list, kept as the only record of those states.

A new place list is built into its own subfolder by `../build_inputs.py`, so it
never overwrites these files.
