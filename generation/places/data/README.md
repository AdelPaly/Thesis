# Place data

The current place set: 295 places from the external export in `external/`,
edited there by hand over several passes. The planner (`../../synthetic/plan.py`)
reads the five files below; `../../synthetic/config.yaml` points at them by
these names.

## What the planner reads

| file | built from | by |
|---|---|---|
| `01_places_final.csv` | the external `01_places_final.csv`, unchanged | copied |
| `05_clue_sets.jsonl` | `external/clue_sets.jsonl` plus `property_label` from `external/clues.jsonl` | `scripts/import_external.py` |
| `04b_place_aliases.jsonl` | `01_places_final.csv` | `scripts/04b_place_aliases.py` |
| `05_name_leakage.csv` | `external/clues.jsonl` | `scripts/05_name_leakage.py` |
| `05b_clue_types.jsonl` | `05_clue_sets.jsonl` | `scripts/05b_clue_types.py` |

A surface form is `ambiguous_alias` (usable for B and NOISE_B) only when
another item **with coordinates, more than 161 km away** carries the same
English label or alias (`geo_shared_by` >= 1). A name shared only with films,
songs or ships ("Cape Town") is `exact`. On 2026-10-20 this left 154 places
with an ambiguous form, 149 after the planner's own filters (was 216).

`05_name_leakage.csv` has a header and no rows: the export already drops clues
that name their place. The planner's own label check (place forms, country,
demonyms, regions) still runs on top of it.

## Rebuilding after editing `external/`

Run from `generation/places/`. Edit the place CSV in `data/` itself; the clue
files in `external/`.

```bash
py -3 scripts/import_external.py
py -3 scripts/05_name_leakage.py --clues data/external/clues.jsonl
py -3 scripts/04b_place_aliases.py
py -3 scripts/05b_clue_types.py
```

`04b` and `05b` query Wikidata; their caches below make a re-run cheap. Then run
`plan.py --replace` from `../synthetic/`.

## `external/`

The source the set was made from. Not read by the planner.

| file | what |
|---|---|
| `clues.jsonl` | every clue for every place, with prominence fields (`clue_inbound_links`, `clue_sitelinks`, `clue_wikipedia_articles`, `thin_record`) |
| `clue_sets.jsonl` | best combinations of 3-5 clues per place |
| `01b_places_enriched.csv` | the place rows plus resolving clues and name-sharing places, for reading |
| `place_exceptions.csv` | the 18 places removed from the original 313, and why |
| `README.md` | the export's own notes |

## Caches and overrides

| file | used by |
|---|---|
| `04b_alias_cache.json` | `04b_place_aliases.py` |
| `05b_type_cache.json` | `05b_clue_types.py` |
| `05b_clue_type_overrides.csv` | `05b_clue_types.py`: hand decisions on whether a clue is a place name |
| `02_clues_cache.json`, `03_clue_totals_cache.json`, `04_score_cache.json`, `05_set_cache.json` | the full clue pipeline (`build_inputs.py`), only for building a new place set from scratch |
| `verify_cache.json`, `verify_usable_cache.json` | `verify_places.py`, `verify_usable.py` |

## Known gaps (2026-10-20)

- 25 places have no `best_sets` in the export (London, Chiang Mai, Mount
  Vesuvius, Lake Constance, ...). With the planner's exclusions, 18 places
  support only `EXPLICIT`.
- Six clue labels give their place away at the start of a longer word:
  Capetonians (Cape Town), Hangötidningen-Hangonlehti (Hanko), Métropole
  lémanique (Lake Geneva), Huronian glaciation (Lake Huron), Hannaea
  superiorensis (Lake Superior), Odinarchaeum yellowstonii (Yellowstone
  National Park). They stay in the data; `plan.py`'s `label_leak` holds them
  back (word-start match on place forms of 4+ letters).
