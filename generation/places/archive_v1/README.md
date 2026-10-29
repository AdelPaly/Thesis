# Place data (archived v1)

> Archived 2026-10-29. Superseded by `generation/out/places.csv`, built by
> `generation/reform_places.py`. The scripts that added places to this set
> (`more_places.py`, `more_nature.py`, `more_balance.py`, `build_batch.py`,
> `merge_batches.py`, `b_extra_places.py`) were deleted and are kept in git history.
> The text below describes the set as it was.

**Since 2026-10-27 (late), `01_places_final.csv` holds only the 737 places the
current plan uses; the other 679 (unused, outside the 30-300 sitelink band, not
environmental) are in `01_places_archive.csv`, same columns. 22 places that span
several countries or are themselves a country or territory (Great Britain,
Borneo, Hispaniola, Saint Martin, Lake Victoria, the Great Lakes, Everest, Mont
Blanc, Curaçao, Guam...) were removed so every place sits in one country
(`removed_places.csv`); the candidate scripts already require one country. The full set as it
was before the split (1,438 places) is kept as `01_places_full_2026-10-27.csv`:
to go back to it, copy it over `01_places_final.csv` and rebuild
`env_places.csv`.** `env_places.csv` is rebuilt from the 710. The clue, alias and
type files still cover every place. The candidate scripts and
`merge_batches.py` treat archived places as taken. Do not run
`b_extra_places.py`: it rewrites the place file.

The current place set: 295 places from the external export in `external/`,
edited there by hand over several passes. The planner (`../../synthetic/plan.py`)
reads the four files below; `../../synthetic/config.yaml` points at them by
these names. On 2026-10-22 the set grew from 295 to 356 places: 16 Arctic places
and 45 more places that pass both A1 and A3 were built with `../build_inputs.py`
(the build folders were removed afterwards) and appended to these files and to
`external/clues.jsonl` and `external/clue_sets.jsonl`. `env_places.csv` is the
filtered view for the environmental-claims posts; the planner does not read it.

On 2026-10-24 a further 27 places that pass both A1 and A3 (towns, islands and
lakes; 72 Wikidata candidates built, 39 survived) were appended to
`01_places_final.csv`, `05_clue_sets.jsonl`, `04b_place_aliases.jsonl`,
`05b_clue_types.jsonl`, `external/clues.jsonl` and `external/clue_sets.jsonl`,
and `env_places.csv` was rebuilt (362 places, 178 A3-eligible). Do not rebuild
`05_clue_sets.jsonl` with `import_external.py` without checking: the current file
holds 31 more places than `external/clue_sets.jsonl`.

On 2026-10-26 A3 went from 178 to 273 eligible places (non-`b_only`). Most A3
failures were not a shortage of clues: stage 5 kept only its top 5 sets, often
all of size 4 or 5, or all holding a toponym or label-leak clue that the planner
then removed. `05_score_sets.py` was re-run over every `env_places.csv` place
from `external/clues.jsonl` (known toponym clues left out), size 3 only, with
`--pool 20 --keep 30`, and the resolving sets not already present were appended
to each place's `best_sets` (6,676 sets over 272 places; existing sets and their
indices are unchanged; `a3_extra_sets` counts them). `05b_clue_types.py` was
re-run on the result. The appended clues also widen the A1 pool (302 to 311
places). Backups of the four files changed are in `a3_sets_run/backup/`.

On 2026-10-27 the set grew to 1,002 environmental places so that every condition
can be planned at 520 items (400 final x 1.3 surplus) with no place used twice in
a condition. `scripts/more_places.py` picked 788 new towns, islands, lakes and
mountains (at least 25 sitelinks, even over continents, fixed sha256 order),
each with a name that a mapped place more than 161 km away also carries, so a
survivor can serve B as well as A1 and A3. `scripts/build_batch.py` built them in
four batches into `more_run/b1`..`b4` (build_inputs plus the size-3 extension
above, applied before survivors are chosen); 640 survived and
`scripts/merge_batches.py` appended them to the six live files (with
`clue_wikipedia_articles` for `external/clues.jsonl`; inbound links not fetched).
The existing places were also re-checked: `more_run/recover.py` gave 19 places
clean size-3 sets (a size-3 search over their clues without toponym or leaking
clues, pool 30, keep 50: London, Tokyo, Kabul...) and 18 places single-clue
sets (place_ambiguity 1 clues that were never in a best set). Then
`build_env_places.py` and `05b_clue_types.py` were re-run. B-only places are now
planned for every condition their clues allow. Feasible places: A1/NOISE_A1
993, A3 844, B/NOISE_B 840, EXPLICIT 1,002 (at least 122 per continent in
each). The new places are 84% towns. Backups: `more_run/backup/`.

Later on 2026-10-27, so that towns are at most half of each condition and the
new places are well known: `scripts/more_nature.py` picked single-country
islands, mountains, lakes, archipelagos, bays and gulfs, deserts and national
parks (rivers were dropped: they serve as clues instead, see
`plan.toponym_clue_categories`). All 288 with a far-namesake name and the 150
best-known others, at least 35 sitelinks, were built into `nature_run/n1`, `n2`;
286 survived and were merged. `01_places_final.csv` has a new `added` column
(2026-10-27 for all 926 places added that day, blank before), which
`plan.min_sitelinks_from` uses so the 30-sitelink floor applies to the new places
only. `env_places.csv`: 1,288 places (748 towns, 307 islands, 77 mountains, 57
lakes, 27 archipelagos, 24 parks, 22 heritage, 16 bays, 10 deserts); the
planner keeps 1,145 after the floor. Dry run: 520 distinct places per spread
condition, 260 of them towns in each, B included. Backups: `nature_run/backup/`.

Clean-up the same day (backup `nature_run/backup/fix_types/`): 17 places typed
island were peninsulas (Wikidata's peninsula class Q34763 sat in the island list
of `type_classes.json`; it now has its own `peninsula` type and themes) and were
relabelled; 24 new places were removed: city hills and sites typed as mountains
(Capitoline, Esquiline, Palatine, Aventine and Primrose Hills, Montmartre,
Calvary, Mount Zion, the Cotswolds), Mount Hermon (contested country), Broad Peak
and Mount Fitz Roy (two countries), and 12 remote uninhabited islands (Baker,
Howland, Bouvet, Kvitøya, Clipperton, Palmyra, Johnston, Heard and McDonald,
Prince Edward Islands, Franz Josef Land, Devon Island, Snake Island), where a
local post makes no sense. `env_places.csv`: 1,264 places; plan: 3,722 items over
1,122 places, 520 per spread condition with 260 towns.

Balanced redesign, also 2026-10-27 (planner side in `../../synthetic/config.yaml`:
`type_groups`, `type_targets`, `core_conditions`, `min_sitelinks`/`max_sitelinks`
30-300 for every place, `min_clue_wikipedia_articles`). `scripts/more_balance.py`
picked 571 well-known single-country parks and protected areas, lakes, mountains,
coasts (bays, peninsulas, archipelagos, beaches, lagoons, reefs), landscapes
(deserts, forests, glaciers, wetlands, canyons, waterfalls, valleys, plateaus) and
World Heritage sites; 57 were struck by hand before building (cities, buildings,
religious or contested sites, unreachable islands), 514 built into
`balance_run/g1`..`g3`, 137 survived (parks and lakes rarely get a resolving clue
set) and were merged. A review of the existing places removed 19 more (stations,
prison islands, Ada Kaleh, Guantánamo Bay, Mount Kumgang, China US, Bethel) and
fixed Tripoli's and Bor's countries and four types. Every removal is in
`removed_places.csv`, which the candidate scripts read so a removed place never
returns. `build_env_places.py` gives every place a set of wide-reach themes
(development on protected land, government drilling plans, protests...); the
prompt's `scale_line` lets the model pick any scale of claim. `env_places.csv`:
1,382 places. Plan: 3,361 items over 761 places; A1, A3 and NOISE_A1 share one
core of 520 (towns 36.7%, islands 18.5%, coasts 11.9%, mountains 9.0%,
landscapes 6.9%, lakes 6.5%, heritage 6.0%, parks 4.4%); B and NOISE_B take core
places first. The planned places are listed in
`../../synthetic/output/planned_places.csv`. Backups: `balance_run/backup/`.

## What the planner reads

| file | built from | by |
|---|---|---|
| `01_places_final.csv` | the external `01_places_final.csv` (295 places), plus 38 rows with `b_only=true` | copied; `scripts/b_extra_places.py` adds the 38 |
| `05_clue_sets.jsonl` | `external/clue_sets.jsonl` plus `property_label` from `external/clues.jsonl` | `scripts/import_external.py` |
| `04b_place_aliases.jsonl` | `01_places_final.csv` | `scripts/04b_place_aliases.py` |
| `05b_clue_types.jsonl` | `05_clue_sets.jsonl` | `scripts/05b_clue_types.py` |

A surface form is `ambiguous_alias` (usable for B and NOISE_B) only when
another item **with coordinates, more than 161 km away** carries the same
English label or alias (`geo_shared_by` >= 1). A name shared only with films,
songs or ships ("Cape Town") is `exact`. On 2026-10-20 this left 154 places
with an ambiguous form, 149 after the planner's own filters (was 216).

Each ambiguous form also records `top_namesake`, the best-known of those other
places (QID and sitelinks), and each record carries the place's own `sitelinks`
fetched in the same pass. In the base set the place is far better known than
any namesake for 77 per cent of B names, so a B result should be split by
this (finalise.py writes `place_is_most_famous`, and the three-way
`rival_prominence`: place_leads_2x / close / rival_leads_2x, the place's
sitelinks over the best-known place with the same name's, each +1).
finalise.py also writes `place_prominence` (every condition: low_under_30 /
mid_30_to_99 / high_100_plus) and `clue_prominence` (A conditions, read from
`external/clues.jsonl`: unknown / niche / well_known, by Wikipedia articles
0 / 1-9 / 10+); the thresholds are at the top of finalise.py.

**B-only places.** `b_extra_places.py` adds the opposite case: 38 places that
share a base-set name with a better-known place (London in Ontario, Paris in
the United States), at least 10 sitelinks, one per name, drawn evenly over
continents in a fixed order. The selection, with each place's famous namesake,
is in `b_extra_places.csv`. They have no clues; the planner uses them for B and
NOISE_B only. One, the Flavian Amphitheater in Pozzuoli, ends up with no
ambiguous form (the Colosseum carries that name only as an alias) and is never
planned. Types come from `type_classes.json`, copied from the place-building
project.

The planner no longer reads a name-leakage file (removed 2026-10-22): with it
switched off the 1,352-line plan audit was identical, because the planner's own
label check (place forms, country, demonyms, regions) already catches every clue
that names its place. `scripts/05_name_leakage.py` is still part of the place
build, where `03_filter_clues.py` uses it.

## Rebuilding after editing `external/`

Run from `generation/places/`. Edit the place CSV in `data/` itself; the clue
files in `external/`.

```bash
py -3 scripts/import_external.py
py -3 scripts/04b_place_aliases.py
py -3 scripts/b_extra_places.py      # re-adds the B-only rows (replaces old ones)
py -3 scripts/04b_place_aliases.py   # again, for the B-only rows
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
