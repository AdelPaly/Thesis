# Place list — column provenance (written for the 147-row candidate list; 01_places_final.csv uses the same columns)

147 places. 90 `origin=retained`, 57 `origin=replacement`.

## Frozen columns

| column | frozen as of | why |
|---|---|---|
| `wikidata_links` | 2026-08-28 (Stage 8) | Inbound-link counts rise as Wikidata grows. Re-deriving on a schedule yields a permanent trickle of one- and two-unit differences that mean nothing and train a reader to ignore the report. Not refreshed. |

`scripts/verify_usable.py` still checks the column, but reports growth as
`inbound_links:drift_since_freeze` rather than as a disagreement. Only a
divergence too large to be growth — over 1% *and* over 2 absolute — fails.
At the freeze date 18 of 147 rows had already drifted, all by ≤3 links.

## Live columns

Re-derived on demand, expected to match Wikidata exactly.

| column | source |
|---|---|
| `area_km2` | Normalised P2046 (`psn:`, square metres ÷ 1e6). Best-rank statement; largest where several share best rank; largest non-deprecated if none is best-rank. Blank means the item has no P2046. Re-derived 2026-08-28 by `scripts/04_fix_areas.py`. |
| `wikimedia_sitelinks` | `wikibase:sitelinks`. Populated only on the 57 replacement rows; blank on all 90 retained. Not frozen — one row (Akaroa) is already stale by +1. |
| `places_sharing_name` | Count of items with `rdfs:label "<place_name>"@en`. |
| `country`, `continent` | P17, and P30 of the P17 country. |

## Clue columns

| column | meaning |
|---|---|
| `clue_count` | Capped sample size — 14 queries × ≤25 clues each, seed `wanderer-2026`. What generation draws from. Saturates; **not** an abundance measure. |
| `clue_total` | Uncapped count over the same 14 queries. Use this for difficulty and any ordinal analysis. Heavy-tailed — rank or log-scale it. |
| `capped` | True where ≥1 query hit the 25 cap, i.e. where `clue_count` understates. 47 of 147 rows. |

## Surface forms — `04b_place_aliases.jsonl`

Written by `scripts/04b_place_aliases.py` from `01_places_final.csv`, reusing
`04_score_clues.py`'s alias and surface-count queries. Collected 2026-09-30.

| field | source |
|---|---|
| `text` | The CSV `place_name` (`source: label`); P1449, P1705 and English `skos:altLabel` of the place item; and `"<place_name>, <country>"` (`source: derived`). A string reached by several sources is written once, under the first of label, P1449, P1705, altLabel, derived — except the derived string, which always wins its own text (6 places also carry it as an altLabel). |
| `shared_by` | For `label`: the CSV's `places_sharing_name`, i.e. items with that English `rdfs:label` **only**. For aliases: 04's surface-ambiguity count, items with the string as an English `rdfs:label` **or** `skos:altLabel`. For `derived`: 1 by definition. The label and alias counts are therefore different measurements; both count Wikidata items of any kind, not only places. |
| `kind` | `ambiguous_alias` if `shared_by` ≥ 2, `exact` if 1, `unmatched` if 0. `unmatched` forms (22, mostly P1705 native labels in other scripts) are kept on record and never used by the planner. |

The five unique-name places that nevertheless have an ambiguous form, all from
aliases: Elamanchili (Yelamanchili), Metropolitan Museum of Art (The Met, MMA),
Yaren District (Yaren, Moqua, Makwa), Río Gallegos (Rio Gallegos), Arroyos y
Esteros (Esteros).

## Clue fields added for stage 6

| field | where | source |
|---|---|---|
| `property_label` | `05_clue_sets.jsonl`, each `best_sets[].clues[]` | Passed through unchanged from `04_clues_scored.jsonl` by `05_score_sets.py`. Re-run from cache 2026-09-30: no queries, output otherwise identical. |
| `leaks_place_name` | the stage 6 plan | `leak_strict` from `05_name_leakage.csv`, keyed on (place, clue, property). |
| `country_forms` | `04b_place_aliases.jsonl`, per place | The CSV `country`, the English label of the place's P17 country, and that country's English demonyms (P1549). Queried 2026-10-01. Medina's P17 returned nothing, so it carries the CSV country only (no "Saudi"). Never offered as a surface form. |
| `region_forms` | `04b_place_aliases.jsonl`, per place | English labels and altLabels of every region the place sits inside (P131+), plus each with a trailing type word dropped ("Övörkhangai Province" → "Övörkhangai"). Forms under four letters and ISO-style codes are left out. Fetched 2026-10-03 in two queries (regions, then their names), because joining both in one timed out on WDQS. 913 forms over 100 places; Sitra and Samarai have no P131. |
| `label_leak` | the stage 6 plan | Why the clue's own label names the place or country, or null: a whole-word, case- and accent-insensitive match against the place name, its surface forms (derived form excluded), its `country_forms` and its `region_forms`. NOISE distractors are checked against the receiving place's forms as well as their own donor's. On 2026-10-01 this flagged 47 (place, clue) pairs over 42 entities, all reviewed as genuine ("Tuvalu Sports Ground", "Rock-cut tombs of Qaw el-Kebir", "USS Mount McKinley"). Stage 5's strict leak test checks the bare place name only and caught none of them. Adding region forms on 2026-10-03 flagged 17 more over 16 entities ("Khövsgöl Province General Hospital", "2005 Songkhla bombings"); one, "Jacinto Lara", is a person whose name the Lara state shares, kept excluded as the cautious reading. |
| `is_toponym` | the stage 6 plan | `toponym` from `05b_clue_types.jsonl`: the entity is an instance (P31/P279*) of human settlement, administrative territorial entity, neighborhood, island, mountain, hill, park, street, body of water, watercourse or landform, with the reviewed overrides in `05b_clue_type_overrides.csv` applied. Queried 2026-10-01. 74 of 614 clue entities; 55 of the 111 reached by P131 and 14 of the 28 reached by P138. The 74 include six IANA time zones and two church jurisdictions whose labels carry the place name in a form the strict leak test misses (`Europe/Tirane`, `America/Bogota`). |
| `discriminativeness` | the stage 6 plan | Always null. No such score exists upstream. |

## Known open items

- **Size rule.** Ten rows exceed 1000 km² (Melbourne 9993 down to Upolu 1125).
  Nothing has ever been removed at that threshold; in the shipped file the rule
  behaved as an advisory note, and inconsistently — 3 of the 10 carry a
  "large area" note, 7 do not. Exclusion remains an open dataset decision.
- **`01_places_final.csv`** (102 rows) shares this file's 15 columns, but how
  the 102 were chosen from the 147 is not recorded here yet.
- **Type mapping** is human judgement. `scripts/verify_usable.py` dumps
  `assigned_type` against the item's P31 labels to `data/verify_types.csv` for
  review and never auto-validates it.
