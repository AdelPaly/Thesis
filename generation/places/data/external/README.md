# Places for post generation

Built by `scripts/08_export_generation.py`; do not edit here, re-run it.
One row/record per place, 295 places (313 sampled, 18 removed; see
`place_exceptions.csv`). Inappropriate places and atrocity clues are removed, as
are clues that name their place in any language and junk labels (codes,
catalogue numbers). Every place has at least one resolving clue.

| file | what |
|---|---|
| `01b_places_enriched.csv` | **Start here.** One row per place: name, `type`, country, continent, counts, plus `resolving_clues` (up to 10 clue labels that point to this place alone, pipe separated), `distinct_facts` (how many different facts those clues really are), `single_family` (true = they collapse to 2 facts or fewer: do not build two-clue items from this place), `top_resolving_clue` (the best-known one, for single-clue items) and `sharing_places` (up to 10 other things with the same name, as `Name (Country)`). |
| `01_places_final.csv` | The same rows without the added columns. |
| `clues.jsonl` | Every clue for every place. `place_ambiguity` = how many places the clue fits (1 = only this one), `surface_ambiguity` = how many things share the clue's label, `property_family` = person / place_or_org / work_or_event / other. Prominence of the clue itself: `clue_inbound_links`, `clue_sitelinks`, `clue_wikipedia_articles`, and `thin_record` (true = 2 or fewer inbound links and no Wikipedia article: an almost empty record, tests memorisation of nothing). |
| `clue_sets.jsonl` | Best combinations of 3-5 clues per place (`best_sets`), each with `intersection` = how many places fit all clues together (1 = unambiguous). `flags` includes `single_family` as in the CSV. |
| `place_aliases.json` | `{place_name: [{qid, countries, has_coord}]}`: every Wikidata item sharing the name. Useful for writing a post that mentions the name but means another place, or for checking a post is not ambiguous. |
| `place_exceptions.csv` | The 18 places removed from the original 313, and why. |

Join on `wikidata_id` (CSV) = `place_qid` (JSONL); `place_aliases.json` is keyed by `place_name`.
