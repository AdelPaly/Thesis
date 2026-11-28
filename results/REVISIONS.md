# Code revisions of the reported runs, and the corrections

Three things are set out here: the correction of the resolver of `gemma4` and
`qwen3_vl`, the audit of 23 November after which six configurations were run
again, and the check that the code revision changed no answer of the runs that
were made once. What the scripts print:

```bash
python scripts/check_revisions.py            # from git and the stored records
python scripts/check_revisions.py --replay   # also resolves the first LLM-only replies again
python scripts/posthoc_report.py --split test   # first runs against the reported runs
```

## Which run was made at which revision

| Revision | Configurations | Records |
|---|---|---|
| `470bcf0` | `majority_class`, `geoparser_irchel`, `refined` | 3,349 each |
| `3bff62c` | `blink` | 3,349 |
| `34d0b50` | `grounded_gemma_linking` | 3,349 |
| `0389a2d` | `population_wikidata`, `retrieve_rerank`, `dense_retrieval`, `gemma4`, `qwen3_vl`, `grounded_gemma` (reported runs, after the audit) | 3,349 each |
| `470bcf0` | `population_wikidata`, `retrieve_rerank`, runs before the audit | 3,349 each |
| `44cacce` | `dense_retrieval`, `grounded_gemma`, runs before the audit | 3,349 each |
| `4a96b21` | `gemma4`, `qwen3_vl`, runs before the audit (corrected resolver) | 3,349 each |
| `3bff62c`, `34d0b50` | `gemma4`, first run (329 and 3,020 records) | 3,349 |
| `34d0b50` | `qwen3_vl`, first run | 3,349 |

No record is marked `+dirty`, and each file has one `config_hash`. The first
runs of the two LLM-only configurations are in `results/posthoc/original/`,
the runs made before the audit in `results/posthoc/before_audit/`.

## The corrected resolver of `gemma4` and `qwen3_vl`

The two LLM-only configurations get text from the model (a city, a country)
and resolve it to a Wikidata item.

1. **First runs** (19 November, prompt v1, rule v1). The rule searched Wikidata
   for the city and took the first result. The country the model named was used
   only when the city found nothing. Test scores: 38.9% and 42.5% exact match.
2. **The fault.** The analysis of the wrong answers showed that in a large
   share the model had named the correct place and country, and the rule had
   returned another item of the same name.
3. **Country only** (21 November, `scripts/posthoc_llm_resolver.py`). The
   stored answers of the first runs were resolved again with the country, with
   no model call: 47.9% and 56.6%. Written on the development split first, no
   variant tried.
4. **Corrected rule** (22 November, `src/resolver.py`, prompt v2). The prompt
   also asks for approximate coordinates, and the rule chooses by country and
   coordinates. It has 63 unit tests on hand-made cases
   (`tests/test_resolver.py`). One setting, the 25 km limit, was chosen on the
   development split. The rule was then fixed
   (`results/posthoc/v2_frozen_rule.sha256`, the two files as at `9b69970`;
   later commits changed only comments in them) and run once on the test split:
   50.8% and 61.9%.
5. **Runs with the corrected rule.** `configs/gemma4.yaml` and
   `configs/qwen3_vl.yaml` were set to prompt v2 and the corrected rule. Those
   runs were the reported ones until the audit below and are now in
   `results/posthoc/before_audit/`. They were written at `4a96b21` from the response cache: no
   model was called, and all 6,698 answers, replies and resolutions are
   identical to the run of step 4. The run time and cost of step 4, when the
   models were called, are in `results/posthoc/first_run_cost.csv`.

The correction was made after the first test scores had been seen. It was
developed on the development split only. `results/posthoc/` holds the first
runs, the country-only answers and the tables that compare the three
(`table_resolver_*.csv`, `table_resolver_differences_*.csv`,
`table_resolution_reasons_*.csv`).

## The audit of 23 November

After the resolver correction the other configurations were checked for the
same kind of fault: a step of this repository's own code, not the method under
test, losing or spoiling an answer. Four were found. Each was corrected on the
development split only, with unit tests (`tests/`, 106 in all). The corrected
code and configurations were then fixed
(`results/posthoc/audit_frozen.sha256`, the files as at commit `0389a2d`;
the next commit changed only comments in `src/resolver.py` and
`src/dense_index.py`) and each configuration was run once over the whole
dataset. The runs made before the
audit are kept in `results/posthoc/before_audit/`. Like the resolver
correction, this was done after test scores had been seen.

| Configuration | Fault | Correction |
|---|---|---|
| `population_wikidata` | The Wikidata search matches the start of a label and any alias, so the most populous result was often a larger unit of a similar name (the province for "Seville"). | `baseline.strategy: most_populous_named`: the most populous among the results named exactly as the toponym, when there are any. |
| `retrieve_rerank` | A candidate with no English description was given the mention as its description, and the reranker scored it high against the post the mention came from. | `src/ranking.py` places candidates without a description after those with one. A change of code, not of a configuration value. |
| `gemma4`, `qwen3_vl` | When none of the ten search results lay in the country the model named, the rule fell back, although the place was further down the results. | `prompt_template: v3`: rule v2 with a second search of fifty results in that case. Same prompt, so the same model replies. |
| `dense_retrieval`, `grounded_gemma` | 185 of the 3,811 pool items had no description or no country (a crater on Mars, disambiguation pages). Their short index text scored high for every post, and 20 items were the answer for 2,222 of the 3,349 posts. | `dense.pool: gold_plus_answers_described` leaves those items out. No gold place is removed. |

Exact match, %:

| Configuration | Dev before | Dev after | Test before | Test after |
|---|---|---|---|---|
| `population_wikidata` | 35.5 | 55.1 | 35.1 | 54.4 |
| `retrieve_rerank` | 18.8 | 19.9 | 21.8 | 23.4 |
| `dense_retrieval` | 2.3 | 10.2 | 3.3 | 12.0 |
| `gemma4` | 50.6 | 51.7 | 50.8 | 51.6 |
| `qwen3_vl` | 58.8 | 61.2 | 61.9 | 63.6 |
| `grounded_gemma` | 11.2 | 32.5 | 13.5 | 32.6 |

**The pool.** The pool of `dense_retrieval` and `grounded_gemma` holds the
final answers of the other configurations, so the two were run last, after the
other four: 3,599 places. When the correction was checked on the development
split the pool still held the answers from before the audit, and
`dense_retrieval` reached 10.6% there; with the final pool it is 10.2%.

**Failed lookups.** The run logs were searched for failed Wikidata searches,
which the code stores as "not found" with no error on the record. There were
two, one post each of `gemma4` and `qwen3_vl`. Both posts were computed again
and the search succeeded. One record of `grounded_gemma` carries an error: the
model chose a number outside its list of ten candidates, which is recorded as
an error by design.

**`refined` and `blink` were not run again.** The change in `src/ranking.py`
also reaches these two, which rerank their candidates with the same function.
Applying the new order to their stored candidates changes the answer of 3
records of `refined` and 5 of `blink` out of 3,349 (2 and 3 more exact matches,
all on the test split, none lost). Their reported scores are those of the
stored runs, without that change.

**Checked and left as it is.** `geoparser_irchel` returns a GeoNames entry with
no Wikidata item for 326 test posts. None of the 394 such entries has an item
on Wikidata (P1566, checked with the query service on 23 November), so these
answers are scored at country level, as the configuration says.

## Revision 3 (code changed on 26 November, run on the development split 26-27 November)

A review of the code and the stored test predictions found four more faults.
The code is changed on top of the audit (`0389a2d`); nothing has been run with
it yet. The plan is the same as before: develop on the development split, fix
the code, run once on the test split, and report the runs before and after.
Like the audit, this comes after test scores had been seen.

1. **A fixed chain chose the place of a linked person or organisation.** A
   linked entity without coordinates of its own was turned into one place by
   the first property along P159, P131, P19, P937, P276 that had one. For a
   person that is the place of birth before the work location, and residence
   (P551) was never read. On clue-only test posts, ReFinED linked the person
   correctly in most of them and the chain then sent most of them elsewhere:
   16.6% exact for residence clues and 13.0% for work-location clues, against
   62.2% for headquarters clues (every person clue in the dataset is tied to
   its place by residence, 532 clues, or work location, 193).

   A residence-first chain (P159, P131, P551, P937, P19, P276) was drafted
   first and replaced before any run, because its order would have matched how
   the clues were chosen. It is kept in `src/wikidata.py` only as the
   documented constant `GEO_PROP_CHAIN`. In its place is a rule with no order
   of preference, `linking.place_rule: all_places`: a linked entity with
   coordinates of its own is still the one candidate; any other gives every
   place it links to through P159, P740, P131, P276, P551, P937, P19 and P20,
   every non-deprecated value of each (end-dated ones included), with the
   one extra P131 hop when a target has no coordinates, deduplicated. The
   configuration's own selection then chooses among them: the E5 reranker for
   `refined`, `blink` and `retrieve_rerank`, the LLM for
   `grounded_gemma_linking`. On the search path of `retrieve_rerank` the caps
   (5 persons, 10 organisations) count entities, not places, so no property
   is cut off by its place in the list. `place_rule: chain_v1`, the default,
   reproduces the reported runs exactly (the chain above on the linking path;
   place of birth for persons and headquarters for organisations on the
   search path, under the same cache keys); `majority_class`,
   `population_wikidata` and `geoparser_irchel` keep it. There is no rule for
   a place that is a country (a residence given as a country is a candidate
   like any other).

   On the development split, from the links stored in the reported runs: a
   linked person gives 3.1 candidate places on average under `all_places`
   (`refined`, 154 links; 28 of them give six or more), against at most one
   under `chain_v1`; a linked organisation gives 1.2 (162 links).
2. **The LLM prompt asked for a city.** Prompt v4 (`src/llm_geo.py`) asks
   for the location itself, "a city, town or village, or any other named
   place", and keeps one refusal rule (refuse only if the text has no
   geographic signal); v1-v3 gave two that disagree. A first draft listed
   islands, lakes, mountains, parks and dams and told the model not to name
   "a nearby town"; both mirrored the dataset's place types and an error
   measured on it, and were removed before any run. Rule v4
   (`src/resolver.py`) is rule v3, the wider search of the audit, on the new
   `place` field (a reply with only a `city` field is read the same way).
   (Revision 3 was first drafted on the code before the audit, where it was
   called v3 and searched 50 results every time; it was renamed v4 and given
   the audit's search instead.) `gemma4` and `qwen3_vl` now name prompt v4
   and new response caches (`*_v4`), so the v2 caches are left as they are.

   Rule v4 also corrects four faults of the resolver, found with hand-made
   cases on names outside the dataset (`tests/test_resolver_outside.py`;
   Richmond, Toledo, Aurangabad). Rules v1-v3 are unchanged.

   a. *A namesake in the named country was taken however far it lay from the
      model's coordinates.* When no namesake in the country among the first
      ten results is within 25 km (the existing `TIE_KM`) of the coordinates,
      whether there is one or several, v4 also searches fifty results and
      takes an in-country namesake within 25 km (the first in search order).
      Only if there is none does v3's rule decide on the ten: one is kept, of
      several the first in search order, so the choice made on the
      development split (search order over coordinates far from every
      candidate) stays the final fallback. ("Richmond, Canada" with
      coordinates in British Columbia gave Richmond, Quebec, 3,800 km away.)
      The correction was first drafted for a single namesake only, and made
      uniform before the development run reached the LLM configurations, so
      that the two cases are not handled two ways.
   b. *The country was the first search hit for its name*, which for "UK" is
      the Ukrainian language (with coordinates in Ukraine), for "Georgia" the
      US state, for "Jordan" and "Chad" a family and a given name. v4 first
      matches the name against the English labels, then the aliases, of the
      Wikidata country items (`src/country_items.json`, 224 items with a
      current P31 of country or sovereign state, built by
      `scripts/build_country_items.py` with one QLever query, 26 November).
      A name that matches no item, or more than one ("The Netherlands"), is
      searched as before. The country fallback answers with the matched item.
   c. *A failed Wikidata request looked like a name that was not found.* v4's
      lookups raise on failure, the resolution is `error` (with `message`),
      nothing about the failure is cached, and the pipeline sets the record's
      `error`, so the runner retries the post on resume (the reply is in the
      response cache, so no model is called again).
   d. *An empty or blocked reply was recorded as `declined`*, the same as a
      model that refuses. The providers now return
      `{"refused": true, "empty_reply": true}` for it (`src/llm_common.py`),
      which v1-v3 and the grounded selector still read as a refusal; v4
      records `empty`, also for the old marker `{"refused": true}` in records
      written before, and for a reply whose fields are all null with
      `"refused": false`. Neither marker is cached.

   Re-resolving the stored development replies of the reported runs (prompt
   v2, 735 posts each, no model called) under v3 and under v4 on the same day's
   Wikidata changes 3 answers of `gemma4` and 5 of `qwen3_vl`. Correction b:
   "Guinea" now gives the country instead of no country (2 posts of
   `gemma4`, both still wrong). Correction a: in 4 posts (1 of `gemma4`, 3 of
   `qwen3_vl`) the ten results held several namesakes in the country, none
   within 25 km, and the wider search found the gold place within 25 km,
   where v3 took the first in search order about 50 km away; in 2 posts of
   `qwen3_vl` ("La Libertad, El Salvador") it moves the answer to a namesake
   12 km from the coordinates instead of one 33 km away, still wrong. Exact
   matches: `gemma4` 376 to 377, `qwen3_vl` 446 to 449 of 735; none became
   wrong. No development reply was empty or all null. One `gemma4` post ended
   in `error` in that pass (a Wikidata request failed while the development
   run was using the API) and resolved as under v3 when resolved again, which
   is the retry correction c provides. Between the stored answers and v3 resolved again today,
   Wikidata itself changed 6 and 4 answers. The effect of the prompt change
   needs replies to prompt v4, so it is measured by the development run.
3. **Demonyms were linked like place names.** `blink` linked every NER tag, so
   in "French tourists" the mention "French" became France even when the post
   named its place; `refined` and `grounded_gemma_linking` never used MISC
   mentions, and `retrieve_rerank` finds nothing for one. A plain exclusion of
   MISC was drafted first and replaced by a fallback, `ner.fallback_types:
   [MISC]`: NER keeps every tag, candidates come from the LOC, PER and ORG
   mentions, and only if those give no place are the MISC mentions
   (demonyms, events) linked, in a second linker call, and resolved. The links
   of the main mentions are therefore those of a run without the fallback.
   Each MISC mention keeps an entry in `linked_entities`, aligned with
   `ner_spans`, marked `fallback: used` or `not_needed`. This applies to
   `refined`, `blink` and `grounded_gemma_linking`; `retrieve_rerank` has no
   fallback (its search path has no handler for MISC), and
   `population_wikidata` stays LOC only. On the development split the
   fallback would be used for 29 posts of `refined` and 10 of `blink` (of 735),
   most of them clue-only or with a noise distractor.
4. **The grounded prompt did not show linked places by name, and asked the
   model to decline when unsure.** A candidate reached through a person or an
   organisation was listed under the mention's name ("Nick Willis") with the
   place's description, so the model never saw the place's own name; dense
   candidates were listed by place name. About a third of the linked
   candidates came through a person or an organisation. Linked candidates now
   carry the place's name (`label`, `src/wikidata.py`) and the list shows it
   for both sources (`src/llm_grounded.py`). The prompt's last rule ("Choosing
   nothing is better than choosing a candidate you do not believe in") is
   removed, so that it declines on the same grounds as prompt v4: when no
   candidate is the location. The old prompt is kept as `PROMPT_V1`; the new
   one writes to its own cache (`*_grounded_<source>_v2`). This reaches
   `grounded_gemma` and `grounded_gemma_linking`.

**The pool of `grounded_gemma` is held fixed (decided 26 November, before the
development run).** Its candidate pool is read from the prediction files in
`results/predictions/` when the configuration runs. For the development run
and for the final test run of revision 3, that pool is the one of the reported
runs: `results/predictions/` as at commit `57d76e7`, 3,673 QIDs before the
coordinate and description filters (3,599 places after them), SHA-256 of the
sorted, comma-joined QIDs
`bd40f02e622ec7788ddffa0bdb4a2824acc4e158472e1d753aee49a7c6f75068` (the
`pool_hash` that `src/dense_index.py` puts in the index cache key). So
`grounded_gemma` is run before any revision 3 run replaces a file in
`results/predictions/`, and only its prompt and the candidate names change
against the reported run.

*Corrected after the development run of 26-27 November:* the pool's QIDs were
held fixed (the development run's index has the `pool_hash` above), but the
index text was not. `src/dense_index.py` fetches each place's label,
description, aliases and country from live Wikidata when it builds the index,
and the index was built again on 27 November (13:03 UTC,
`index_3530d31fa83aacb7.pkl`, 3,599 places, none unfetched, all 955 gold
places in it). Between 23 and 27 November places changed label, description
or country (Portuguese civil parishes got new descriptions; Q967746 was
labelled "lima perú" and is now "Morón"): of the 1,263 places in the
development top-10 lists of the reported run and of `dev_rev3`, 1,235 are in
both, and 14 of those are listed with another label, description or country
(counted from the stored candidate records). So 148 of the 735 development
top-10 lists of `grounded_gemma`
differ from those of the reported run; the gold place entered the top 10 in 8
more posts and left it in none. The reported `dense_retrieval` was not run
again, so `grounded_gemma` (revision 3) and `dense_retrieval` no longer shared
a retriever, which breaks the planned comparison between them. The index is
therefore pinned (below), and `dense_retrieval` is part of revision 3.

**A new configuration, `qwen3_vl_32b` (added 26 November, first run on the
development split 26-27 November).**
Qwen3-VL 32B Instruct (dense), on the same gateway, with the prompt and rule
of revision 3 (prompt v4, rule v4) and a cache of its own (`qwen3_vl_32b_v4`).
It compares size within one family: the reported `qwen3_vl` is Qwen3-VL
235B-A22B, a mixture of experts with 235B parameters in total and about 22B
active per token, and `gemma4` is a dense 12B. It is compared with the
revision 3 runs of `qwen3_vl` and `gemma4`, not with their reported runs. The
pair `qwen3_vl` against `qwen3_vl_32b` is a tested comparison
(`configs/comparisons.yaml`), added with the configuration, after test scores
of the other configurations had been seen; it enlarges the Holm family by one,
so the adjusted p-values of the other pairs change slightly from the reported
ones. (After the development run the pair was moved out of the Holm family;
see below.)
Its predictions were kept out of `results/predictions/` until `grounded_gemma`
had run, so that the pool of `grounded_gemma` stayed the one above. With the
pinned index (below) the pool is no longer read, so where its predictions
are written no longer matters.

A run of `refined` or `blink` with this code also applies the reranking order
of the audit (`src/ranking.py`), which their reported runs predate (3 and 5
records, see above).

**Configuration hashes.** The two new fields are left out of the hash while
they hold their defaults (`_LATE_FIELDS` in `src/config.py`), so no other
configuration's hash moved: the hashes of `refined`, `blink`,
`retrieve_rerank` and `grounded_gemma_linking` changed (they set the new
fields), those of the other seven did not, and the five of those that were not
changed in revision 3 still equal the hash of their reported prediction file.
`gemma4` and `qwen3_vl` already had new hashes from their new cache names;
the prompt text is not hashed, the response-cache fingerprint is. The change in
(4) does not change the `config_hash` of `grounded_gemma`, so the runner would
resume its stored records as they are: run revision 3 into a new `--out-dir`
(or with `--overwrite`). The new tests are in `tests/test_chain.py`,
`tests/test_fallback.py`, `tests/test_prompt.py`, `tests/test_grounded.py`,
`tests/test_resolver_outside.py` (with `tests/fixtures/namesakes.json`, copied
from Wikidata) and at the end of `tests/test_resolver.py`. The corrections to
rule v4 change no configuration hash: the rule is chosen by
`prompt_template: v4`, and no record was written with v4 before them.

### Revision 3, after the development run of 26-27 November

Revision 3 was run on the development split on 26-27 November into
`results/archive/dev_rev3/` (code `57d76e7+dirty`, the code of commit `3ac1a37`). A
review of that run found the faults below. **Every choice here was made on
development results**; no test-split run was made with any of it. The runs
after these changes are in `results/archive/dev_rev3b/`, and `results/archive/dev_rev3/` is
kept as it is. Prompts and rules v1-v4, list v2 and every existing response
cache are unchanged.

1. **Prompt v4 lost the settlement.** Asked for "the location itself", on
   clue-only posts the models named a street, venue, club or person that often
   has no Wikidata item ("Pervez Musharraf Street", "Kossa F.C.", "CONMEBOL"),
   and rule v4 then fell straight back to the country. Reported run against
   `dev_rev3`: exact on condition A, `qwen3_vl` 10 to 3 of 71 posts (14% to
   4%) and `gemma4` 6 to 2 (8% to 3%); status `not_found`, `gemma4` 39 to 91
   and `qwen3_vl` 24 to 68 of 735. Prompt v5 (`src/llm_geo.py`) is v4 with one
   more JSON field after `place`: `"settlement": "<the city, town or village
   the location is in, or the location itself if it is one, or null if it is
   not in one>"`; nothing else in the wording changed. Rule v5
   (`resolve_place_settlement`, `src/resolver.py`) runs rule v4 on `place`;
   only when the place is `not_found` or `ambiguous`, and `settlement` holds
   a name that is not the place's (compared without case, accents or
   punctuation), it runs the same search and choice on the settlement, with
   the same country and coordinates. A settlement that resolves is the answer
   (`field: settlement`, the place's status kept as `place_status`);
   otherwise the answer is v4's (the country, then the continent). A place
   that resolves never triggers a settlement search (`field: place`), and a
   failed Wikidata request in either search ends in `error`. `gemma4`,
   `qwen3_vl` and `qwen3_vl_32b` name prompt v5 and new caches
   (`gemma4_12b_v5`, `qwen3_vl_235b_v5`, `qwen3_vl_32b_v5`).
2. **The dense index is pinned.** The index of 27 November 13:03 UTC (above)
   is the revision 3 index: `src/.dense_index/index_3530d31fa83aacb7.pkl`,
   SHA-256 `7a93c61c576426c22733b4781e25ebc524ca1546227cad8b6165bce683660f6f`,
   recorded next to it (`index_3530d31fa83aacb7.sha256`) and copied outside
   `src/.dense_index` (`../dense_index_backup/`, beside the repository). A new
   field, `dense.index_signature` (default null, in `_LATE_FIELDS`, so no
   other configuration's hash moved), names it in `dense_retrieval` and
   `grounded_gemma`. With it set, `src/dense_index.py` loads exactly that
   file, checks its SHA-256 and the settings it was built with, and never
   builds an index: a missing or different file is a `ConfigError`, which
   aborts the run. The pool on disk is not read at all. Each prediction
   record of the two configurations carries the signature
   (`models.dense_index`).
3. **The grounded list lost the mention.** Change (4) above shows the place's
   name but dropped the mention it was reached from: for a post about the club
   "C.R. Caála" the list showed only "Caála — Angola — city in Angola", and
   the model declined. `grounded_gemma_linking` declined with the gold place
   in its list in 29 development posts in the reported run and 44 in
   `dev_rev3`; of the 33 posts newly declined that way, 14 were right in the
   reported run (6 reached through an ORG mention, 4 through PER, 4 through
   LOC). List v3 (`src/llm_grounded.py`) adds one field at the end of a
   line, `from "<mention>"`, when a candidate has a `label` and the mention is
   not that name (compared as above); nothing else in the prompt changed. The
   list version is in the cache name per candidate source
   (`LIST_VERSIONS`): linking `_v3` (`gemma4_12b_grounded_linking_v3`), dense
   `_v2`. A dense candidate has no `label`, so its list is byte-identical to
   v2 (`tests/test_grounded.py` checks every place of the pinned index), and
   `grounded_gemma` keeps its cache. Stored candidates now carry `label`.
4. **Containment of runs outside `results/predictions/`.**
   `scripts/build_containment.py` read only `results/predictions/`, so the
   wrong answers of `results/archive/dev_rev3/` were never looked up and "exact or
   inside" came out too low for it: 216 distinct (predicted, gold) pairs, in
   356 records. It takes `--predictions-dir` (repeatable; default
   `results/predictions`), and `--all` now looks up again only the pairs of the
   directories read, keeping every other row of `results/containment.csv`.
   Run on `results/archive/dev_rev3` on 27 November.
5. **Failed lookups under `all_places` were silent.** In `src/wikidata.py`,
   `resolve_qid_to_candidates` threw away the completeness flag of the place
   fetch and of the country lookup, so a failed request silently dropped
   places, the fault rule v4 had fixed for the LLMs. Under `all_places` a
   failed fetch or country/continent lookup now raises `WikidataError`, nothing
   is cached for the request, the pipeline sets the record's `error`, and the
   runner retries the post on resume. The same holds in `get_coords` under
   `all_places`, for its search too (for every mention type, as the search
   path's cache entries of `all_places` are its own). `chain_v1` stays as it
   ran. Check: the linked QIDs of the `dev_rev3` records that have a link and
   no candidate place (`refined` 7, `blink` 10, `grounded_gemma_linking` 7)
   were resolved again with the day's Wikidata and empty caches on
   27 November; all lookups completed and none gives a place (demonyms and
   languages such as Croatian or Japanese, and BLINK links such as Tripoli
   Q1568779 with no place property), so none of them was a failed request.
6. **`qwen3_vl` against `qwen3_vl_32b` is exploratory.** The pair was added
   after test scores had been seen, and inside the Holm family it moved the
   adjusted p-values of the planned pairs. `configs/comparisons.yaml` keeps
   it in a separate list, `exploratory:`; `scripts/analyse.py` computes the
   same statistics for it outside the family, with an unadjusted p, labelled
   exploratory (column `kind` of `table_stats_<split>.csv`). The planned
   family is again the 14 pairs chosen in advance (`tests/test_comparisons.py`).
7. **Stale texts.** The revision 3 notes of the configurations say "run on
   dev 26-27 Nov"; the design texts of the three LLM configurations state rule
   v5, with the statuses `empty` and `error`; the resolver's docstring says
   that level `city` means the named place in v4 and v5.

**Configuration hashes.** `dense.index_signature` is a late field, so only the
two configurations that set it moved: `grounded_gemma` and `dense_retrieval`.
`gemma4`, `qwen3_vl` and `qwen3_vl_32b` moved with their new caches. The
hashes of `refined`, `blink`, `retrieve_rerank` and `grounded_gemma_linking`
equal those of their `dev_rev3` records (the list version of
`grounded_gemma_linking` is in its cache name, not its hash, so it is run into
a new `--out-dir`). The exact code of the `dev_rev3b` runs, whose records say
`+dirty`, is in `results/archive/dev_rev3b/code.diff`. New tests: the end of
`tests/test_resolver.py`, `tests/test_prompt.py`, `tests/test_grounded.py`,
`tests/test_chain.py`, and the new `tests/test_dense_pin.py` and
`tests/test_comparisons.py`.

**Development results of `results/archive/dev_rev3b/`** (run 27 November, 735
development posts per configuration, no record with an error; containment
looked up for `results/archive/dev_rev3` and `results/archive/dev_rev3b`). Exact match, in %,
`dev_rev3` to `dev_rev3b`:

| configuration | all | A | B | NOISE_A | NOISE_B | exact or inside, all |
|---|---|---|---|---|---|---|
| `gemma4` | 52.9 → 50.5 | 2.8 → 2.8 | 72.6 → 59.5 | 4.2 → 2.8 | 66.7 → 59.5 | 56.5 → 56.9 |
| `qwen3_vl` | 61.9 → 59.7 | 4.2 → 8.5 | 85.7 → 71.4 | 4.2 → 11.3 | 82.1 → 71.4 | 66.4 → 67.6 |
| `qwen3_vl_32b` | 60.3 → 56.5 | 7.0 → 4.2 | 82.1 → 64.3 | 4.2 → 4.2 | 82.1 → 69.0 | 62.9 → 63.8 |
| `grounded_gemma_linking` | 55.8 → 58.8 | 28.2 → 42.3 | 65.5 → 67.9 | 12.7 → 31.0 | 63.1 → 64.3 | 64.8 → 67.5 |

The reported runs had, for A: `gemma4` 8.5, `qwen3_vl` 14.1,
`grounded_gemma_linking` 31.0.

- *Prompt v5 and rule v5 did not do what they were for.* `not_found` fell
  (`gemma4` 91 to 60, `qwen3_vl` 68 to 30, `qwen3_vl_32b` 37 to 36), and 117,
  112 and 107 answers resolved through the settlement (63, 62 and 54 of them
  exact). But with a settlement field to fill, the models moved the town into
  `settlement` and named a landmark inside it as `place` ("Aga" became "Hideya
  Station", "Blantyre" "Chichiri Museum", "Stari Grad" "Hvar Airport"). The
  landmark resolves, so the settlement is never searched, and the answer lies
  inside the gold place without being it: exact on B and NOISE_B fell by 10 to
  18 points for all three models, while "exact or inside" stayed level or
  rose. Of the posts of B and NOISE_B that were exact in `dev_rev3` and are not
  in `dev_rev3b`, the place field resolved in 17 of 19 (`gemma4`), 21 of 21
  (`qwen3_vl`) and 26 of 27 (`qwen3_vl_32b`). Exact on A came back only for
  `qwen3_vl`, and only part of the way. No decision has been taken on v5.
- *List v3 helped `grounded_gemma_linking`.* Declines with the gold place in
  the list: 29 reported, 44 `dev_rev3`, 19 `dev_rev3b`; exact on A 28.2 to
  42.3, the highest of any run.
- *The pinned index.* The top-10 list of `dense_retrieval` in `dev_rev3b`
  equals that of `grounded_gemma` in `dev_rev3` for all 735 posts. Its exact
  match is that of the reported run (10.2%).

#### Prompt v5 as run in results/archive/dev_rev3b/ (replaced)

As the model saw it (in `src/llm_geo.py` the braces of the JSON line are
doubled):

```
You are a geolocation assistant. Given a text, identify the single most relevant real-world location the text refers to.

The location can be a city, town or village, or any other named place. Name the location itself, not a person, organisation or event mentioned in the text.

Respond with JSON only, no explanation:
{"place": "<name of the location, or null if unknown>", "settlement": "<the city, town or village the location is in, or the location itself if it is one, or null if it is not in one>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "latitude": <approximate latitude of the location in decimal degrees, or null if unknown>, "longitude": <approximate longitude in decimal degrees, or null if unknown>, "refused": false}

Set "refused" to true only if the text contains no geographic signal at all.

Text: {text}
```

The prompt below keeps the name v5 and replaces that text. It is the final LLM
prompt of revision 3. It was chosen on dev results, after v4 and the first v5.
Its dev results are reported whatever they are. After its dev run, only clear
bugs are fixed.

#### Prompt v5 (final), run in results/archive/dev_rev3c/

```
You are a geolocation assistant. Given a text, identify the single most relevant real-world location the text refers to.

The location can be a city, town or village, or any other named place. If the text is about a building, street or venue, name the city, town or village it is in. If it is about a person, organisation or event, name the place where they are.

Respond with JSON only, no explanation:
{"place": "<name of the location, or null if unknown>", "country": "<country name or null if unknown>", "continent": "<continent name or null if unknown>", "latitude": <approximate latitude of the location in decimal degrees, or null if unknown>, "longitude": <approximate longitude in decimal degrees, or null if unknown>, "refused": false}

Set "refused" to true only if the text contains no geographic signal at all.

Text: {text}
```

It is v4 with the second paragraph replaced and v4's JSON fields (no
`settlement`). Rule v5 needs the settlement field, so `gemma4`, `qwen3_vl`
and `qwen3_vl_32b` resolve it with rule v4, through a new field,
`llm.resolver_rule` (default null: the rule named like the prompt; in
`_LATE_FIELDS`, so no other configuration's hash moved; recorded in each
record as `models.llm_resolver_rule`). Rule v5 stays in the code with its
tests. New caches: `gemma4_12b_v5b`, `qwen3_vl_235b_v5b`, `qwen3_vl_32b_v5b`.
The exact code of the `dev_rev3c` runs is in `results/archive/dev_rev3c/code.diff`.

**Development results of `results/archive/dev_rev3c/`** (run 27 November, 735
development posts per configuration; containment looked up for
`results/archive/dev_rev3c`). The `gemma4` run stopped at post 268 when the machine ran
low on memory and was resumed from its records. One record ended in a
Wikidata `cirrussearch-too-busy-error`; a resume looked it up again from the
cached reply, and no record has an error. Columns: reported run (v2),
`dev_rev3` (v4), `dev_rev3b` (first v5), `dev_rev3c` (v5), in %.

| configuration | metric | all | A | B | NOISE_A | NOISE_B |
|---|---|---|---|---|---|---|
| `gemma4` | exact | 51.7 / 52.9 / 50.5 / 54.3 | 8.5 / 2.8 / 2.8 / 7.0 | 71.4 / 72.6 / 59.5 / 76.2 | 8.5 / 4.2 / 2.8 / 7.0 | 64.3 / 66.7 / 59.5 / 70.2 |
| | exact or inside | 53.3 / 56.5 / 56.9 / 56.1 | 11.3 / 18.3 / 21.1 / 14.1 | 71.4 / 75.0 / 73.8 / 76.2 | 11.3 / 14.1 / 15.5 / 11.3 | 65.5 / 67.9 / 70.2 / 70.2 |
| | country | 63.5 / 63.5 / 63.3 / 64.8 | 39.4 / 36.6 / 35.2 / 42.3 | 81.0 / 81.0 / 79.8 / 83.3 | 28.2 / 31.0 / 32.4 / 28.2 | 73.8 / 72.6 / 77.4 / 76.2 |
| | no result | 8.6 / 7.8 / 12.0 / 3.5 | 29.6 / 32.4 / 39.4 / 16.9 | 3.6 / 3.6 / 4.8 / 1.2 | 22.5 / 21.1 / 29.6 / 9.9 | 2.4 / 3.6 / 4.8 / 0.0 |
| `qwen3_vl` | exact | 61.2 / 61.9 / 59.7 / 63.4 | 14.1 / 4.2 / 8.5 / 12.7 | 84.5 / 85.7 / 71.4 / 86.9 | 14.1 / 4.2 / 11.3 / 9.9 | 79.8 / 82.1 / 71.4 / 85.7 |
| | exact or inside | 64.4 / 66.4 / 67.6 / 66.8 | 18.3 / 21.1 / 29.6 / 22.5 | 88.1 / 88.1 / 88.1 / 89.3 | 19.7 / 22.5 / 32.4 / 23.9 | 83.3 / 84.5 / 84.5 / 86.9 |
| | country | 74.8 / 74.6 / 74.0 / 75.5 | 46.5 / 46.5 / 45.1 / 53.5 | 94.0 / 94.0 / 95.2 / 94.0 | 43.7 / 46.5 / 47.9 / 50.7 | 91.7 / 91.7 / 90.5 / 92.9 |
| | no result | 4.5 / 4.9 / 5.3 / 3.8 | 23.9 / 29.6 / 31.0 / 22.5 | 0.0 / 0.0 / 0.0 / 0.0 | 16.9 / 15.5 / 16.9 / 11.3 | 0.0 / 0.0 / 0.0 / 0.0 |
| `qwen3_vl_32b` | exact | - / 60.3 / 56.5 / 59.9 | - / 7.0 / 4.2 / 8.5 | - / 82.1 / 64.3 / 81.0 | - / 4.2 / 4.2 / 7.0 | - / 82.1 / 69.0 / 79.8 |
| | exact or inside | - / 62.9 / 63.8 / 61.8 | - / 14.1 / 21.1 / 12.7 | - / 83.3 / 82.1 / 82.1 | - / 14.1 / 19.7 / 12.7 | - / 83.3 / 82.1 / 81.0 |
| | country | - / 71.8 / 71.7 / 72.0 | - / 42.3 / 45.1 / 49.3 | - / 89.3 / 89.3 / 89.3 | - / 36.6 / 36.6 / 36.6 | - / 90.5 / 89.3 / 88.1 |
| | no result | - / 3.4 / 3.5 / 1.5 | - / 23.9 / 23.9 / 8.5 | - / 0.0 / 0.0 / 0.0 | - / 9.9 / 11.3 / 5.6 | - / 0.0 / 0.0 / 0.0 |

EXPLICIT and NOISE_EXPLICIT moved by at most 2 points against `dev_rev3`
(full table in `results/archive/dev_rev3c/report.txt`). Exact / exact or inside by
gold type (column `type` of `generation/out/places.csv`; 626 settlements,
39 islands, 30 natural, 40 manmade), `dev_rev3` → `dev_rev3c`:

| configuration | settlement | island | natural | manmade |
|---|---|---|---|---|
| `gemma4` | 52.1/55.1 → 53.5/55.1 | 71.8/79.5 → 69.2/76.9 | 46.7/53.3 → 50.0/50.0 | 52.5/57.5 → 55.0/55.0 |
| `qwen3_vl` | 61.5/66.1 → 62.8/66.3 | 74.4/82.1 → 74.4/82.1 | 63.3/66.7 → 66.7/66.7 | 55.0/55.0 → 60.0/60.0 |
| `qwen3_vl_32b` | 60.1/62.6 → 60.1/61.8 | 76.9/84.6 → 71.8/79.5 | 56.7/56.7 → 56.7/56.7 | 50.0/50.0 → 47.5/47.5 |

Resolution status (`dev_rev3` → `dev_rev3c`): `gemma4` not_found 91 → 68,
ambiguous 13 → 5, declined 40 → 21, resolved 591 → 641; `qwen3_vl` not_found
68 → 47, ambiguous 4 → 3, declined 30 → 22, resolved 633 → 663;
`qwen3_vl_32b` not_found 37 → 22, declined 25 → 11, resolved 673 → 702.

- Exact on B is back at or above v4 (`gemma4` 76.2, `qwen3_vl` 86.9;
  `qwen3_vl_32b` 81.0 against 82.1), and the islands keep v4's gain
  (`qwen3_vl_32b` loses 2 of 39).
- Exact on A rose from v4 for all three (`gemma4` 2 to 5 of 71, `qwen3_vl` 3
  to 9, `qwen3_vl_32b` 5 to 6) but stays below the reported run for `gemma4`
  (6) and `qwen3_vl` (10). Country on A rose for all three, and no result on
  A fell (`qwen3_vl_32b` 23.9 to 8.5). "Exact or inside" on A fell back from
  the first v5 to about v4, as the models now name the town rather than a
  landmark inside it.
- On A the models still name a person's or a club's place from the wrong
  country (Pervez Musharraf Street → Islamabad for Delhi; Vital'O FC →
  Sutton Coldfield or Lyon for Bujumbura; `results/archive/dev_rev3c/report.txt`
  lists 30 sampled answers).

### Final run of revision 3

Code frozen at commit `13187fe` ("Freeze revision 3", 27 November 2026). The
final run, all 3,349 posts of `refined`, `blink`, `retrieve_rerank`,
`dense_retrieval`, `grounded_gemma`, `grounded_gemma_linking`, `gemma4`,
`qwen3_vl` and `qwen3_vl_32b`, was made on 27-28 November 2026 into
`results/final_rev3/`.

## What changed between the revisions of the other runs

Only changes under `src/` and `configs/` can change an answer.

| Step | Files changed | Configurations it can reach | Effect on the stored answers |
|---|---|---|---|
| `470bcf0` to `3bff62c` | `src/blink_linker.py` | `blink` | None on the others. `blink` was run after the change. |
| `3bff62c` to `7cd7ba4` | `src/llm_common.py`: LLM replies limited to 512 output tokens | the four LLM configurations | The two grounded configurations were run after the change. The five configurations run at `470bcf0` and `blink` call no LLM. For the first run of `gemma4` see below. |
| `7cd7ba4` to `34d0b50` | nothing under `src/` or `configs/` | none | None. |
| `34d0b50` to `44cacce` | `src/dense_index.py`, `src/config.py` (one more allowed value for the dense pool), `configs/dense_retrieval.yaml`, `configs/grounded_gemma.yaml` | `dense_retrieval`, `grounded_gemma` | None on the others. Both were run after the change. |
| `44cacce` to `4a96b21` | `src/resolver.py`, `src/llm_geo.py`, `configs/gemma4.yaml`, `configs/qwen3_vl.yaml` | `gemma4`, `qwen3_vl` | None on the others. |
| `4a96b21` to `0389a2d` | the audit: `src/gazetteer.py`, `src/ranking.py`, `src/resolver.py`, `src/llm_geo.py`, `src/dense_index.py`, `src/pipeline.py`, `src/config.py` and five configurations | the six configurations run again, and `refined` and `blink` through `src/ranking.py` | The six were run after the change. `refined`: 3 answers would differ, `blink`: 5 (see the audit above). None on `majority_class`, `geoparser_irchel`, `grounded_gemma_linking`. |

So each of the five configurations that were run once would have been given
the same code at any of these revisions, as far as the code it uses goes, with
the exception of the reranking order for `refined` and `blink` stated above.
For the configurations that call no LLM this rests on the list of changed
files, not on running them again.

## The first runs

The first runs of `gemma4` and `qwen3_vl` (prompt v1) are kept as the record of
the resolver correction.

**Two revisions in one file.** The first run of `gemma4` stopped after 329
posts and was resumed eight hours later, after the 512-token limit had been
added. The 329 replies written before the limit are at most 119 characters
long, far below 512 tokens, so the limit would not have cut any of them.

**Failed lookups.** With `--replay` the stored replies of the first runs
(6,698 records) are resolved again with rule v1, once with `src/llm_geo.py` as
it stood at `44cacce` and once with the current code. The two versions of the
code give the same answer on all 6,698 records. Eight stored answers differ
from what the code gives now, seven of `gemma4` and one of `qwen3_vl`: in each
a Wikidata lookup failed during the run and the answer fell back to the
country or to nothing. None of the eight is scored correct, and five would be
correct if resolved today.

| Item | Configuration | Stored | Resolved now |
|---|---|---|---|
| `a0595c64955f6d2d` | `gemma4` | Q30 | Q515719 |
| `315cc8b9ca9ccde2` | `gemma4` | Q45 | Q83247 |
| `d0225e963003afc4` | `gemma4` | none | Q884 |
| `fd08b4a0ccf29926` | `gemma4` | none | Q2256382 |
| `04bb3724fdaacc7e` | `gemma4` | none | Q28520 |
| `6b7787f49d7cfb2d` | `gemma4` | Q3359409 | Q319836 |
| `e091651003066fd1` | `gemma4` | Q32 | Q1842 |
| `9107f8eb39526e88` | `qwen3_vl` | Q145 | Q107393 |

A failed lookup leaves no trace on a record of the first runs. The reported
runs record the outcome of every resolution (`resolution` on the record), so a
lookup that finds nothing is visible there as `not_found`.

## The result tables

The tables in `results/summary/` and `results/posthoc/` name the revision they
were written at and the version of `configs/comparisons.yaml` in their last two
columns. The list of planned comparisons is unchanged; all of them were computed again
with the runs made after the audit.
