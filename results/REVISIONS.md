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

## Revision 3 (code changed on 26 November, not yet run)

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
against the reported run. `dense_retrieval` is not part of revision 3.

**A new configuration, `qwen3_vl_32b` (added 26 November, not yet run).**
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
ones.
Its predictions are kept out of `results/predictions/` until `grounded_gemma`
has run, so that the pool of `grounded_gemma` stays the one above.

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
