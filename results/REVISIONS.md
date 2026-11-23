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
