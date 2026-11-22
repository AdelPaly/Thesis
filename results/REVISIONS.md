# Code revisions of the reported runs, and the corrected resolver

Two things are set out here: the correction of the resolver of `gemma4` and
`qwen3_vl`, which were run twice, and the check that the code revision changed
no answer of the other runs. What the scripts print:

```bash
python scripts/check_revisions.py            # from git and the stored records
python scripts/check_revisions.py --replay   # also resolves the first LLM-only replies again
python scripts/posthoc_report.py --split test   # first runs against the reported runs
```

## Which run was made at which revision

| Revision | Configurations | Records |
|---|---|---|
| `470bcf0` | `majority_class`, `population_wikidata`, `geoparser_irchel`, `retrieve_rerank`, `refined` | 3,349 each |
| `3bff62c` | `blink` | 3,349 |
| `34d0b50` | `grounded_gemma_linking` | 3,349 |
| `44cacce` | `dense_retrieval`, `grounded_gemma` | 3,349 each |
| `4a96b21` | `gemma4`, `qwen3_vl` (reported runs, corrected resolver) | 3,349 each |
| `3bff62c`, `34d0b50` | `gemma4`, first run (329 and 3,020 records) | 3,349 |
| `34d0b50` | `qwen3_vl`, first run | 3,349 |

No record is marked `+dirty`, and each file has one `config_hash`. The first
runs are in `results/posthoc/original/`.

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
5. **Reported runs.** `configs/gemma4.yaml` and `configs/qwen3_vl.yaml` now use
   prompt v2 and the corrected rule, and the files in `results/predictions/`
   are those runs. They were written at `4a96b21` from the response cache: no
   model was called, and all 6,698 answers, replies and resolutions are
   identical to the run of step 4. The run time and cost of step 4, when the
   models were called, are in `results/posthoc/first_run_cost.csv`.

The correction was made after the first test scores had been seen. It was
developed on the development split only. `results/posthoc/` holds the first
runs, the country-only answers and the tables that compare the three
(`table_resolver_*.csv`, `table_resolver_differences_*.csv`,
`table_resolution_reasons_*.csv`).

Two configurations were run before the correction and depend on the answers of
the others: the candidate pool of `dense_retrieval` and `grounded_gemma` holds
the other configurations' answers as they were on 19 November, the first-run
answers of `gemma4` and `qwen3_vl` among them. They were not run again.

## What changed between the revisions of the other runs

Only changes under `src/` and `configs/` can change an answer.

| Step | Files changed | Configurations it can reach | Effect on the stored answers |
|---|---|---|---|
| `470bcf0` to `3bff62c` | `src/blink_linker.py` | `blink` | None on the others. `blink` was run after the change. |
| `3bff62c` to `7cd7ba4` | `src/llm_common.py`: LLM replies limited to 512 output tokens | the four LLM configurations | The two grounded configurations were run after the change. The five configurations run at `470bcf0` and `blink` call no LLM. For the first run of `gemma4` see below. |
| `7cd7ba4` to `34d0b50` | nothing under `src/` or `configs/` | none | None. |
| `34d0b50` to `44cacce` | `src/dense_index.py`, `src/config.py` (one more allowed value for the dense pool), `configs/dense_retrieval.yaml`, `configs/grounded_gemma.yaml` | `dense_retrieval`, `grounded_gemma` | None on the others. Both were run after the change. |

So each of the nine configurations that were run once would have been given
the same code at any of these revisions, as far as the code it uses goes. For
the configurations that call no LLM this rests on the list of changed files,
not on running them again.

## The first runs

They are no longer the reported runs, and are kept as the record of the
correction.

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
columns. The list of planned comparisons is unchanged; the comparisons that
involve `gemma4` or `qwen3_vl` were computed again with the reported runs.
