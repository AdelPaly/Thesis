# Code revisions of the reported runs

The eleven reported configurations were not all run at one code revision, and
`gemma4` was written at two. This note sets out the check that no answer
depends on that. Everything in it is printed by

```bash
python scripts/check_revisions.py            # from git and results/predictions/
python scripts/check_revisions.py --replay   # also resolves the LLM-only replies again
```

The runs were not repeated. The test split is reported once, and a second run
would also pick up changes in Wikidata's live search that have nothing to do
with the code.

## Which run was made at which revision

| Revision | Configurations | Records |
|---|---|---|
| `470bcf0` | `majority_class`, `population_wikidata`, `geoparser_irchel`, `retrieve_rerank`, `refined` | 3,349 each |
| `3bff62c` | `blink` | 3,349 |
| `3bff62c` | `gemma4`, first part | 329 |
| `34d0b50` | `gemma4`, second part | 3,020 |
| `34d0b50` | `qwen3_vl`, `grounded_gemma_linking` | 3,349 each |
| `44cacce` | `dense_retrieval`, `grounded_gemma` | 3,349 each |

Every configuration except `gemma4` was run whole at one revision. No record
is marked `+dirty`, and each file has one `config_hash`.

## What changed between the revisions

Only changes under `src/` and `configs/` can change an answer.

| Step | Files changed | Configurations it can reach | Effect on the stored answers |
|---|---|---|---|
| `470bcf0` to `3bff62c` | `src/blink_linker.py` | `blink` | None on the others. `blink` was run after the change. |
| `3bff62c` to `7cd7ba4` | `src/llm_common.py`: LLM replies limited to 512 output tokens | the four LLM configurations | Reaches `gemma4`, see below. The other three were run after the change. The five configurations run at `470bcf0` and `blink` call no LLM. |
| `7cd7ba4` to `34d0b50` | nothing under `src/` or `configs/` | none | None. |
| `34d0b50` to `44cacce` | `src/dense_index.py`, `src/config.py` (one more allowed value for the dense pool), `configs/dense_retrieval.yaml`, `configs/grounded_gemma.yaml` | `dense_retrieval`, `grounded_gemma` | None on the others. Both were run after the change. |

So each configuration other than `gemma4` would have been given the same code
at any of the four revisions, as far as the code it uses goes.

## `gemma4`: two revisions in one file

The run stopped after 329 posts and was resumed eight hours later, after the
512-token limit had been added. The 329 replies written before the limit are
at most 119 characters long (a JSON object with a city, a country and a
continent), far below 512 tokens, so the limit would not have cut any of them.
All 329 are posts of condition A. Nothing else that `gemma4` uses changed
between the two revisions.

## The stored LLM-only answers against the code

With `--replay` the stored replies of `gemma4` and `qwen3_vl` (6,698 records)
are resolved again, once with `src/llm_geo.py` as it stood at `44cacce` and
once with the current code (`src/resolver.py`, rule v1):

- the two versions of the code give the same answer on all 6,698 records;
- 8 stored answers differ from what the code gives now: 7 of `gemma4` and 1 of
  `qwen3_vl`.

The 8 are not a difference in code. In each, a Wikidata lookup failed while
the run was being made, and the answer fell back to the country or to nothing;
the same reply resolves to a place today. Three of the stored answers are no
answer at all, four are the country and one is another item.

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

The reported numbers use the stored answers and were left as they are. None
of the 8 stored answers is scored correct. Resolved today, 4 of the 7 `gemma4`
answers and the `qwen3_vl` answer would be the correct place, so the reported
exact match is lower than the code would give by at most 4 of 3,349 posts
(0.12 points) for `gemma4` and 1 of 3,349 (0.03 points) for `qwen3_vl`.

## What this does not show

- The check for the configurations that call no LLM rests on the list of
  changed files, not on running them again.
- A failed lookup leaves no trace on a record, so the count of 8 covers only
  the two LLM-only configurations, where the reply is stored and can be
  resolved again.

## The post-hoc runs

`gemma4_v2` and `qwen3_vl_v2` (`results/posthoc/`) were first run at
`41a88f3+dirty`, before the resolver was committed. After the commit
(`9b69970`) both were written again from the response cache with
`--overwrite`: no model was called, and all 6,698 answers, replies and
resolutions are identical to the first run. The records now carry `9b69970`;
their run time and cost fields are those of the replay, not of the first run.
The rule had been fixed before the test run
(`results/posthoc/v2_frozen_rule.sha256`).

The result tables in `results/summary/` and `results/posthoc/` name the
revision they were written at and the version of `configs/comparisons.yaml`
in their last two columns.
