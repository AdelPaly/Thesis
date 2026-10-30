# Model comparison

1 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 100% | - | - | - | - | - | **100%** |
| vse:qwen3.5:122b | 100% | - | - | - | - | - | **100%** |

## Drop reasons

| model | reason | n |
|---|---|---|

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign | statue | media_filler |
|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |
| vse:qwen3.5:122b | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 1.4 | 1.4 | 1936 | 96 | $0.00017 | $0.17 |
| vse:qwen3.5:122b | 1.9 | 1.9 | 1693 | 131 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 0.83 | 1.00 | "so the" | 100% | 0.00 | 0 | 18.0 |
| vse:qwen3.5:122b | 0.88 | 1.00 | "draining the" | 100% | 0.00 | 0 | 24.0 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

