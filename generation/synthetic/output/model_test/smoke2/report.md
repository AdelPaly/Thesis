# Model comparison

30 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 60% | 20% | 20% | 40% | 0% | 0% | **23%** |
| vse:qwen3.5:122b | 100% | 80% | 60% | 80% | 60% | 100% | **80%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| deepseek/deepseek-v4.1-flash | no_environment | 12 |
| deepseek/deepseek-v4.1-flash | noise_missing | 4 |
| deepseek/deepseek-v4.1-flash | clue_missing | 3 |
| deepseek/deepseek-v4.1-flash | wrong_toponym | 2 |
| deepseek/deepseek-v4.1-flash | length | 2 |
| vse:qwen3.5:122b | no_environment | 4 |
| vse:qwen3.5:122b | clue_missing | 2 |

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign |
|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 5 of 30 | 4 of 30 | 0 of 30 |
| vse:qwen3.5:122b | 2 of 30 | 4 of 30 | 1 of 30 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.9 | 1.2 | 1284 | 76 | $0.00017 | $0.17 |
| vse:qwen3.5:122b | 2.2 | 2.3 | 1068 | 99 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.63 | 0.93 | "bins out" | 7% | 0.05 | 0 | 17.1 |
| vse:qwen3.5:122b | 0.63 | 0.94 | "tired of" | 7% | 0.04 | 0 | 18.3 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

