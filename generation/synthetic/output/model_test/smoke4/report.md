# Model comparison

30 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0% | 20% | 60% | 40% | 20% | 0% | **23%** |
| vse:qwen3.5:122b | 80% | 100% | 60% | 80% | 40% | 0% | **60%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| deepseek/deepseek-v4.1-flash | no_environment | 16 |
| deepseek/deepseek-v4.1-flash | clue_missing | 2 |
| deepseek/deepseek-v4.1-flash | noise_missing | 2 |
| deepseek/deepseek-v4.1-flash | length | 1 |
| deepseek/deepseek-v4.1-flash | missing_name | 1 |
| deepseek/deepseek-v4.1-flash | wrong_toponym | 1 |
| vse:qwen3.5:122b | no_environment | 9 |
| vse:qwen3.5:122b | wrong_toponym | 2 |
| vse:qwen3.5:122b | clue_missing | 1 |

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign |
|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 1 of 30 | 0 of 30 | 0 of 30 |
| vse:qwen3.5:122b | 4 of 30 | 0 of 30 | 0 of 30 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.9 | 1.2 | 1525 | 79 | $0.00017 | $0.17 |
| vse:qwen3.5:122b | 3.0 | 3.2 | 1309 | 98 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.67 | 0.94 | "bins still" | 7% | 0.04 | 0 | 17.6 |
| vse:qwen3.5:122b | 0.67 | 0.95 | "heading to" | 3% | 0.04 | 0 | 18.7 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

