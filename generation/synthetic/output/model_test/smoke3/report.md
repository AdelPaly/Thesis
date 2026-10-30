# Model comparison

30 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 60% | 0% | 20% | 40% | 0% | 0% | **20%** |
| vse:qwen3.5:122b | 40% | 60% | 40% | 80% | 60% | 20% | **50%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| deepseek/deepseek-v4.1-flash | clue_missing | 10 |
| deepseek/deepseek-v4.1-flash | no_environment | 9 |
| deepseek/deepseek-v4.1-flash | wrong_toponym | 3 |
| deepseek/deepseek-v4.1-flash | length | 1 |
| deepseek/deepseek-v4.1-flash | noise_missing | 1 |
| vse:qwen3.5:122b | no_environment | 11 |
| vse:qwen3.5:122b | clue_missing | 4 |

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign |
|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 3 of 30 | 0 of 30 | 0 of 30 |
| vse:qwen3.5:122b | 6 of 30 | 0 of 30 | 0 of 30 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.9 | 1.3 | 1470 | 74 | $0.00016 | $0.16 |
| vse:qwen3.5:122b | 2.7 | 2.9 | 1256 | 96 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.65 | 0.93 | "bins still" | 10% | 0.04 | 0 | 16.7 |
| vse:qwen3.5:122b | 0.64 | 0.94 | "grab your" | 3% | 0.04 | 0 | 17.9 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

