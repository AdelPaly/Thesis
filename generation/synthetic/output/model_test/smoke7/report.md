# Model comparison

30 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 80% | 60% | 60% | 40% | 40% | 60% | **57%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| vse:qwen3.5:122b | no_environment | 9 |
| vse:qwen3.5:122b | clue_missing | 2 |
| vse:qwen3.5:122b | wrong_toponym | 1 |
| vse:qwen3.5:122b | length | 1 |

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign | statue | media_filler |
|---|---|---|---|---|---|
| vse:qwen3.5:122b | 0 of 30 | 0 of 30 | 0 of 30 | 7 of 30 | 0 of 30 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 3.4 | 3.8 | 1444 | 134 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 0.58 | 0.92 | "saw the" | 17% | 0.06 | 0 | 20.8 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

