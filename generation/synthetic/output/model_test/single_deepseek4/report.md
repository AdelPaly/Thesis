# Model comparison

3 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 67% | - | - | - | - | - | **67%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | no_environment | 1 |

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign | statue | media_filler |
|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 0 of 3 | 0 of 3 | 0 of 3 | 0 of 3 | 0 of 3 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 1.1 | 1.1 | 1938 | 99 | $0.00022 | $0.22 |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| openrouter:deepseek/deepseek-v4.1-flash | 0.73 | 0.85 | "stuck in" | 33% | 0.16 | 0 | 18.7 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

