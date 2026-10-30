# Model comparison

1 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 100% | - | - | - | - | - | **100%** |

## Drop reasons

| model | reason | n |
|---|---|---|

## Phrasing flags

Posts the checks keep or drop alike, but that read badly: a named person as an imagined judge ("X would be furious"), the clue's relation copied ("born here"), the distractor marked as foreign ("from elsewhere").

| model | imagined_judge | relation_echo | marked_foreign | statue | media_filler |
|---|---|---|---|---|---|
| vse:qwen3.5:122b | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 | 0 of 1 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 1.7 | 1.7 | 1687 | 107 | not reported | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| vse:qwen3.5:122b | 0.94 | 1.00 | "that concrete" | 100% | 0.00 | 0 | 18.0 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

