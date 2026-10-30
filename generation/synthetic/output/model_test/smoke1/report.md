# Model comparison

30 items per model (item_plan.jsonl). Pass = kept by the finalise.py checks.

## Pass rate

| model | A1 | A3 | B | EXPLICIT | NOISE_A1 | NOISE_B | all |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 20% | 20% | 40% | 40% | 40% | 60% | **37%** |
| vse:qwen3.5:122b | 0% | 0% | 0% | 0% | 0% | 0% | **0%** |

## Drop reasons

| model | reason | n |
|---|---|---|
| deepseek/deepseek-v4.1-flash | error | 16 |
| deepseek/deepseek-v4.1-flash | no_environment | 3 |
| vse:qwen3.5:122b | error | 30 |

## Cost and speed

| model | median latency s | p90 latency s | prompt tok / post | output tok / post | cost / post | cost / 1,000 posts |
|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 9.6 | 12.2 | 1312 | 2756 | $0.00192 | $1.92 |
| vse:qwen3.5:122b | - | - | - | - | - | - |

## Diversity

Distinct-n: unique share of all words / word pairs (higher is more varied). Near-duplicates: pairs of posts sharing at least 50% of their words.

| model | distinct-1 | distinct-2 | most common opening | share | mean Jaccard | near-dup pairs | mean words |
|---|---|---|---|---|---|---|---|
| deepseek/deepseek-v4.1-flash | 0.78 | 0.99 | "bins still" | 7% | 0.03 | 0 | 17.0 |

## Human rating

Fill realism, naturalness and fit (1-5) in rating_sheet.csv without opening rating_key.csv, then run `python compare_models.py --score-ratings --run <run>`.

