# `scripts/`

Everything here is on the path from raw data to reported results, in the order
it runs.

Scripts are standalone (`python scripts/<name>.py`) and resolve paths relative
to the repository root.

For dataset provenance see [`../DATASETS.md`](../DATASETS.md). For the
decisions behind the numbers — the split, the tiers, the failure taxonomy — see
[`../METHODOLOGY.md`](../METHODOLOGY.md).

---

## 0. Corpus acquisition

These build and clean the Bluesky dump: ~380 MB, not committed, and
regenerable only from here. No configured dataset consumes it *yet* -- it is
intended as the source of style exemplars for the next dataset, so treat it as
on the path rather than as a leftover.

| Script | Purpose |
|--------|---------|
| `fetch_bluesky_en.py` | Filters the 2.1M-post Bluesky dump down to its English entries (~1.04M rows) with `langdetect`, since the dataset has no language field. |
| `filter_adult_spam.py` | Strips solicitation spam from that dump using a two-tier vocabulary: unambiguous terms drop a post alone, ambiguous ones only alongside the "follow me / dm me / link in bio" grammar. Ordinary profanity is deliberately kept. |

## 1. Dataset construction

| Script | Purpose |
|--------|---------|
| `generate_synthetic.py` | Regenerates a synthetic dataset from the spec at [`../prompt`](../prompt) via OpenRouter, in batches of signal type × continent, resumable, logging every request/response plus a `.meta.json` with model, date and spec hash. |
| `dataset_qids.py validate \| repair` | Checks each row's `qid` against Wikidata (item exists, English label matches, `P17` country agrees) and optionally repairs it by searching for a candidate whose country matches. Either action runs on any dataset. |
| `dataset_coords.py add \| verify` | Fills `latitude`/`longitude` from Wikidata `P625` for each row's `qid`, or checks the stored values still match (~1.1 km tolerance). |

Both use the one coordinate client, `src/wikidata.coords_for_qids`, and the one
shared cache in `src/.wikidata_cache/`.

## 2. The dev/test split

| Script | Purpose |
|--------|---------|
| `make_split.py` | Writes `../splits/place_split.csv` (the assignment) and `../splits/item_split.csv` (joined to item ids). Both committed. |

The unit is a **place, not an item**, so posts about the same place cannot
straddle the boundary. Global across datasets, stratified by
(dataset, condition), deterministic from `sha256(SALT + place_key)`.
Configurations and thresholds are chosen on **dev**; **test is reported once**.
`--force` moves places between sides and invalidates every threshold fitted on
dev, which is why `SALT` and `DEV_FRACTION` are constants.

## 3. Inference

| Entry point | Purpose |
|--------|---------|
| `../src/runner.py` | Runs one configuration (`../configs/*.yaml`) over one dataset and writes per-item records to `../results/predictions/<config>__<dataset>.jsonl`. **Computes no metrics.** Resumable; a failed item is recorded with an `error`, never skipped. |
| `run_all.py` | Drives the runner over a set of (configuration, dataset) pairs, one at a time, logging each to `../results/logs/`. Sequential on purpose: two runners contend for the Wikidata rate limiter and clobber each other's caches. |

`configs/` is the registry — adding a file is enough to get a configuration
run. `ORDER` in `run_all.py` is an ordering preference only; anything in
`configs/` but not in `ORDER` still runs.

```bash
python src/runner.py --list
python scripts/run_all.py
```

## 4. Scoring

| Script | Purpose |
|--------|---------|
| `score.py` | Reads the prediction files and computes every metric. Imports nothing from `../src/`, loads no model, makes no network call. Sole home of the haversine implementation. |

```bash
python scripts/score.py --split dev          # choose here
python scripts/score.py --split test         # report once
python scripts/score.py --by condition
```

## 5. Analysis

| Script | Purpose |
|--------|---------|
| `analyse.py` | Failure attribution, the grounded-selection report, paired statistics and the report tables. Subcommands `failures`, `grounding`, `tables`, `stats`, `all`. Groups configurations by the `paradigm` on their records rather than by any hardcoded list. |
| `stats.py` | Statistical primitives, all clustered by place: McNemar, Wilcoxon signed-rank, cluster bootstrap, paired effect intervals. |

```bash
python scripts/analyse.py all --split test --out results/summary
```

Failure attribution classifies each wrong prediction by the stage where it
broke, read from recorded fields rather than inferred: `no_ner_span`,
`span_not_linked`, `wrong_entity`, `wrong_candidate_selected`,
`correct_candidate_not_first`, `returned_nothing`, plus
`no_intermediate_stages` for architectures with no stages to attribute to.
Statistics pair and resample by **place**, never by item.

## API keys

No script here contains a key, and the Wikidata endpoints are public.
`generate_synthetic.py` needs `OPENROUTER_API_KEY` / `OPENROUTER_MODEL`, and
the LLM configurations need `GOOGLE_API_KEY`; all are read
from a git-ignored `src/.env`.
