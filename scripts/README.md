# `scripts/`

Everything here is on the path from raw data to reported results, in the order
it runs.

Scripts are standalone (`python scripts/<name>.py`) and resolve paths relative
to the repository root.

Setup is in the [top-level README](../README.md). How the dataset was made is
in [`../generation/README.md`](../generation/README.md).

---

## 1. Corpus for the style references

These build and clean the Bluesky dump (~380 MB, not committed) that
`generation/synthetic/build_exemplars.py` picks tone references from. Only
needed to build those references again.

| Script | Purpose |
|--------|---------|
| `fetch_bluesky_en.py` | Filters the 2.1M-post Bluesky dump down to its English entries (~1.04M rows) with `langdetect`, since the dataset has no language field. |
| `filter_adult_spam.py` | Strips solicitation spam from that dump using a two-tier vocabulary: unambiguous terms drop a post alone, ambiguous ones only alongside the "follow me / dm me / link in bio" grammar. Ordinary profanity is deliberately kept. |

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

Every record says how it was made: `config_hash` and `code_rev`, the `models`
and decoding settings by role, `run_utc`, the `dataset_release` it was computed
on with the file's `dataset_sha256`, the `seconds` the item took and its `cost`
(LLM calls, prompt and output tokens, Wikidata requests). An LLM answer
replayed from the response cache counts as `llm_cache_hits` and has no tokens:
the cache keeps the answer, not what it cost or when it was fetched.

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
| `score.py` | Reads the prediction files and computes every metric. Imports nothing from `../src/`, loads no model, makes no network call. |

```bash
python scripts/score.py --split dev          # choose here
python scripts/score.py --split test         # report once
python scripts/score.py --by condition
```

## 5. Analysis

| Script | Purpose |
|--------|---------|
| `analyse.py` | Failure attribution, the grounded-selection report, paired statistics and the report tables. Subcommands `failures`, `grounding`, `tables`, `stats`, `layers`, `bins`, `all`. `bins` breaks every configuration down by condition and by each bin on the records (prominence, theme, noise kind, name ambiguity, register, scale), with a place-resampled interval on exact match. Groups configurations by the `paradigm` on their records, orders rows by the `tier` in `configs/*.yaml`, and Holm-adjusts the by-place McNemar p-values over the `COMPARISONS` family. |
| `stats.py` | Statistical primitives, all clustered by place: McNemar, Holm adjustment, cluster bootstrap, paired effect intervals. |
| `error_layers.py` | Gives every item one layer of its configuration's chain where it was lost (no mention, gold not among the candidates, wrong selection, ...). Also `analyse.py layers`. |
| `simulate_designs.py` | Exploratory: cascades, gates and fallbacks built item by item from the saved predictions of two configurations. Not part of the planned comparisons. |
| `check_rescore.py` | Re-runs scoring and analysis into a temporary folder and compares with the tables in `results/summary/`, and with a second fresh run. |

```bash
python scripts/analyse.py all --split test --out results/summary
```

Failure attribution classifies each wrong prediction by the stage where it
broke, read from recorded fields rather than inferred. Ten levels: `error`
(a runner exception, reported separately from pipeline failures), `no_ner_span`,
`span_not_linked`, `wrong_entity`, `gold_not_retrieved`,
`wrong_candidate_selected`, `correct_candidate_not_first`, `abstained`,
`selection_violation`, `returned_nothing`, plus `no_intermediate_stages` for
architectures with no stages to attribute to (a level an architecture cannot
produce prints `n/a`, not 0.0).
Statistics pair and resample by **place**, never by item. The `stats` output
has a `p_holm` column: the Holm-adjusted by-place McNemar p over the family of
comparisons that ran (family size printed). `--out` writes the failure, table,
statistics, bootstrap, bin and grounding tables as CSV, with `n_boot` and `seed`
(10,000 replicates, seed 20260917 unless `--boot` / `--seed` say otherwise).

## API keys

No script here contains a key, and the Wikidata endpoints are public. The LLM
configurations need `VSE_API_KEY`, read from a git-ignored `src/.env`. The
synthetic dataset is generated in `../generation/`, which reads its own key.

## Environment

- **Python 3.13.3** (CPython, Windows 11 x64). `requirements.lock` (repo root)
  is the exact `pip freeze` of the environment; `requirements.txt` is the
  short list. `python install.py` creates `.venv` and installs the lock.
  The lock pins ReFinED to a git commit and `en_core_web_md` to its release
  wheel. torch is the CPU build. Scoring and analysis (`score.py`,
  `analyse.py`, `check_rescore.py`) need only PyYAML, numpy and scipy.
- **Irchel Geoparser** (`geoparser_irchel`) runs in a second venv,
  `.venv-geoparser` (or `GEOPARSER_PYTHON`), with `geoparser==0.6.1`
  (`python install.py --geoparser`). It needs transformers>=5, which conflicts
  with Flair, so it is not in the lock.

Large external files, none committed (sizes as on the original machine):

| What | Used by | Where it comes from | Size |
|------|---------|---------------------|------|
| BLINK models + entity encodings (`biencoder_wiki_large.*`, `crossencoder_wiki_large.*`, `all_entities_large.t7`, `entity.jsonl`) | `blink` | BLINK's `download_blink_models.sh`, into `.venv/src/blink/models` (or `BLINK_MODELS`); see the top-level README | ~30 GB |
| ReFinED `wikipedia_model` + `wikipedia_data` + `roberta-base` | `refined`, `grounded_gemma_linking` | downloaded by ReFinED on first use to `~/.cache/refined` | ~8.6 GB |
| Flair `flair/ner-large` (ner-english-large) | every config with Flair NER | Hugging Face hub, auto-download to `~/.flair` | ~6.3 GB |
| `intfloat/multilingual-e5-base` | rerank and dense retrieval (`configs/shared.yaml`) | Hugging Face hub, auto-download | ~2.2 GB |
| GeoNames gazetteer `geonames.db` | `geoparser_irchel` | built by the geoparser's gazetteer setup, in `%LOCALAPPDATA%/geoparser/gazetteers` | ~9.6 GB |
| `dguzh/geo-all-MiniLM-L6-v2`, spaCy `en_core_web_sm` | `geoparser_irchel` | Hugging Face hub / spaCy, in `.venv-geoparser` | ~0.1 GB |
| Wikidata response cache `src/.wikidata_cache/` | linkers, gazetteer, dense index | built by the runner from public Wikidata endpoints | ~0.6 GB |
| Bluesky English dump (`generation/synthetic/data/bluesky_en.jsonl`) | synthetic-data generation only | `scripts/fetch_bluesky_en.py` from the HF dataset `alpindale/two-million-bluesky-posts` | ~0.4 GB |

The LLM configurations (`gemma4`, `qwen3_vl`, `grounded_gemma`,
`grounded_gemma_linking`) call the VSE gateway (`VSE_API_KEY`) and replay from
`src/.llm_cache/` where a response is cached.
None of the above is needed to rescore: `python scripts/check_rescore.py`
reads only `results/predictions/`, `splits/` and `configs/`.
