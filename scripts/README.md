# `scripts/` — dataset construction, validation, and analysis

Utility scripts used to build, validate, and analyse the datasets in `src/`.
They are standalone (run with `python scripts/<name>.py`) and assume the
project layout, resolving paths relative to the repository root.

For dataset provenance and generation metadata (which model, when, validation
results), see [`../DATASETS.md`](../DATASETS.md).

## Generation

The synthetic datasets were generated **interactively** with Claude Opus 4.x
from a written specification — there is no generation script. The specification
is the reproducible artifact and lives at the repository root in
[`../prompt`](../prompt). See `../DATASETS.md` for details.

### Programmatic generation via OpenRouter

`generate_synthetic.py` regenerates a dataset from the same spec through
[OpenRouter](https://openrouter.ai), using the client in `../src/llm_openrouter.py`.
Set `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` (an OpenRouter model slug) in
`src/.env`, or pass `--model` to override.

```bash
python src/llm_openrouter.py          # one-request smoke test of key + model
python scripts/generate_synthetic.py --format social_media --out src/social_media_synthetic.csv
python scripts/generate_synthetic.py --format journalistic --out src/journalistic_synthetic.csv
```

It generates in batches of signal type x continent (8 each, 144 rows total),
tells each batch which locations are already used, and resumes if you
interrupt it. Next to the CSV it writes a `.raw.jsonl` log of every
request/response and a `.meta.json` with the model, date and spec hash
(spec section 9.1). Rows are only checked for structure, so run
`dataset_qids.py validate` and `dataset_coords.py add` on the output afterwards.

### Real social media posts (English)

`fetch_bluesky_en.py` filters the Bluesky dump
[`alpindale/two-million-bluesky-posts`](https://huggingface.co/datasets/alpindale/two-million-bluesky-posts)
(2,107,530 raw firehose posts) down to its English entries.

```bash
python scripts/fetch_bluesky_en.py
```

The dataset has a single config (`default`) and **no language field** — its
schema is `text, created_at, author, uri, has_images, reply_to` — so English is
detected locally with `langdetect`. That costs ~2.2 ms/post, which is ~72
minutes single-core over 2M posts, hence `--num-proc` (default 10).

Every row detected as English is kept with all six columns intact: no
deduplication, no length limits, no content filtering. About **49.5%** of the
dump is English, giving ~1.04M rows (~380 MB) at `../src/bluesky_en.jsonl`.
Use `--format parquet` for a much smaller file, or `--limit N` for a trial run.

## Validation

| Script | Purpose |
|--------|---------|
| `dataset_qids.py validate <csv>` | Check every row's `qid` against Wikidata: the item exists, its English label matches the expected name, and its country (`P17`) matches the `country` column. Read-only. |
| `dataset_qids.py repair <csv> [--out f]` | The same three checks, plus a Wikidata search for a candidate whose `P17` matches, rewriting `qid`. In place unless `--out`. Its run over the 144-row GeoCorpora sample produced `../results/logs/fix_qids_geocorpora_144.log`. |

```bash
python scripts/dataset_qids.py validate src/social_media_dataset.csv
python scripts/dataset_qids.py validate src/journalistic_dataset.csv
python scripts/dataset_qids.py repair   src/geocorpora_eval.csv
```

This replaces `validate_dataset.py` and `fix_qids.py`, which were the same
capability split by dataset: one could only validate, the other could repair,
and each carried its own copy of the SPARQL boilerplate. Either action now
runs on any dataset.

Uses only the **public Wikidata SPARQL / search APIs** — no API key needed.

## Coordinate enrichment

| Script | Purpose |
|--------|---------|
| `dataset_coords.py add <csv>` | Fill `latitude`/`longitude` from Wikidata `P625` for each row's `qid`, in place. Sources from the QID currently in the CSV, so manual QID edits flow through. |
| `dataset_coords.py verify <csv>...` | Read-only check that stored coordinates still match `P625` (tolerance ~1.1 km). |

```bash
python scripts/dataset_coords.py add    src/social_media_dataset.csv
python scripts/dataset_coords.py verify src/social_media_dataset.csv src/journalistic_dataset.csv
```

This replaces `add_coords.py` and `verify_coords.py`. There is now **one
coordinate client** — `src/wikidata.coords_for_qids`, reading `P625` through
the same rate-limited `wbgetentities` path and the same `src/.wikidata_cache`
the pipeline uses. The old pair reached the same fact through a SPARQL endpoint
with a second cache at `src/.qid_coords_cache.pkl`; that transport and that
cache are gone.

`add_qids.py` (GeoNames `P1566` → QID) is deleted: it imported
`prepare_geocorpora.py`, which exists nowhere in the repo or its history, so it
could not run. `dataset_qids.py repair` is the surviving QID resolver.

## Evaluation and analysis

Evaluation itself lives outside this directory and is split in two, so that
changing a metric never re-runs a model:

| Entry point | Purpose |
|--------|---------|
| `../src/runner.py` | Inference only. Runs one configuration (`../configs/*.yaml`) over one dataset and writes `../results/predictions/<config>__<dataset>.jsonl`. Computes no metrics. |
| `score.py` | Metrics only. Reads those prediction files and computes everything. Imports nothing from `../src/`, loads no model, makes no network call. Sole home of the haversine implementation. |

```bash
python src/runner.py --list
python src/runner.py --config refined --dataset social_media_clean
python scripts/score.py --by condition
```

The old `eval_acc161.py` + `_eval_acc161_worker.py` pair is deleted. It was a
second copy of the inference loop with its own 8-column schema, existing only
because some corpora have coordinates but no gold QIDs. The single record
format allows a null `place_qid`, and `score.py` reports `exact` over the
items that have one (`nQID` column). **Accuracy@161 km itself is unchanged** —
it is the field standard and `score.py` reports it.

| Script | Purpose |
|--------|---------|
| `run_all.py` | Drives `../src/runner.py` over a set of (configuration, dataset) pairs, one at a time, logging each to `../results/logs/run_<config>__<dataset>.log`. Sequential on purpose: two runners contend for the Wikidata rate limiter and clobber each other's caches. Resumable. |
| `oracle_router.py` | Analysis-only. Per item, takes whichever component got it right — the ceiling for the routing idea. Computed from existing prediction files; it is **not** a configuration and has no YAML, because it chooses using the gold answer. |
| `fit_gate.py` | Sweeps the routing gate's `ambiguity_threshold` on the **dev** split and reports what each value would do. Reads stored `routing_features` and component predictions, so no model runs. Imports the rule from `../src/pipeline.route_from_features` rather than reimplementing it. |
| `analyze_failures.py` | Failure-mode analysis across the retired chapter-4 CSVs. Superseded by `analyse.py`. |
| `diagnose_ner_recall.py` | NER-recall diagnostic — attributes pipeline misses to the Flair NER stage. |
| `archive/` | Completed one-off scripts, kept as provenance for files on disk. See `archive/README.md`. |

## The dev/test split

`make_split.py` writes `../splits/place_split.csv` (the assignment) and
`../splits/item_split.csv` (the same assignment joined to item ids). Both are
committed, because the split has to be stable across runs.

The unit is a **place, not an item**: posts about the same place cannot
straddle the boundary. The assignment is global across datasets, stratified by
(dataset, condition), and deterministic — a place's rank inside its stratum is
`sha256(SALT + place_key)`, so it does not depend on row order.

```bash
python scripts/make_split.py               # refuses to overwrite silently
python scripts/score.py --split dev        # choose configurations and thresholds here
python scripts/score.py --split test       # report once
```

Rebuilding with `--force` moves places between sides and invalidates every
threshold fitted on dev. `SALT` and `DEV_FRACTION` in `make_split.py` are
constants for that reason.

## API keys

None of the validation or enrichment scripts here contain hard-coded API keys —
they use the public Wikidata endpoints. The *evaluation pipeline* in `src/`
(not these scripts) reads its keys from environment variables loaded from a
local `src/.env`, which is git-ignored and is never committed.
