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

## Validation (the scripts your reviewer asked for)

| Script | Purpose |
|--------|---------|
| `validate_dataset.py` | Validate a synthetic dataset against Wikidata: checks each `qid` exists, its English label matches the expected name, and its country (`P17`) matches. Reports failing rows. Run per dataset. |
| `fix_qids.py` | The GeoCorpora validator: checks/repairs `qid`s by searching Wikidata and matching country (`P17`). Its run over the 144-row GeoCorpora sample produced `../results/logs/fix_qids_geocorpora_144.log`. |

```bash
python scripts/validate_dataset.py src/social_media_dataset.csv
python scripts/validate_dataset.py src/journalistic_dataset.csv
```

Both use only the **public Wikidata SPARQL / search APIs** — no API key needed.

## Coordinate / QID enrichment

| Script | Purpose |
|--------|---------|
| `add_qids.py` | Add a `qid` column by resolving a `geoNameId` to its Wikidata QID via `P1566`. |
| `add_coords.py` | Add `latitude`/`longitude` by looking up Wikidata `P625` for each row's QID (batched, disk-cached). |
| `add_noisy_coords.py` | Join clean coordinates onto `social_media_noisy.csv` by `qid` (the noisy set is row-aligned with the clean set) and re-verify against `P625`. |
| `verify_coords.py` | Read-only check that `latitude`/`longitude` in an eval CSV match Wikidata `P625` for the row's `qid`. |

## Evaluation / analysis helpers

| Script | Purpose |
|--------|---------|
| `eval_acc161.py` + `_eval_acc161_worker.py` | Distance-only (within-161 km) evaluation for CSVs that have ground-truth coordinates but no QIDs (e.g. raw GeoCorpora rows). |
| `enrich_within_161km.py` | Back-fill the `within_161km` column on `eval_FINAL_*` CSVs produced before `evaluate.py` started writing it. |
| `restore_geocorpora_eval.py` | Reconstruct `src/geocorpora_eval.csv` from the ground-truth columns of an existing `eval_FINAL_*_geocorpora.csv`, so all modes are evaluated on the same row set. |
| `analyze_failures.py` | Failure-mode analysis across the chapter-4 evaluations. |
| `diagnose_ner_recall.py` | NER-recall diagnostic — attributes pipeline misses to the Flair NER stage. |

## API keys

None of the validation or enrichment scripts here contain hard-coded API keys —
they use the public Wikidata endpoints. The *evaluation pipeline* in `src/`
(not these scripts) reads its keys from environment variables loaded from a
local `src/.env`, which is git-ignored and is never committed.
