# Geolocating short texts: a controlled comparison

Code and data for a thesis that compares methods for finding the place a short
social-media post is about. Each method reads a post and returns a Wikidata
item (QID). The methods range from baselines through a geoparser, entity
linkers and retrieval to LLMs with and without retrieved candidates.

The evaluation data is a synthetic dataset of 3,349 posts about 955 places.
Every post belongs to one condition that fixes how the place appears in it
(named, hidden behind a clue, ambiguous, with or without a misleading signal),
so results can be read per condition.

## Install

You need Python 3.13 and git.

```bash
git clone https://github.com/AdelPaly/Thesis.git
cd Thesis
python install.py
```

`install.py` creates `.venv`, installs the pinned packages from
`requirements.lock` (several GB, mostly PyTorch), writes empty `.env` files
from the templates and checks that the runner starts. Running it again is safe.

## Run

Use the environment's Python: `.venv/Scripts/python` on Windows,
`.venv/bin/python` on Linux and macOS (written as `python` below).

```bash
python src/runner.py --list                                              # configurations and datasets
python src/runner.py --config majority_class --dataset synthetic --limit 20   # quick try
python scripts/score.py                                                  # metrics for what has been run
```

The full evaluation:

```bash
python scripts/run_all.py                                   # every configuration, one at a time, resumable
python scripts/score.py --split dev                         # choose settings here
python scripts/score.py --split test --by condition         # report once
python scripts/analyse.py all --split test --out results/summary
```

The runner only writes predictions (`results/predictions/`, one JSON line per
post). Scoring and analysis read those files and need no model, key or
network, so a metric can be changed without running anything again. No
predictions are committed: the first run downloads models and fills a local
Wikidata cache, and takes a while.

## Configurations

One YAML file in `configs/` per method. Adding a file adds a method.

| Tier | Configuration | Method | Needs beyond `install.py` |
|---|---|---|---|
| 0 | `majority_class` | most frequent place in dev | nothing |
| 0 | `population_wikidata` | Flair NER, most populous candidate | nothing |
| 1 | `geoparser_irchel` | Irchel Geoparser as published | second environment, see below |
| 2 | `retrieve_rerank` | Wikidata search and rerank | nothing |
| 2 | `refined` | ReFinED entity linking | nothing (8.6 GB download on first use) |
| 2 | `blink` | BLINK entity linking | BLINK checkout and models, see below |
| 3 | `dense_retrieval` | dense retrieval over place texts | nothing |
| 4 | `gemma4` | LLM on the whole text | `GOOGLE_API_KEY` |
| 5 | `grounded_gemma` | LLM choosing among dense candidates | `GOOGLE_API_KEY` |
| 5 | `grounded_gemma_linking` | LLM choosing among linked candidates | `GOOGLE_API_KEY` |

Models are downloaded on first use (Flair `ner-large` 6.3 GB,
`intfloat/multilingual-e5-base` 2.2 GB, ReFinED 8.6 GB). Run a subset with
`python scripts/run_all.py --configs refined gemma4`.

### Keys

Keys go in `src/.env` (git-ignored; `src/.env.example` lists them). Only the
Gemma configurations need one: a Google AI Studio key as `GOOGLE_API_KEY`.
Wikidata is public; set `WDQS_CONTACT` to your email or URL so its operators
can reach you.

### Irchel Geoparser (optional)

It needs a newer `transformers` than Flair allows, so it has its own
environment:

```bash
python install.py --geoparser
.venv-geoparser/Scripts/python -m geoparser install geonames     # about 10 GB, once
```

### BLINK (optional)

BLINK is not on PyPI and is not installed by `install.py`. The results were
produced with a checkout of `https://github.com/facebookresearch/BLINK` at
commit `5fe254d` in `.venv/src/blink`, put on the import path with a `.pth`
file in `site-packages`, and its models (about 30 GB) fetched with its
`download_blink_models.sh` into `.venv/src/blink/models`. Set `BLINK_MODELS`
if the models are elsewhere.

## Dataset

`generation/synthetic/output/synthetic_dataset.csv`, registered as `synthetic`
in `configs/datasets.yaml`.

| Condition (`signal_type`) | Posts | The post contains |
|---|---|---|
| `EXPLICIT` | 955 | the place name only |
| `A` | 363 | no place name, one clue entity tied to the place |
| `B` | 363 | a place name shared with a place in another country, and one clue |
| `NOISE_EXPLICIT` | 942 | as `EXPLICIT`, plus one misleading signal from another country |
| `NOISE_A` | 363 | as `A`, plus one misleading signal |
| `NOISE_B` | 363 | as `B`, plus one misleading signal |

735 posts are dev and 2,614 test. The split is by place (`splits/`), so no
place is on both sides. The posts were written by an LLM from a plan and
checked by a second model; they are plausible, not factual. This is release
`synthetic-v1`: `generation/synthetic/output/release.json` holds its date, file
hashes, models and code revision. How the dataset was built, and how to build
it again, is in [`generation/README.md`](generation/README.md).

## Layout

```
install.py           setup
requirements.lock    exact package versions (requirements.txt: the short list)
configs/             one file per method, plus datasets, shared values, planned comparisons
src/                 the pipeline and the runner
scripts/             run_all, score, analyse, statistics, the dev/test split
splits/              dev/test assignment per place and per post
results/             predictions, logs and summary tables (written when you run)
generation/          how the dataset was made: place selection, plan, writer, judge
```

More on the scripts and the large external files:
[`scripts/README.md`](scripts/README.md).
