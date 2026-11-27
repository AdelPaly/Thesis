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

That one command creates `.venv` with the three packages scoring needs, then
re-computes the result tables from the predictions in the repository and
compares them with the tables in `results/summary/`. It takes a few minutes
and needs no key and no model. When it prints `PASS`, the reported numbers
reproduce on your machine.

To run the methods themselves:

```bash
python install.py --run          # the pinned packages (several GB, mostly PyTorch)
```

Running either command again is safe. The packages were pinned on Windows 11
with the CPU build of PyTorch; other platforms are untested.

## Run

Use the environment's Python: `.venv/Scripts/python` on Windows,
`.venv/bin/python` on Linux and macOS (written as `python` below).

Scores from the predictions in the repository (after `python install.py`):

```bash
python scripts/score.py --split test --by condition
python scripts/check_rescore.py                             # do the tables in results/summary/ reproduce?
```

Running a method (after `python install.py --run`):

```bash
python src/runner.py --list                                              # configurations and datasets
python src/runner.py --config majority_class --dataset synthetic --limit 20 --out-dir my_run   # quick try
```

The full evaluation:

```bash
python scripts/run_all.py                                   # every configuration, one at a time, resumable
python scripts/build_containment.py                         # Wikidata lookup for the "exact or inside" score
python scripts/score.py --split dev                         # choose settings here
python scripts/score.py --split test --by condition         # report once
python scripts/analyse.py all --split test --out results/summary
```

The runner only writes predictions (`results/predictions/`, one JSON line per
post). Scoring and analysis read those files and need no model, key or
network, so a metric can be changed without running anything again.

The predictions behind the reported results are committed, all 11
configurations over all 3,349 posts, with the summary tables
(`results/summary/`) and the Wikidata containment lookup
(`results/containment.csv`). The runner resumes from an existing prediction
file and skips posts already in it, so pass `--out-dir` or `--overwrite` to
compute something again. A fresh run downloads models, fills a local Wikidata
cache and takes hours; Wikidata changes, so its answers can differ slightly
from the committed ones.

## Run on your own data

### Posts without a gold location

A CSV with one text column is enough:

```bash
python scripts/predict.py --config refined --input posts.csv --out-dir my_predictions
python scripts/predict.py --config gemma4 --input posts.csv --text-column body --out-dir my_predictions
```

It writes `<config>__<input>.predict.csv`, one row per post, with the
predicted item (`predicted_label`, `predicted_qid`, `lat`, `lon`, `country`,
`continent`) and a `status` saying what happened: for the linkers, search and
baselines `no_mention`, `no_link`, `no_place` or `predicted` (with `fallback`
when the MISC fallback was used); for the LLMs the resolution status
(`resolved`, `declined`, `unreadable`, `not_found`, `ambiguous`) and its
`level`; for the grounded LLM `predicted`, `abstained` or `violation`.
A `.predict.jsonl` next to it holds the full record of each post. Only the
text column is read, nothing is scored, and no split is needed.

Three configurations need gold data and stop with a message:
`majority_class` (it predicts the most frequent gold place of a labelled dev
split), and `dense_retrieval` and `grounded_gemma` (their candidate pool is
built from the gold places in `configs/datasets.yaml` and other
configurations' answers).

The named-entity recogniser (Flair `ner-large`, CoNLL-03) is English-only,
and so are the reranker's descriptions and Wikidata's English labels. All
configurations except the LLMs assume English posts.

### A new labelled dataset

To score methods on another labelled dataset:

1. A CSV with the columns `text`, `qid` (the gold Wikidata item),
   `location_name`, `country`, `continent`, `latitude`, `longitude`,
   `signal_type` (the condition; any labels, or one value for all) and
   `split` (`dev` or `test`, assigned by place: every post about one place on
   the same side). The optional columns in `BIN_COLUMNS` (`scripts/score.py`)
   are specific to the synthetic set and can be left out.
2. An entry in `configs/datasets.yaml`.
3. `python scripts/make_split.py --force`, which rewrites `splits/` for all
   registered datasets (the synthetic rows come out the same).
4. `python src/runner.py --config <name> --dataset <new>`, then
   `python scripts/build_containment.py` for the new (prediction, gold) pairs,
   then `scripts/score.py --dataset <new>`.

Specific to the synthetic dataset: the condition names and order of
`scripts/analyse.py` (`EXPLICIT`, `A`, `B` and their `NOISE_` variants), the
`BIN_COLUMNS`, the continent codes the scorer expects for gold continents
(`AF`, `EU`, ...; full names are mapped too), and the per-place bootstrap of
`scripts/stats.py`, which assumes several posts per place. Registering a
dataset also adds its gold places to the dense pool of every dataset
(`src/dense_index.py`), which changes `dense_retrieval` and `grounded_gemma` on
the synthetic set too.

## Configurations

One YAML file in `configs/` per method. Adding a file adds a method.

| Tier | Configuration | Method | Needs beyond `install.py` |
|---|---|---|---|
| 0 | `majority_class` | most frequent place in dev | nothing |
| 0 | `population_wikidata` | Flair NER, most populous candidate of that name | nothing |
| 1 | `geoparser_irchel` | Irchel Geoparser as published | second environment, see below |
| 2 | `retrieve_rerank` | Wikidata search and rerank | nothing |
| 2 | `refined` | ReFinED entity linking | nothing (8.6 GB download on first use) |
| 2 | `blink` | BLINK entity linking | BLINK checkout and models, see below |
| 3 | `dense_retrieval` | dense retrieval over place texts | nothing |
| 4 | `gemma4` | LLM on the whole text (Gemma 4 12B) | `VSE_API_KEY` |
| 4 | `qwen3_vl` | the same with a second model (Qwen3-VL 235B-A22B: 235B in total, 22B active) | `VSE_API_KEY` |
| 4 | `qwen3_vl_32b` | the same family, smaller (Qwen3-VL 32B, dense); revision 3, not yet run | `VSE_API_KEY` |
| 5 | `grounded_gemma` | LLM choosing among dense candidates | `VSE_API_KEY` |
| 5 | `grounded_gemma_linking` | LLM choosing among linked candidates | `VSE_API_KEY` |

The tiers: 0 baselines, 1 a published geoparser, 2 mention detection and
entity linking, 3 dense retrieval with no mention detection, 4 an LLM alone,
5 an LLM choosing among retrieved candidates.

Models are downloaded on first use (Flair `ner-large` 6.3 GB,
`intfloat/multilingual-e5-base` 2.2 GB, ReFinED 8.6 GB). Run a subset with
`python scripts/run_all.py --configs refined gemma4`.

`dense_retrieval` and `grounded_gemma` retrieve from a pool made of the gold
places and the final answers of the other configurations, so they run last and
their results depend on which other prediction files are present.

### Keys

Keys go in `src/.env` (git-ignored; `src/.env.example` lists them). Only the
LLM configurations need one: `VSE_API_KEY`, for the university's model gateway
(litellm.vse.cz), which served Gemma 4 12B and Qwen3-VL 235B for the results.
Both models have open weights; to run them elsewhere, set `llm.provider` in the
configuration to `openrouter` or `google` and give that provider's key
(`OPENROUTER_API_KEY`, `GOOGLE_API_KEY`). That changes the configuration's
hash, so its results are a new run, not a continuation of the committed one.
Without a key the four LLM configurations cannot be run; their committed
predictions can still be scored.
Wikidata is public; set `WDQS_CONTACT` to your email or URL so its operators
can reach you.

### Irchel Geoparser (optional)

It needs a newer `transformers` than Flair allows, so it has its own
environment. This creates it and downloads its GeoNames gazetteer (about
10 GB, once):

```bash
python install.py --geoparser
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

## Words used here

| Word | Meaning |
|---|---|
| place, gold place | the correct answer of a post: one Wikidata item. `place_qid` on a prediction record, `qid` in the dataset |
| condition | how the place appears in the post (table above). The dataset column is `signal_type`; prediction records call it `condition` |
| clue | an entity tied to the place: a club based there, a person born there |
| noise, distractor | one misleading name from a place in another country |
| mention | a span of the post that a recogniser marked as a name (`ner_spans`) |
| linked entity | the Wikidata item a mention was resolved to (`linked_entities`) |
| candidate place | a place a method considered for its answer (`candidate_places`) |
| prediction | the one item a method returned (`predicted_qid`); "no result" when it returned none |
| paradigm | the kind of method, stored on every record: `baseline`; `geoparser`; `ner_dependent` (mention detection, then entity linking); `dense_retrieval` (the whole post embedded, no mention detection); `ner_free` (an LLM reading the whole post, nothing else); `grounded_llm` (an LLM choosing among candidates another stage retrieved) |
| `item_id` | on prediction records and in `splits/`, a hash of the post text. The dataset has a column of the same name that holds the id of the planned item; join the two on `text` |
| `config_hash`, `code_rev` | on a prediction record: a hash of the configuration that produced it and the git revision of the code |

The scores (`scripts/score.py`):

| Column | Meaning |
|---|---|
| `exact` | the prediction is the gold place |
| `ex+in` | exact, or the prediction lies inside the gold place on Wikidata (a station in the right town) |
| `coarser` | share of posts where the prediction is a larger unit containing the gold place (a prefecture for its capital). Not counted as correct |
| `country` | exact, or at least in the gold place's country |
| `cont` | that, or at least on its continent |
| `no-res` | share of posts with no prediction |
| `err` | share of posts where the run itself failed |

## Layout

```
install.py           setup
requirements.lock    exact package versions (requirements.txt: the short list)
configs/             one file per method, plus datasets, shared values, planned comparisons
src/                 the pipeline and the runner
scripts/             run_all, score, analyse, statistics, the dev/test split
splits/              dev/test assignment per place and per post
results/             the predictions, the summary tables and the containment lookup
generation/          how the dataset was made: place selection, plan, writer, judge
```

More on the scripts and the large external files:
[`scripts/README.md`](scripts/README.md).
