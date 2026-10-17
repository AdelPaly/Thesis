# Synthetic dataset generation

Makes social-media-style posts about known real places, so the evaluation
pipeline (`../src/runner.py`) can be tested on text where the correct answer is
known exactly. Each post hides its place behind clues (a person born there, a
club based there, an event held there), an ambiguous place name, or both.

Two folders, used in this order:

| folder | what it does | when you need it |
|---|---|---|
| `places/` | Turns a list of places into clues and name variants from Wikidata | Only when you change the set of places |
| `synthetic/` | Plans each post, has an LLM write it, checks it, writes the dataset CSV | Every time you generate |

The current place set (102 places, 17 per continent) is already built, so to
generate you only need `synthetic/`.

## Generating the dataset

Run from `synthetic/`. The OpenRouter key goes in `synthetic/.env` as
`OPENROUTER_API_KEY=...`.

```bash
cd synthetic

# 1. Style references from real Bluesky posts (once; about $0.10)
python build_exemplars.py

# 2. The plan: which place, clues, condition and style each post gets.
#    Already made (984 posts). Only re-run to start over:
python plan.py --replace

# 3. Write the posts (under $2 for the full plan; resumes if interrupted)
python generate.py --dry-run --limit 3      # look at the prompts first
python generate.py
python generate.py --retry-errors           # re-try any that failed

# 4. Check the posts and write the dataset
python finalise.py
```

Result: `synthetic/output/synthetic_dataset.csv`, in the same columns as the
other datasets (`text, qid, location_name, country, continent, signal_type,
latitude, longitude`). Add it to `../configs/datasets.yaml` to evaluate it.

`finalise.py` drops posts that broke their instructions, such as naming the
place when the condition forbids it or leaving out a clue. Every planned post
and its verdict is in `output/finalise_report.csv`.

### The conditions (`signal_type`)

| condition | the post contains |
|---|---|
| `A1` | one clue entity, no place name |
| `A3` | three clue entities, no place name |
| `B` | an ambiguous place name (shared by several places), no clue |
| `AB` | an ambiguous place name plus one clue |
| `NOISE_A1` | `A1` plus a distractor entity from another continent |
| `NOISE_B` | `B` plus a distractor entity from another continent |
| `CEILING` | the place named plainly: the easy upper bound |

What you would change, and where: model, prompt wording, number of posts per
condition, writing styles and their slot values are all in
`synthetic/config.yaml`.

## Using different places

```bash
cd places
python build_inputs.py my_places.csv --out-dir data/my_run --per-continent 17
```

`my_places.csv` needs only a `wikidata_id` column; name, country, continent
and the other columns are filled from Wikidata. The script runs the clue
stages in order and keeps the places that have a set of clues pointing to them
alone. Expect to lose roughly a third, so start with more than you need. It
ends by printing the five lines to paste into `synthetic/config.yaml`; then
run `plan.py --replace`.

## What is in each folder

```
places/
  build_inputs.py      runs everything below for a new place list
  scripts/             the individual stages (02_clues ... 05b_clue_types),
                       plus checks: verify_places.py / verify_usable.py
                       re-derive the place columns from Wikidata, 06_status.py
                       reports how far each place got
  data/                the current place set, its clues and the query caches
                       (see data/README.md)

synthetic/
  config.yaml          everything configurable
  plan.py              decides every post; no network, no model
  generate.py          sends each planned post to the model
  build_exemplars.py   picks real posts as tone references
  finalise.py          checks the posts, writes the dataset CSV
  sampler/, state/     random.org numbers the plan draws from, and the cursor
  data/                bluesky_en.jsonl (real posts), style_exemplars.jsonl
  output/              plan, raw generations, dataset CSV, report;
                       archive/ holds superseded plans
```
