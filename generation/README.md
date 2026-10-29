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
| `B` | an ambiguous place name (shared by several places) plus one `A1`-style clue that picks the right namesake |
| `EXPLICIT` | the bare place name, no clue, for every place; `name_ambiguous` says whether others share it, and `rival_prominence` bins it like `B` when they do |
| `NOISE_A1` | `A1` plus a distractor entity from another continent |
| `NOISE_B` | `B` plus a distractor entity from another continent |

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

### Environmental posts

The posts are about the environment: each one makes a plausible (not
necessarily true) claim about its place's surroundings.

- `places/data/env_places.csv` is the place set (`synthetic/config.yaml`
  `paths.places_csv`), made by `places/scripts/build_env_places.py`: towns,
  islands, lakes, mountains, parks and heritage sites, with an `env_themes`
  column of topics that fit each type (flooding, falling water levels, fracking,
  fast fashion, greenwashing, permafrost thaw and so on). The script cuts a
  theme where it is wrong for the place (coral bleaching outside the tropics,
  glacier retreat on a mountain with no glacier, fracking where there is none);
  edit its rules and re-run it, not the CSV. The clue, alias and clue-type files
  cover a few more places than this one; only the places in it are planned.
- Every post is between 7 and 30 words (`prompt.length_bands`, enforced by
  `finalise.min_words` and `max_words`; a post outside is dropped as `length`).
  The slot values (what is bothering them, the event, what the picture is of)
  are worded to sit beside an environmental claim and to read naturally.
- `plan.py` draws one theme per item from its place's `env_themes` (`env_theme`
  in the plan). `generate.py` adds it to the prompt with `prompt.environment_line`.
- `finalise.py` drops a post as `no_environment` when it has none of the
  `finalise.environment_keywords`. That is a presence check, not a claim check,
  so read a sample of the kept posts. The theme is written to the dataset as
  `env_theme`.
- To use real environmental posts as tone references: `python csv_to_dump.py
  ../climate_en.csv data/environment_en.jsonl`, then `python build_exemplars.py
  --topic environment --dump data/environment_en.jsonl` (calls the labelling
  model), then point `paths.style_exemplars` at
  `data/style_exemplars_environment.jsonl`. Until then the generic exemplars
  are used.
- A full plan on the 362-place set uses about 16,600 integers of
  `sampler/geo_random.txt`. After the 2026-10-24 plan the cursor is at 73,376
  of 82,000, so a new plan needs more integers appended first (generate them in
  the configured range and log the batch in `sampler/geo_random_batches.csv`).

## What is in each folder

```
places/
  build_inputs.py      runs everything below for a new place list
  scripts/             the individual stages (02_clues ... 05b_clue_types),
                       plus checks: verify_places.py / verify_usable.py
                       re-derive the place columns from Wikidata
  data/                the current place set, its clues and the query caches
                       (see data/README.md)

synthetic/
  config.yaml          everything configurable
  plan.py              decides every post; no network, no model
  generate.py          sends each planned post to the model
  build_exemplars.py   picks real posts as tone references
  finalise.py          checks the posts, writes the dataset CSV
  sampler/, state/     the pre-generated integers the plan draws from (with a
                       per-batch source log), and the cursor
  data/                bluesky_en.jsonl (real posts), style_exemplars.jsonl
  output/              plan, raw generations, dataset CSV, report;
                       archive/ holds superseded plans
```
