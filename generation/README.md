# Synthetic dataset generation

Makes social-media-style posts about known real places, so the evaluation
pipeline (`../src/runner.py`) can be tested on text where the correct answer is
known exactly. Each post hides its place behind clues (a person born there, a
club based there, an event held there), an ambiguous place name, or both.

Two steps, used in this order:

| step | what it does | when you need it |
|---|---|---|
| `reform_places.py`, then `place_inputs.py` | Pick the place set, its clues, namesakes and condition pools from Wikidata (`out/places.csv`), then fetch the names, clue relations and themes the generator needs (`out/place_inputs.jsonl`) | Only when you change the set of places |
| `synthetic/` | Plans each post, has an LLM write it, checks it, writes the dataset CSV | Every time you generate |

The current place set (939 places, `out/places.csv`, run of 2026-10-29) is
already built, so to generate you only need `synthetic/`.

## Generating the dataset

Run from `synthetic/`. The OpenRouter key goes in `synthetic/.env` as
`OPENROUTER_API_KEY=...`.

```bash
cd synthetic

# 1. Style references from real Bluesky posts (once; about $0.10)
python build_exemplars.py

# 2. The plan: one item per (place, condition) that out/places.csv asks
#    for, each with its clues, style slots and theme. Only re-run to start over:
python plan.py --replace

# 3. Write the posts (resumes if interrupted)
python generate.py --dry-run --limit 3      # look at the prompts first
python generate.py
python generate.py --retry-errors           # re-try any that failed
python generate.py --pool reserve           # reserve items, if finalise comes up short

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
| `A1` | the place's most documented clue entity, no place name |
| `A3` | its first three clue entities (at least two kinds), no place name |
| `B` | the place's name, shared with a namesake more than 161 km away, plus one clue that has no link to that namesake |
| `EXPLICIT` | the bare place name, no clue, for every place; `name_ambiguous` says whether others share it, and `rival_prominence` bins it like `B` when they do |
| `NOISE_A1` | `A1` plus a distractor entity from a place in another country |
| `NOISE_B` | `B` plus a distractor entity from a place in another country |

Clues that are themselves place names (a municipality, a river, a mountain
range) are allowed in the post by name; every other place name is not.

What you would change, and where: model, prompt wording, writing styles and
their slot values are in `synthetic/config.yaml`. The place set, its type and
continent balance and the posts per condition (`FINAL`) are set at the top of
`reform_places.py`.

## Changing the places

```bash
python reform_places.py --seed 20261028 --prefer old_qids.txt --exclude excluded_qids.txt
python place_inputs.py
cd synthetic && python plan.py --replace
```

Set `SSL_CERT_FILE` first if Python's HTTPS to qlever.dev fails. Both scripts
cache every query in `out/cache`, so a rerun is fast. `reform_places.py`
writes `out/places.csv`: per place its type, country, continent, dev/test
split, clues, namesake ("rival") and whether it is in the A and B sets as
`plan` or `reserve`. `place_inputs.py` adds what the generator needs: the
place's English names, its country's names and demonyms, its regions, which
property links each clue, and the environmental themes. It also checks the B
rule: a B clue must have no clue relation (born, located in, headquartered…)
with the namesake. B places left with no such clue are listed in
`out/b_need_clues.csv` and are not planned for B until they get one.

### Environmental posts

The posts are about the environment: each one makes a plausible (not
necessarily true) claim about its place's surroundings.

- Each place's `env_themes` in `out/place_inputs.jsonl` are topics that fit its
  kind (flooding, falling water levels, fracking, nuclear waste, greenwashing and
  so on), set by `THEMES` and `KIND` in `place_inputs.py`, plus a few wide ones
  any place can carry. Coral bleaching is cut outside the tropics and fracking
  where there is none; edit the rules there and re-run it, not the file.
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
- The plan draws from `sampler/geo_random.txt` (style slots, register, theme
  and noise distractor per item). When it runs out, append integers generated
  in the configured range and log the batch in `sampler/geo_random_batches.csv`.

## What is in each folder

```
reform_places.py       picks the place set (out/places.csv, out/report.md)
place_inputs.py        names, clue relations, B clues, themes (out/place_inputs.jsonl)
out/                   the place set, its run log and the query cache

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
