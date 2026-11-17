# Synthetic dataset generation

Makes social-media-style posts about known real places, so the evaluation
pipeline (`../src/runner.py`) can be tested on text where the correct answer is
known exactly. Each post hides its place behind clues (a club based there, an
event held there), an ambiguous place name, or both.

The dataset is already built: `synthetic/output/synthetic_dataset.csv`
(3,349 posts, 955 places, release `synthetic-v1`). **You do not need anything here to run the
evaluation.** This folder is the record of how the dataset was made, with
every input kept, so each step can be checked or run again.

| step | what it does | output (kept) |
|---|---|---|
| `reform_places.py` | picks the places, their clues, namesakes and condition pools from Wikidata | `out/places.csv` (1,129 places) |
| `place_inputs.py` | fetches the names, clue relations and themes the generator needs | `out/place_inputs.jsonl` |
| `synthetic/plan.py` | decides every post: place, condition, clues, style, theme. No network, no model | `synthetic/output/item_plan.jsonl` (4,370 items: 3,500 planned, 870 reserve) |
| `synthetic/validate_anchors.py` | checks every planned clue and distractor against Wikidata and repairs the plan | `synthetic/output/anchor_checks.jsonl` |
| `synthetic/generate.py` | the writer model writes each post, the judge model (`judge.py`) checks it | `synthetic/output/raw_generations.jsonl`, `swaps.jsonl` |
| `synthetic/finalise.py` | applies the checks, balances the conditions, writes the dataset | `synthetic/output/synthetic_dataset.csv`, `finalise_report.csv`, `release.json` |
| `synthetic/audit.py` | a model that took no part in generation reads every kept post again | `synthetic/output/audit/<model>.jsonl` |
| `synthetic/quality_report.py` | yield, balance against the plan, rule checks, length and the audit as numbers | `synthetic/output/quality_report.csv` |

The writer was Claude Haiku 5.5 and the judge DeepSeek v4.1 Flash, both through
OpenRouter (2026-11-12 to 13). Neither is among the evaluated systems. The
judge decides what enters the dataset, so it is a filter, not an independent
check of it.

The audit (2026-11-17) was run with Qwen3.8-27B. It flags many acceptable
posts, so its flags are candidates for review, not verdicts: off-theme posts
and links the model paraphrased instead of quoting are not counted unless
asked for (`--strict-theme`, `--strict-quotes`). The flagged `A` and `B` posts
were read with an AI assistant and seven were rewritten with
`synthetic/regenerate.py`, which sends an item back to the writer with the
reason its post was removed. The same script wrote 39 `NOISE_EXPLICIT` posts
that had failed their first three tries (`--failed NOISE_EXPLICIT`). No
documented manual sample of the posts was rated.

## Running a step again

`finalise.py` reads only the kept files, so it costs nothing:

```bash
cd synthetic
python finalise.py
```

Writing posts again calls paid models. The OpenRouter key goes in
`synthetic/.env` as `OPENROUTER_API_KEY=...` (template: `synthetic/.env.example`).

```bash
cd synthetic
python plan.py --replace                    # a new plan; only to start over
python validate_anchors.py
python generate.py --dry-run --limit 3      # look at the prompts first
python generate.py                          # resumes if interrupted
python generate.py --retry-errors           # re-try any that failed
python generate.py --pool reserve           # reserve items, if finalise comes up short
python finalise.py
```

`finalise.py` drops posts that broke their instructions, such as naming the
place when the condition forbids it or leaving out a clue. Every planned post
and its verdict is in `output/finalise_report.csv`.

Model, prompt wording, writing styles and their slot values are in
`synthetic/config.yaml`. The comments there keep the history of the choices,
including models tried and dropped.

## The conditions (`signal_type`)

| condition | the post contains |
|---|---|
| `A` | the place's most documented clue entity, no place name |
| `B` | the place's name, whose best-known namesake (more than 161 km away) is in another country, plus one clue that has no link to that namesake (the place needs two such clues: one used, one spare) |
| `EXPLICIT` | the bare place name, no clue, for every place; `name_ambiguous` says whether others share it, and `rival_prominence` bins it like `B` when they do |
| `NOISE_A` | `A` plus one distractor from a place in another country: half get one of its clue entities, half its country's demonym ("French tourists"); `noise_kind` says which |
| `NOISE_B` | `B` plus one distractor in the same way, never from the rival's country |
| `NOISE_EXPLICIT` | `EXPLICIT` plus one distractor in the same way, on the same places and with the same theme, valence, scale, register and length as its `EXPLICIT` post |

`A`, `B`, `NOISE_A` and `NOISE_B` are balanced to 363 posts each. `EXPLICIT`
(955) and `NOISE_EXPLICIT` (942) keep every post that passed.

Clues that are themselves place names (a municipality, a river, a mountain
range) are allowed in the post by name; every other place name is not.

## The place set

`out/places.csv` holds, per place, its type, country, continent, dev/test
split, clues, namesake ("rival") and whether it is in the A and B sets as
`plan` or `reserve`. It was put together from three runs of
`reform_places.py`: Europe from a run with the final rules, North America from
the run of 2026-11-05, and the other continents from a top-up run on
2026-11-10 that put environmental places first:

```bash
PYTHONHASHSEED=0 python reform_places.py --seed 20261028 --exclude excluded_qids.txt \
    --alloc AS=95,AF=95,SA=95,OC=95 --env --file places_topup.csv
```

The partial files and the script that merged them are not kept, so a new run
of `reform_places.py` gives a different set, not this one. To build a new set:

```bash
python reform_places.py --seed 20261028 --prefer old_qids.txt --exclude excluded_qids.txt
python place_inputs.py
cd synthetic && python plan.py --replace
```

Set `SSL_CERT_FILE` first if Python's HTTPS to qlever.dev fails. Both scripts
cache every query in `out/cache` (not committed), so a rerun is fast.
`place_inputs.py` also checks the B rule: a B clue must have no clue relation
(born, located in, headquartered…) with the namesake. B places left with no
such clue are listed in `out/b_need_clues.csv` and are not planned for B.

## Environmental posts

The posts are about the environment: each one makes a plausible (not
necessarily true) claim about its place's surroundings.

- Each place's `env_themes` in `out/place_inputs.jsonl` are topics that fit its
  kind (flooding, falling water levels, fracking, nuclear waste, greenwashing and
  so on), set by `THEMES` and `KIND` in `place_inputs.py`, plus a few wide ones
  any place can carry. Edit the rules there and re-run it, not the file.
- `plan.py` draws one theme per item from its place's `env_themes`.
  `generate.py` adds it to the prompt with `prompt.environment_line`.
- `finalise.py` drops a post as `no_environment` when it has none of the
  `finalise.environment_keywords`. That is a presence check, not a claim check;
  the judge model checks the claim.
- The plan draws from `sampler/geo_random.txt` (style slots, register, theme
  and noise distractor per item). When it runs out, append integers generated
  in the configured range and log the batch in `sampler/geo_random_batches.csv`.

## Style references

`synthetic/data/style_exemplars.jsonl` holds real Bluesky posts used as tone
references in the prompts, picked by `build_exemplars.py`
(`exemplar_labels.jsonl` is its label cache). Building them again needs the
Bluesky dump (~380 MB, not committed):

```bash
python ../scripts/fetch_bluesky_en.py       # from the repository root: scripts/
python ../scripts/filter_adult_spam.py
cd synthetic
python length_dist.py                       # post lengths, sampler/length_dist.json
python build_exemplars.py                   # calls the labelling model, about $0.10
```

## What is in each folder

```
reform_places.py       picks the place set (out/places.csv)
place_inputs.py        names, clue relations, B clues, themes (out/place_inputs.jsonl)
old_qids.txt           places preferred by reform_places.py --prefer
excluded_qids.txt      places kept out by --exclude
out/                   the place set and its inputs for the generator

synthetic/
  config.yaml          everything configurable
  plan.py              decides every post
  validate_anchors.py  checks the planned clues against Wikidata
  generate.py          sends each planned post to the writer model
  judge.py             the second model's check on each post
  finalise.py          checks the posts, writes the dataset CSV
  audit.py             a second judge model reads the finished dataset
  regenerate.py        rewrites posts removed by hand or failed in one condition
  quality_report.py    the quality criteria as numbers
  build_exemplars.py   picks real posts as tone references
  length_dist.py       word-count distribution of real posts
  sampler/, state/     the pre-generated integers the plan draws from (with a
                       per-batch source log), and the cursor
  data/                style_exemplars.jsonl and its label cache
  output/              plan, anchor checks, raw generations, dataset CSV, reports,
                       release.json, the audit log
```
