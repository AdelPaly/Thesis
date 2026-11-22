# Synthetic dataset generation

Makes social-media-style posts about known real places, so the evaluation
pipeline (`../src/runner.py`) can be tested on text where the correct answer is
known exactly. Each post hides its place behind clues (a club based there, an
event held there), an ambiguous place name, or both.

The dataset is already built: `synthetic/output/synthetic_dataset.csv`
(3,349 posts, 955 places, release `synthetic-v1`). **You do not need anything here to run the
evaluation.** This folder is the record of how the dataset was made, with
every input kept, so each step can be checked or run again.

| step | what it does | output |
|---|---|---|
| `reform_places.py` | picks the places, their clues, namesakes and condition pools from Wikidata | `out/places.csv` (1,129 places) |
| `place_inputs.py` | fetches the names, clue relations and themes the generator needs | `out/place_inputs.jsonl` |
| `synthetic/plan.py` | decides every post: place, condition, clues, style, theme. No network, no model | `synthetic/output/item_plan.jsonl` (4,370 items: 3,500 planned, 870 reserve) |
| `synthetic/validate_anchors.py` | checks every planned clue and distractor against Wikidata and repairs the plan | `synthetic/output/anchor_checks.jsonl` |
| `synthetic/generate.py` | the writer model writes each post, the judge model (`judge.py`) checks it | `synthetic/output/raw_generations.jsonl`, `swaps.jsonl` |
| `synthetic/finalise.py` | applies the checks, balances the conditions, writes the dataset | `synthetic/output/synthetic_dataset.csv`, `finalise_report.csv`, `release.json` |
| `synthetic/clue_types.py` | looks up what kind of thing each clue is (Wikidata "instance of") for the prompt | `synthetic/output/clue_types.json` |
| `synthetic/regenerate.py` | writes a post again: removed by hand, failed, or copying a stock scene | appends to `raw_generations.jsonl` |
| `synthetic/audit.py` | a model that took no part in generation reads every kept post again | `synthetic/output/audit/<model>.jsonl` (local), `<model>_summary.csv` (kept) |
| `synthetic/check_burials.py` | finds posts that speak of a person's tomb where, by Wikidata, they are not buried | `synthetic/output/audit/burial_check.csv` (local) |
| `synthetic/quality_report.py` | yield, balance against the plan, rule checks, length and the audit as numbers | `synthetic/output/quality_report.csv` |

The writer was Claude Haiku 5.5 and the judge DeepSeek v4.1 Flash, both through
OpenRouter (2026-11-12 to 13). Neither is among the evaluated systems. The
judge decides what enters the dataset, so it is a filter, not an independent
check of it.

The audit (2026-11-17) was run with Qwen3.8-27B. It is a filtering tool and
not an accurate one: its flags are candidates for review, not verdicts. On the
sample of 100 rated posts it flagged seven, six of which were kept, and it
missed all four rejected posts. Its per-post log stays local; the counts per
condition and reason are in
`synthetic/output/audit/vse_Qwen3.8_27B_summary.csv`. Told what each clue is,
it did find 21 posts that used a name as the wrong kind of thing, and 19 were
rewritten.
Off-theme posts and links the model paraphrased instead of quoting are not
counted unless asked for (`--strict-theme`, `--strict-quotes`). Each flag is
marked as being about the clue (worth reading) or about the scene (the
distractor, an invented detail of the setting), which the post rules allow.
`synthetic/check_burials.py` covers one fault no judge sees: a post that
speaks of a person's tomb in a place where, by Wikidata, they are not buried
or did not die. It found 97 such posts, and all were rewritten; where no post
without a tomb passed, the place's spare clue was used instead.

`synthetic/output/audit/manual_sample_100.csv` is a sample of 100 posts drawn
at random from the whole dataset (seed 20261118; drawn once by hand, no script
in the repository does it), each with one answer, keep or not, and a note where
the answer is no. 96 were kept and 4 rejected: two for a
tomb in the wrong city, one for a beach written as a building and one for a tie
that is only a street sign. The first three faults were then searched for in
the whole dataset as described here; the sheet shows the posts as they were
rated.

The flagged `A` and `B` posts
were read with an AI assistant and seven were rewritten with
`synthetic/regenerate.py`, which sends an item back to the writer with the
reason its post was removed. The same script wrote 39 `NOISE_EXPLICIT` posts
that had failed their first three tries (`--failed NOISE_EXPLICIT`).

A third of the first generation shared two scenes that the prompt had given as
worked examples: vehicles idling outside an office, and a crowd leaving cups
and bottles behind (34.1 % of posts, over half of `NOISE_A` and `NOISE_B`). The
writer had also treated stations, parks and a boat as companies, because the
prompt gave only their names. On 2026-11-17 the examples were taken out of the
prompt, `synthetic/clue_types.py` fetched what each clue is from Wikidata
("railway station") for the writer and the judge, and
`regenerate.py --stock-scenes` rewrote the posts that matched either scene
(`stock_scene()` in that script; two rounds). 1,095 posts were replaced; a
post that failed its tries kept its old text. 3.3 % of posts still match a
stock scene, most of them on a topic that is about traffic fumes or litter.
The dataset therefore holds posts from three versions of the prompt: the
original one (2,173 posts), the one without the worked examples (1,060) and
the one that also forbids writing about a tomb (116). Every row of
`raw_generations.jsonl` carries its `config_hash`, `release.json` counts all
three, and `quality_report.csv` (section `repetition`) gives the shares.

Nothing linked to terrorism, extremism, Nazi crimes, killers or massacres may
appear in a post, as a clue or as a distractor. `synthetic/banned_entities.txt`
lists the entities found in the plan (by name, Wikidata type, party, membership
and conviction); `finalise.py` drops any post that uses one. Ten kept posts
did: their distractor, or for one place its clue, was swapped in the plan for
another entity of the same place (`noise_swaps`, `banned_clue_swaps` on the
plan item) and the posts were written again.

## Running a step again

What can be run again from the repository alone: everything up to and
including `finalise.py` and `quality_report.py`. The working files of the
repairs are committed too (the per-post audit log, the burial check, the hand
checks and the list of posts removed by hand, which is empty once its posts
are rewritten); only the raw model replies (`raw_responses.jsonl`, 33 MB) stay
local. [`MANUAL_EDITS.md`](MANUAL_EDITS.md) lists everything a person decided,
and `synthetic/manual_edits.py` rebuilds the list of posts removed by hand
from the generation log. The tests are in `synthetic/tests/`
(`python -m pytest generation/synthetic/tests`).

Rows written from 2026-11-22 on carry the `code_rev` of the generation code.
The rows of `synthetic-v1` are older and are left unchanged;
`synthetic/infer_code_rev.py` writes for them the commits each row falls
between (`synthetic/output/raw_generations_code_rev.csv`).

`finalise.py` reads only the kept files, so it costs nothing:

```bash
cd synthetic
python finalise.py
```

Writing posts again calls paid models. The OpenRouter key goes in
`synthetic/.env` as `OPENROUTER_API_KEY=...` (template: `synthetic/.env.example`).
The audit with a `vse:` model reads `VSE_API_KEY` from the same file.

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

## Words used here

| word | meaning |
|---|---|
| place | the gold answer of a post: one Wikidata item with coordinates |
| condition | how the place appears in the post; the column is `signal_type` |
| clue | an entity tied to the place (a club based there, a person born there). The code also says **anchor** for a clue or distractor that is checked against Wikidata, and **hook** for the kind of tie |
| noise, distractor, misleading signal | the same thing: one name from a place in another country. The **donor** is that place |
| rival, namesake | the best-known other place with the same name |
| plan, reserve | items planned for the dataset, and spare items used when a planned one fails |
| writer, judge | the model that writes a post and the model that accepts or rejects it during generation |
| audit | a later reading of the finished posts by a third model |

## The columns of `synthetic_dataset.csv`

| column | meaning |
|---|---|
| `text` | the post |
| `qid`, `location_name`, `country`, `continent`, `latitude`, `longitude` | the gold place: Wikidata item, name, country, continent code, coordinates |
| `signal_type` | the condition (table above). The evaluation calls it `condition` |
| `signal` | what kind of evidence the post carries, a coarser grouping: `explicit` (the name), `A` (a clue), `A+B` (an ambiguous name and a clue). Not the condition |
| `surface_form` | the place name as written in the post; empty when the post has none |
| `clue_qids`, `clue_labels`, `clue_forms` | the clue: Wikidata item, its label, and how the post writes it |
| `clue_name_shared` | 1 if the clue's name is also the name of something elsewhere (allowed in `B` and `NOISE_B` only) |
| `clue_link_elsewhere` | other places the clue is also tied to |
| `noise_entity`, `noise_kind` | the distractor as written, and whether it is an `entity` or a `demonym` ("French") |
| `noise_donor_qid`, `noise_donor_place`, `noise_donor_country_qid` | the place the distractor comes from, and its country |
| `register`, `scale`, `valence`, `angle`, `role` | style of the post: kind of post, how wide the claim is, positive or negative, the idea given to the writer, the role of a person clue |
| `topic` | the environmental theme of the post (called `env_theme` in the plan and on prediction records) |
| `target_len`, `len` | planned and actual length, in words |
| `place_sitelinks`, `rival_sitelinks` | number of Wikipedia language versions of the place and of its namesake |
| `place_prominence`, `clue_prominence` | those counts in three bins |
| `rival_qid`, `name_ambiguous`, `place_is_most_famous`, `rival_prominence` | the namesake, whether the name is shared, whether the place has more sitelinks than its namesake, and by how much |
| `tries` | writer attempts until the judge accepted |
| `clue_swapped`, `swapped_from` | `clue_swapped`: clue replacements made after planning, as old>new (empty in this release). `swapped_from`: for a post on a reserve place, the planned place it stands in for |
| `split` | `dev` or `test`, by place |
| `pool` | whether the item came from the `plan` or the `reserve` |
| `item_id` | the id of the planned item. **Not** the `item_id` on prediction records, which is a hash of the text (`src/keys.py`); join predictions to the dataset on `text` |
| `model` | the writer model |

`out/places.csv` has short headers: `i` is the place's Wikidata item, `c` its
country, `sub` its subtype, `sl` its sitelinks, `rival_*` its namesake, and
`A`, `B`, `EXPLICIT` say whether the place is in that condition's `plan` or
`reserve`. The remaining short columns (`pbin`, `cbin`, `rbin`, `sbin`,
`n_fam`, `n_dom`, `hooks`, `old`, `rank`, `n_names`) are working columns of
`reform_places.py`, explained where that script writes them.

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

Both scripts (and `clue_types.py`, `check_burials.py`) cache every query in `out/cache` (not committed), so a rerun is fast.
`place_inputs.py` also checks the B rule: a B clue must have no clue relation
(born, located in, headquartered…) with the namesake. B places left with no
such clue are listed in `out/b_need_clues.csv` and are not planned for B. For
this place set the file is empty (header only).

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

`synthetic/data/style_exemplars.jsonl` holds 110 real Bluesky posts used as
tone references in the prompts, picked by `build_exemplars.py`
(`exemplar_labels.jsonl` is its label cache). They come from the public
Hugging Face dataset `alpindale/two-million-bluesky-posts`; each keeps the
address of the original post (`at://did:plc:…`) as its source. They are shown
to the writer model as examples of tone and appear in no post of the dataset. Building them again needs the
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
  banned_entities.txt  entities no post may use; finalise.py drops such posts
  audit.py             a second judge model reads the finished dataset
  check_burials.py     finds posts that place a tomb where the person is not buried
  regenerate.py        rewrites posts removed by hand, failed, or copying a stock scene
  manual_edits.py      lists the posts removed by hand, from the generation log
  infer_code_rev.py    the commits each old generation row falls between
  clue_types.py        what each clue is, from Wikidata, for the prompt
  quality_report.py    the quality criteria as numbers
  build_exemplars.py   picks real posts as tone references
  length_dist.py       word-count distribution of real posts
  sampler/, state/     the pre-generated integers the plan draws from (with a
                       per-batch source log), and the cursor
  data/                style_exemplars.jsonl and its label cache
  output/              plan (item_plan.jsonl) and the items the plan rejected
                       (plan_rejections.jsonl), anchor checks, raw generations,
                       dataset CSV, reports, release.json; in audit/ the audit
                       summary and the sample of 100 rated posts
  .env.example         template for the keys (copy to .env)
```
