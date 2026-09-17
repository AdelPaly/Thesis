# Methodology notes

Notes that belong with the reported numbers rather than with the code. For
dataset provenance see [`DATASETS.md`](DATASETS.md); for how to run things see
[`scripts/README.md`](scripts/README.md).

## Evaluation design

Inference and scoring are separate programs. `src/runner.py` runs one
configuration over one dataset and writes per-item records to
`results/predictions/<config>__<dataset>.jsonl`; it computes no metrics.
`scripts/score.py` reads those records and computes every metric; it imports
nothing from `src/`, loads no model and makes no network call. A change to a
metric definition therefore never re-runs a model, and every reported number
is recomputable from the committed prediction files alone.

Each record carries the intermediate stages, not just the winning prediction:
the NER spans with character offsets, the linked entities with the linker's
confidence and its full ranked candidate set, the scored candidate places, and
(for routed configurations) the branch taken and the input features the gate
read. Failure attribution is therefore *attributed* from what the system
recorded rather than *inferred* from the shape of its output.

A configuration is a file in `configs/`. No model identity, threshold or
parameter is hardcoded anywhere in the pipeline, and `config_hash` — a sha256
over the canonicalised configuration — is stored on every record, so a record
can always be traced to the exact configuration that produced it.

## Tiers

The evaluation is organised by **what kind of signal a method can consume**,
not by model family. Tiers 0 and 1 need a toponym in the text and score zero by
construction on entity-only conditions; tiers 2 upward do not.

| Tier | Method | Configuration |
|---|---|---|
| 0 | population baseline, majority | `baseline_population`, `baseline_majority` |
| 1 | geoparser | `mordecai3` (cited from literature, not re-run — see below) |
| 2a | retrieve + rerank | `two_stage` |
| 2b | entity linking | `refined` |
| 3 | dense retrieval | `dense_retrieval` |
| 4 | LLM-only | `gemma`, `groq_oss` (model-sensitivity contrast) |
| 5 | grounded LLM | `grounded_gemma`, `grounded_gemma_linking` |
| combo | routed | `routed_gemma` (reported as a negative result) |
| — | oracle | analysis only, not a configuration |

A configuration declares its tier in its config file. `tier` is presentation
metadata: it orders report rows and is **excluded from `config_hash`**, so it
cannot change a prediction and cannot force a re-run.

### Tier 3 — dense retrieval, and why it exists

Without it, "NER-free" and "LLM" are the same category in the results, because
every system that lacks a mention-detection stage happens to be an LLM. The
headline contrast could not then separate *no mention detection* from
*parametric knowledge*. Tier 3 is that missing cell: no mention detection, no
entity linking, no LLM — embed the post, embed a text description of every
candidate place, take nearest neighbours.

It uses the **same encoder as the tier 2 reranker**
(`intfloat/multilingual-e5-base`), so the comparison is not confounded by a
different encoder. E5 is trained with asymmetric prefixes for retrieval, so
`query: ` and `passage: ` are set explicitly in the config rather than left
implicit in the encoder call.

**What goes into the index** (`dense.index_fields`) defines what the tier can
possibly retrieve — a place whose indexed text says nothing about football
cannot be retrieved from a post about football. It is a configuration
parameter and it is recorded in the index metadata:

> English label · English description · English aliases (up to 8) ·
> country label · continent label

**The candidate pool** determines the difficulty: retrieving from 150 places is
a different task from retrieving from 100,000. The `gold_plus_predicted` pool
is every gold place across all configured datasets, plus every QID any
configuration ever proposed as a candidate place, plus every QID in a linker
candidate set, filtered to entities that actually carry coordinates. Sizes are
recorded in the index metadata and reported.

Two properties of that pool must be stated wherever this tier is reported:

* **The gold place is in the pool by construction.** So "gold not in the
  top-k" is a *ranking* failure, never a coverage failure, and 100% recall is
  reachable in principle. This is a closed-set retrieval evaluation.
* **The distractors are endogenous** — they are the places other systems
  actually confused for the gold. That makes them hard negatives, but the pool
  lacks the long tail of world geography, so scores here are optimistic
  relative to retrieval over a full gazetteer.

### Tier 5 — grounded LLM

Retrieval-augmented, not routed. Candidates are retrieved first, handed to the
model as a numbered list with QID, label, country and description, and the
model selects one or declines. It may not name a place outside the list.

The motivation is the routing result: the oracle shows ~20 pp of headroom that
routing does not capture, and the rebuilt gate underperforms both its
components. Retrieval-augmented generation is the direction adjacent tasks have
taken (Img2Loc for image geolocation, Spatial-RAG for spatial QA); the
text-geolocation equivalent is unexplored. Combining the paradigms is a cleaner
contribution than choosing between them.

Three design points:

* **The retriever is an existing tier, never a new one.** `grounded_gemma`
  takes candidates from tier 3, `grounded_gemma_linking` from tier 2b. A new
  retriever would confound the comparison against the tier it is built on.
* **Abstention is a first-class answer.** Declining when the right place is
  absent from the candidates is correct behaviour, not failure. Tier 2's high
  no-result rate makes abstention a real axis here, so refusal is recorded as
  refusal.
* **Selection is validated.** An index out of range, or a QID not in the list,
  is recorded as a violation in the record's `error`. A grounded system that
  ignores its grounding must be visible rather than scored as an ordinary
  wrong guess.

The two variants answer a question neither can answer alone: **does grounding
help because of the LLM or because of the retriever?** Same model, same prompt,
same selection contract, different candidate source. Chosen on dev.

### Failure attribution for the new tiers

Both tiers get labels appropriate to their architecture, and the table
distinguishes **impossible** cells from **empty** ones: a level whose stage
does not exist in an architecture prints `n/a`, while `0.0` means the stage
exists and the failure never occurred. Dense retrieval has no NER and no
linking stage, so those cells are `n/a`, not zero.

Two levels are specific to retrieval-then-selection:

| level | meaning |
|---|---|
| `gold_not_retrieved` | the gold place was not among the retrieved candidates — a retrieval failure |
| `wrong_candidate_selected` | the gold place *was* in the supplied list and the model chose another — a selection failure |
| `abstained` | the model declined although the gold place was in the list |
| `selection_violation` | the model answered outside the candidate list |

That split is the number tier 5 exists to produce: it separates retrieval
failure from selection failure, which is what says whether grounding is
working.

## Dev/test split

There is a single split, committed at `splits/place_split.csv` (with an
item-level view at `splits/item_split.csv`).

The unit is a **place, not an item**: every item about the same place is on the
same side, so a configuration cannot be tuned on one post about a place and
then reported on another post about the same place. The assignment is global
across datasets, stratified by (dataset, condition), and deterministic from
`sha256(SALT + place_key)` rather than from row order.

Configurations and every threshold are chosen on **dev**. **Test is reported
once.** Paired tests and bootstrap intervals resample **by place**, never by
item, because items within a place are not independent.

## Failure attribution

Every wrong prediction is classified by where it broke, using the stage fields
the runner recorded. The previous analysis inferred a granularity taxonomy
(continent wrong / country wrong / far / not exact) from stored booleans; this
one attributes to a stage:

| level | meaning |
|---|---|
| `no_ner_span` | no mention was detected at all |
| `span_not_linked` | mentions detected, the linker resolved none |
| `wrong_entity` | linked, but the gold place is nowhere in the pipeline — not among the linker's candidates, not among the scored places |
| `wrong_candidate_selected` | the linker *had* the gold place among its candidates and chose another |
| `correct_candidate_not_first` | the gold place was in the final scored list, and the reranker did not put it first |
| `returned_nothing` | stages completed, no prediction came out |
| `no_intermediate_stages` | the configuration has no stages to attribute to |

Three things to state when reporting this:

**The gold annotation is a place, not an entity.** So `wrong_entity` and
`wrong_candidate_selected` are separated by whether the gold place appears in
the linker's candidate set, which is the closest observable proxy for "it had
the right entity in hand". This is only possible because the records keep
ReFinED's full ranked candidate set, not just the winner.

**`no_intermediate_stages` is a result, not a gap.** A single-call LLM
configuration, and the constant majority baseline, produce an answer with no
observable intermediate state. Their failures cannot be localised, which is
the cost of that design and should be reported as such rather than hidden.

**A routed configuration is attributed per branch.** Because the branch is
recorded, its two populations separate: failures on the linking branch get a
stage, failures on the LLM branch get `no_intermediate_stages`. Without the
branch on the record this split would be impossible, which was the original
reason for recording it.

## Statistical approach

Every test and interval is clustered by **place**, because items about the same
place are not independent — the datasets contain several items per place, and
the synthetic sets were generated per place.

* **Paired accuracy**: McNemar, where each *place* contributes one outcome per
  system (whether that system got the majority of the place's items right).
  Places where the systems tie are concordant and drop out. An exact binomial
  p-value is used rather than the chi-square approximation, because the
  discordant count is small at this sample size. The item-level test is printed
  alongside and labelled: its pairs are not independent, so its p-value is
  anti-conservative.
* **Paired distances**: Wilcoxon signed-rank over places, each place reduced to
  the median distance of its items per system. Only items where *both* systems
  produced a coordinate can be paired; the count of usable pairs is always
  reported, because substituting a number for a missing coordinate is precisely
  what the Null Island bug did.
* **Intervals**: cluster bootstrap, resampling places with replacement and
  taking each sampled place with all of its items.
* **Effect sizes**: the paired difference with a bootstrap interval, reported
  next to every p-value. A test says whether a difference is distinguishable
  from zero; only the interval says whether it is large enough to matter.

## Scorer correctness check

Before the new scorer was trusted, it was run over retired result CSVs
converted into the new record format (`scripts/check_scorer.py`). The purpose
is to catch sign errors and column mix-ups, which are invisible in a
well-formatted table: a flipped haversine sign, lat and lon swapped, or gold
and predicted transposed all still produce plausible numbers.

The conversion applies the Null Island fix explicitly: a retired row whose
predicted coordinate is exactly 0.0/0.0 with no predicted QID had no
coordinate, and becomes `predicted_latlon: null`.

Result, over three files chosen to include both a non-LLM and the
worst-affected LLM run:

| file | rows | disagreements | on Null Island rows | unexplained |
|---|---|---|---|---|
| `eval_FINAL_refined_clean_sm.csv` (no LLM) | 144 | 0 | 0 | **0** |
| `eval_FINAL_two_stage_clean_journ.csv` (no LLM) | 72 | 0 | 0 | **0** |
| `eval_FINAL_groq_llama_noisy_sm.csv` (LLM) | 144 | 9 (all `dist_km`) | 9 | **0** |

On the two non-LLM files the new scorer reproduces the retired verdicts
*exactly* — every one of exact, country, continent, within-161 km and the
distance itself, to within 0.1 km, and identical aggregate metrics (46.5% /
53.5% / 56.2%, mean 759.5 km). That is the sign-error and column-mix-up check
passing: both implementations read the same coordinates and agree.

On the LLM file the only disagreements are the 9 distance fields on Null Island
rows, which is the fix working. Mean error falls from 2757.8 km to 2094.8 km
(−663 km, −24%) and the count of items carrying a distance falls from 141 to
132 — those 9 items now correctly have no distance rather than a ~6,000 km one.

## The geoparser tier was not re-run locally

Mordecai3 is included as a configuration (`configs/mordecai3.yaml`) with a
complete adapter (`src/geoparser.py`), but **its numbers are cited from the
published literature rather than reproduced here**, because of dependency
conflicts that cannot be resolved in this environment:

* Mordecai3 pins `torch<2.0`, `pandas<2.0` and `numpy<2.0`. The installed
  stack is torch 2.10 / pandas 2.2 / numpy 2.4, which is what flair, refined
  and sentence-transformers are built against.
* `torch<2.0` has **no cp313 wheel**, so the pin is unsatisfiable on Python
  3.13 at any level of effort — this is not a version-juggling problem.
* At runtime it additionally requires a local Elasticsearch 7.x instance with
  the GeoNames gazetteer loaded (`elastic_utilities.py` connects to
  `localhost:9200` and searches `index="geonames"`).

Reproducing it would mean a second Python ≤3.11 environment plus a
GeoNames-loaded Elasticsearch container, for one row in one table, on a tier
of system that is structurally limited on most of these conditions anyway
(see below). The configuration is kept because a configuration that fails with
a recorded explanation documents the decision better than an absent one, and
because it will run unchanged if that environment is ever built.

**Edinburgh Geoparser** and **CLAVIN** are not deployed either, for the same
class of reason: both are Java and both need local GeoNames instances. Their
numbers are cited from the EUPEG comparison.

Published figures to cite (fill the unverified cells from the source papers —
they are deliberately not reproduced from memory here):

| System | Corpus | Mean error | Accuracy@161km | Source |
|---|---|---|---|---|
| Mordecai3 | GeoWebNews | 184 km | 94 | Halterman (2023) |
| spaCy NER + population heuristic | GeoWebNews | 210 km | 95 | Gritta et al. |
| Edinburgh Geoparser | *(per corpus)* | — | — | EUPEG (Wang & Hu) |
| CLAVIN | *(per corpus)* | — | — | EUPEG (Wang & Hu) |

### Why this costs less than it looks

The population baseline (`configs/baseline_population.yaml`) covers most of
what the geoparser tier was there to establish.

One difference to state when citing it: this baseline takes population from
**Wikidata P1082**, not from the GeoNames dump that Gritta et al. used, so it
is the same heuristic over a different gazetteer rather than a reproduction of
the published baseline. P1082 coverage is sparser for small settlements, and
candidates with no population figure at all fall to a sitelink-count tie-break,
which GeoNames would not need. Treat the published 95 Accuracy@161km as the
heuristic's ceiling on a toponym-rich corpus, not as a number this
configuration should be expected to hit. It is a toponym-plus-gazetteer
system — NER extracts toponyms, the gazetteer is consulted, the most populous
candidate always wins — and on GeoWebNews that design beat several published
geoparsers. Unlike Mordecai3 it runs in this environment, on these datasets,
under the same runner and the same scorer as everything else.

### A geoparser cannot score on most of these conditions

A geoparser is a *toponym resolver*: it needs a place name in the text. On
conditions where no toponym is present it cannot score above zero **by
construction**. That is the expected outcome and the reason for including the
tier — it bounds what toponym resolution alone can achieve on text where the
location is implied rather than named. It must not be reported as a defect of
the system.

## Why the retired results are not comparable

`results/raw/eval_FINAL_*.csv` are kept as a historical record only. Two
defects in the code that produced them, both since fixed, make their numbers
non-comparable to anything reported now — and the two compounded.

**1. Null Island.** The LLM path's continent-only fallback returned
`qid=""`, `lat=0.0`, `lon=0.0`. The scorer read that as a real coordinate, so
a single item was simultaneously counted as "no result", "continent correct",
and carrying a ~6,000 km error. 42 rows across 11 result files are affected.
On `groq_llama_noisy_sm` the reported mean error is inflated by 663 km, about
32%.

**2. Uncached gazetteer search under a rate limit.** `_search_entities` was
the one uncached call on the hot path, so every configuration re-queried
Wikidata for the same surface forms at a 0.5 s floor. The committed run logs
record **178 `Search error: 429 Too Many Requests after retries`** across the
final runs, touching 159 items.

A 429 returned `[]`, which is indistinguishable from "no such place", so the
LLM path's city → country → continent cascade fell through a level. The two
defects therefore interact, and the interaction is measurable:

* There are 42 Null Island rows in total. 41 of them sit in the 15 runs whose
  logs record at least one transient failure, and **32 of those 41 (78%) are
  on items that hit one**.
* Within those runs, an item that hit a 429 was **9.5×** likelier to land on
  Null Island than the base rate (20.1% vs 2.1%).
* The direction is confirmed, not assumed. The response caches still hold what
  each model actually said, keyed by the same text hash. On the 42 Null Island
  rows the model named a **city in 12 cases and a country in 21** — 79% — and
  refused in none. The continent-only fallback was resolution failing, not the
  model declining to answer. In several of those cases the named country was
  the gold country, so the item should have scored country-correct with a real
  coordinate and was instead recorded as a no-result with a ~6,000 km error.

All three checks are reproducible: `python scripts/audit_retired_results.py`.

Search results are now disk-cached (`src/.wikidata_cache/search.pkl`), and
failed searches are deliberately *not* cached, so a transient error cannot
become a permanent empty answer. The retired LLM distance statistics were
pessimistic for infrastructure reasons unrelated to the method.

**3. Placeholder names read as place names.** A few responses put the *string*
`"null"` where JSON `null` was asked for. A truthy string passed the emptiness
check, so the resolver searched Wikidata for the literal word "null", and a
`continent: "null"` could be emitted as a prediction that can never be correct
yet still counts as a prediction, suppressing the no-result rate. Measured
across the three response caches (1,646 responses): 5 city, 3 country, 1
continent. Placeholder strings are now normalised to absent.

## Framing notes for the write-up

**Every non-LLM configuration uses the same Flair NER.** Mention detection is
held constant across `two_stage`, `refined` and the routed configurations, which is what makes a linker comparison a comparison of linkers
rather than of pipelines. State this as a design strength.

This was checked rather than assumed, because the configurations do not filter
tags identically: `two_stage` sets `ner.entity_types: null` (keep every tag)
while `refined` sets `[LOC, PER, ORG]`. On `social_media_clean` the effective
mention set is nevertheless **identical on 144/144 items** — `two_stage`
produced 4 MISC spans in total, and none of them yielded a candidate place,
because the retrieval step ignores tags outside LOC/PER/ORG. So Table 3 varies
only the linking mechanism.

`baseline_population` is deliberately *not* part of that comparison: it sets
`ner.entity_types: [LOC]`, since a population heuristic over a gazetteer is a
toponym system and feeding it person and organisation mentions would not be the
published baseline.

**"End-to-end" is retired as a label for ReFinED.** ReFinED here runs in
constrained-span mode over mentions supplied by Flair. It performs
disambiguation; Flair performs detection. That is a two-stage pipeline, and
calling it end-to-end describes neither what it does nor what varies between
the configurations being compared.
