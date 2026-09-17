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
what the geoparser tier was there to establish. It is a toponym-plus-gazetteer
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
held constant across `two_stage`, `refined`, `refined_search`,
`baseline_population` and the routed configurations, which is what makes a
linker comparison a comparison of linkers rather than of pipelines. State this
as a design strength.

**"End-to-end" is retired as a label for ReFinED.** ReFinED here runs in
constrained-span mode over mentions supplied by Flair. It performs
disambiguation; Flair performs detection. That is a two-stage pipeline, and
calling it end-to-end describes neither what it does nor what varies between
the configurations being compared.
