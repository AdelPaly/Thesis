# Datasets: provenance, generation metadata, and validation

This file documents how the evaluation datasets in `src/` were created, with which model, when, and what the validation produced. It is the single "generation metadata" record for the project.

## Datasets

| File | Entries | Description |
|------|--------:|-------------|
| `src/social_media_dataset.csv` | 144 | Synthetic social-media posts (clean). |
| `src/social_media_noisy.csv` | 144 | Noise-injected variant of the above, row-aligned (same `qid`s and coordinates, text rewritten with distractors). |
| `src/journalistic_dataset.csv` | 72 | Synthetic journalistic articles. |
| `src/Geocorpora.csv`, `src/geocorpora_eval.csv` | n/a | External GeoCorpora reference data and the 144-row sample used for cross-checking (not LLM-generated). |

Each synthetic CSV has the columns:
`text, qid, location_name, country, continent, signal_type, ambiguous_name, other_referents, latitude, longitude`.

The three **signal types** (`A`, `B`, `C`) are defined in full in the generation specification (see below).

## Generation

- **Model:** Claude Opus 4.7
- **Method:** Interactive. The specification was given to the model via the Claude chat interface (claude.ai), not a programmatic API, so there is no generation script. The reproducible artifact is the specification itself (see below). Re-running it would yield different entries (LLM output is non-deterministic) but follow the same procedure.
- **Specification:** the complete spec is committed at the repository root in the `prompt` file. It defines the output schema, the three signal types (A = entity linking, B = toponym disambiguation, C = entity disambiguation), the text formats (social-media post vs. journalistic article), the continent distribution (8 entries per continent per signal type), the location/entity selection criteria, and the quality-control checklist applied to every entry.
- **Approximate generation dates** (from working-file timestamps):
  - `social_media_dataset.csv`: 2026-04-29
  - `journalistic_dataset.csv`: 2026-04-29
  - `social_media_noisy.csv`: 2026-05-03 (noise step, see below)
- **Wikidata verification date:** on/near the generation dates above (QIDs and entity-location links can change over time, so they were verified close to generation; see the validation section).

## Noise injection (`social_media_noisy.csv`)

Created from `social_media_dataset.csv` interactively (in the same Claude Opus 4.x chat session as the clean generation) by rewriting the text of each clean row in place, injecting distractor signals (references to other places, nationalities, or people who are not the target location, for example "some Brazilian tourists near me left at halftime" or "texting my mate in Washington") while keeping the post's true target location unchanged.

Like the clean generation, the noise step was interactive, not scripted: the 144 clean rows were rewritten into their noisy variants in the chat, following the noise specification below.

The noisy set is **row-aligned** with the clean set: same number of rows, same `qid`s in the same order, identical `latitude`/`longitude`. Only the `text` column differs. This alignment is verified by `scripts/add_noisy_coords.py`, which joins the clean coordinates onto the noisy rows by `qid` and re-checks them against Wikidata `P625`.

### Noise specification

The distractor design (the noise "prompt" applied to each row) held the genuine Tier-1 signal and the true target location fixed, and added misleading distractors drawn from Gritta et al.'s associative-toponym categories. In substance:

> Keep each clean post's genuine signal and its true target location unchanged. Rewrite the text so it additionally contains at least two misleading geographic signals, each from a different category below, each pointing to a geographically plausible but incorrect location (ideally on the same continent as the target, to maximise difficulty). Distractors must read as natural, incidental detail: a fluent reader with full context should still identify the genuine signal and discount the noise.

Categories (Gritta et al. associative toponyms):

- **Demonyms / nationality adjectives**: e.g. "British tourists", "German engineers"
- **Non-literal modifiers** (geographic string on a non-locative head): e.g. "German Shepherd", "French press coffee"
- **Metonymic toponyms** (institution-as-place): e.g. "Brussels announced", "Washington responded"
- **Embedded associative toponyms** (non-toponym in a larger name): e.g. "US Supreme Court", "Sydney Lottery"

> Change only the text column. Leave `qid`, `location_name`, `country`, `continent`, `signal_type`, `ambiguous_name`, `other_referents`, `latitude`, `longitude` identical to the clean row.

In practice each rewritten post carried two to three distractors, typically one nationality/demonym distractor plus one "place X is famous for Y" comparison (e.g. a Brazilian-carnival or Italian-derby reference). These are deliberately prototypical "obviously wrong" signals, which test the strong failure mode (the pipeline latching onto a salient foreign reference) rather than near-miss confusions between culturally adjacent places.

## Validation

Every synthetic entry was validated against Wikidata with `scripts/validate_dataset.py`, which checks, per row, that (1) the `qid` exists, (2) its English label matches the expected name, and (3) its country (`P17`) is consistent with the `country` column. One known edge case in the `P17` check is sovereign-territory mapping: e.g. Nouméa (Q9733), whose `P17` is France rather than New Caledonia, which strict `P17` equality flags as a mismatch; a `P131` traversal or a sovereign-territory allowlist avoids this.

**First-pass validation results** (entries that failed QID existence or label matching and were subsequently regenerated/corrected):

| Dataset | First-pass failures | Share |
|---------|--------------------:|------:|
| social_media_dataset.csv | 71 / 144 | 49% |
| journalistic_dataset.csv | 34 / 72 | 47% |

The failing rows were regenerated/fixed until the datasets passed validation.

The same family of checks was applied to the **GeoCorpora** reference sample by `scripts/fix_qids.py`; its run log is committed at `results/logs/fix_qids_geocorpora_144.log`.

## Reproducing the validation

```bash
pip install -r requirements.txt
python scripts/validate_dataset.py src/social_media_dataset.csv
python scripts/validate_dataset.py src/journalistic_dataset.csv
```

No API key is required for validation; it uses the public Wikidata SPARQL endpoint. See `scripts/README.md` for the full list of scripts.
