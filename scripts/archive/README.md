# `scripts/archive/` — completed one-offs

These scripts have already done their job and are not part of any current
workflow. They are kept, rather than deleted, because they are the provenance
of files that are still on disk: if you need to know how a column or a CSV got
there, the answer is here.

| Script | What it did |
|--------|-------------|
| `add_noisy_coords.py` | Joined clean coordinates onto `src/social_media_noisy.csv` by `qid` (the noisy set is row-aligned with the clean set) and re-verified them against `P625`. Provenance for the coordinate columns of the noisy dataset. |
| `enrich_within_161km.py` | Back-filled the `within_161km` column on `results/raw/eval_FINAL_*.csv` files produced before the old `evaluate.py` started writing it. Provenance for that column in the retired results. |
| `analyze_failures.py` | Produced `results/summary/table_FINAL_failure_modes.csv` (committed) from the retired `results/raw/eval_FINAL_*.csv`. Its six-level taxonomy was a *granularity* breakdown (continent wrong / country wrong / beyond 161 km / not exact) inferred from stored booleans. Superseded by `scripts/analyse.py failures`, which attributes to a *stage* from recorded fields instead of inferring. Archived rather than deleted because it is the generator of that committed summary table. |
| `restore_geocorpora_eval.py` | Reconstructed `src/geocorpora_eval.csv` from the ground-truth columns of an existing `eval_FINAL_*_geocorpora.csv`, so every mode was evaluated on the same row set. Provenance for that dataset file. |

All three did `from add_coords import fetch_coords`, and `add_coords.py` is
gone, so **they no longer import**. That is deliberate: restoring them to
working order would mean restoring the second coordinate client the
consolidation removed. They are here to record what was done to the files on
disk, not to be re-run. If one ever needs to run again, point it at
`src/wikidata.coords_for_qids`, which returns the same `{qid: (lat, lon)}`
mapping `fetch_coords` did.

They are not maintained. `enrich_within_161km.py` carries its own copy of the
haversine formula and its own hardcoded mode list; both are the reason it is
archived rather than kept live. The live implementations are
`scripts/score.py` and `configs/`.
