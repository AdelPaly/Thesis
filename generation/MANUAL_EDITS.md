# Manual edits to the dataset

Everything in release `synthetic-v1` that a person decided, as opposed to a
rule or a judge model. No post text was typed or corrected by hand: a person
only chose which posts to take out and wrote the reason, and the writer model
wrote the replacement, which then passed the same checks as any other post.

The list of removals is not kept by hand. `synthetic/manual_edits.py` reads it
from the generation log and writes `synthetic/output/manual_edits.csv`, with
the removed post, the reason, the date and the post the dataset has now:

```bash
python generation/synthetic/manual_edits.py
```

It reports 47 removals of 44 items, all entered on 2026-11-17 and all replaced
the same day. All 44 items are in the dataset, none with the removed text.

## 1. Posts removed by hand (44 items)

All were made by the author on 2026-11-17, the posts read together with an AI
assistant. A removal is a line in `synthetic/output/manual_removed.csv`; then
`python generation/synthetic/regenerate.py` rewrites the item and takes the
line out again, which is why that file is empty now.

| Group | Items | How the posts were chosen | Where the choice is recorded |
|---|---|---|---|
| Flagged `A` and `B` posts read by hand | 7 | The second judge (`synthetic/audit.py`) flagged 173 posts. The 22 flagged `A` and `B` posts about the clue were read, 12 were written down, 7 called "remove" and 5 "borderline" and kept. Faults: a tie that is only a visit (3), a clue written as the wrong kind of thing (3), a dead person acting today (1). | `synthetic/output/audit/hand_check_A_B.csv`, `vse_Qwen3.8_27B_flagged.csv` |
| A banned entity as clue or distractor | 10 | Every kept post whose clue or distractor is in `synthetic/banned_entities.txt`. The entity was swapped in the plan for another one of the same place (`noise_swaps`, `banned_clue_swaps`) and the post written again. | `synthetic/banned_entities.txt`, the plan item |
| A tomb in the wrong place, clue replaced | 9 | From `synthetic/check_burials.py`: posts about a person's tomb where no new post without a tomb passed with `regenerate.py --burials`. The place's spare clue was used instead. | the reason on the removal names the person; `synthetic/output/audit/burial_check.csv` is empty now, since no such post is left |
| A clue written as the wrong kind of thing | 19 | After `synthetic/clue_types.py` had fetched from Wikidata what each clue is (a station, a dam, a beach), posts that treated the clue as something else, for example a station with offices and trucks. | reason on the removal |

One item is in two groups (10 + 9 + 19 + 7 = 45 for 44 items): its distractor
was banned and its new post then misused the new one.

The last group was chosen by reading, not by a script: no file records which
posts were looked at and passed. That is the one selection here that cannot be
run again.

## 2. Rewrites that a script chose

These are in the README of this folder and are listed here only to keep them
apart from the edits above. No person chose the single posts.

| Rewrite | Posts | Chosen by |
|---|---|---|
| Stock scenes | 1,095 | `regenerate.py --stock-scenes` (`stock_scene()`) |
| Tombs in the wrong place | 89 | `check_burials.py`, then `regenerate.py --burials` |
| `NOISE_EXPLICIT` posts that had failed their tries | 39 | `regenerate.py --failed NOISE_EXPLICIT` |

## 3. Lists written by hand

| File | What | When | How chosen |
|---|---|---|---|
| `synthetic/banned_entities.txt` | 19 Wikidata items no post may use | 2026-11-17 | The plan was searched by name, Wikidata type, party, membership and conviction for anything linked to terrorism, extremism, Nazi crimes, killers or massacres. The matches were read and the list written by hand. |
| `excluded_qids.txt` | places kept out of the place selection (`reform_places.py --exclude`) | before the place selection | see the README of this folder |
| `synthetic/output/audit/manual_sample_100.csv` | 100 posts rated keep or not: 96 kept, 4 rejected | committed 2026-11-17 | Drawn at random from the whole dataset with seed 20261118, once, by hand; no script in the repository draws it. The four faults were then searched for in the whole dataset (groups above). |

## What is not recorded

- The posts that were read and left in, except the five "borderline" ones in
  `hand_check_A_B.csv` and the 96 kept ones in `manual_sample_100.csv`.
- Who typed each line of `manual_removed.csv`: the file had no column for it.
  All lines were entered by the author.
- The code revision of each rewrite. The rows carry `config_hash`; for the
  revision see `synthetic/output/raw_generations_code_rev.csv`.
