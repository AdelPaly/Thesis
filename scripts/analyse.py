"""Analysis and reporting, reading the prediction files.

    python scripts/analyse.py failures --split test
    python scripts/analyse.py tables   --split test
    python scripts/analyse.py stats    --split test
    python scripts/analyse.py all      --split test --out results/summary

Reads results/predictions/*.jsonl and splits/item_split.csv, plus configs/*.yaml
for report row order (tier) and each architecture's stage set. Loads no model,
makes no network call, and imports nothing from src/ -- metrics come from
score.py and statistics from stats.py, so there is one definition of each.

Failure attribution
-------------------
For every wrong prediction, where did it break? The ten levels are attributed
from the stage fields the runner recorded, not inferred from the shape of the
output:

  error                     the runner recorded an exception (API failure,
                            timeout): infrastructure, not a pipeline fault
  no_ner_span               no mention was detected at all
  span_not_linked           mentions detected, the linker resolved none
  wrong_entity              linked, but the gold place is nowhere in the
                            pipeline -- not among the linker's candidates and
                            not among the scored places
  gold_not_retrieved        retrieval (dense or grounded) never supplied the
                            gold place
  wrong_candidate_selected  the gold place was among the candidates (the
                            linker's, or the grounded list) and another was
                            chosen
  correct_candidate_not_first  the gold place was in the final scored list but
                            the reranker did not put it first
  abstained                 grounded model declined with the gold in front of it
  selection_violation       grounded model named something outside the list
  returned_nothing          stages completed (or the architecture has none to
                            fail), no prediction came out
  no_intermediate_stages    answered, but the architecture records no stages

Two honest limits on this taxonomy:

* The datasets annotate a gold PLACE, not a gold entity. "Linked to the wrong
  entity" and "correct entity, wrong candidate" are therefore separated by
  whether the gold place appears in the linker's candidate set, which is the
  closest observable proxy.
* ner_free (LLM) configurations have no intermediate stages by construction,
  so their failures are attributed to `no_intermediate_stages`. That is not a
  gap in the analysis -- it is the finding that a single-call LLM pipeline
  cannot be debugged stage by stage.

Statistics
----------
Exact-match comparisons are McNemar by place. The p-values are Holm-adjusted
over the family of comparisons run (printed as "p_holm" with the family size);
the bootstrap intervals are not adjusted.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from score import (attach_split, find_prediction_files, group_by, load_predictions, print_table,
                   score_record, summarise_metrics)
import stats as st

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"

# Bootstrap replicates for this report. stats.DEFAULT_N_BOOT is larger than a
# pure-Python report over every configuration needs.
DEFAULT_BOOT = 2000
RULE = "=" * 100

# Report order. A configuration declares its tier in its config file; the tier
# is presentation metadata excluded from config_hash, so it orders rows without
# being able to change a prediction. Rank falls back to the paradigm on the
# records for anything with no config file left.
TIER_RANK = {"0": 0, "2a": 2, "2b": 3, "3": 4, "4": 5, "5": 6}
PARADIGM_RANK = {"baseline": 0, "ner_dependent": 3, "dense_retrieval": 4,
                 "ner_free": 5, "grounded_llm": 6}


def _config_stages(raw: dict) -> set[str]:
    """Stages an architecture has, read from its config."""
    paradigm = raw.get("paradigm")
    stages = set()
    if (raw.get("ner") or {}).get("model"):
        stages.add("ner")
    # wikidata_search is retrieval, not a linker: the records never carry links.
    if (raw.get("linking") or {}).get("backend") not in (None, "none", "wikidata_search"):
        stages.add("linking")
    if paradigm in ("ner_dependent", "dense_retrieval", "grounded_llm"):
        stages.add("candidates")
    if paradigm == "grounded_llm":
        stages.add("selection")
    return stages


def load_config_meta(config_dir: Path = CONFIG_DIR) -> dict[str, dict]:
    """{config name: {"tier", "stages"}} read straight from the YAML.

    Reading config files for row order and stage sets is not the scorer loading
    a model: no model, no network, no import from src/. score.py stays free of
    this entirely. A malformed file is warned about and skipped; a missing
    PyYAML is an error, not a silent fallback.
    """
    meta = {}
    for path in sorted(Path(config_dir).glob("*.yaml")):
        if path.name == "datasets.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            print(f"warning: skipping unreadable config {path.name}: {exc}", file=sys.stderr)
            continue
        if isinstance(raw, dict) and raw.get("name"):
            meta[raw["name"]] = {"tier": str(raw.get("tier", "") or ""),
                                 "stages": _config_stages(raw)}
    return meta


def pct_str(a: int, b: int) -> str:
    return f"{a / b * 100:.1f}%" if b else "-"


def sort_key(name: str, records: list[dict], meta: dict) -> tuple:
    """Tier first, then paradigm, then name."""
    tier = meta.get(name, {}).get("tier", "")
    if tier in TIER_RANK:
        return (TIER_RANK[tier], name)
    paradigms = {r.get("paradigm") for r in records}
    paradigm = sorted(paradigms)[0] if paradigms else ""
    return (PARADIGM_RANK.get(paradigm, 9), name)


LEVELS = ["error", "no_ner_span", "span_not_linked", "wrong_entity",
          "gold_not_retrieved", "wrong_candidate_selected",
          "correct_candidate_not_first", "abstained", "selection_violation",
          "returned_nothing", "no_intermediate_stages"]

# Which pipeline stage a level presupposes. A level whose stage does not exist
# in an architecture is STRUCTURALLY IMPOSSIBLE there, and the table prints
# `n/a` rather than 0.0 -- "this cannot happen here" and "this never happened"
# are different statements and must not look alike.
LEVEL_REQUIRES = {
    "error": None,                       # any architecture can hit an API error
    "no_ner_span": "ner",
    "span_not_linked": "linking",
    "wrong_entity": "linking",
    "gold_not_retrieved": "candidates",
    "wrong_candidate_selected": "candidates",
    "correct_candidate_not_first": "candidates",
    "abstained": "selection",
    "selection_violation": "selection",
    "returned_nothing": None,            # any architecture can return nothing
    "no_intermediate_stages": None,
}


def stages_present(records: list[dict]) -> set[str]:
    """Which stages the records show an architecture has. Unioned with the
    config's stage set, because records alone miss a stage whose every item
    failed."""
    present = set()
    for rec in records:
        if rec.get("ner_spans"):
            present.add("ner")
        if rec.get("linked_entities"):
            present.add("linking")
        if rec.get("candidate_places"):
            present.add("candidates")
        if rec.get("selection") is not None:
            present.add("selection")
    return present


# ------------------------------------------------------- failure attribution

def _linker_candidate_qids(rec: dict) -> set[str]:
    """Every QID the linker had on the table, across all mentions."""
    out = set()
    for le in rec.get("linked_entities") or []:
        if le.get("qid"):
            out.add(le["qid"])
        for cand in le.get("candidates") or []:
            if cand.get("qid"):
                out.add(cand["qid"])
    return out


def _place_qids(rec: dict) -> list:
    return [p.get("qid") for p in rec.get("candidate_places") or []]


def attribute(rec: dict, has_ner_stage: bool = True) -> str | None:
    """Where a wrong prediction broke. None for a correct prediction.

    `has_ner_stage` is False for architectures with no mention detection, so an
    empty result is `returned_nothing` there rather than `no_ner_span`.
    """
    if rec.get("exact"):
        return None
    if rec.get("error") is not None:
        return "error"

    gold = rec.get("place_qid")
    spans = rec.get("ner_spans") or []
    links = rec.get("linked_entities") or []
    places = rec.get("candidate_places") or []
    has_pred = rec.get("has_prediction")
    selection = rec.get("selection")
    gold_in_places = bool(gold and gold in _place_qids(rec))

    # --- retrieval-then-selection architectures (tier 5) -------------------
    # Attributed first, because for these the question is specifically whether
    # retrieval or selection failed, and that is the number the tier exists to
    # produce.
    if selection is not None:
        if selection.get("violation"):
            return "selection_violation"
        if not gold_in_places:
            return "gold_not_retrieved"      # nothing retrieved, or not the gold
        if selection.get("abstained"):
            return "abstained"               # declined with the gold in front of it
        return "wrong_candidate_selected"    # had it, chose another

    # --- dense retrieval (tier 3): no NER, no linking ----------------------
    if rec.get("paradigm") == "dense_retrieval":
        if not places:
            return "returned_nothing"
        return "correct_candidate_not_first" if gold_in_places else "gold_not_retrieved"

    # No observable stage at all. Two different situations:
    #   * it still produced an answer -- a single-call LLM or the constant
    #     majority baseline. Nothing in the record can say where it went wrong,
    #     which is the finding rather than a gap.
    #   * it produced nothing either: `no_ner_span` if the architecture has a
    #     mention-detection stage, otherwise it simply returned nothing.
    if not spans and not places:
        if has_pred:
            return "no_intermediate_stages"
        if has_ner_stage and rec.get("paradigm") in ("ner_dependent", "baseline"):
            return "no_ner_span"
        return "returned_nothing"

    if not spans:
        return "no_ner_span"

    linked_any = any(le.get("qid") for le in links)
    # A configuration that does no linking (the population baseline) reaches
    # candidate places without ever linking, so an empty link list is only a
    # failure if nothing else was produced.
    if links and not linked_any and not places:
        return "span_not_linked"

    if not has_pred:
        return "returned_nothing"

    if gold:
        if gold_in_places:
            # It was retrieved and scored, and something else won.
            return "correct_candidate_not_first"
        if gold in _linker_candidate_qids(rec):
            return "wrong_candidate_selected"

    return "wrong_entity"


def attribute_all(scored: list[dict], meta: dict) -> list[dict]:
    for rec in scored:
        stages = meta.get(rec.get("config_name"), {}).get("stages")
        rec["failure_level"] = attribute(rec, has_ner_stage=stages is None or "ner" in stages)
    return scored


def crosstab(scored: list[dict], row_key: str, meta: dict, only_failures: bool = True,
             all_records: list[dict] | None = None):
    """rows = row_key value, columns = failure level, cells = counts.

    Also returns, per row, the set of stages that row's architecture has, so
    the printer can tell an impossible cell from an empty one.
    """
    table = defaultdict(Counter)
    rows_records = defaultdict(list)
    for rec in (all_records if all_records is not None else scored):
        rows_records[rec.get(row_key)].append(rec)
    for rec in scored:
        level = rec.get("failure_level")
        if only_failures and level is None:
            continue
        table[rec.get(row_key)][level or "correct"] += 1
    stages = {}
    for key, recs in rows_records.items():
        stages[key] = stages_present(recs).union(
            *(meta.get(r.get("config_name"), {}).get("stages", set()) for r in recs))
    return table, stages


def print_crosstab(result, row_label: str, percent: bool = True) -> None:
    """Row percentages. A cell an architecture cannot produce prints as `n/a`,
    never as 0.0."""
    table, stages = result
    cols = [c for c in LEVELS if any(c in row for row in table.values())]
    width = max([len(row_label)] + [len(str(k)) for k in table]) + 1
    header = f"{row_label:{width}} {'n':>5} " + " ".join(f"{c[:11]:>12}" for c in cols)
    print(header)
    print("-" * len(header))
    for key in sorted(table, key=lambda k: str(k)):
        row = table[key]
        total = sum(row.values())
        have = stages.get(key, set())
        cells = []
        for c in cols:
            required = LEVEL_REQUIRES.get(c)
            if required is not None and required not in have and not row[c]:
                cells.append(f"{'n/a':>12}")
            elif percent:
                cells.append(f"{row[c] / total * 100:>12.1f}" if total else f"{'-':>12}")
            else:
                cells.append(f"{row[c]:>12}")
        print(f"{str(key):{width}} {total:>5} " + " ".join(cells))
    print("  n/a = the architecture has no such stage, so the level is impossible;")
    print("  0.0 = the stage exists and the failure never occurred.")
    print()


def write_rows(path: Path, header: list[str], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"wrote {path}")


def cmd_failures(scored: list[dict], args, meta: dict) -> None:
    attribute_all(scored, meta)
    failures = [r for r in scored if r.get("failure_level") is not None]
    errors = Counter(r.get("config_name") for r in failures
                     if r["failure_level"] == "error")
    print(RULE)
    print(f"FAILURE ATTRIBUTION  (split={args.split})")
    print(f"{len(failures)} wrong prediction(s) of {len(scored)} scored item(s)")
    print(RULE)
    if errors:
        print("Runner errors (infrastructure, not pipeline failures): "
              + ", ".join(f"{c} {n}" for c, n in sorted(errors.items())))
    else:
        print("Runner errors: none.")
    print("Row percentages across the levels; each row sums to 100.\n")

    print("-- by configuration --")
    print_crosstab(crosstab(failures, "config_name", meta, all_records=scored), "config")
    print("-- by paradigm --")
    print_crosstab(crosstab(failures, "paradigm", meta, all_records=scored), "paradigm")
    print("-- by condition --")
    print_crosstab(crosstab(failures, "condition", meta, all_records=scored), "condition")
    print("-- by paradigm x condition --")
    for rec in scored:
        rec["_pxc"] = f"{rec.get('paradigm')} / {rec.get('condition')}"
    print_crosstab(crosstab(failures, "_pxc", meta, all_records=scored),
                   "paradigm / condition")

    if args.out:
        counts = Counter((r.get("config_name"), r.get("dataset"),
                          r.get("paradigm"), r.get("condition"),
                          r.get("failure_level"))
                         for r in failures)
        rows = [[*key, n] for key, n in sorted(counts.items(), key=lambda kv: str(kv[0]))]
        write_rows(Path(args.out) / f"table_failure_attribution_{args.split}.csv",
                  ["config", "dataset", "paradigm", "condition", "failure_level", "n"], rows)


# ------------------------------------------------------- grounding report

def _gold_supplied(recs: list[dict]) -> list[dict]:
    """Records whose candidate list contained the gold place."""
    return [r for r in recs
            if r.get("place_qid")
            and r["place_qid"] in {c.get("qid") for c in r.get("candidate_places") or []}]


def _selected_gold(supplied: list[dict]) -> int:
    return sum(1 for r in supplied if (r.get("selection") or {}).get("qid") == r.get("place_qid"))


def cmd_grounding(scored: list[dict], args) -> None:
    """Separate retrieval failure from selection failure for tier 5.

    This is the number that says whether grounding is working: of the items
    where the gold place WAS in the supplied candidate list, how often did the
    model actually pick it? A low rate means the model is the bottleneck; a low
    "gold supplied" rate means the retriever is.
    """
    grounded = [r for r in scored if r.get("selection") is not None]
    if not grounded:
        return
    print(RULE)
    print(f"GROUNDED SELECTION: retrieval failure vs selection failure "
          f"(split={args.split})")
    print(RULE)

    by_config = group_by(grounded, "config_name")
    csv_rows = []

    header = (f"{'configuration':26} {'n':>5} {'gold supplied':>14} "
              f"{'selected gold':>14} {'abstained':>10} {'abst. w/ gold':>14} "
              f"{'violations':>11}")
    print(header)
    print("-" * len(header))
    for name, recs in sorted(by_config.items()):
        n = len(recs)
        supplied = _gold_supplied(recs)
        sel = _selected_gold(supplied)
        abstained = sum(1 for r in recs if (r.get("selection") or {}).get("abstained"))
        abst_gold = sum(1 for r in supplied if (r.get("selection") or {}).get("abstained"))
        viol = sum(1 for r in recs if (r.get("selection") or {}).get("violation"))
        print(f"{name:26} {n:>5} "
              f"{f'{len(supplied)}/{n} ' + pct_str(len(supplied), n):>14} "
              f"{f'{sel}/{len(supplied)} ' + pct_str(sel, len(supplied)):>14} "
              f"{pct_str(abstained, n):>10} "
              f"{f'{abst_gold}/{len(supplied)}':>14} "
              f"{viol:>11}")
        csv_rows.append([name, "", n, len(supplied), sel, abstained, abst_gold, viol])
    print()
    print("  gold supplied = the retriever put the gold place in the list at all")
    print("  selected gold = of those, the model chose it. This is the selection")
    print("                  step in isolation: retrieval had already succeeded.")
    print("  abst. w/ gold = declined although the gold place was in front of it")
    print("  violations    = named something outside the supplied list")
    print()

    # By condition, since the tiers differ most by what signal the text carries.
    for name, recs in sorted(by_config.items()):
        conds = sorted({r.get("condition") for r in recs if r.get("condition")})
        if not conds:
            continue
        print(f"-- {name}, by condition --")
        print(f"{'condition':12} {'n':>5} {'gold supplied':>14} {'selected gold':>14}")
        for cond in conds:
            sub = [r for r in recs if r.get("condition") == cond]
            supplied = _gold_supplied(sub)
            sel = _selected_gold(supplied)
            print(f"{str(cond):12} {len(sub):>5} "
                  f"{f'{len(supplied)}/{len(sub)} ' + pct_str(len(supplied), len(sub)):>14} "
                  f"{f'{sel}/{len(supplied)} ' + pct_str(sel, len(supplied)):>14}")
            csv_rows.append([name, cond, len(sub), len(supplied), sel, "", "", ""])
        print()

    if args.out:
        write_rows(Path(args.out) / f"table_grounding_{args.split}.csv",
                  ["config", "condition", "n", "gold_supplied", "selected_gold",
                   "abstained", "abstained_with_gold", "violations"], csv_rows)


# ---------------------------------------------------------------- tables

METRIC_COLS = ["n", "n_gold_qid", "exact", "country", "continent",
               "no_result_rate", "error_rate"]


def _print_metric_table(rows: list, label: str = "configuration", note: str | None = None) -> None:
    print_table(rows, label)
    if note:
        print(f"\n  {note}")
    print()


def _config_dataset(rec: dict) -> tuple:
    return rec.get("config_name"), rec.get("dataset")


def _span_signature(rec: dict) -> tuple:
    return tuple((s.get("text"), s.get("start"), s.get("end"), s.get("type"))
                 for s in rec.get("ner_spans") or [])


def _ner_agreement_note(groups: dict, names: list[str]) -> str:
    """Computed, not asserted: on the items two configurations share, how many
    have identical NER spans."""
    parts = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pairs = st.pair_records(groups[a], groups[b])
            same = sum(1 for _, ra, rb in pairs if _span_signature(ra) == _span_signature(rb))
            parts.append(f"{a} vs {b}: NER spans identical on {same}/{len(pairs)} shared items.")
    return " ".join(parts)


def cmd_tables(scored: list[dict], args, meta: dict) -> None:
    groups = group_by(scored, "config_name")
    present = set(groups)

    def ordered(names) -> list[str]:
        return sorted((n for n in names if n in present),
                      key=lambda n: sort_key(n, groups[n], meta))

    # Grouped by the paradigm ON THE RECORDS rather than by listing
    # configuration names, so a configuration added or removed needs no edit here.
    by_paradigm = defaultdict(list)
    for name, recs in groups.items():
        paradigms = {r.get("paradigm") for r in recs}
        by_paradigm[sorted(paradigms)[0] if paradigms else "unknown"].append(name)
    for key in by_paradigm:
        by_paradigm[key] = ordered(by_paradigm[key])

    ner_dependent = by_paradigm.get("ner_dependent", [])
    dense = by_paradigm.get("dense_retrieval", [])
    ner_free = by_paradigm.get("ner_free", [])
    grounded = by_paradigm.get("grounded_llm", [])

    print(RULE)
    print("TABLE 1  Baselines vs everything  -- is the task non-trivial?")
    print(f"(split={args.split})")
    print(RULE)
    # Tier order: what kind of signal each can consume, cheapest first.
    note = None
    if args.split == "test" and "baseline_majority" in present:
        note = ("A place is either dev or test, so the most frequent DEV place "
                "cannot be a test gold place: baseline_majority is structurally "
                "0% exact on test. That is the split working, not a defect.")
    _print_metric_table([(n, summarise_metrics(groups[n])) for n in ordered(present)], note=note)

    print(RULE)
    print("TABLE 2  NER-dependent vs NER-free, by condition")
    print("          -- does having a mention-detection stage matter?")
    print(RULE)
    # "NER-free" and "LLM" are separate axes: dense retrieval has no mention
    # detection and no LLM, so it is the cell that lets the contrast separate
    # "no mention detection stage" from "parametric knowledge".
    tagged = ([(n, "[mention-detect]") for n in ner_dependent]
              + [(n, "[no-mention-detect]") for n in dense]
              + [(n, "[llm, no-mention-detect]") for n in ner_free]
              + [(n, "[llm, grounded]") for n in grounded])
    for cond in sorted({r.get("condition") for r in scored if r.get("condition")}):
        print(f"-- condition {cond} --")
        _print_metric_table([(f"{n}  {tag}",
                              summarise_metrics([r for r in groups[n] if r.get("condition") == cond]))
                             for n, tag in tagged])
    print("-- pooled over conditions --")
    _print_metric_table([(f"{n}  {t}", summarise_metrics(groups[n])) for n, t in tagged])

    print(RULE)
    print("TABLE 3  Linker comparison -- which linker?")
    print(RULE)
    notes = []
    if len(ner_dependent) > 1:
        notes.append(_ner_agreement_note(groups, ner_dependent))
    if {"two_stage", "refined"} <= set(ner_dependent):
        notes.append("two_stage retrieves with wbsearchentities and reranks with E5; refined "
                     "disambiguates with ReFinED. Both detect mentions with Flair; the "
                     "difference is purely the linker only where the NER spans above agree.")
    _print_metric_table([(n, summarise_metrics(groups[n])) for n in ner_dependent],
                        note=" ".join(notes) or None)

    print(RULE)
    print("APPENDIX  All configurations")
    print(RULE)
    _print_metric_table([(n, summarise_metrics(groups[n])) for n in sorted(present)])

    print("APPENDIX  All configurations x dataset")
    print(RULE)
    pairs = group_by(scored, _config_dataset)
    _print_metric_table([(f"{c} / {d}", summarise_metrics(v)) for (c, d), v in sorted(pairs.items())])

    if args.out:
        rows = []
        for (c, d), v in sorted(pairs.items()):
            m = summarise_metrics(v)
            rows.append([c, d] + [round(m.get(k), 4) if isinstance(m.get(k), float) else m.get(k)
                                  for k in METRIC_COLS])
        write_rows(Path(args.out) / f"table_appendix_all_configs_{args.split}.csv",
                  ["config", "dataset"] + METRIC_COLS, rows)


# ---------------------------------------------------------------- statistics

# Explicit pairs, because which comparisons are worth making is an analysis
# decision, not something to enumerate automatically. Pairs naming a
# configuration with no predictions on disk are skipped silently. The Holm
# family is the comparisons that actually run.
COMPARISONS = [
    ("refined", "two_stage", "which linker: ReFinED vs search+rerank"),
    ("refined", "baseline_population", "does the linker beat the population floor"),
    ("gemma4", "baseline_population", "does the LLM beat the population floor"),
    ("refined", "gemma4", "mention detection vs parametric knowledge"),
    ("dense_retrieval", "refined", "dense retrieval vs entity linking"),
    ("dense_retrieval", "gemma4", "no mention detection: retrieval vs parametric"),
    ("grounded_gemma", "gemma4", "does grounding help (same model, matched control)"),
    ("grounded_gemma", "dense_retrieval", "grounded LLM vs its retriever alone"),
    ("grounded_gemma_linking", "grounded_gemma", "grounding from linking vs from dense"),
    ("grounded_gemma_linking", "refined", "grounded LLM vs its retriever alone"),
]


def _exact(rec: dict):
    return rec.get("exact")


def run_comparisons(groups: dict, n_boot: int, seed: int) -> list[dict]:
    """Run every comparison that has data on both sides, then Holm-adjust the
    by-place McNemar p-values over that family."""
    results = []
    for a_name, b_name, question in COMPARISONS:
        if a_name not in groups or b_name not in groups:
            continue
        pairs = st.pair_records(groups[a_name], groups[b_name])
        if not pairs:
            continue
        results.append({
            "a": a_name, "b": b_name, "question": question,
            "n_pairs": len(pairs), "n_places": len({p[0] for p in pairs}),
            "mc_place": st.mcnemar(pairs, _exact, unit="place"),
            "mc_item": st.mcnemar(pairs, _exact, unit="item"),
            "effect": st.paired_effect_ci(pairs, st.exact_rate, n_boot=n_boot, seed=seed),
        })
    adjusted = st.holm_adjust([r["mc_place"]["p"] for r in results])
    for r, p_holm in zip(results, adjusted):
        r["p_holm"] = p_holm
    return results


def cmd_stats(scored: list[dict], args) -> None:
    groups = group_by(scored, "config_name")
    print(RULE)
    print(f"PAIRED STATISTICS  (split={args.split}, n_boot={args.boot}, seed={args.seed})")
    print(RULE)
    print("Everything below is clustered by PLACE. Items about the same place are")
    print("not independent, so places are the unit paired by McNemar and the unit")
    print("resampled by the bootstrap.")
    print("Item-level McNemar is shown alongside for reference only; its pairs are")
    print("not independent and its p-value is anti-conservative.\n")

    results = run_comparisons(groups, args.boot, args.seed)
    family = len(results)
    print(f"Family of {family} comparison(s). p_holm is the Holm-adjusted p of the")
    print("McNemar-by-place test over that family; bootstrap intervals are not adjusted.\n")

    for r in results:
        mp, mi = r["mc_place"], r["mc_item"]
        print(f"--- {r['a']}  vs  {r['b']} ---")
        print(f"    {r['question']}")
        print(f"    {r['n_pairs']} paired item(s), {r['n_places']} place(s)")
        print(f"    exact match, McNemar by place : b={mp['b']} c={mp['c']} "
              f"p={st.fmt_p(mp['p'])}  p_holm={st.fmt_p(r['p_holm'])} (family of {family})")
        print(f"    exact match, McNemar by item  : b={mi['b']} c={mi['c']} "
              f"p={st.fmt_p(mi['p'])}  (reference only)")
        print(f"    exact match difference        : "
              f"{st.fmt_ci(r['effect'], pct=True, unit=' pp')}")
        print()

    print(RULE)
    print("BOOTSTRAP INTERVALS, resampled by place")
    print(RULE)
    header = f"{'configuration':30} {'exact %':>22}"
    print(header)
    print("-" * len(header))
    boot_rows = []
    for name in sorted(groups):
        ci = st.cluster_bootstrap_ci(groups[name], st.exact_rate, n_boot=args.boot, seed=args.seed)
        print(f"{name:30} {st.fmt_ci(ci, pct=True):>22}")
        boot_rows.append([name, ci.get("point"), ci.get("lo"), ci.get("hi"),
                          ci.get("n_places"), ci.get("n_boot"), args.boot, args.seed])
    print()

    if args.out:
        out = Path(args.out)
        write_rows(out / f"table_stats_{args.split}.csv",
                  ["config_a", "config_b", "question", "n_pairs", "n_places",
                   "mcnemar_place_b", "mcnemar_place_c", "p_place", "p_holm", "family_size",
                   "mcnemar_item_b", "mcnemar_item_c", "p_item",
                   "exact_diff", "exact_diff_lo", "exact_diff_hi", "n_boot", "seed"],
                  [[r["a"], r["b"], r["question"], r["n_pairs"], r["n_places"],
                    r["mc_place"]["b"], r["mc_place"]["c"], r["mc_place"]["p"], r["p_holm"], family,
                    r["mc_item"]["b"], r["mc_item"]["c"], r["mc_item"]["p"],
                    r["effect"].get("point"), r["effect"].get("lo"), r["effect"].get("hi"),
                    args.boot, args.seed] for r in results])
        write_rows(out / f"table_bootstrap_{args.split}.csv",
                  ["config", "exact", "lo", "hi", "n_places", "n_boot_valid", "n_boot", "seed"],
                  boot_rows)


# ---------------------------------------------------------------- entry

def _warn_uneven_coverage(scored: list[dict]) -> None:
    """Pooling a configuration over fewer datasets than its neighbours makes a
    table quietly incomparable, so say so loudly instead."""
    per_config = defaultdict(set)
    for r in scored:
        per_config[r.get("config_name")].add(r.get("dataset"))
    if not per_config:
        return
    all_datasets = set().union(*per_config.values())
    short = {c: sorted(all_datasets - ds) for c, ds in per_config.items()
             if ds != all_datasets}

    # Item counts too, not just which datasets are present: a run still being
    # written appears complete by dataset while being short by items, and a
    # row pooled over fewer items is quietly incomparable in exactly the same
    # way as a row pooled over fewer datasets.
    counts = defaultdict(Counter)
    for r in scored:
        counts[r.get("dataset")][r.get("config_name")] += 1
    partial = defaultdict(list)
    for dataset, per_cfg in counts.items():
        expected = max(per_cfg.values())
        for cfg, n in per_cfg.items():
            if n < expected:
                partial[cfg].append(f"{dataset} {n}/{expected}")

    if not short and not partial:
        return
    print("!" * 92)
    print("WARNING: uneven coverage. Rows pooled over different datasets or different")
    print("item counts are not comparable; treat these rows as partial.")
    for cfg in sorted(set(short) | set(partial)):
        bits = []
        if short.get(cfg):
            bits.append("missing datasets: " + ", ".join(short[cfg]))
        if partial.get(cfg):
            bits.append("short: " + ", ".join(sorted(partial[cfg])))
        print(f"  {cfg:28} {'; '.join(bits)}")
    print("!" * 92)
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command",
                   choices=["failures", "tables", "stats", "grounding", "all"])
    p.add_argument("--split", default="test", choices=["all", "dev", "test"])
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="directory for CSV output")
    p.add_argument("--boot", type=int, default=DEFAULT_BOOT,
                   help=f"bootstrap replicates (default {DEFAULT_BOOT})")
    p.add_argument("--seed", type=int, default=st.DEFAULT_SEED,
                   help=f"bootstrap seed (default {st.DEFAULT_SEED})")
    args = p.parse_args()

    paths = find_prediction_files(None, None)
    if args.configs:
        paths = [q for q in paths if q.stem.split("__", 1)[0] in args.configs]
    if args.datasets:
        paths = [q for q in paths if q.stem.split("__", 1)[1] in args.datasets]
    if not paths:
        raise SystemExit("no prediction files found")

    records = attach_split(load_predictions(paths))
    unsplit = [r for r in records if r.get("split") is None]
    if unsplit:
        by_ds = Counter(r.get("dataset") for r in unsplit)
        print(f"warning: {len(unsplit)} of {len(records)} record(s) are missing from the "
              f"split index (splits/item_split.csv stale or absent): "
              + ", ".join(f"{d} {n}" for d, n in sorted(by_ds.items())), file=sys.stderr)
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
    if not records:
        raise SystemExit(f"no records for split={args.split} in {len(paths)} file(s)")
    scored = [score_record(r) for r in records]
    meta = load_config_meta()
    print(f"loaded {len(scored)} scored record(s) from {len(paths)} file(s), "
          f"split={args.split}\n")
    _warn_uneven_coverage(scored)

    if args.command in ("failures", "all"):
        cmd_failures(scored, args, meta)
    if args.command in ("grounding", "all"):
        cmd_grounding(scored, args)
    if args.command in ("tables", "all"):
        cmd_tables(scored, args, meta)
    if args.command in ("stats", "all"):
        cmd_stats(scored, args)


if __name__ == "__main__":
    main()
