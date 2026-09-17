"""Analysis and reporting, reading the prediction files.

    python scripts/analyse.py failures --split test
    python scripts/analyse.py tables   --split test
    python scripts/analyse.py stats    --split test
    python scripts/analyse.py all      --split test --out results/summary

Reads only results/predictions/*.jsonl and splits/item_split.csv. Loads no
model, makes no network call, and imports nothing from src/ -- metrics come
from score.py and statistics from stats.py, so there is one definition of each.

Failure attribution
-------------------
For every wrong prediction, where did it break? The six levels are attributed
from the stage fields the runner recorded, not inferred from the shape of the
output:

  no_ner_span               no mention was detected at all
  span_not_linked           mentions detected, the linker resolved none
  wrong_entity              linked, but the gold place is nowhere in the
                            pipeline -- not among the linker's candidates and
                            not among the scored places
  wrong_candidate_selected  the linker HAD the gold place among its candidates
                            for some mention and chose another
  correct_candidate_not_first  the gold place was in the final scored list but
                            the reranker did not put it first
  returned_nothing          stages completed, no prediction came out

Two honest limits on this taxonomy:

* The datasets annotate a gold PLACE, not a gold entity. "Linked to the wrong
  entity" and "correct entity, wrong candidate" are therefore separated by
  whether the gold place appears in the linker's candidate set, which is the
  closest observable proxy.
* ner_free (LLM) configurations have no intermediate stages by construction,
  so their failures are attributed to `no_intermediate_stages`. That is not a
  gap in the analysis -- it is the finding that a single-call LLM pipeline
  cannot be debugged stage by stage.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (PREDICTIONS_DIR, aggregate, attach_split,      # noqa: E402
                   find_prediction_files, load_predictions, score_record)
import stats as st                                                # noqa: E402

# Report order. A configuration declares its tier in its config file; the tier
# is presentation metadata excluded from config_hash, so it orders rows without
# being able to change a prediction. Rank falls back to the paradigm on the
# records for anything with no config file left -- refined_gemma, whose
# configuration was removed but whose result is still reported.
TIER_RANK = {"0": 0, "1": 1, "2a": 2, "2b": 3, "3": 4, "4": 5, "5": 6,
             "combo": 7, "oracle": 8}
PARADIGM_RANK = {"baseline": 0, "geoparser": 1, "ner_dependent": 3,
                 "dense_retrieval": 4, "ner_free": 5, "grounded_llm": 6,
                 "routed": 7, "oracle": 8}


def load_tiers(config_dir=CONFIG_DIR):
    """{config name: tier} read straight from the YAML.

    Reading config files for row order is not the scorer loading a model: no
    model, no network, no import from src/. score.py stays free of this
    entirely; only the report ordering needs it.
    """
    tiers = {}
    try:
        import yaml
    except ImportError:
        return tiers
    for path in sorted(Path(config_dir).glob("*.yaml")):
        if path.name == "datasets.yaml":
            continue
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:                                  # noqa: BLE001
            continue
        if isinstance(raw, dict) and raw.get("name"):
            tiers[raw["name"]] = str(raw.get("tier", "") or "")
    return tiers


def sort_key(name, records, tiers):
    """Tier first, then paradigm, then name."""
    tier = tiers.get(name, "")
    if tier in TIER_RANK:
        return (TIER_RANK[tier], name)
    paradigms = {r.get("paradigm") for r in records}
    paradigm = sorted(paradigms)[0] if paradigms else ""
    return (PARADIGM_RANK.get(paradigm, 9), name)


LEVELS = ["no_ner_span", "span_not_linked", "wrong_entity",
          "gold_not_retrieved", "wrong_candidate_selected",
          "correct_candidate_not_first", "abstained", "selection_violation",
          "returned_nothing", "no_intermediate_stages"]

# Which pipeline stage a level presupposes. A level whose stage does not exist
# in an architecture is STRUCTURALLY IMPOSSIBLE there, and the table prints a
# dash rather than 0.0 -- "this cannot happen here" and "this never happened"
# are different statements and must not look alike.
LEVEL_REQUIRES = {
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


def stages_present(records):
    """Which stages an architecture actually has, read off its records."""
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

def _linker_candidate_qids(rec):
    """Every QID the linker had on the table, across all mentions."""
    out = set()
    for le in rec.get("linked_entities") or []:
        if le.get("qid"):
            out.add(le["qid"])
        for cand in le.get("candidates") or []:
            if cand.get("qid"):
                out.add(cand["qid"])
    return out


def attribute(rec):
    """Where a wrong prediction broke. None for a correct prediction."""
    if rec.get("exact"):
        return None

    gold = rec.get("place_qid")
    spans = rec.get("ner_spans") or []
    links = rec.get("linked_entities") or []
    places = rec.get("candidate_places") or []
    has_pred = rec.get("has_prediction")
    selection = rec.get("selection")
    paradigm = rec.get("paradigm")

    # --- retrieval-then-selection architectures (tier 5) -------------------
    # Attributed first, because for these the question is specifically whether
    # retrieval or selection failed, and that is the number the tier exists to
    # produce.
    if selection is not None:
        place_qids = [p.get("qid") for p in places]
        gold_present = bool(gold and gold in place_qids)
        if selection.get("violation"):
            return "selection_violation"
        if not places:
            return "gold_not_retrieved"      # nothing was retrieved at all
        if not gold_present:
            return "gold_not_retrieved"      # retrieval failure, not the model
        if selection.get("abstained"):
            return "abstained"               # declined with the gold in front of it
        return "wrong_candidate_selected"    # had it, chose another

    # --- dense retrieval (tier 3): no NER, no linking ----------------------
    if paradigm == "dense_retrieval":
        if not places:
            return "returned_nothing"
        place_qids = [p.get("qid") for p in places]
        if gold and gold in place_qids:
            return "correct_candidate_not_first"
        return "gold_not_retrieved"

    # No observable stage at all. Two different situations:
    #   * it still produced an answer -- a single-call LLM, the constant
    #     majority baseline, or an oracle row inherited from one of those.
    #     Nothing in the record can say where it went wrong, which is the
    #     finding rather than a gap.
    #   * it produced nothing either, so the configuration simply returned
    #     nothing.
    # Distinguishing them on `has_prediction` rather than on the paradigm keeps
    # baseline_majority (no NER stage, always answers) from being blamed for
    # "no NER span", which it cannot have.
    if not spans and not places:
        if has_pred:
            return "no_intermediate_stages"
        if rec.get("paradigm") in ("ner_dependent", "routed", "geoparser", "baseline"):
            return "no_ner_span"
        return "returned_nothing"

    if not spans:
        return "no_ner_span"

    linked_any = any(le.get("qid") for le in links)
    # A configuration that does no linking (wikidata_search retrieval, the
    # population baseline) reaches candidate places without ever linking, so
    # an empty link list is only a failure if nothing else was produced.
    if links and not linked_any and not places:
        return "span_not_linked"

    if not has_pred:
        return "returned_nothing"

    if gold:
        place_qids = [p.get("qid") for p in places]
        if gold in place_qids:
            # It was retrieved and scored, and something else won.
            return "correct_candidate_not_first"
        if gold in _linker_candidate_qids(rec):
            return "wrong_candidate_selected"

    return "wrong_entity"


def attribute_all(scored):
    for rec in scored:
        rec["failure_level"] = attribute(rec)
    return scored


def crosstab(scored, row_key, only_failures=True, all_records=None):
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
    stages = {key: stages_present(recs) for key, recs in rows_records.items()}
    return table, stages


def print_crosstab(result, row_label, percent=True):
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


def cmd_failures(scored, args):
    attribute_all(scored)
    failures = [r for r in scored if r.get("failure_level") is not None]
    print("=" * 100)
    print(f"FAILURE ATTRIBUTION  (split={args.split})")
    print(f"{len(failures)} wrong prediction(s) of {len(scored)} scored item(s)")
    print("=" * 100)
    print("Row percentages across the levels; each row sums to 100.\n")

    print("-- by configuration --")
    print_crosstab(crosstab(failures, "config_name", all_records=scored), "config")
    print("-- by paradigm --")
    print_crosstab(crosstab(failures, "paradigm", all_records=scored), "paradigm")
    print("-- by condition --")
    print_crosstab(crosstab(failures, "condition", all_records=scored), "condition")
    print("-- by paradigm x condition --")
    for rec in scored:
        rec["_pxc"] = f"{rec.get('paradigm')} / {rec.get('condition')}"
    print_crosstab(crosstab(failures, "_pxc", all_records=scored),
                   "paradigm / condition")
    print("-- by routing branch (routed configurations only) --")
    routed = [r for r in failures if r.get("routing_branch")]
    if routed:
        for rec in scored:
            if rec.get("routing_branch"):
                rec["_cxb"] = f"{rec.get('config_name')} / {rec.get('routing_branch')}"
        branch_records = [r for r in scored if r.get("routing_branch")]
        print_crosstab(crosstab(routed, "_cxb", all_records=branch_records),
                       "config / branch")
    else:
        print("(none)\n")

    if args.out:
        path = Path(args.out) / f"table_failure_attribution_{args.split}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["config", "dataset", "paradigm", "condition",
                        "routing_branch", "failure_level", "n"])
            counts = Counter((r.get("config_name"), r.get("dataset"),
                              r.get("paradigm"), r.get("condition"),
                              r.get("routing_branch"), r.get("failure_level"))
                             for r in failures)
            for key, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
                w.writerow([*key, n])
        print(f"wrote {path}")


# ------------------------------------------------------- grounding report

def cmd_grounding(scored, args):
    """Separate retrieval failure from selection failure for tier 5.

    This is the number that says whether grounding is working: of the items
    where the gold place WAS in the supplied candidate list, how often did the
    model actually pick it? A low rate means the model is the bottleneck; a low
    "gold supplied" rate means the retriever is.
    """
    grounded = [r for r in scored if r.get("selection") is not None]
    if not grounded:
        return
    print("=" * 100)
    print(f"GROUNDED SELECTION: retrieval failure vs selection failure "
          f"(split={args.split})")
    print("=" * 100)

    by_config = defaultdict(list)
    for rec in grounded:
        by_config[rec.get("config_name")].append(rec)

    header = (f"{'configuration':26} {'n':>5} {'gold supplied':>14} "
              f"{'selected gold':>14} {'abstained':>10} {'abst. w/ gold':>14} "
              f"{'violations':>11}")
    print(header)
    print("-" * len(header))
    for name, recs in sorted(by_config.items()):
        n = len(recs)
        supplied = [r for r in recs
                    if r.get("place_qid")
                    and r["place_qid"] in [c.get("qid") for c in (r.get("candidate_places") or [])]]
        sel = r_sel = 0
        for r in supplied:
            chosen = (r.get("selection") or {}).get("qid")
            if chosen and chosen == r.get("place_qid"):
                sel += 1
        abstained = sum(1 for r in recs if (r.get("selection") or {}).get("abstained"))
        abst_gold = sum(1 for r in supplied if (r.get("selection") or {}).get("abstained"))
        viol = sum(1 for r in recs if (r.get("selection") or {}).get("violation"))
        pct = lambda a, b: f"{a / b * 100:.1f}%" if b else "-"          # noqa: E731
        print(f"{name:26} {n:>5} "
              f"{f'{len(supplied)}/{n} ' + pct(len(supplied), n):>14} "
              f"{f'{sel}/{len(supplied)} ' + pct(sel, len(supplied)):>14} "
              f"{pct(abstained, n):>10} "
              f"{f'{abst_gold}/{len(supplied)}':>14} "
              f"{viol:>11}")
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
            supplied = [r for r in sub
                        if r.get("place_qid")
                        and r["place_qid"] in [c.get("qid") for c in (r.get("candidate_places") or [])]]
            sel = sum(1 for r in supplied
                      if (r.get("selection") or {}).get("qid") == r.get("place_qid"))
            pct = lambda a, b: f"{a / b * 100:.1f}%" if b else "-"      # noqa: E731
            print(f"{str(cond):12} {len(sub):>5} "
                  f"{f'{len(supplied)}/{len(sub)} ' + pct(len(supplied), len(sub)):>14} "
                  f"{f'{sel}/{len(supplied)} ' + pct(sel, len(supplied)):>14}")
        print()


# ---------------------------------------------------------------- tables

METRIC_COLS = ["n", "n_gold_qid", "exact", "country", "continent",
               "acc@161km", "median_km", "mean_km", "no_result_rate"]


def _row(label, records):
    m = aggregate(records)
    return label, m


def _print_metric_table(rows, label="configuration", note=None):
    header = (f"{label:36} {'n':>4} {'exact':>7} {'country':>8} {'cont':>7} "
              f"{'a@161':>7} {'med km':>8} {'mean km':>9} {'no-res':>7}")
    print(header)
    print("-" * len(header))
    for name, m in rows:
        if not m or m.get("n", 0) == 0:
            print(f"{name:36} {0:>4}")
            continue

        def f(key, pct=False, width=7):
            v = m.get(key)
            if v is None:
                return f"{'-':>{width}}"
            return f"{v * 100:>{width}.1f}" if pct else f"{v:>{width}.1f}"

        print(f"{name:36} {m['n']:>4} {f('exact', True)} {f('country', True, 8)} "
              f"{f('continent', True)} {f('acc@161km', True)} "
              f"{f('median_km', width=8)} {f('mean_km', width=9)} "
              f"{f('no_result_rate', True)}")
    if note:
        print(f"\n  {note}")
    print()


def _by_config(scored):
    groups = defaultdict(list)
    for r in scored:
        groups[r.get("config_name")].append(r)
    return groups


def cmd_tables(scored, args):
    groups = _by_config(scored)
    present = set(groups)

    def rows_for(names, suffix=""):
        return [_row(n + suffix, groups[n]) for n in names if n in present]

    # Grouped by the paradigm ON THE RECORDS, not by a hardcoded list of
    # configuration names. A list here would be a fifth copy of the mode
    # registry, and it would go stale the moment a configuration is added or
    # removed -- which is exactly what happened to the four copies that change
    # 8 removed.
    by_paradigm = defaultdict(list)
    for name, recs in groups.items():
        paradigms = {r.get("paradigm") for r in recs}
        by_paradigm[sorted(paradigms)[0] if paradigms else "unknown"].append(name)
    tiers_map = load_tiers()
    for names_ in by_paradigm.values():
        names_.sort(key=lambda n: sort_key(n, groups[n], tiers_map))

    baselines = by_paradigm.get("baseline", [])
    ner_dependent = by_paradigm.get("ner_dependent", [])
    dense = by_paradigm.get("dense_retrieval", [])
    ner_free = by_paradigm.get("ner_free", [])
    grounded = by_paradigm.get("grounded_llm", [])
    geoparser = by_paradigm.get("geoparser", [])
    routed = by_paradigm.get("routed", [])
    oracle = by_paradigm.get("oracle", [])

    print("=" * 100)
    print(f"TABLE 1  Baselines vs everything  -- is the task non-trivial?")
    print(f"(split={args.split})")
    print("=" * 100)
    # Tier order: what kind of signal each can consume, cheapest first.
    tiers = load_tiers()
    order = sorted(present, key=lambda n: sort_key(n, groups[n], tiers))
    _print_metric_table(rows_for(order),
                        note="A place is either dev or test, so the most frequent DEV "
                             "place cannot be a test gold place: baseline_majority is "
                             "structurally 0% exact on test. That is the split working, "
                             "not a defect.")

    print("=" * 100)
    print("TABLE 2  NER-dependent vs NER-free, by condition")
    print("          -- does having a mention-detection stage matter?")
    print("=" * 100)
    # "NER-free" and "LLM" are separate axes: dense retrieval has no mention
    # detection and no LLM, so it is the cell that lets the contrast separate
    # "no mention detection stage" from "parametric knowledge".
    tagged = ([(n, "[mention-detect]") for n in ner_dependent]
              + [(n, "[no-mention-detect]") for n in dense]
              + [(n, "[llm, no-mention-detect]") for n in ner_free]
              + [(n, "[llm, grounded]") for n in grounded])
    conditions = sorted({r.get("condition") for r in scored if r.get("condition")})
    for cond in conditions:
        print(f"-- condition {cond} --")
        rows = []
        for name, tag in tagged:
            if name not in present:
                continue
            sub = [r for r in groups[name] if r.get("condition") == cond]
            rows.append(_row(f"{name}  {tag}", sub))
        _print_metric_table(rows)
    print("-- pooled over conditions --")
    _print_metric_table([_row(f"{n}  {t}", groups[n]) for n, t in tagged
                         if n in present])

    print("=" * 100)
    print("TABLE 3  Linker comparison, NER held constant -- which linker?")
    print("=" * 100)
    _print_metric_table(
        rows_for(ner_dependent),
        note="Both use the same Flair ner-large over the same mentions (verified "
             "identical on 144/144 items), so what varies is only how a mention "
             "becomes a place: wbsearchentities retrieval + E5 rerank (two_stage) "
             "versus ReFinED disambiguation (refined). ReFinED here is a TWO-STAGE "
             "pipeline, not end-to-end: Flair detects, ReFinED disambiguates.")

    print("=" * 100)
    print("TABLE 4  Routed vs best component vs oracle")
    print("          -- does combining help, and how much headroom remains?")
    print("=" * 100)
    components = [n for n in ["refined", "gemma"] if n in present]
    rows = rows_for(components, "  [component]")
    rows += rows_for([n for n in grounded if n in present], "  [grounded]")
    rows += rows_for([n for n in routed if n in present], "  [routed]")
    rows += rows_for(oracle, "  [oracle]")
    _print_metric_table(rows,
                        note="The oracle is analysis-only: it picks per item using the "
                             "gold answer, so it is a ceiling, not a system. The gap "
                             "between a routed row and the oracle is the headroom the "
                             "routing idea still has.")

    print("=" * 100)
    print("APPENDIX  All configurations")
    print("=" * 100)
    _print_metric_table([_row(n, groups[n]) for n in sorted(present)])

    print("APPENDIX  All configurations x dataset")
    print("=" * 100)
    pairs = defaultdict(list)
    for r in scored:
        pairs[(r.get("config_name"), r.get("dataset"))].append(r)
    _print_metric_table([_row(f"{c} / {d}", v)
                         for (c, d), v in sorted(pairs.items())])

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"table_appendix_all_configs_{args.split}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config", "dataset"] + METRIC_COLS)
            w.writeheader()
            for (c, d), v in sorted(pairs.items()):
                m = aggregate(v)
                row = {"config": c, "dataset": d}
                for k in METRIC_COLS:
                    val = m.get(k)
                    row[k] = round(val, 4) if isinstance(val, float) else val
                w.writerow(row)
        print(f"wrote {path}")


# ---------------------------------------------------------------- statistics

# Explicit pairs, because which comparisons are worth making is an analysis
# decision, not something to enumerate automatically. Pairs naming a
# configuration with no predictions on disk are skipped silently.
COMPARISONS = [
    ("refined", "two_stage", "which linker: ReFinED vs search+rerank"),
    ("refined", "baseline_population", "does the linker beat the population floor"),
    ("gemma", "baseline_population", "does the LLM beat the population floor"),
    ("gemma", "groq_oss", "LLM model sensitivity"),
    ("refined", "gemma", "mention detection vs parametric knowledge"),
    ("dense_retrieval", "refined", "dense retrieval vs entity linking"),
    ("dense_retrieval", "gemma", "no mention detection: retrieval vs parametric"),
    ("grounded_gemma", "gemma4", "does grounding help (same model, matched control)"),
    ("grounded_gemma", "dense_retrieval", "grounded LLM vs its retriever alone"),
    ("grounded_gemma_linking", "grounded_gemma", "grounding from linking vs from dense"),
    ("grounded_gemma_linking", "refined", "grounded LLM vs its retriever alone"),
    ("routed_gemma", "refined", "does the rebuilt router beat its best component"),
    ("routed_gemma", "refined_gemma", "rebuilt gate vs the legacy output-reading gate"),
    ("oracle_refined_gemma", "routed_gemma", "headroom left for the router"),
]


def cmd_stats(scored, args):
    groups = _by_config(scored)
    print("=" * 100)
    print(f"PAIRED STATISTICS  (split={args.split})")
    print("=" * 100)
    print("Everything below is clustered by PLACE. Items about the same place are")
    print("not independent, so places are the unit paired by McNemar, the unit")
    print("ranked by Wilcoxon, and the unit resampled by the bootstrap.")
    print("Item-level McNemar is shown alongside for reference only; its pairs are")
    print("not independent and its p-value is anti-conservative.\n")

    for a_name, b_name, question in COMPARISONS:
        if a_name not in groups or b_name not in groups:
            continue
        pairs = st.pair_records(groups[a_name], groups[b_name])
        if not pairs:
            continue
        print(f"--- {a_name}  vs  {b_name} ---")
        print(f"    {question}")
        print(f"    {len(pairs)} paired item(s), "
              f"{len({p[0] for p in pairs})} place(s)")

        mc_place = st.mcnemar(pairs, lambda r: r.get("exact"), unit="place")
        mc_item = st.mcnemar(pairs, lambda r: r.get("exact"), unit="item")
        print(f"    exact match, McNemar by place : b={mc_place['b']} c={mc_place['c']} "
              f"p={st.fmt_p(mc_place['p'])}")
        print(f"    exact match, McNemar by item  : b={mc_item['b']} c={mc_item['c']} "
              f"p={st.fmt_p(mc_item['p'])}  (reference only)")

        eff = st.paired_effect_ci(pairs, st.exact_rate, n_boot=args.boot)
        print(f"    exact match difference        : "
              f"{st.fmt_ci(eff, pct=True, unit=' pp')}")

        eff161 = st.paired_effect_ci(pairs, st.acc_at(161.0), n_boot=args.boot)
        print(f"    acc@161km difference          : "
              f"{st.fmt_ci(eff161, pct=True, unit=' pp')}")

        wil = st.wilcoxon_distance(pairs, unit="place")
        if wil.get("p") is None:
            print(f"    error distance, Wilcoxon      : {wil.get('note')}")
        else:
            print(f"    error distance, Wilcoxon      : n={wil['n_pairs']} places "
                  f"p={st.fmt_p(wil['p'])} median diff={wil['median_diff']:.1f} km")
        effmed = st.paired_effect_ci(pairs, st.median_distance, n_boot=args.boot)
        print(f"    median distance difference    : {st.fmt_ci(effmed, unit=' km')}")
        print()

    print("=" * 100)
    print("BOOTSTRAP INTERVALS, resampled by place")
    print("=" * 100)
    header = f"{'configuration':30} {'exact %':>22} {'acc@161 %':>22} {'median km':>20}"
    print(header)
    print("-" * len(header))
    for name in sorted(groups):
        recs = groups[name]
        ex = st.cluster_bootstrap_ci(recs, st.exact_rate, n_boot=args.boot)
        a161 = st.cluster_bootstrap_ci(recs, st.acc_at(161.0), n_boot=args.boot)
        med = st.cluster_bootstrap_ci(recs, st.median_distance, n_boot=args.boot)
        print(f"{name:30} {st.fmt_ci(ex, pct=True):>22} "
              f"{st.fmt_ci(a161, pct=True):>22} {st.fmt_ci(med):>20}")
    print()


# ---------------------------------------------------------------- entry

def _warn_uneven_coverage(scored):
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
    counts = defaultdict(lambda: defaultdict(int))
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command",
                   choices=["failures", "tables", "stats", "grounding", "all"])
    p.add_argument("--split", default="test", choices=["all", "dev", "test"])
    p.add_argument("--configs", nargs="*", default=None)
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--out", default=None, help="directory for CSV output")
    p.add_argument("--boot", type=int, default=2000,
                   help="bootstrap replicates (default 2000)")
    args = p.parse_args()

    paths = find_prediction_files(None, None)
    if args.configs:
        paths = [q for q in paths if q.stem.split("__", 1)[0] in args.configs]
    if args.datasets:
        paths = [q for q in paths if q.stem.split("__", 1)[1] in args.datasets]
    if not paths:
        raise SystemExit("no prediction files found")

    records = attach_split(load_predictions(paths))
    if args.split != "all":
        records = [r for r in records if r.get("split") == args.split]
    scored = [score_record(r) for r in records]
    print(f"loaded {len(scored)} scored record(s) from {len(paths)} file(s), "
          f"split={args.split}\n")
    _warn_uneven_coverage(scored)

    if args.command in ("failures", "all"):
        cmd_failures(scored, args)
    if args.command in ("grounding", "all"):
        cmd_grounding(scored, args)
    if args.command in ("tables", "all"):
        cmd_tables(scored, args)
    if args.command in ("stats", "all"):
        cmd_stats(scored, args)


if __name__ == "__main__":
    main()
