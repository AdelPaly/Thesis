"""Error by layer: which stage of a configuration's chain an item was lost in.

    python scripts/error_layers.py                     # all items
    python scripts/error_layers.py --split test
    python scripts/error_layers.py --out results/summary

Reads results/predictions/*.jsonl (through score.py, so "correct" is score.py's
`exact`: predicted_qid == place_qid) and configs/*.yaml (for which stages a
configuration has). Loads no model, makes no network call.

Every item gets exactly one layer, checked in this order:

  error                   the runner recorded an exception (not a selection
                          violation)
  selection_violation     grounded LLM answered outside its candidate list
                          (stored in `error` with a "selection violation" prefix)
  correct                 exact QID match
  no_mention              mention-based configuration with no usable mention.
                          detail: no_span (NER/recogniser found nothing),
                          no_link (spans, none linked to a QID), no_place
                          (linked/retrieved, nothing resolved to a place)
  gold_not_in_candidates  candidates exist, the gold QID is in none of them
                          (candidate_places, linked QIDs, linker candidate
                          lists). detail: predicted | abstained
  wrong_selection         the gold QID was available and another was picked.
                          detail: ranker (in candidate_places, not first --
                          for the geoparser "first" is reading order),
                          resolve (linked to the gold QID, but it never became
                          a candidate place),
                          linker (only in a linker's candidate list: the linker
                          chose another entity for the mention), selector
                          (grounded LLM chose another candidate)
  abstained               no prediction and no error with nothing upstream to
                          blame. detail: gold_in_candidates (grounded LLM
                          declined with the gold in front of it), no_candidates
                          (retriever returned nothing), no_answer (no stages:
                          LLM refused / unresolvable)
  wrong                   configuration with no candidate stage (LLM only,
                          majority constant) predicted another place

Configurations without candidates (ner_free, majority_class) can only be
correct / wrong / abstained / error.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

from score import (attach_split, find_prediction_files, load_predictions, row_order,
                   score_record)
from analyse import load_config_meta

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "summary"

LAYERS = ["correct", "no_mention", "gold_not_in_candidates", "wrong_selection",
          "abstained", "wrong", "selection_violation", "error"]
SHORT = {"correct": "correct", "no_mention": "no_ment", "gold_not_in_candidates": "gold_absent",
         "wrong_selection": "wrong_sel", "abstained": "abstain", "wrong": "wrong",
         "selection_violation": "violat", "error": "error"}
ALL = "ALL"
VIOLATION_PREFIX = "selection violation"


def _linked_qids(rec: dict) -> set[str]:
    return {le["qid"] for le in rec.get("linked_entities") or [] if le.get("qid")}


def _linker_alternatives(rec: dict) -> set[str]:
    return {c["qid"] for le in rec.get("linked_entities") or []
            for c in le.get("candidates") or [] if c.get("qid")}


def _place_qids(rec: dict) -> list:
    return [c.get("qid") for c in rec.get("candidate_places") or []]


def classify(rec: dict, stages: set[str]) -> tuple[str, str]:
    """(layer, detail) for one scored record. `stages` is the configuration's
    stage set from analyse.load_config_meta (ner, linking, candidates,
    selection)."""
    err = rec.get("error")
    if err is not None:
        if str(err).startswith(VIOLATION_PREFIX):
            return "selection_violation", ""
        return "error", ""
    if rec.get("exact"):
        return "correct", ""

    gold = rec.get("place_qid")
    has_pred = rec.get("has_prediction")
    spans = rec.get("ner_spans") or []
    places = _place_qids(rec)
    selection = rec.get("selection") or {}

    if "candidates" not in stages:
        return ("wrong", "") if has_pred else ("abstained", "no_answer")

    if "ner" in stages:
        if not spans:
            return "no_mention", "no_span"
        if not places:
            if "linking" in stages and not _linked_qids(rec):
                return "no_mention", "no_link"
            return "no_mention", "no_place"

    if not places:
        return "abstained", "no_candidates"

    in_places = gold in places
    in_linker = gold in _linked_qids(rec) or gold in _linker_alternatives(rec)
    if not (in_places or in_linker):
        return "gold_not_in_candidates", "predicted" if has_pred else "abstained"

    if not has_pred:
        if selection.get("abstained"):
            return ("abstained", "gold_in_candidates") if in_places else \
                   ("gold_not_in_candidates", "abstained")
        return "abstained", "no_answer"
    if "selection" in stages and in_places:
        return "wrong_selection", "selector"
    if in_places:
        return "wrong_selection", "ranker"
    # Linked to the gold QID but it never became a candidate place: resolution
    # lost it. Otherwise the gold was only one of the linker's alternatives.
    return "wrong_selection", "resolve" if gold in _linked_qids(rec) else "linker"


def config_stages(meta: dict) -> dict[str, set[str]]:
    """analyse.py's stage sets, plus `candidates` for the population baseline:
    it scores gazetteer hits, which analyse.py does not count as a stage."""
    out = {}
    for name, m in meta.items():
        stages = set(m.get("stages", set()))
        if name == "population_wikidata":
            stages.add("candidates")
        out[name] = stages
    return out


def classify_all(scored: list[dict], meta: dict) -> list[dict]:
    stage_sets = config_stages(meta)
    for rec in scored:
        stages = stage_sets.get(rec.get("config_name"), set())
        rec["layer"], rec["layer_detail"] = classify(rec, stages)
    return scored


def layer_table(scored: list[dict]) -> list[dict]:
    """Long rows: config, dataset, condition (and ALL), layer, detail, n, share
    of the group's items. Layers a configuration cannot reach are absent."""
    groups = defaultdict(list)
    for rec in scored:
        cond = rec.get("condition") or "-"
        groups[(rec["config_name"], rec["dataset"], cond)].append(rec)
        groups[(rec["config_name"], rec["dataset"], ALL)].append(rec)
    rows = []
    for (cfg, ds, cond), recs in groups.items():
        total = len(recs)
        counts = Counter((r["layer"], r["layer_detail"]) for r in recs)
        by_layer = Counter(r["layer"] for r in recs)
        for (layer, detail), n in counts.items():
            rows.append({"config": cfg, "dataset": ds, "condition": cond,
                         "layer": layer, "detail": detail or "", "n": n,
                         "n_layer": by_layer[layer], "n_items": total,
                         "share": round(n / total, 4),
                         "share_layer": round(by_layer[layer] / total, 4)})
    return rows


def _sort_rows(rows: list[dict], order: dict) -> list[dict]:
    def key(r):
        return (row_order(r["config"], order), r["dataset"],
                r["condition"] != ALL, r["condition"],
                LAYERS.index(r["layer"]), r["detail"])
    return sorted(rows, key=key)


def print_overview(scored: list[dict], order: dict, by_condition: bool) -> None:
    groups = defaultdict(list)
    for rec in scored:
        groups[(rec["config_name"], rec["dataset"], ALL)].append(rec)
        if by_condition:
            groups[(rec["config_name"], rec["dataset"], rec.get("condition") or "-")].append(rec)
    cols = [l for l in LAYERS if any(r["layer"] == l for r in scored)]
    head = (f"{'config / dataset / cond':44} {'n':>4} "
            + " ".join(f"{SHORT[c]:>11}" for c in cols) + f" {'| no_pred':>9}")
    print(head)
    print("-" * len(head))
    keys = sorted(groups, key=lambda k: (row_order(k[0], order), k[1], k[2] != ALL, k[2]))
    for cfg, ds, cond in keys:
        recs = groups[(cfg, ds, cond)]
        c = Counter(r["layer"] for r in recs)
        label = f"{cfg} / {ds}" if cond == ALL else f"    {cond}"
        print(f"{label:44} {len(recs):>4} "
              + " ".join(f"{c[l] / len(recs) * 100:>11.1f}" if c[l] else f"{'.':>11}"
                         for l in cols)
              + f" {sum(not r['has_prediction'] for r in recs) / len(recs) * 100:>9.1f}")
    print("  row % of items; '.' = none in that layer (or the layer is unreachable).")
    print("  no_pred = items with no prediction, whatever their layer (an abstention")
    print("  with the gold absent from the candidates counts as gold_absent).")


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["config", "dataset", "condition", "layer", "detail", "n", "n_layer",
              "n_items", "share", "share_layer"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def report(scored: list[dict], meta: dict, split: str, out: Path | None = DEFAULT_OUT,
           by_condition: bool = False) -> list[dict]:
    """Classify already-scored records, print the overview, write the CSV.
    analyse.py calls this with its own scored records."""
    scored = classify_all(scored, meta)
    order = {k: (v.get("tier"), v.get("variant", "")) for k, v in meta.items()}
    print(f"ERROR BY LAYER  (split={split}, {len(scored)} items; correct = exact QID)\n")
    print_overview(scored, order, by_condition)
    rows = _sort_rows(layer_table(scored), order)
    if out is not None:
        write_csv(rows, Path(out) / f"table_error_layers_{split}.csv")
    return rows


def run(split: str = "all", out: Path | None = DEFAULT_OUT,
        by_condition: bool = False) -> list[dict]:
    paths = find_prediction_files()
    if not paths:
        raise SystemExit("no prediction files found")
    records = attach_split(load_predictions(paths))
    if split != "all":
        records = [r for r in records if r.get("split") == split]
    return report([score_record(r) for r in records], load_config_meta(), split, out,
                  by_condition)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="all", choices=["all", "dev", "test"])
    p.add_argument("--out", default=str(DEFAULT_OUT),
                   help="directory for table_error_layers_<split>.csv ('' to skip)")
    p.add_argument("--by-condition", action="store_true",
                   help="print per-condition rows in the overview (always in the CSV)")
    args = p.parse_args()
    run(args.split, Path(args.out) if args.out else None, args.by_condition)


if __name__ == "__main__":
    sys.exit(main())
