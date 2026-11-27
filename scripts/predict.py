"""Predict-only mode: run one configuration over a CSV that has only a text column.

    python scripts/predict.py --config refined --input posts.csv --out-dir my_predictions
    python scripts/predict.py --config gemma4 --input posts.csv --text-column body --out-dir out

For data with no gold location. There is no split, no scoring and no dataset
registry: only the text column of the input is read (pandas `usecols`), so no
gold column can reach the pipeline even if the file has one. Each
configuration runs as it is configured, including revision 3
(linking.place_rule all_places, the MISC fallback, prompt v4).

Writes, into --out-dir:

  <config>__<input stem>.predict.csv    one row per post: row, item_id, text,
      config, config_hash, code_rev, status, level, fallback, predicted_label,
      predicted_qid, lat, lon, country, continent, error
  <config>__<input stem>.predict.jsonl  the full record of each post (NER spans,
      links, candidate places, raw LLM reply, resolution), without gold fields

`status` says what happened to the post, in terms that fit every method:

  linkers, search, population baseline, geoparser:
      no_mention   the recogniser found no mention
      no_link      mentions, but none was linked to a Wikidata item
      no_place     linked (or searched), but nothing resolved to a place
      predicted
  LLM alone (gemma4, qwen3_vl): the resolution status of src/resolver.py,
      resolved | declined | unreadable | not_found | ambiguous; `level` says
      whether the answer is a place, a country or a continent
  grounded LLM on linked candidates: the linker statuses above when there was
      nothing to choose from, else predicted | abstained | violation
  any method: error (the item failed; `error` holds the message)

`fallback` (configurations with ner.fallback_types): "used" when no LOC/PER/ORG
mention gave a place and the MISC mentions were linked, "not_needed" when the
post had MISC mentions that were not needed, empty otherwise.

Configurations that need gold data stop before anything runs:
majority_class (the most frequent gold place of a labelled dev split) and
dense_retrieval / grounded_gemma (their candidate pool is built from the gold
places of configs/datasets.yaml and other configurations' answers,
src/dense_index.py).

Caches are the runner's (src/.wikidata_cache, src/.llm_cache, ...): they are
keyed by text, not by dataset, and hold no gold data.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(SRC))

from config import PipelineConfig, load_config   # noqa: E402
from keys import item_id                          # noqa: E402

log = logging.getLogger("predict")

COLUMNS = ["row", "item_id", "text", "config", "config_hash", "code_rev", "status", "level",
           "fallback", "predicted_label", "predicted_qid", "lat", "lon", "country", "continent",
           "error"]
LINKER_STATUSES = ("no_mention", "no_link", "no_place", "predicted")


def needs_gold(cfg: PipelineConfig) -> str | None:
    """Why this configuration cannot run without gold data, or None."""
    if cfg.paradigm == "baseline" and cfg.baseline.strategy == "majority_dev_place":
        return (f"{cfg.name} predicts the most frequent gold place of the dev split of a "
                "labelled dataset; data with no gold location has none")
    if cfg.paradigm == "dense_retrieval" or (cfg.paradigm == "grounded_llm"
                                             and cfg.grounded.candidate_source == "dense"):
        return (f"{cfg.name} retrieves from the dense pool (dense.pool = {cfg.dense.pool}), "
                "which is built from the gold places of every dataset in configs/datasets.yaml "
                "and the answers of other configurations (src/dense_index.py); it cannot run "
                "on data without gold until the pool is a fixed gazetteer")
    return None


def _linker_status(out) -> str:
    if not out.ner_spans:
        return "no_mention"
    if out.linked_entities and not any(l.get("qid") for l in out.linked_entities):
        # Geoparser entries without a QID still give a GeoNames candidate.
        if not out.candidate_places:
            return "no_link"
    if not out.candidate_places:
        return "no_place"
    return "predicted" if out.prediction else "no_place"


def status_of(cfg: PipelineConfig, out) -> tuple[str, str]:
    """(status, level) of one PipelineOutput; see the module docstring."""
    if out.error and not (out.selection or {}).get("violation"):
        return "error", ""
    if cfg.paradigm == "ner_free":
        res = out.resolution or {}
        return res.get("status") or "error", res.get("level") or ""
    if cfg.paradigm == "grounded_llm":
        sel = out.selection or {}
        if sel.get("no_candidates") or not out.candidate_places:
            status = _linker_status(out)
            return (status if status != "predicted" else "no_place"), ""
        if sel.get("violation"):
            return "violation", ""
        if sel.get("abstained") or not out.prediction:
            return "abstained", ""
        return "predicted", ""
    return _linker_status(out), ""


def fallback_of(out) -> str:
    marks = {l.get("fallback") for l in out.linked_entities or []}
    if "used" in marks:
        return "used"
    return "not_needed" if "not_needed" in marks else ""


def labels_for(qids: list[str]) -> dict[str, str]:
    """English labels of the predicted items (Wikidata, cached)."""
    from wikidata import fetch_entities
    ents = fetch_entities(sorted(set(q for q in qids if q)))
    return {q: e.get("labels", {}).get("en", {}).get("value", "") for q, e in ents.items()}


def read_texts(path: Path, column: str) -> list[str]:
    """The text column and nothing else."""
    header = pd.read_csv(path, nrows=0).columns
    if column not in header:
        raise SystemExit(f"{path} has no column {column!r} (columns: {list(header)}); "
                         "pass --text-column")
    df = pd.read_csv(path, usecols=[column], dtype=str, keep_default_na=False)
    return [t for t in df[column].tolist()]


def _number(value):
    return None if value is None else float(value)


def predict(config: str, input_path: Path, out_dir: Path, text_column: str = "text",
            limit: int | None = None, pipeline_factory=None) -> tuple[Path, Path]:
    cfg = load_config(config)
    reason = needs_gold(cfg)
    if reason:
        raise SystemExit(f"cannot run {cfg.name} in predict-only mode: {reason}")

    texts = read_texts(input_path, text_column)
    if limit is not None:
        texts = texts[:limit]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{cfg.name}__{Path(input_path).stem}.predict"
    csv_path, jsonl_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.jsonl"

    from runner import code_rev, models_of
    if pipeline_factory is None:
        from pipeline import Pipeline as pipeline_factory
    pipe = pipeline_factory(cfg, dataset=None)
    pipe.prepare([t for t in texts if t.strip()])

    rev = code_rev()
    results = []
    try:
        for i, text in enumerate(texts):
            out = pipe.run(text) if text.strip() else None
            results.append((i, text, out))
            log.info("[%d/%d] %s", i + 1, len(texts),
                     ((out.prediction or {}).get("qid") if out else None) or "-")
    finally:
        from wikidata import save_caches
        save_caches()
        pipe.save_llm_cache()

    labels = labels_for([(o.prediction or {}).get("qid") for _, _, o in results if o])
    with csv_path.open("w", encoding="utf-8", newline="") as fc, \
            jsonl_path.open("w", encoding="utf-8") as fj:
        writer = csv.DictWriter(fc, fieldnames=COLUMNS)
        writer.writeheader()
        for i, text, out in results:
            if out is None:
                row = {"row": i, "item_id": item_id(text), "text": text, "config": cfg.name,
                       "config_hash": cfg.config_hash, "code_rev": rev, "status": "no_text"}
                writer.writerow(row)
                fj.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            pred = out.prediction or {}
            status, level = status_of(cfg, out)
            qid = pred.get("qid")
            # The item's English label; without an item (a continent-only LLM
            # answer, a GeoNames entry with no Wikidata item) the candidate's
            # own name. Not `entity`: for a linker that is the mention text.
            label = (labels.get(qid) if qid else None) or pred.get("description") or ""
            row = {"row": i, "item_id": item_id(text), "text": text, "config": cfg.name,
                   "config_hash": cfg.config_hash, "code_rev": rev, "status": status,
                   "level": level, "fallback": fallback_of(out), "predicted_label": label or "",
                   "predicted_qid": qid or "", "lat": _number(pred.get("lat")),
                   "lon": _number(pred.get("lon")), "country": pred.get("country") or "",
                   "continent": pred.get("continent") or "", "error": out.error or ""}
            writer.writerow(row)
            fj.write(json.dumps({**row, "models": models_of(cfg), "paradigm": cfg.paradigm,
                                 "ner_spans": out.ner_spans,
                                 "linked_entities": out.linked_entities,
                                 "candidate_places": out.candidate_places,
                                 "llm_raw_response": out.llm_raw_response,
                                 "selection": out.selection, "resolution": out.resolution},
                                ensure_ascii=False) + "\n")
    return csv_path, jsonl_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configuration name or configs/*.yaml path")
    p.add_argument("--input", required=True, type=Path, help="CSV with a text column")
    p.add_argument("--text-column", default="text")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--limit", type=int, default=None, help="only the first N rows")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    load_dotenv(SRC / ".env")
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    csv_path, jsonl_path = predict(args.config, args.input, args.out_dir, args.text_column,
                                   args.limit)
    print(f"wrote {csv_path}\n      {jsonl_path}")


if __name__ == "__main__":
    main()
