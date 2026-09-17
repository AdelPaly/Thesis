"""Oracle router -- analysis only.

    python scripts/oracle_router.py --components refined gemma
    python scripts/oracle_router.py --components refined gemma --name oracle_refined_gemma

For every item, take whichever component got it right. This is the ceiling for
the routing idea: the gap between a real router and this number is how much
headroom the routing idea has left, and if the gap is small the router is
already close to the best that switching between these components can do.

It is NOT a system and it is NOT a configuration. There is no
configs/oracle_*.yaml, because it cannot be run on an unlabelled input: it
chooses using the gold answer. It is computed from prediction files that
already exist, so it costs nothing, and it is written into
results/predictions/ in the same record format only so that the one scorer
reports it alongside everything else. `paradigm` is "oracle" to keep it
distinguishable from anything runnable, and `routing_branch` records which
component the oracle picked, so its choices can be compared with the real
gate branch-by-branch.

"Got it right" is a lexicographic preference, best first:

  1. exact QID match
  2. within 161 km
  3. smaller error distance
  4. correct country
  5. correct continent
  6. produced any prediction at all

Ties keep the component named first on the command line, so the ordering of
--components is the tie-break and is recorded in the output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from score import (PREDICTIONS_DIR, find_prediction_files, load_predictions,
                   score_record)


def _preference(scored):
    """Higher is better. -distance so that a smaller distance ranks higher; a
    missing distance ranks below every real distance but above nothing."""
    d = scored["distance_km"]
    return (
        1 if scored["exact"] else 0,
        1 if scored["within_161km"] else 0,
        -d if d is not None else float("-inf"),
        1 if scored["country_correct"] else 0,
        1 if scored["continent_correct"] else 0,
        1 if scored["has_prediction"] else 0,
    )


def build(components, dataset, name, directory=PREDICTIONS_DIR):
    """One oracle record per item the components share."""
    per_component = {}
    for comp in components:
        paths = find_prediction_files(comp, dataset, directory)
        if not paths:
            raise SystemExit(f"no predictions for component {comp!r} on {dataset!r}")
        per_component[comp] = {r["item_id"]: r for r in load_predictions(paths)}

    shared = set.intersection(*(set(v) for v in per_component.values()))
    missing = set().union(*(set(v) for v in per_component.values())) - shared
    if missing:
        print(f"  {dataset}: {len(missing)} item(s) not present for every "
              f"component, skipped")

    out = []
    for item_id in sorted(shared):
        best_comp, best_rec, best_key = None, None, None
        for comp in components:                       # command-line order = tie-break
            rec = per_component[comp][item_id]
            key = _preference(score_record(rec))
            if best_key is None or key > best_key:
                best_comp, best_rec, best_key = comp, rec, key
        out.append({
            **best_rec,
            "config_name": name,
            "config_hash": "oracle:" + "+".join(components),
            "paradigm": "oracle",
            "routing_branch": best_comp,
            "routing_rule": f"oracle over {', '.join(components)} -> {best_comp}",
        })
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", nargs="+", required=True,
                   help="configuration names to take the per-item best of")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--name", default=None,
                   help="output configuration name (default oracle_<components>)")
    p.add_argument("--out-dir", default=str(PREDICTIONS_DIR))
    args = p.parse_args()

    name = args.name or ("oracle_" + "_".join(args.components))
    out_dir = Path(args.out_dir)

    datasets = args.datasets
    if not datasets:
        datasets = sorted({p.stem.split("__", 1)[1]
                           for p in find_prediction_files(args.components[0], None,
                                                          out_dir)})
    if not datasets:
        raise SystemExit("no datasets found for the first component")

    for ds in datasets:
        records = build(args.components, ds, name, out_dir)
        path = out_dir / f"{name}__{ds}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        chosen = {}
        for rec in records:
            chosen[rec["routing_branch"]] = chosen.get(rec["routing_branch"], 0) + 1
        picks = ", ".join(f"{k}={v}" for k, v in sorted(chosen.items()))
        print(f"{path.name}: {len(records)} item(s)  oracle picks: {picks}")


if __name__ == "__main__":
    main()
