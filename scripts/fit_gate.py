"""Fit the routing gate's threshold on the dev split.

    python scripts/fit_gate.py --gate-config routed_gemma \
        --linking refined --llm gemma
    python scripts/fit_gate.py --gate-config routed_gemma --write

The gate's rule is hand-specified (see pipeline.route_from_features). The only
free parameter is `routing.ambiguity_threshold`: how many gazetteer candidates
a toponym needs before it counts as ambiguous. This sweeps that threshold on
the DEV split and reports what each value would do.

No model runs and no API call is made. The gate's input features are stored
per item (`routing_features`), and each branch's outcome is already in the
component prediction files, so a threshold is evaluated by re-deciding the
branch and reading the component's answer. That also means the threshold can
be refitted at any time for free.

The rule itself is imported from src/pipeline, not reimplemented, so a sweep
cannot silently evaluate a different rule from the one the pipeline runs.

Test is never touched here. `--split test` exists only so the fitted choice
can be reported once at the end, after it has been chosen on dev.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score import (aggregate, attach_split, find_prediction_files,   # noqa: E402
                   load_predictions, print_table, score_record)
from pipeline import route_from_features                             # noqa: E402


def _index(config_name, datasets=None):
    paths = find_prediction_files(config_name, None)
    if datasets:
        paths = [p for p in paths if p.stem.split("__", 1)[1] in datasets]
    if not paths:
        raise SystemExit(f"no prediction files for {config_name!r}")
    records = attach_split(load_predictions(paths))
    return {(r["item_id"], r["dataset"]): r for r in records}


def simulate(features_by_item, components, threshold, fallback="llm",
             has_geoparser=False, split="dev"):
    """Re-decide every item's branch at this threshold and take that
    component's stored answer. Returns (scored_records, branch_counts)."""
    out, counts = [], {}
    for key, rec in features_by_item.items():
        if split != "all" and rec.get("split") != split:
            continue
        features = rec.get("routing_features")
        if not features:
            continue
        branch, _why = route_from_features(
            features, fallback=fallback, ambiguity_threshold=threshold,
            has_geoparser=has_geoparser)
        counts[branch] = counts.get(branch, 0) + 1
        source = components.get(branch)
        if source is None or key not in source:
            # No component stands in for this branch; the item produces
            # nothing rather than being dropped.
            out.append(score_record({**rec, "predicted_qid": None,
                                     "predicted_latlon": None,
                                     "predicted_country": None,
                                     "predicted_continent": None,
                                     "routing_branch": branch,
                                     "error": f"no predictions for branch {branch}"}))
            continue
        out.append(score_record({**source[key], "routing_branch": branch}))
    return out, counts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate-config", default="routed_gemma",
                   help="routed configuration whose stored routing_features are used")
    p.add_argument("--linking", default="refined",
                   help="configuration standing in for the linking branch")
    p.add_argument("--llm", default="gemma",
                   help="configuration standing in for the LLM branch")
    p.add_argument("--geoparser", default=None,
                   help="configuration standing in for the geoparser branch")
    p.add_argument("--datasets", nargs="*", default=None)
    p.add_argument("--thresholds", nargs="*", type=int,
                   default=[1, 2, 3, 4, 5, 6, 8, 10])
    p.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    p.add_argument("--write", action="store_true",
                   help="write the best dev threshold into the gate config file")
    args = p.parse_args()

    features = _index(args.gate_config, args.datasets)
    components = {"linking": _index(args.linking, args.datasets),
                  "llm": _index(args.llm, args.datasets)}
    if args.geoparser:
        components["geoparser"] = _index(args.geoparser, args.datasets)

    n_side = sum(1 for r in features.values() if r.get("split") == args.split
                 or args.split == "all")
    print(f"gate features from : {args.gate_config}")
    print(f"linking branch     : {args.linking}")
    print(f"llm branch         : {args.llm}")
    print(f"split              : {args.split} ({n_side} items)\n")

    rows, best = [], None
    for t in args.thresholds:
        scored, counts = simulate(features, components, t,
                                  has_geoparser=bool(args.geoparser),
                                  split=args.split)
        if not scored:
            continue
        m = aggregate(scored)
        label = (f"threshold={t}  " +
                 " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        rows.append((label, m))
        # Chosen on exact match, then acc@161km, then mean error. Fitted on dev.
        key = (m["exact"] or 0, m["acc@161km"] or 0, -(m["mean_km"] or 1e9))
        if best is None or key > best[0]:
            best = (key, t, m, counts)

    print_table(rows, group_label="gate setting")

    if best:
        _key, t, m, counts = best
        print(f"\nbest on {args.split}: ambiguity_threshold={t}  "
              f"exact={(m['exact'] or 0)*100:.1f}%  "
              f"acc@161={(m['acc@161km'] or 0)*100:.1f}%  "
              f"branches: {dict(sorted(counts.items()))}")
        if args.write:
            path = REPO_ROOT / "configs" / f"{args.gate_config}.yaml"
            text = path.read_text(encoding="utf-8")
            import re
            new_text, n = re.subn(r"(?m)^(  ambiguity_threshold: )\d+$",
                                  rf"\g<1>{t}", text)
            if n != 1:
                raise SystemExit(f"could not find ambiguity_threshold in {path}")
            path.write_text(new_text, encoding="utf-8")
            print(f"wrote ambiguity_threshold: {t} to {path.relative_to(REPO_ROOT)}")
            print("note: this changes the config_hash, so the routed "
                  "configuration must be re-run")


if __name__ == "__main__":
    main()
