"""Look up, for every wrong answer, whether it lies inside the gold place.

    python scripts/build_containment.py          # adds the pairs not yet looked up
    python scripts/build_containment.py --all    # looks the pairs read up again
    python scripts/build_containment.py --predictions-dir results/archive/dev_rev3
                                                 # the pairs of runs written elsewhere

A configuration often answers with something that is in the right place but is
not the place: the station it found for a town, the lighthouse for an island.
Exact match counts that as wrong. scripts/score.py therefore reports a second
score, "exact or inside", and this script fetches what it needs: for each
(predicted item, gold place) pair in results/predictions/ where the two differ,
the relation between them on Wikidata. Runs written to another directory
(runner --out-dir) are read with --predictions-dir, which can be given more
than once; the pairs of every directory go into the one results/containment.csv.

    inside     the predicted item is located in the gold place: a chain of
               P131 (located in the administrative territorial entity), P276
               (location) or P706 (located on terrain feature) leads from it
               to the gold place
    contains   the other way round: the gold place lies in the predicted
               item, which is a larger unit (a prefecture named after its
               capital, a country). Reported as a percentage of items (the
               scorer's `coarser` column), never as correct.
    none       neither

The rule was fixed on the dev split on 2026-11-18, after a first look at dev
results and before any test result was scored.

Scoring stays offline: this is the only step that calls Wikidata, and it
writes results/containment.csv, which the scorer reads. Wikidata changes, so
each row carries the date it was fetched; rows already in the file are kept as
they are unless --all is given, which looks up again the pairs of the
directories read and keeps every other row.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS_DIR = REPO_ROOT / "results" / "predictions"
OUT = REPO_ROOT / "results" / "containment.csv"
ENDPOINT = "https://qlever.dev/api/wikidata"
BATCH = 40
FIELDS = ["predicted_qid", "gold_qid", "relation", "fetched_utc"]
PATH = "(wdt:P131|wdt:P276|wdt:P706)+"


def tls_context():
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def pairs_in_predictions(dirs=(PREDICTIONS_DIR,)) -> set[tuple[str, str]]:
    found = set()
    paths = []
    for d in dirs:
        d = Path(d) if Path(d).is_absolute() else REPO_ROOT / d
        if not d.is_dir():
            raise SystemExit(f"no predictions directory {d}")
        paths += sorted(d.glob("*.jsonl"))
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pred, gold = rec.get("predicted_qid"), rec.get("place_qid")
                if pred and gold and pred != gold:
                    found.add((pred, gold))
    return found


def ancestors(qids: list[str]) -> dict[str, set[str]]:
    """{item: everything a chain of P131, P276 or P706 leads to from it}."""
    sparql = ("PREFIX wd: <http://www.wikidata.org/entity/> PREFIX wdt: <http://www.wikidata.org/prop/direct/> "
              "SELECT ?a ?x WHERE { VALUES ?a { " + " ".join("wd:" + q for q in qids) + " } ?a " + PATH + " ?x }")
    request = urllib.request.Request(ENDPOINT + "?" + urllib.parse.urlencode({"query": sparql}),
                                     headers={"Accept": "application/sparql-results+json",
                                              "User-Agent": "containment-lookup/1.0"})
    rows = None
    for attempt in range(8):           # the endpoint answers 429 when asked too fast
        try:
            with urllib.request.urlopen(request, timeout=300, context=tls_context()) as response:
                rows = json.load(response)["results"]["bindings"]
            break
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 7:
                raise
            time.sleep(min(120, 5 * 2 ** attempt))
    found: dict[str, set[str]] = {q: set() for q in qids}
    for row in rows:
        found[row["a"]["value"].rsplit("/", 1)[-1]].add(row["x"]["value"].rsplit("/", 1)[-1])
    return found


def save(known: dict) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(known[k] for k in sorted(known))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="look up every pair again")
    parser.add_argument("--predictions-dir", action="append", default=None,
                        help="directory of prediction files to read (repeatable; "
                             "default results/predictions)")
    args = parser.parse_args()
    dirs = args.predictions_dir or [PREDICTIONS_DIR]

    # Rows of pairs outside the directories read are always kept: --all looks
    # up again the pairs of those directories only.
    known: dict[tuple[str, str], dict] = {}
    if OUT.exists():
        with OUT.open(encoding="utf-8", newline="") as f:
            known = {(r["predicted_qid"], r["gold_qid"]): r for r in csv.DictReader(f)}
    pairs = pairs_in_predictions(dirs)
    todo = sorted(pairs if args.all else pairs - set(known))
    print(f"{len(known)} pairs known, {len(todo)} to look up")
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    # One question per item, not per pair: what each predicted item and each
    # gold place lies in. A pair is "inside" when the gold place is among the
    # predicted item's ancestors, "contains" when it is the other way round.
    items = sorted({q for pair in todo for q in pair})
    up: dict[str, set[str]] = {}
    for start in range(0, len(items), BATCH):
        up.update(ancestors(items[start:start + BATCH]))
        print(f"  {min(start + BATCH, len(items))}/{len(items)} items", flush=True)
        time.sleep(0.5)
    for pred, gold in todo:
        relation = "inside" if gold in up[pred] else "contains" if pred in up[gold] else "none"
        known[(pred, gold)] = {"predicted_qid": pred, "gold_qid": gold, "relation": relation, "fetched_utc": stamp}
    save(known)
    counts = {}
    for row in known.values():
        counts[row["relation"]] = counts.get(row["relation"], 0) + 1
    print(f"wrote {OUT} ({counts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
