#!/usr/bin/env python3
"""
Check every planned anchor (clue, spare clue, distractor) against Wikidata
and repair the plan before anything is generated.

    python validate_anchors.py            # check, repair output/item_plan.jsonl, write the report
    python validate_anchors.py --dry-run  # check and report only

An anchor is rejected when (2026-11-02, user, after smoke13):

    elsewhere      it is filed under another place as well, one that is neither
                   inside the place nor above it (Goyang Zaicro FC: P131 Goyang,
                   Gimpo, Ansan, Iksan; P159 Goyang). The post would be about
                   Goyang as much as Ansan.
    defunct        a club, company or other body that has been dissolved (P576)
                   and so cannot do anything today
    past_event     a dated one-off occurrence (P585, P580 or P582: the 2000
                   Summer Olympics, a 7th-century expedition). It cannot be a
                   present venue or a present cause.
    blocked_class  a religious site or body, an atrocity, a battle, war or
                   military operation, a memorial, camp or prison
    names_other_place  its name contains the name of another place the dataset
                   holds (a place, its other names, regions, countries, place
                   clues): "Seoul Semiconductor" for Ansan
    alias_leak     one of its names (label or alias; for people the label only)
                   contains a name of the place or a region it lies in
                   ("Sydney 2000")

People and species are not checked for "elsewhere": people live in many places
and species range widely; the planner already linked them to the place.
Distractors (noise) are checked for everything but "elsewhere" and
"alias_leak", which are about the place the post is set in.

Repair: a rejected clue is replaced by the item's first spare clue that passes;
a rejected distractor by the best-known clue of the same donor place that
passes, else by one from another donor the planner allows (plan.noise_donors,
in an order fixed by the item id), so a noise item is never lost while its
plain twin is kept (2026-11-04, user). An item left without enough anchors, and every item of a
condition no longer in config plan.conditions, is removed from the plan and
written to output/plan_rejections.jsonl. The old plan is archived first. Writes
output/anchor_checks.jsonl (one verdict per place and anchor), which
finalise.check also reads, so a post written from a bad anchor fails even if
the plan was not repaired.

Queries go through ../reform_places.py (QLever, cached in ../out/cache).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import reform_places as rp   # noqa: E402  (sparql, batched, block classes)

import generate as gen       # noqa: E402
import plan                  # noqa: E402

CACHE = str(HERE.parent / "out" / "cache")
LOCATING = ["P131", "P159", "P276", "P115"]           # where a thing is said to be
DATED = ["P585", "P580", "P582"]                       # point in time, start, end
OCCURRENCE = "Q1190554"
MILITARY = ["Q831663", "Q645883", "Q180684", "Q350604", "Q124734", "Q10931", "Q188055", "Q2001676",
            "Q1384277", "Q124490", "Q115431196"]
# military campaign, military operation, conflict, armed conflict, rebellion, revolution, siege, offensive,
# military expedition, violence, violent conflict
BLOCKED = sorted(set(rp.BLOCK_CLASSES + rp.ATROCITY + rp.CLUE_BLOCK + MILITARY))
NOT_LOCATED = {"person", "species"}                    # entity types skipped for "elsewhere"
CHECKS_PATH = HERE / "output" / "anchor_checks.jsonl"


def contains(text: str, form: str) -> bool:
    return len(plan.fold(form)) > 2 and plan.contains_word(text, form)


def wikidata_facts(qids: list[str]) -> dict[str, dict[str, Any]]:
    facts: dict[str, dict[str, Any]] = {q: {"classes": set(), "dated": False, "dissolved": False,
                                            "located": set(), "occurrence": False} for q in qids}
    for r in rp.batched(lambda b: f"SELECT DISTINCT ?i ?x WHERE {{ VALUES ?i {{{rp.V(b)}}} "
                                  f"VALUES ?x {{{rp.V(BLOCKED + [OCCURRENCE])}}} ?i wdt:P31/wdt:P279* ?x }}", qids, CACHE):
        if r["x"] == OCCURRENCE:
            facts[r["i"]]["occurrence"] = True
        else:
            facts[r["i"]]["classes"].add(r["x"])
    for r in rp.batched(lambda b: f"SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{rp.V(b)}}} "
                                  f"{{ ?i wdt:P585 [] }} UNION {{ ?i wdt:P580 [] }} UNION {{ ?i wdt:P582 [] }} }}", qids, CACHE):
        facts[r["i"]]["dated"] = True
    for r in rp.batched(lambda b: f"SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i wdt:P576 [] }}", qids, CACHE):
        facts[r["i"]]["dissolved"] = True
    for prop in LOCATING:
        for r in rp.batched(lambda b: f"SELECT DISTINCT ?i ?x ?xl WHERE {{ VALUES ?i {{{rp.V(b)}}} ?i wdt:{prop} ?x. "
                                      f'OPTIONAL {{ ?x rdfs:label ?xl FILTER(LANG(?xl) = "en") }} }}', qids, CACHE):
            facts[r["i"]]["located"].add((r["x"], r.get("xl") or r["x"]))
    return facts


def place_name_index(inputs: dict[str, dict[str, Any]], places_csv: Path) -> dict[str, set[str]]:
    """Every place name the dataset itself holds -> the places it belongs to: each
    place's label, other names, regions and country names (place_inputs.jsonl,
    places.csv) and the clues that are themselves places. No Wikidata query
    (2026-11-02, user)."""
    index: dict[str, set[str]] = {}
    # one-word names need 5 letters: "Salt" (Jordan) and "Finn" matched "Salt Spring dollar", "Finn Brothers"
    add = lambda name, q: index.setdefault(name, set()).add(q) if name and len(name) >= (4 if " " in name else 5) else None  # noqa: E731
    for q, r in inputs.items():
        for name in [*(r.get("name_forms") or []), *(r.get("region_forms") or []), *(r.get("country_forms") or [])]:
            add(name, q)
        for c in r.get("clues") or []:
            if c.get("is_toponym"):
                add(c["clue_label"], q)
    with places_csv.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            add(row["label"], row["i"])
    return index


def names_in_labels(labels: list[str], index: dict[str, set[str]]) -> dict[str, list[tuple[set[str], str]]]:
    """Label -> [(places the name belongs to, name)] for every dataset place name
    written, capitalised, as whole words in it ("Seoul Semiconductor" -> Seoul)."""
    by_first: dict[str, list[str]] = {}
    for name in index:
        by_first.setdefault(plan.fold(name).split()[0], []).append(name)
    found: dict[str, list[tuple[set[str], str]]] = {}
    for label in labels:
        for w in {plan.fold(w) for w in re.findall(r"[^\W\d_][\w'’-]*", label) if w[0].isupper()}:
            for name in by_first.get(w, ()):
                if plan.fold(name) != plan.fold(label) and plan.contains_word(label, name):
                    found.setdefault(label, []).append((index[name], name))
    return found


def related(pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """The (x, place) pairs where x lies inside the place or the place inside x
    (P131 chain either way), or x is a country."""
    up: dict[str, set[str]] = {}
    nodes = sorted({q for pair in pairs for q in pair})
    for r in rp.batched(lambda b: f"SELECT DISTINCT ?i ?u WHERE {{ VALUES ?i {{{rp.V(b)}}} "
                                  f"{{ ?i (wdt:P131|wdt:P276)+ ?u }} UNION {{ ?i wdt:P17 ?u }} }}", nodes, CACHE):
        up.setdefault(r["i"], set()).add(r["u"])
    countries = {r["i"] for r in rp.batched(lambda b: f"SELECT DISTINCT ?i WHERE {{ VALUES ?i {{{rp.V(b)}}} "
                                                      f"?i wdt:P31 wd:Q6256 }}", nodes, CACHE)}
    return {(x, p) for x, p in set(pairs)
            if p in up.get(x, ()) or x in up.get(p, ()) or x in countries}


def judge_anchor(clue: dict[str, Any], place_qid: str, facts: dict[str, Any], near: set[tuple[str, str]],
                 leak_forms: list[str], noise: bool) -> tuple[str, str] | None:
    f = facts[clue["clue_qid"]]
    if f["classes"]:
        return "blocked_class", ",".join(sorted(f["classes"]))
    if f["occurrence"] and f["dated"]:
        return "past_event", "a dated one-off event"
    if f["dissolved"] and not clue.get("is_human"):
        return "defunct", "dissolved (P576)"
    if not clue.get("is_human"):
        other = [n for owners, n in f.get("place_names", []) if place_qid not in owners]   # own names: alias_leak
        if other:
            return "names_other_place", f'"{clue["clue_label"]}" contains the place name "{other[0]}"'
    if noise:
        return None
    kind = "person" if clue.get("is_human") else clue.get("entity_type")
    if kind not in NOT_LOCATED:
        other = sorted(label for x, label in f["located"] if x != place_qid and (x, place_qid) not in near)
        if other:
            return "elsewhere", "also filed under " + ", ".join(other[:4])
    for name in [clue["clue_label"], *([] if clue.get("is_human") else clue.get("aliases") or [])]:
        for form in leak_forms:
            if contains(name, form):
                return "alias_leak", f'"{name}" contains "{form}"'
    return None


def main(argv: list[str] | None = None) -> int:
    plan.configure_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=HERE / "config.yaml")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    config = gen.load_config(args.config)
    plan_path = Path(config["paths"]["item_plan"])
    items = gen.read_jsonl(plan_path)
    inputs = {r["place_qid"]: r for r in gen.read_jsonl(Path(config["paths"]["place_inputs"]))}
    conditions = set(config["plan"]["conditions"])

    def leak_forms(qid: str, item: dict[str, Any]) -> list[str]:
        r = inputs.get(qid, {})
        # The place's own names and the regions it lies in. Not the country: a national broadcaster or
        # university named after its country ("Television Malta") points to the country, not the place.
        return [item["place_name"], *(r.get("name_forms") or []), *(r.get("region_forms") or [])]

    anchors: list[tuple[str, dict[str, Any], bool]] = []   # (place the post is about, anchor, is noise)
    for i in items:
        anchors += [(i["place_qid"], c, False) for c in i["clues"] + (i.get("spare_clues") or [])]
        if i.get("noise_entity"):
            n = i["noise_entity"]
            anchors.append((i["place_qid"], n, True))
            anchors += [(i["place_qid"], {**c, "donor_place_qid": n["donor_place_qid"]}, True)
                        for c in inputs.get(n["donor_place_qid"], {}).get("clues") or []]
    qids = sorted({c["clue_qid"] for _, c, _ in anchors})
    print(f"{len(items)} items, {len(qids)} anchor entities: querying Wikidata", flush=True)
    facts = wikidata_facts(qids)
    labels = {c["clue_qid"]: c["clue_label"] for _, c, _ in anchors}
    name_index = place_name_index(inputs, Path(config["paths"]["places_csv"]))
    in_labels = names_in_labels(sorted(set(labels.values())), name_index)
    for q, label in labels.items():
        facts[q]["place_names"] = in_labels.get(label, [])
    near = related([(x, p) for p, c, noise in anchors if not noise for x, _ in facts[c["clue_qid"]]["located"]])

    verdicts: dict[tuple[str, str, bool], tuple[str, str] | None] = {}
    place_name = {i["place_qid"]: i["place_name"] for i in items}
    for p, c, noise in anchors:
        key = (p, c["clue_qid"], noise)
        if key not in verdicts:
            forms = leak_forms(p, {"place_name": place_name[p]})
            verdicts[key] = judge_anchor(c, p, facts, near, forms, noise)

    bad = lambda p, c, noise=False: verdicts.get((p, c["clue_qid"], noise))  # noqa: E731

    # Distractor repair draws on other donor places (2026-11-04, user: NOISE_A1 and
    # NOISE_B must cover the same places as A1 and B). Their clues are judged here,
    # with the same rules, when first needed.
    places = plan.load_places(Path(config["paths"]["places_csv"]), Path(config["paths"]["place_inputs"]))

    def fetch_facts(clues: list[dict[str, Any]]) -> None:
        new = sorted({c["clue_qid"] for c in clues} - set(facts))
        if not new:
            return
        facts.update(wikidata_facts(new))
        found = names_in_labels(sorted({c["clue_label"] for c in clues if c["clue_qid"] in new}), name_index)
        for c in clues:
            labels.setdefault(c["clue_qid"], c["clue_label"])
            if c["clue_qid"] in new:
                facts[c["clue_qid"]]["place_names"] = found.get(c["clue_label"], [])

    def distractor_options(item: dict[str, Any], donor_qid: str) -> list[dict[str, Any]]:
        """The donor's clues that may stand in for the item's distractor, best known first (as
        the planner draws them: plan.Place.distractor_clues, no toponyms)."""
        place, current = places[item["place_qid"]], item["noise_entity"]["clue_qid"]
        return sorted((c for c in place.distractor_clues(places[donor_qid])
                       if c["clue_qid"] != current and not c.get("is_toponym")),
                      key=lambda c: -(c.get("sitelinks") or 0))

    def donor_order(item: dict[str, Any]) -> list[str]:
        """The item's own donor first, then every other donor the planner allows (plan.noise_donors:
        not in the post's country) in an order fixed by the item id."""
        first = item["noise_entity"]["donor_place_qid"]
        rank = lambda q: hashlib.sha256(f'{item["item_id"]}:{q}'.encode()).hexdigest()  # noqa: E731
        rest = sorted((q for q in plan.noise_donors(places[item["place_qid"]], places) if q != first), key=rank)
        return ([first] if first in places else []) + rest

    def replacement_distractor(item: dict[str, Any]) -> dict[str, Any] | None:
        p = item["place_qid"]
        forms = leak_forms(p, {"place_name": place_name[p]})
        for qid in donor_order(item):
            options = distractor_options(item, qid)
            fetch_facts(options)
            for c in options:
                key = (p, c["clue_qid"], True)
                if key not in verdicts:
                    verdicts[key] = judge_anchor(c, p, facts, near, forms, True)
                if verdicts[key] is None:
                    return refresh(qid, plan.noise_entity_from(places[p], places[qid], c))
        return None

    # One Wikidata round for the likely fallback donors, rather than one per donor.
    stuck = [i for i in items if i["condition"] in conditions and i.get("noise_entity")
             and i["place_qid"] in places and bad(i["place_qid"], i["noise_entity"], True)
             and all(bad(i["place_qid"], c, True) for c in distractor_options(i, i["noise_entity"]["donor_place_qid"]))]
    fetch_facts([c for i in stuck for q in donor_order(i)[1:11] for c in distractor_options(i, q)])
    # Role and description as place_inputs.py now gives them (2026-11-02: role by most occupations, the
    # description's closing bracket kept); the plan copied them when it was written.
    fresh = {(r["place_qid"], c["clue_qid"]): c for r in inputs.values()
             for c in r.get("clues") or []}

    def refresh(p: str, c: dict[str, Any]) -> dict[str, Any]:
        f = fresh.get((p, c["clue_qid"]))
        return {**c, "role": f.get("role"), "description": f.get("description")} if f else c

    refreshed = 0
    for i in items:
        before = json.dumps([i["clues"], i.get("spare_clues"), i.get("noise_entity")], sort_keys=True)
        i["clues"] = [refresh(i["place_qid"], c) for c in i["clues"]]
        i["spare_clues"] = [refresh(i["place_qid"], c) for c in i.get("spare_clues") or []]
        if i.get("noise_entity"):
            i["noise_entity"] = refresh(i["noise_entity"]["donor_place_qid"], i["noise_entity"])
        refreshed += before != json.dumps([i["clues"], i.get("spare_clues"), i.get("noise_entity")], sort_keys=True)
    kept, rejected, changes = [], [], Counter()
    for item in items:
        p = item["place_qid"]
        if item["condition"] not in conditions:
            rejected.append({"item_id": item["item_id"], "place_qid": p, "condition": item["condition"],
                             "reason": "condition_dropped", "detail": "not in config plan.conditions"})
            changes["condition_dropped"] += 1
            continue
        spares = [c for c in item.get("spare_clues") or [] if not bad(p, c)]
        if item["condition"] in ("B", "NOISE_B"):
            spares = [c for c in spares if not c.get("linked_to_rival")]
        clues, swaps, failed = [], [], None
        for c in item["clues"]:
            v = bad(p, c)
            if v is None:
                clues.append(c)
                continue
            new = next((s for s in spares if s["clue_qid"] not in {x["clue_qid"] for x in clues + item["clues"]}), None)
            if new is None:
                failed = (v[0], f'{c["clue_label"]}: {v[1]}')
                break
            spares.remove(new)
            clues.append(new)
            swaps.append({"from_qid": c["clue_qid"], "from_label": c["clue_label"], "to_qid": new["clue_qid"],
                          "to_label": new["clue_label"], "reason": v[0]})
        noise = item.get("noise_entity")
        if failed is None and noise and bad(p, noise, True):
            v = bad(p, noise, True)
            new = replacement_distractor(item) if p in places else None
            if new is None:
                failed = (v[0], f'distractor {noise["clue_label"]}: {v[1]}')
            else:
                swaps.append({"from_qid": noise["clue_qid"], "from_label": noise["clue_label"], "to_qid": new["clue_qid"],
                              "to_label": new["clue_label"], "reason": v[0], "noise": True,
                              "from_donor": noise["donor_place_qid"], "to_donor": new["donor_place_qid"]})
                changes["distractor_new_donor"] += new["donor_place_qid"] != noise["donor_place_qid"]
                noise = new
        if failed:
            rejected.append({"item_id": item["item_id"], "place_qid": p, "condition": item["condition"],
                             "reason": "anchor_" + failed[0], "detail": failed[1]})
            changes["dropped_" + failed[0]] += 1
            continue
        if swaps:
            changes["repaired"] += 1
            item = {**item, "clues": clues, "spare_clues": spares, "noise_entity": noise,
                    "anchor_swaps": item.get("anchor_swaps", []) + swaps,
                    "role": ";".join(dict.fromkeys(c.get("role") or "other" for c in clues)) if clues else item.get("role")}
        kept.append(item)

    checks = [{"place_qid": p, "clue_qid": q, "noise": noise, "status": v[0] if v else "ok", "reason": v[1] if v else ""}
              for (p, q, noise), v in sorted(verdicts.items())]
    for c in checks:
        c["clue_label"] = labels[c["clue_qid"]]
    plan.write_jsonl(CHECKS_PATH, checks)
    print("anchor verdicts:", dict(Counter(c["status"] for c in checks if not c["noise"])),
          "| distractors:", dict(Counter(c["status"] for c in checks if c["noise"])))
    print("plan changes:", dict(changes), f"| {len(kept)} of {len(items)} items kept"
          f" | role or description refreshed on {refreshed}")
    if args.dry_run:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = plan_path.parent / "archive" / f"item_plan.{stamp}.jsonl"
    archive.parent.mkdir(exist_ok=True)
    shutil.copyfile(plan_path, archive)
    plan.write_jsonl(plan_path, kept)
    rej_path = Path(config["paths"]["plan_rejections"])
    old = gen.read_jsonl(rej_path) if rej_path.exists() else []
    plan.write_jsonl(rej_path, old + [{**r, "stage": "validate_anchors", "utc": stamp} for r in rejected])
    print(f"old plan archived as {archive}\nwrote {plan_path}, {rej_path}, {CHECKS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
