"""Resolve the first LLM-only runs again with the country, calling no model.

    python scripts/posthoc_llm_resolver.py --split dev     # where the rule was fixed
    python scripts/posthoc_llm_resolver.py --split test    # applied once, afterwards

A record of how the resolver of gemma4 and qwen3_vl was corrected. Their first
runs (results/posthoc/original/) turned the model's answer into a Wikidata
item by searching for the city name alone and taking the first result. The
country the model gave was not used unless the city found nothing, so
"Worcester, South Africa" became the city in the United States. This script
resolves those stored answers (`llm_raw_response`) again with the country:

  1. search Wikidata for the city name, ten results;
  2. take the first result that has coordinates, lies in the country the model
     named (P17 equals the item the country name resolves to) and is not that
     country itself;
  3. if there is none, or the model named no country, keep the first answer.

It was the first step of the correction and shows how much the country alone
recovers from the same replies. The reported configurations now use the rule
in src/resolver.py (country and coordinates, prompt v2). The records written
here go to results/posthoc/ under the names gemma4_country_resolver and
qwen3_vl_country_resolver and are not reported configurations.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import wikidata as wd  # noqa: E402
from llm_common import parse_json_object  # noqa: E402
from llm_geo import _clean_name  # noqa: E402
from score import attach_split, load_predictions, score_record, summarise_metrics  # noqa: E402

ORIGINAL = REPO_ROOT / "results" / "posthoc" / "original"
OUT_DIR = REPO_ROOT / "results" / "posthoc"
CONFIGS = ["gemma4", "qwen3_vl"]
SEARCH_LIMIT = 10


def country_aware(city: str, country: str) -> dict | None:
    """The first search result for `city` that lies in `country`, or None."""
    country_hits = wd.search_entities(country, limit=1)
    if not country_hits:
        return None
    country_qid = country_hits[0]
    hits = wd.search_entities(city, limit=SEARCH_LIMIT)
    ents = wd.fetch_entities(hits) if hits else {}
    for qid in hits:
        ent = ents.get(qid) or {}
        coords = wd.get_coords_of(ent)
        if qid == country_qid or not coords or wd._claim_id(ent, "P17") != country_qid:
            continue
        label, continent = wd.resolve_country_continent({qid: ent}).get(qid, ("", ""))
        return {"qid": qid, "latlon": [coords[0], coords[1]], "country": label, "continent": continent}
    return None


def re_resolve(rec: dict) -> tuple[dict, bool]:
    """(the record with its prediction re-resolved, whether it changed)."""
    parsed = parse_json_object(rec.get("llm_raw_response") or "")
    if not parsed or parsed.get("refused"):
        return rec, False
    city, country = _clean_name(parsed.get("city")), _clean_name(parsed.get("country"))
    if not city or not country:
        return rec, False
    found = country_aware(city, country)
    if found is None or found["qid"] == rec.get("predicted_qid"):
        return rec, False
    return {**rec, "predicted_qid": found["qid"], "predicted_latlon": found["latlon"],
            "predicted_country": found["country"], "predicted_continent": found["continent"]}, True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True, choices=["dev", "test"])
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for config in CONFIGS:
        records = attach_split(load_predictions([ORIGINAL / f"{config}__synthetic.jsonl"]))
        records = [r for r in records if r.get("split") == args.split]
        new, changed = [], 0
        for i, rec in enumerate(records, 1):
            out, did = re_resolve(rec)
            changed += did
            new.append({**out, "config_name": f"{config}_country_resolver"})
            if i % 200 == 0:
                wd.save_caches()
                print(f"  {config}: {i}/{len(records)}", flush=True)
        wd.save_caches()
        path = OUT_DIR / f"{config}_country_resolver__synthetic__{args.split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for rec in new:
                f.write(json.dumps({k: v for k, v in rec.items() if k not in ("split", "place_key")},
                                   ensure_ascii=False) + "\n")
        before = [score_record(r) for r in records]
        after = [score_record(r) for r in new]
        gained = sum(a["exact"] and not b["exact"] for a, b in zip(after, before))
        lost = sum(b["exact"] and not a["exact"] for a, b in zip(after, before))
        print(f"\n{config} / {args.split}: {len(records)} posts, {changed} answers changed, "
              f"{gained} became correct, {lost} became wrong")
        for label, scored in (("reported", before), ("country-aware", after)):
            m = summarise_metrics(scored)
            print(f"  {label:14} exact {100 * m['exact']:5.1f}  country {100 * m['country']:5.1f}  "
                  f"no answer {100 * m['no_result_rate']:5.1f}")
        for cond in ("EXPLICIT", "NOISE_EXPLICIT", "B", "NOISE_B", "A", "NOISE_A"):
            b = [s for s in before if s["condition"] == cond]
            a = [s for s in after if s["condition"] == cond]
            print(f"    {cond:15} n={len(b):4}  {100 * summarise_metrics(b)['exact']:5.1f} -> "
                  f"{100 * summarise_metrics(a)['exact']:5.1f}")


if __name__ == "__main__":
    main()
