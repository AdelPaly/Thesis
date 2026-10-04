"""Generate a synthetic dataset through OpenRouter from the `prompt` spec.

The spec at the repository root is sent as the system prompt. Generation is
split into batches of (signal type x continent), 8 entries each by default,
matching the spec's continent distribution (6 continents x 3 signal types x 8
= 144 rows per dataset). Each batch is told which locations the dataset
already uses so the model doesn't repeat them.

The run is resumable: batches already present in the output CSV (counted by
signal_type + continent) are skipped, so an interrupted run can be restarted
with the same command.

Alongside the CSV it writes:
  <out>.raw.jsonl   every request/response pair (raw text + OpenRouter meta)
  <out>.meta.json   generation metadata per spec section 9.1 (model, date,
                    spec sha256); wikidata_verification_date is left for you
                    to fill in after running scripts/dataset_qids.py validate

The model is OPENROUTER_MODEL from src/.env unless --model is given.

Usage:
    python scripts/generate_synthetic.py --format social_media --out src/social_media_synthetic.csv
    python scripts/generate_synthetic.py --format journalistic --out src/journalistic_synthetic.csv
    python scripts/generate_synthetic.py --format social_media --out trial.csv \
        --signals A --continents Europe --per-batch 2        # small trial run

Rows are only checked structurally here (schema, enums, QID shape, referent
count). Run dataset_qids.py validate on the output afterwards, then dataset_coords.py add
for latitude/longitude.
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from llm_openrouter import OpenRouterClient, OpenRouterError, parse_json_reply  # noqa: E402

SPEC_PATH = PROJECT_ROOT / "prompt"

COLUMNS = ["text", "qid", "location_name", "country", "continent",
           "signal_type", "ambiguous_name", "other_referents"]
CONTINENTS = ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"]
SIGNALS = ["A", "B", "C"]
FORMATS = {
    "social_media": "Social Media post",
    "journalistic": "Journalistic Article",
}

BATCH_PROMPT = """Generate {n} entries for the {format_label} dataset.

Signal type: {signal}
Continent: {continent}

Follow the specification exactly, including the quality control checklist for
this signal type. Each entry must use a different target location.
{avoid}
Respond with a single JSON object and nothing else:
{{"entries": [{{"text": "...", "qid": "Q...", "location_name": "...", "country": "...",
"continent": "{continent}", "signal_type": "{signal}", "ambiguous_name": "...",
"other_referents": "referent 1; referent 2; referent 3"}}]}}

For signal type A, ambiguous_name and other_referents must be empty strings."""


def load_existing(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_meta(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def validate_entry(entry, signal, continent):
    """Structural checks only. Returns a list of problems; empty means accepted."""
    if not isinstance(entry, dict):
        return ["not an object"]
    problems = []
    missing = [c for c in COLUMNS if c not in entry]
    if missing:
        problems.append(f"missing {missing}")
    qid = str(entry.get("qid", ""))
    if not (qid.startswith("Q") and qid[1:].isdigit()):
        problems.append(f"bad qid {qid!r}")
    if entry.get("signal_type") != signal:
        problems.append(f"signal_type {entry.get('signal_type')!r} != {signal}")
    if entry.get("continent") != continent:
        problems.append(f"continent {entry.get('continent')!r} != {continent}")
    if not str(entry.get("text", "")).strip():
        problems.append("empty text")
    if signal in ("B", "C"):
        if not str(entry.get("ambiguous_name", "")).strip():
            problems.append("empty ambiguous_name")
        refs = entry.get("other_referents", "")
        if isinstance(refs, str):
            refs = refs.split(";")
        if len([r for r in refs if str(r).strip()]) < 3:
            problems.append("fewer than 3 other_referents")
    return problems


def normalise(entry):
    row = {c: entry.get(c, "") for c in COLUMNS}
    if isinstance(row["other_referents"], list):
        row["other_referents"] = "; ".join(map(str, row["other_referents"]))
    return {c: ("" if v is None else str(v).strip()) for c, v in row.items()}


def append_rows(path, rows):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Generate a synthetic dataset via OpenRouter.")
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", help="override OPENROUTER_MODEL")
    ap.add_argument("--per-batch", type=int, default=8)
    ap.add_argument("--signals", nargs="+", default=SIGNALS, choices=SIGNALS)
    ap.add_argument("--continents", nargs="+", default=CONTINENTS, choices=CONTINENTS)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="re-ask a batch up to this many times if it comes back short/invalid")
    ap.add_argument("--no-json-mode", action="store_true",
                    help="don't send response_format=json_object (for models that reject it)")
    args = ap.parse_args()

    load_dotenv(PROJECT_ROOT / "src" / ".env")
    load_dotenv(PROJECT_ROOT / ".env")

    spec =SPEC_PATH.read_text(encoding="utf-8")
    client = OpenRouterClient(model=args.model)
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    raw_path = out.with_suffix(".raw.jsonl")
    meta_path = out.with_suffix(".meta.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = load_existing(out)
    served_models = set()
    print(f"model={client.model}  format={args.format}  existing rows={len(rows)}")

    for signal in args.signals:
        for continent in args.continents:
            have = sum(1 for r in rows
                       if r["signal_type"] == signal and r["continent"] == continent)
            need = args.per_batch - have
            if need <= 0:
                print(f"[{signal}/{continent}] complete ({have}), skipping")
                continue

            for attempt in range(1, args.max_attempts + 1):
                used = sorted({r["location_name"] for r in rows})
                avoid = (f"Do not use any of these locations, already in the dataset: "
                         f"{', '.join(used)}.\n" if used else "")
                prompt = BATCH_PROMPT.format(n=need, format_label=FORMATS[args.format],
                                             signal=signal, continent=continent, avoid=avoid)
                print(f"[{signal}/{continent}] requesting {need} (attempt {attempt})...",
                      end="", flush=True)
                try:
                    text, meta = client.complete(prompt, system_prompt=spec,
                                                 temperature=args.temperature,
                                                 max_tokens=args.max_tokens,
                                                 json_mode=not args.no_json_mode)
                except OpenRouterError as e:
                    print(f" failed: {e}")
                    if attempt == args.max_attempts:
                        raise
                    continue

                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        "signal_type": signal, "continent": continent,
                        "prompt": prompt, "response": text, "meta": meta,
                    }, ensure_ascii=False) + "\n")
                if meta.get("model"):
                    served_models.add(meta["model"])

                try:
                    parsed = parse_json_reply(text)
                    entries = parsed.get("entries", []) if isinstance(parsed, dict) else parsed
                except (ValueError, AttributeError) as e:
                    print(f" unparseable reply ({e}); finish_reason={meta.get('finish_reason')}")
                    continue

                accepted = []
                seen = {r["location_name"] for r in rows}
                for entry in entries:
                    problems = validate_entry(entry, signal, continent)
                    row = None if problems else normalise(entry)
                    if row and row["location_name"] in seen:
                        problems = [f"duplicate location {row['location_name']!r}"]
                    if problems:
                        print(f"\n    rejected: {'; '.join(problems)}", end="")
                        continue
                    seen.add(row["location_name"])
                    accepted.append(row)
                    if len(accepted) == need:
                        break

                append_rows(out, accepted)
                rows.extend(accepted)
                need -= len(accepted)
                usage = meta.get("usage") or {}
                print(f" +{len(accepted)} rows (tokens in/out: "
                      f"{usage.get('prompt_tokens')}/{usage.get('completion_tokens')})")
                if need <= 0:
                    break
            else:
                print(f"[{signal}/{continent}] still short by {need} after "
                      f"{args.max_attempts} attempts; rerun to top up")

    meta_out = load_meta(meta_path)
    meta_out.update({
        "format": args.format,
        "requested_model": client.model,
        "served_models": sorted(set(meta_out.get("served_models", [])) | served_models),
        "generation_date": datetime.now(timezone.utc).date().isoformat(),
        "spec_file": "prompt",
        "spec_sha256": hashlib.sha256(spec.encode("utf-8")).hexdigest(),
        "temperature": args.temperature,
        "per_batch": args.per_batch,
        "rows": len(rows),
        "wikidata_verification_date": meta_out.get("wikidata_verification_date"),
    })
    meta_path.write_text(json.dumps(meta_out, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(rows)} rows in {out}\nmetadata: {meta_path}\nraw log:  {raw_path}")


if __name__ == "__main__":
    main()
