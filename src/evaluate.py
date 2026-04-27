import pandas as pd
import argparse
from dotenv import load_dotenv
load_dotenv()
from main import geolocate, geolocate_refined, geolocate_refined_search, geolocate_refined_gemma
from wikidata import save_caches

def _print_summary(label, sub_df):
    total = len(sub_df)
    if total == 0:
        print(f"{label}: (no rows)")
        return
    no_result = (sub_df["pred_qid"] == "").sum()
    exact = sub_df["exact"].sum()
    country = sub_df["country_match"].sum()
    continent = sub_df["continent_match"].sum()
    print(f"{label} (n={total})")
    print(f"  Exact location:  {exact:>3}/{total}  ({exact/total*100:5.1f}%)")
    print(f"  Country level:   {country:>3}/{total}  ({country/total*100:5.1f}%)")
    print(f"  Continent level: {continent:>3}/{total}  ({continent/total*100:5.1f}%)")
    print(f"  No result:       {no_result:>3}/{total}  ({no_result/total*100:5.1f}%)")

def evaluate(dataset_path, limit=None, mode="two_stage", output="eval_results.csv"):
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows_out = []
    llm_save = None
    # gemma and refined_gemma both use the LLM cache; load the saver lazily
    if mode in ("gemma", "refined_gemma"):
        from llm_gemma import geolocate_gemma, save_cache as _sc
        llm_save = _sc

    predict = {
        "two_stage": geolocate,
        "refined": geolocate_refined,
        "refined_search": geolocate_refined_search,
        "refined_gemma": geolocate_refined_gemma,
        "gemma": geolocate_gemma if mode in ("gemma", "refined_gemma") else None,
    }[mode]

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['location_name']}...", end=" ", flush=True)
        if i > 0 and i % 10 == 0:
            save_caches()
            if llm_save:
                llm_save()

        try:
            pred = predict(row["text"])
        except Exception as e:
            print(f"ERROR ({type(e).__name__})")
            pred = None

        if pred is None:
            print("NO RESULT")
            rows_out.append({
                "text": row["text"],
                "signal_type": row.get("signal_type", ""),
                "true_qid": row["qid"],
                "true_country": row["country"],
                "true_continent": row["continent"],
                "pred_qid": "",
                "pred_country": "",
                "pred_continent": "",
                "exact": False,
                "country_match": False,
                "continent_match": False,
            })
            continue

        exact = pred.get("qid", "") == row["qid"]
        country_match = pred.get("country", "").lower() == row["country"].lower() or exact
        continent_match = (
            pred.get("continent", "").lower() == row["continent"].lower()
            or country_match
        )

        status = "EXACT" if exact else ("COUNTRY" if country_match else ("CONTINENT" if continent_match else "MISS"))
        print(f"{status} | pred={pred.get('qid','')} ({pred.get('country','')}) | true={row['qid']} ({row['country']})")

        rows_out.append({
            "text": row["text"],
            "signal_type": row.get("signal_type", ""),
            "true_qid": row["qid"],
            "true_country": row["country"],
            "true_continent": row["continent"],
            "pred_qid": pred.get("qid", ""),
            "pred_country": pred.get("country", ""),
            "pred_continent": pred.get("continent", ""),
            "exact": exact,
            "country_match": country_match,
            "continent_match": continent_match,
        })

    results_df = pd.DataFrame(rows_out)

    print("\n" + "=" * 60)
    print("Pipeline performance overview")
    print("=" * 60)
    _print_summary("OVERALL", results_df)

    if "signal_type" in results_df.columns:
        print("\n" + "-" * 60)
        print("Per-signal breakdown")
        print("-" * 60)
        for signal in sorted(s for s in results_df["signal_type"].unique() if s != ""):
            _print_summary(f"Signal {signal}", results_df[results_df["signal_type"] == signal])
    print("=" * 60)

    if llm_save:
        llm_save()
    results_df.to_csv(output, index=False)
    print(f"Detailed results saved to {output}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sample_100.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows")
    parser.add_argument("--mode", choices=["two_stage", "refined", "refined_search", "refined_gemma", "gemma"], default="two_stage")
    parser.add_argument("--output", default="eval_results.csv", help="Path to write detailed results CSV")
    args = parser.parse_args()
    evaluate(args.dataset, limit=args.limit, mode=args.mode, output=args.output)
