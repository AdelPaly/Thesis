import pandas as pd
import argparse
from main import geolocate, geolocate_hybrid, geolocate_conditional, geolocate_refined, geolocate_routed, geolocate_gated, geolocate_coverage
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

def evaluate(dataset_path, limit=None, mode="original", output="eval_results.csv"):
    df = pd.read_csv(dataset_path)
    if limit is not None:
        df = df.head(limit)
    rows_out = []
    predict = {
        "hybrid": geolocate_hybrid,
        "conditional": geolocate_conditional,
        "original": geolocate,
        "refined": geolocate_refined,
        "routed": geolocate_routed,
        "gated": geolocate_gated,
        "coverage": geolocate_coverage,
    }[mode]

    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] {row['location_name']}...", end=" ", flush=True)
        if i > 0 and i % 10 == 0:
            save_caches()

        pred = predict(row["text"])

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

    results_df.to_csv(output, index=False)
    print(f"Detailed results saved to {output}")

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="journalistic_dataset.csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only first N rows")
    parser.add_argument("--mode", choices=["original", "hybrid", "conditional", "refined", "routed", "gated", "coverage"], default="routed")
    parser.add_argument("--output", default="eval_results.csv", help="Path to write detailed results CSV")
    args = parser.parse_args()
    evaluate(args.dataset, limit=args.limit, mode=args.mode, output=args.output)
