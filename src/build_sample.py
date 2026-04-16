"""Build a 100-row sample stratified by signal type and source dataset."""
import pandas as pd

SEED = 42
PER_SIGNAL_PER_SOURCE = {"A": 17, "B": 17, "C": 16}  # 50 + 50 = 100


def build():
    j = pd.read_csv("journalistic_dataset.csv")
    s = pd.read_csv("social_media_dataset.csv")
    j["source"] = "journalistic"
    s["source"] = "social_media"

    parts = []
    for df in (j, s):
        for sig, n in PER_SIGNAL_PER_SOURCE.items():
            sub = df[df["signal_type"] == sig].sample(n=n, random_state=SEED)
            parts.append(sub)

    sample = pd.concat(parts).reset_index(drop=True)
    sample.to_csv("sample_100.csv", index=False)
    print(f"Wrote sample_100.csv ({len(sample)} rows)")
    print(sample.groupby(["source", "signal_type"]).size())


if __name__ == "__main__":
    build()
