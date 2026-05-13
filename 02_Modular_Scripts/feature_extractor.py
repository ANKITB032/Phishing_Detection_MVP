import pandas as pd
import re

INPUT  = "../01_Datasets/cleaned_data.csv"
OUTPUT = "../01_Datasets/features.csv"

SPECIAL_CHARS = "@-?=.#%+&_~/!"

def extract_features(url: str) -> dict:
    return {
        "url_length":        len(url),
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_at":            url.count("@"),
        "num_query_params":  url.count("?") + url.count("&"),
        "has_https":         int(url.startswith("https")),
        "has_ip":            int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", url))),
        "subdomain_depth":   max(0, len(url.split("/")[0].split(".")) - 2),
        "path_depth":        url.count("/"),
    }

if __name__ == "__main__":
    df = pd.read_csv(INPUT)

    features_df = pd.DataFrame(df["url"].map(extract_features).tolist())
    out = pd.concat([features_df, df["label"]], axis=1)

    out.to_csv(OUTPUT, index=False)
    print(f"Saved {len(out):,} rows × {len(out.columns)} cols → {OUTPUT}")
    print(out.drop(columns="label").describe().round(2).to_string())
