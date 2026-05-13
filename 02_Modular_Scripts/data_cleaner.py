import pandas as pd

INPUT  = "../01_Datasets/malicious_phish.csv"
OUTPUT = "../01_Datasets/cleaned_data.csv"

LABEL_MAP = {"benign": 0, "defacement": 1, "phishing": 1, "malware": 1}

def clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    df["label"] = df["type"].map(LABEL_MAP)
    df.drop(columns=["type"], inplace=True)
    return df

if __name__ == "__main__":
    df = clean(INPUT)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {len(df):,} rows → {OUTPUT}")
    print(df["label"].value_counts().rename({0: "benign", 1: "malicious"}))
