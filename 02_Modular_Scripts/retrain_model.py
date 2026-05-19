"""
retrain_model.py — PhishGuard v3.5 Model Retraining Script
============================================================
Retrains the Random Forest classifier with four features that fix blind
spots identified in v3.0 and confirmed via FN error analysis.

Features added beyond v3.0 baseline:
  - is_ip        : bool — host is a raw IPv4 address (e.g. http://213.190.128.11/paypal/)
  - path_brand   : bool — URL path contains a spoofed brand name
  - url_in_query : bool — query string embeds a second URL/domain (open redirect)
                          Identified from FN ranks 4 & 10: ?ref=http://legit.com/login
  - tld_risk     : int  — ordinal TLD risk score (0=low, 1=medium, 2=high)
                          Encodes .info/.biz/.name (medium) and .tk/.ml/.ga/.cf/.gq/
                          .xyz/.top/.club/.work/.icu/.buzz/.cam (high) phishing rates

Run:
    python3 retrain_model.py --input ../01_Datasets/malicious_phish.csv \
                             --output ../04_Saved_Models/phishing_model_v3_5.joblib

A note on StandardScaler + Random Forest:
    RF is invariant to monotonic feature transforms, so StandardScaler does
    not improve split quality. It IS included here because:
      (a) it is required by the task spec, and
      (b) it future-proofs the pipeline if a linear model (LR, SVM) is ever
          added to an ensemble — the scaler is saved inside the Pipeline
          object so inference is always consistent.
"""

import re
import argparse
import warnings
import joblib
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics         import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_MAP: dict[str, int] = {
    "benign":      0,
    "defacement":  1,
    "phishing":    1,
    "malware":     1,
}

SPECIAL_CHARS = "@-?=.#%+&_~/!"

# Top-20 brand names most commonly spoofed in phishing URL paths
PATH_BRANDS: list[str] = [
    "paypal", "microsoft", "apple", "google", "amazon",
    "facebook", "netflix", "instagram", "linkedin", "twitter",
    "dropbox", "github", "spotify", "adobe", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "ebay", "walmart",
]

_IP_RE     = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
_IP_IN_URL = re.compile(r"(\d{1,3}\.){3}\d{1,3}")

# Detects a second embedded URL in the query string (open redirect bait).
# e.g. ?ref=http://legit.com/login  or  ?next=//evil.com
_URL_IN_QUERY_RE = re.compile(
    r"(https?://|//|www\.)[a-z0-9\-]+\.[a-z]{2,}",
    re.IGNORECASE,
)

# TLD ordinal risk score derived from phishing-campaign abuse frequency.
# 0 = common/low-risk  |  1 = medium-risk  |  2 = high-risk (blocklist)
_TLD_RISK: dict[str, int] = {
    # medium risk
    ".info": 1, ".biz": 1, ".name": 1, ".mobi": 1, ".pro": 1,
    # high risk (same set as predictor.py SUSPICIOUS_TLDS)
    ".top": 2, ".xyz": 2, ".tk": 2, ".ml": 2, ".ga": 2,
    ".cf": 2, ".gq": 2, ".buzz": 2, ".club": 2, ".work": 2,
    ".icu": 2, ".cam": 2, ".rest": 2, ".surf": 2, ".monster": 2, ".sbs": 2,
}


# ── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(url: str) -> dict:
    """
    Extract all features from a single URL string.

    Structural features (carry over from v3.0):
        url_length, num_special_chars, num_dots, num_hyphens, num_at,
        num_query_params, has_https, subdomain_depth, path_depth

    Features added v3.5 (original):
        is_ip       — True if the host is a bare IPv4 address.
        path_brand  — True if the URL path contains a spoofed brand keyword.

    Features added v3.5 (FN error-analysis patch):
        url_in_query — Query string embeds a second URL/domain (open redirect).
                       Caught FN ranks 4 & 10: ?ref=http://us.battle.net/login/
        tld_risk     — Ordinal TLD risk score: 0=benign, 1=medium, 2=high-abuse.
                       Directly encodes .info (rank 2 FN) and the SUSPICIOUS_TLDS
                       blocklist already used by predictor.py heuristics.
    """
    try:
        parsed   = urlparse(url if "://" in url else f"https://{url}")
        hostname = (parsed.hostname or "").lower()
        path     = (parsed.path or "").lower()
        query    = (parsed.query or "").lower()
    except Exception:
        hostname = path = query = ""

    is_ip      = bool(_IP_RE.match(hostname)) or bool(_IP_IN_URL.search(hostname))
    path_lower = path + "/" + query
    path_brand = any(brand in path_lower for brand in PATH_BRANDS)

    # url_in_query: open redirect / URL embedding in query string
    url_in_query = bool(_URL_IN_QUERY_RE.search(query))

    # tld_risk: look up the last label of the hostname
    tld = "." + hostname.split(".")[-1] if "." in hostname else ""
    tld_risk = _TLD_RISK.get(tld, 0)

    return {
        # ── Structural (v3.0) ──────────────────────────────────────────────
        "url_length":        len(url),
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_at":            url.count("@"),
        "num_query_params":  url.count("?") + url.count("&"),
        "has_https":         int(url.startswith("https")),
        "subdomain_depth":   max(0, len(hostname.split(".")) - 2),
        "path_depth":        url.count("/"),
        # ── v3.5 original ─────────────────────────────────────────────────
        "is_ip":             int(is_ip),
        "path_brand":        int(path_brand),
        # ── v3.5 FN patch ─────────────────────────────────────────────────
        "url_in_query":      int(url_in_query),
        "tld_risk":          tld_risk,
    }


FEATURE_COLS: list[str] = list(extract_features("http://x.com"))


# ── Data Loading & Cleaning ──────────────────────────────────────────────────

def load_and_clean(csv_path: str) -> pd.DataFrame:
    print(f"[1/5] Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    required = {"url", "type"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required}. Found: {set(df.columns)}")

    before = len(df)
    df.dropna(subset=["url", "type"], inplace=True)
    df.drop_duplicates(subset=["url"], inplace=True)
    after = len(df)

    df["label"] = df["type"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)          # drop unknown type values
    df["label"] = df["label"].astype(int)

    print(f"    Rows: {before:,} raw -> {after:,} clean ({before - after:,} dropped)")
    print(f"    Label dist: {df['label'].value_counts().to_dict()}")
    return df


# ── Feature Matrix Construction ──────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print(f"[2/5] Extracting {len(FEATURE_COLS)} features from {len(df):,} URLs …")
    features = pd.DataFrame(df["url"].map(extract_features).tolist(), index=df.index)
    X = features[FEATURE_COLS]
    y = df["label"]
    print(f"    Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} cols")
    return X, y


# ── Model Training ───────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    StandardScaler → RandomForestClassifier wrapped in a sklearn Pipeline.
    The Pipeline is saved as the model artefact, so predict_url() passes
    raw feature DataFrames and scaling is applied automatically at inference.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,         # more trees → lower variance
            max_depth=None,           # full depth — RF handles overfitting via bagging
            min_samples_leaf=2,       # slight regularisation
            class_weight="balanced",  # corrects 2:1 benign/malicious imbalance
            random_state=42,
            n_jobs=-1,
        )),
    ])


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    print("\n[4/5] Evaluation on held-out test set (20%)")
    print("─" * 60)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    auc     = roc_auc_score(y_test, y_proba)

    print(classification_report(y_test, y_pred, target_names=["benign", "malicious"]))
    print(f"ROC-AUC : {auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix:")
    print(f"  True  Negatives (benign correct) : {tn:>8,}")
    print(f"  False Positives (false alarms)   : {fp:>8,}")
    print(f"  False Negatives (missed threats) : {fn:>8,}")
    print(f"  True  Positives (caught threats) : {tp:>8,}")

    # Feature importance — key deliverable for verifying is_ip weight
    rf         = pipeline.named_steps["rf"]
    importances = rf.feature_importances_
    feat_rank  = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)

    print(f"\nFeature Importance Ranking (Random Forest Gini):")
    print(f"  {'Feature':<22}  {'Importance':>10}")
    print(f"  {'─'*22}  {'─'*10}")
    for feat, imp in feat_rank:
        marker = "  ← NEW" if feat in ("is_ip", "path_brand") else ""
        marker = "  ← FN PATCH" if feat in ("url_in_query", "tld_risk") else marker
        print(f"  {feat:<22}  {imp:>10.4f}{marker}")


# ── Manual Spot-Check ────────────────────────────────────────────────────────

def spot_check(pipeline: Pipeline) -> None:
    """Verify the known failure cases that motivated the v3.5 retraining."""
    print("\n[5/5] Spot-check on previously failing URLs")
    print("─" * 60)

    cases = [
        ("http://213.190.128.11/paypal/",               1, "raw IP + brand in path"),
        ("http://192.168.1.1/login@evil.ru?redirect=1", 1, "raw IP + encoded redirect"),
        ("https://paypal.com/signin",                   0, "legit PayPal"),
        ("https://google.com",                          0, "legit Google"),
        ("http://paypa1-secure.login-verify.com/",      1, "typosquat + keywords"),
        ("https://amazon.com/dp/B09G9FPHY6",            0, "legit Amazon product"),
        # FN patch regression tests
        ("http://www.hsqxx.com/js/?us.battle.net/login/en/?ref=http",  1, "open redirect in query"),
        ("http://secure-verify.tk/account/update",      1, "high-risk TLD .tk"),
        ("http://calendarscripts.info/index.php?action=product", 1, "medium-risk TLD .info"),
    ]

    for url, expected_label, note in cases:
        feats = pd.DataFrame([extract_features(url)])[FEATURE_COLS]
        pred  = pipeline.predict(feats)[0]
        prob  = pipeline.predict_proba(feats)[0]
        conf  = prob[pred] * 100
        ok    = "✓" if pred == expected_label else "✗ WRONG"
        verdict = "Safe" if pred == 0 else "Phishing"
        print(f"  {ok}  [{note}]")
        print(f"     {url}")
        print(f"     → {verdict} ({conf:.1f}% confidence)")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PhishGuard v3.5 Model Retraining")
    parser.add_argument(
        "--input",  "-i",
        default="../01_Datasets/malicious_phish.csv",
        help="Path to the raw Kaggle CSV (url, type columns)",
    )
    parser.add_argument(
        "--output", "-o",
        default="../04_Saved_Models/phishing_model_v3_5.joblib",
        help="Output path for the trained Pipeline (.joblib)",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run 5-fold stratified cross-validation before final fit (slower)",
    )
    args = parser.parse_args()

    # 1. Load
    df = load_and_clean(args.input)

    # 2. Features
    X, y = build_feature_matrix(df)

    # 3. Split — stratified to preserve 2:1 class ratio
    print("[3/5] Splitting 80/20 (stratified, random_state=42)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"    Train: {len(X_train):,}   Test: {len(X_test):,}")

    # Optional cross-validation
    if args.cv:
        print("    Running 5-fold stratified CV …")
        cv_pipeline = build_pipeline()
        skf   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(cv_pipeline, X_train, y_train,
                                 cv=skf, scoring="roc_auc", n_jobs=-1)
        print(f"    CV ROC-AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    # Final fit on full training set
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 4. Evaluate
    evaluate(pipeline, X_test, y_test)

    # 5. Spot-check
    spot_check(pipeline)

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    joblib.dump(pipeline, args.output)
    print(f"\n✓ Model saved → {args.output}")
    print("  Update MODEL_PATH in predictor.py to point to the new .joblib file.")
    print("  Also update main.py version string to 3.5.0.")


if __name__ == "__main__":
    main()
