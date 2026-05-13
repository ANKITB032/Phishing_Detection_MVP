import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

INPUT      = "../01_Datasets/features.csv"
MODEL_OUT  = "../04_Saved_Models/phishing_model.joblib"
FEATURES   = ["url_length","num_special_chars","num_dots","num_hyphens",
              "num_at","num_query_params","has_https","has_ip",
              "subdomain_depth","path_depth"]

df = pd.read_csv(INPUT)
X, y = df[FEATURES], df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", n_jobs=-1),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                   random_state=42, n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f"\n{'='*50}\n{name}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["benign","malicious"]))
    print(f"ROC-AUC: {auc:.4f}")
    results[name] = (auc, model)

best_name, (best_auc, best_model) = max(results.items(), key=lambda x: x[1][0])
joblib.dump(best_model, MODEL_OUT)
print(f"\n✓ Best model: {best_name} (AUC={best_auc:.4f}) → saved as '{MODEL_OUT}'")
