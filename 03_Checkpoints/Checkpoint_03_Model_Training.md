Project: Phishing URL detection MVP
Pipeline state:

data_cleaner.py → cleaned_data.csv (641,125 rows, url + label)
feature_extractor.py → features.csv (641,125 rows, 10 features + label)
train_model.py → phishing_model.joblib (Random Forest, 100 estimators, class_weight="balanced")

Saved model: phishing_model.joblib
Best model performance: Accuracy 91%, ROC-AUC 0.9648
Expected input for inference — exactly these 10 features in order:
url_length, num_special_chars, num_dots, num_hyphens, num_at,
num_query_params, has_https, has_ip, subdomain_depth, path_depth
Exact next step: predictor.py — load phishing_model.joblib + reuse extract_features() from feature_extractor.py to build a single-function inference interface: predict_url(url: str) -> dict returning {"label": 0|1, "confidence": float, "verdict": "benign"|"malicious"}. Optionally wrap in a minimal CLI (argparse) or FastAPI endpoint for the MVP demo.