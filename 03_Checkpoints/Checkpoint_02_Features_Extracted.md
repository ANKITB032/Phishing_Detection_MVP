Project: Phishing URL detection MVP (Python/pandas → scikit-learn)

Data state:

cleaned_data.csv — 641,125 rows, 2 cols (url, label)
features.csv — 641,125 rows, 11 cols (10 numeric features + label)
Features: url_length, num_special_chars, num_dots, num_hyphens, num_at, num_query_params, has_https, has_ip, subdomain_depth, path_depth
All features are numeric (int/float); no nulls; ready to feed directly into sklearn
Class balance: 428,080 benign (0) vs 213,045 malicious (1) — ~2:1 imbalance still present in features.csv

Exact next step: model_trainer.py — load features.csv, split into train/test (stratified to preserve class ratio), train a baseline RandomForestClassifier, evaluate with classification_report + ROC-AUC, and save the model with joblib. Consider passing class_weight="balanced" to the classifier to handle the 2:1 imbalance without resampling.

