Project: Phishing URL detection MVP (Python/pandas)
Data state:

Source: malicious_phish.csv — 651,191 rows, 2 columns (url, type)
Cleaned: cleaned_data.csv — 641,125 rows, 2 columns (url, label)
Label encoding: benign → 0, defacement/phishing/malware → 1
10,066 duplicates removed; 0 nulls found

Class balance: 428,080 benign (0) vs 213,045 malicious (1) — imbalanced ~2:1
Exact next step: Feature engineering — extract structured features from raw URL strings (e.g., URL length, number of dots, presence of IP address, special char counts, use of HTTPS, subdomain depth, TLD extraction) and build a feature_extractor.py module that outputs a numeric feature matrix ready for scikit-learn.