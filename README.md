# PhishGuard MVP — v4.0.0

AI-powered phishing URL detection combining a trained Random Forest classifier with a seven-rule deterministic Heuristic Override Engine and live HTML content analysis.

**Live demo:** [phishguard.ankitband.me](https://phishguard.ankitband.me) · **API:** [ab2403-phishguard-api.hf.space](https://ab2403-phishguard-api.hf.space)

---

## Architecture

```
URL Input
    │
    ├─ Step 0a: Infrastructure Abuse Pre-Check (free-hosting subdomain entropy/keyword scan)
    ├─ Step 0:  Trusted-Domain Whitelist Fast Path (50+ curated domains → 100% benign, early exit)
    │
    ├─ Rule 0 (Module L): Live HTML Title Brand Impersonation          ← EARLY SHORT-CIRCUIT
    │   └─ Fetches live page <title> (3 s timeout, 50 KB cap, browser User-Agent)
    │       If title contains a TARGET_BRAND keyword (PayPal, Netflix, Apple, Microsoft …)
    │       AND the URL domain ≠ official brand domain
    │       → INSTANT return: label=1, verdict=malicious, confidence=99.0%
    │       → Skips all remaining heuristic modules and ML inference entirely
    │
    ├─ Heuristic Modules (all run in parallel, populate threat_flags[])
    │   ├─ A  Typosquatting        Levenshtein ≤ 2 vs 20 brand names
    │   ├─ B  Suspicious TLD       27-entry blocklist (.tk .xyz .tech .live …)
    │   ├─ C  Phishing Keywords    30+ credential-harvesting terms in path/query
    │   ├─ D  URL Complexity       dot count > 5 or length > 100 chars
    │   ├─ E  IPFS Gateway         7 known gateways + CID extraction
    │   ├─ F  URL Shortener        12 shortener domains → HTTP HEAD expand → re-analyse
    │   ├─ G  Brand-in-Subdomain   brand label on untrusted eTLD+1 (facebook.evil.net)
    │   ├─ H  Hyphenated Creds     appieid-enable, drive-google-com patterns in hostname
    │   └─ I  Abused Free Host     21 platforms (000webhostapp, firebaseapp, netlify …)
    │
    ├─ ML Inference
    │   └─ Random Forest (13 features, class_weight=balanced) → label + P(benign) / P(malicious)
    │
    ├─ Trust Protocol
    │   └─ ML=malicious + zero heuristic flags → flip to benign (eliminates FP on clean URLs)
    │
    ├─ Heuristic Override Engine  ← core FN fix
    │   ├─ Rule 1  Typosquat            → label=1, conf≥88%
    │   ├─ Rule 2  Brand-in-subdomain   → label=1, conf≥90%
    │   ├─ Rule 3  Hyphenated creds     → label=1, conf≥87%
    │   ├─ Rule 4  Abused free host     → label=1, conf≥85%
    │   ├─ Rule 5  Suspicious TLD + keyword (compound) → label=1, conf≥83%
    │   ├─ Rule 6  Suspicious TLD (standalone)         → label=1, conf≥76%
    │   └─ Rule 7  Phishing keywords + ≥2 flags        → label=1, conf≥78%
    │
    └─ Final Verdict  {label, verdict, confidence%, reason, security_analysis{}}
```

---

## ML Model

| Property | Value |
|---|---|
| Algorithm | `RandomForestClassifier(class_weight='balanced', random_state=42)` |
| Tuning | `RandomizedSearchCV(n_iter=15, cv=StratifiedKFold(5), scoring='roc_auc')` |
| Pipeline | `StandardScaler → RandomForestClassifier` (saved as sklearn Pipeline) |
| Dataset | Kaggle malicious-phish.csv — 641,119 URLs (428K benign / 213K malicious) |
| Split | 80/20 stratified |
| ROC-AUC | **0.9645** |
| Accuracy | 90% |
| False Negatives (held-out) | 4,305 / 42,608 malicious (10.1%) — covered by Override Engine |

### Features (13)

| Feature | Description |
|---|---|
| `url_length` | Total character count |
| `num_special_chars` | Count of `@-?=.#%+&_~/!` |
| `num_dots` | `.` count |
| `num_hyphens` | `-` count |
| `num_at` | `@` count |
| `num_query_params` | `?` + `&` count |
| `has_https` | 1 if URL starts with `https` |
| `subdomain_depth` | Labels left of eTLD+1 |
| `path_depth` | `/` count |
| `is_ip` | 1 if host is a bare IPv4 address |
| `path_brand` | 1 if path/query contains a spoofed brand name |
| `url_in_query` | 1 if query string embeds a second URL (open redirect) |
| `tld_risk` | 0=benign / 1=medium (.info .biz) / 2=high-abuse (.tk .xyz .tech …) |

---

## API

Base URL: `https://ab2403-phishguard-api.hf.space`

### `POST /predict`

**Rate limit:** 5 requests / minute per IP · **Input limit:** 2048 chars

**Request**
```json
{ "url": "http://appieid-enable.com/verify" }
```

**Response — full pipeline**
```json
{
  "url": "http://appieid-enable.com/verify",
  "label": 1,
  "verdict": "malicious",
  "confidence": 90.9,
  "reason": "Hyphenated Impersonation: hostname segment 'appieid-enable' mimics a trusted brand via hyphen-obfuscation — a classic credential-harvesting pattern.",
  "security_analysis": {
    "live_content": { "title": null, "fetched": false, "error": "..." },
    "brand_title_impersonation": { "fired": false, "brand": null, "title": null, "flag": null },
    "threat_flags": ["Hyphenated credential domain: 'appieid-enable' mimics a brand via hyphen obfuscation."],
    "override_engine_fired": true,
    "override_confidence": 87.0,
    "typosquatting": { "is_typosquat": false },
    "hyphenated_creds": { "found": true, "matched": "appieid-enable" }
  }
}
```

**Response — Rule 0 short-circuit** (brand impersonation in live page title)
```json
{
  "url": "https://netflix-login.pages.dev/signin",
  "label": 1,
  "verdict": "malicious",
  "confidence": 99.0,
  "reason": "Brand Impersonation: Page title claims to be 'Netflix — Watch TV Shows Online' (contains 'netflix') but domain 'pages.dev' does not match official 'netflix.com' and is hosted on free platform 'pages.dev'.",
  "security_analysis": {
    "live_content": { "title": "Netflix — Watch TV Shows Online", "fetched": true, "error": null },
    "brand_title_impersonation": { "fired": true, "brand": "Netflix", "title": "...", "flag": "..." },
    "threat_flags": ["Brand Impersonation: ..."],
    "note": "Short-circuited at Rule 0 — brand impersonation in live page title."
  }
}
```

**429 Too Many Requests**
```json
{ "error": "Rate limit exceeded: 5 per 1 minute" }
```

**422 Unprocessable Entity**
```json
{ "detail": "URL exceeds maximum length of 2048 characters (3100 received)." }
```

---

### `POST /report`

Submit a user correction to the data feedback loop. Appends a row to `01_Datasets/user_reported_corrections.csv` (file is created with headers on first write). Intended for MLOps retraining of v5.0.

**Rate limit:** 10 requests / minute per IP

**Request**
```json
{
  "url": "https://example.com/safe-page",
  "correct_verdict": "benign",
  "comments": "False positive — legitimate site."
}
```

| Field | Type | Required | Values |
|---|---|---|---|
| `url` | string | ✓ | Any non-empty URL |
| `correct_verdict` | string | ✓ | `"benign"` or `"malicious"` |
| `comments` | string | — | Optional free-text note |

**Response (`201 Created`)**
```json
{
  "status": "accepted",
  "message": "Thank you — your correction has been recorded.",
  "data": {
    "timestamp": "2026-06-09T04:30:00+00:00",
    "url": "https://example.com/safe-page",
    "correct_verdict": "benign",
    "comments": "False positive — legitimate site."
  }
}
```

---

### `GET /health`

**Rate limit:** 60 requests / minute per IP

```json
{ "status": "ok", "model": "Random Forest (AUC 0.9645)" }
```

---

## Pydantic Response Schema

The `SecurityAnalysis` response model uses `model_config = {"extra": "allow"}` with all heuristic sub-fields typed `Optional[Any] = None`. This flexible schema supports three distinct response shapes without triggering a `ResponseValidationError`:

| Code path | Shape of `security_analysis` |
|---|---|
| Trusted-domain fast path | Plain string `"Bypassed via trusted domain whitelist"` |
| Rule 0 short-circuit | Dict with `live_content`, `brand_title_impersonation`, `threat_flags`, `note` only |
| Full pipeline | All 18+ heuristic module fields |

`PredictionResponse.security_analysis` is typed `Any = None` so FastAPI never attempts to coerce any of the three shapes into a fixed schema.

---

## Deployment

| Layer | Platform | Notes |
|---|---|---|
| Frontend (`index.html`) | Vercel | Static deploy via `vercel.json` |
| Backend (`main.py`) | Hugging Face Spaces | Docker container, port 7860 |
| Model (`*.joblib`) | Git LFS | ~440 MB, tracked via `.gitattributes` |

### Local development

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API:  http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Retraining

```bash
# Paths are __file__-anchored — works from any working directory
python 02_Modular_Scripts/retrain_model.py
# Output: 04_Saved_Models/phishing_model_v4_0.joblib
```

---

## Project Structure

```
Phishing_Detection_MVP/
├── main.py                    FastAPI backend (rate limiting, input validation, /predict, /report)
├── predictor.py               Full pipeline (whitelist → Rule 0 → heuristics → ML → override)
├── requirements.txt           Python dependencies
├── Dockerfile                 HF Spaces container (python:3.11-slim, port 7860)
├── vercel.json                Vercel static deploy config
├── index.html                 Frontend UI (IBM Plex fonts, Anime.js, report feedback loop UI)
├── 01_Datasets/
│   ├── malicious_phish.csv              Raw Kaggle dataset (651K URLs)
│   ├── cleaned_data.csv                 Deduplicated, labelled
│   └── user_reported_corrections.csv    User feedback CSV (appended by POST /report)
├── 02_Modular_Scripts/
│   ├── data_cleaner.py        Cleaning pipeline
│   ├── feature_extractor.py   Feature extraction + fetch_live_content() (bs4)
│   └── retrain_model.py       Retraining script (RandomizedSearchCV, n_iter=15, StratifiedKFold)
├── 03_Checkpoints/            Progress logs
└── 04_Saved_Models/
    └── phishing_model_v4_0.joblib   Active model (Git LFS)
```

---

## Detection Coverage

| Attack Vector | Detection Method | Confidence |
|---|---|---|
| Brand impersonation in live page title | Rule 0 — Module L (live HTML fetch) | **99%** (instant short-circuit) |
| Typosquatting (`paypa1.com`) | Module A + Override Rule 1 | ≥ 88% |
| Brand in subdomain (`facebook.evil.net`) | Module G + Override Rule 2 | ≥ 90% |
| Hyphenated impersonation (`appieid-enable.com`) | Module H + Override Rule 3 | ≥ 87% |
| Free-host abuse (`verify.000webhostapp.com`) | Module I + Override Rule 4 | ≥ 85% |
| High-abuse TLD (`secure-login.tk`) | Module B + Override Rule 5/6 | ≥ 76% |
| IPFS-hosted phishing | Module E | 65% (capped — immutable hosting) |
| URL shorteners | Module F → expand → re-analyse | ≥ 65% |
| Open redirect in query | ML `url_in_query` feature | Model-driven |
| Infrastructure abuse (`verify.firebaseapp.com`) | Infra patch (Step 0a) | ≥ 85% |
| Raw IP host (`http://1.2.3.4/paypal/`) | Hard override | ≥ 80% |
| Encoded payloads (Base64/hex in query) | Crypto analysis module | Threat flag |
| SQL injection / XSS / path traversal in URL | Vuln pattern scanner | Threat flag |