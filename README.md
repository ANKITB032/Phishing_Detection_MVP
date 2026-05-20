# PhishGuard MVP — v3.5.0

AI-powered phishing URL detection combining a trained Random Forest classifier with a seven-rule deterministic Heuristic Override Engine.

**Live demo:** [phishguard.ankitband.me](https://phishguard.ankitband.me) · **API:** [ab2403-phishing-detection-mvp.hf.space](https://ab2403-phishing-detection-mvp.hf.space)

---

## Architecture

```
URL Input
    │
    ├─ Step 0a: Infrastructure Abuse Pre-Check (free-hosting subdomain entropy/keyword scan)
    ├─ Step 0:  Trusted-Domain Whitelist Fast Path (50+ curated domains → 100% benign, early exit)
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
    │   └─ Random Forest (200 trees, 13 features) → label + P(benign) / P(malicious)
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
| Algorithm | `RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced')` |
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

### `POST /predict`

**Rate limit:** 5 requests / minute per IP · **Input limit:** 2048 chars

**Request**
```json
{ "url": "http://appieid-enable.com/verify" }
```

**Response**
```json
{
  "url": "http://appieid-enable.com/verify",
  "label": 1,
  "verdict": "malicious",
  "confidence": 90.9,
  "reason": "Hyphenated Impersonation: hostname segment 'appieid-enable' mimics a trusted brand via hyphen-obfuscation — a classic credential-harvesting pattern.",
  "security_analysis": {
    "threat_flags": ["Hyphenated credential domain: 'appieid-enable' mimics a brand via hyphen obfuscation."],
    "override_engine_fired": true,
    "override_confidence": 87.0,
    "typosquatting": { "is_typosquat": false, ... },
    "hyphenated_creds": { "found": true, "matched": "appieid-enable" },
    ...
  }
}
```

**429 Too Many Requests** (rate limit exceeded)
```json
{ "error": "Rate limit exceeded: 5 per 1 minute" }
```

**422 Unprocessable Entity** (input validation failure)
```json
{ "detail": "URL exceeds maximum length of 2048 characters (3100 received)." }
```

### `GET /health`

**Rate limit:** 60 requests / minute per IP

```json
{ "status": "ok", "model": "Random Forest (AUC 0.9645)" }
```

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
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

### Retraining

```bash
python 02_Modular_Scripts/retrain_model.py \
  --input 01_Datasets/malicious_phish.csv \
  --output 04_Saved_Models/phishing_model_v3_5.joblib
```

---

## Project Structure

```
Phishing_Detection_MVP/
├── main.py                    FastAPI backend (rate limiting, input validation, endpoints)
├── predictor.py               Full detection pipeline (whitelist → heuristics → ML → override)
├── requirements.txt           Python dependencies
├── Dockerfile                 HF Spaces container (python:3.11-slim, port 7860)
├── vercel.json                Vercel static deploy config
├── index.html                 Frontend UI (IBM Plex fonts, Anime.js, typewriter results)
├── 01_Datasets/
│   ├── malicious_phish.csv    Raw Kaggle dataset (651K URLs)
│   └── cleaned_data.csv       Deduplicated, labelled
├── 02_Modular_Scripts/
│   ├── data_cleaner.py        Cleaning pipeline
│   ├── feature_extractor.py   Standalone feature extraction (10 features, v3.0)
│   └── retrain_model.py       Full retraining script (13 features, v3.5)
├── 03_Checkpoints/            Progress logs
└── 04_Saved_Models/
    └── phishing_model_v3_5.joblib   Active model (Git LFS)
```

---

## Detection Coverage

| Attack Vector | Detection Method | Confidence |
|---|---|---|
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