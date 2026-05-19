# Checkpoint 04 — v3.5: Heuristic Expansion & Infrastructure Abuse Patch

**Date:** 2026-05-15
**Version:** v3.5.0
**Status:** In Progress — patch implemented, retraining pending on Kaggle dataset

---

## Project State

| Artefact | File | Status |
|---|---|---|
| Data cleaning | `02_Modular_Scripts/data_cleaner.py` | ✓ Complete |
| Feature extraction (v3.0) | `02_Modular_Scripts/feature_extractor.py` | ✓ Complete |
| Model training (v3.0) | `02_Modular_Scripts/train_model.py` | ✓ Complete |
| Inference pipeline | `predictor.py` | ✓ Patched (v3.5) |
| FastAPI backend | `main.py` | ✓ Running on HF Spaces |
| Frontend | `index.html` | ✓ Live on Vercel |
| **Infra abuse patch** | `infra_abuse_patch.py` | ✓ Implemented |
| **Retraining script (v3.5)** | `retrain_model.py` | ✓ Written, pending run |
| Saved model (v3.0) | `04_Saved_Models/phishing_model.joblib` | Active in prod |
| Saved model (v3.5) | `04_Saved_Models/phishing_model_v3_5.joblib` | Pending retrain |

---

## Full System Architecture

### Layer 1 — Trust Protocol (Pre-ML Gate)

- **Whitelist fast-path:** `is_trusted_domain(url)` extracts the eTLD+1 and checks it against `TRUSTED_DOMAINS` (50+ curated entries). If matched and no override flag is set, returns `Safe` at 100% confidence immediately.
- **Override condition:** The whitelist is bypassed if any of the following are true:
  - `infra_abuse_flag` is set (new in v3.5 — see Module G below)
  - `heuristic_override` is set (≥ 2 threat flags fired, or Module A/B escalated)
- **Design intent:** Eliminates false positives on `google.com`, `paypal.com`, etc. without slowing them through the full pipeline.

---

### Layer 2 — Heuristic Modules (A–F + Infrastructure Patch)

Modules run sequentially on the final URL (post-shortener expansion). Each module appends to `threat_flags[]`. When `len(threat_flags) >= 2`, `heuristic_override` is set, forcing ML inference even on whitelisted domains.

#### Module A — Typosquatting Detection
- Levenshtein edit-distance check (threshold ≤ 2) of the eTLD+1 label against 20 major brand names.
- Also catches exact brand name on wrong TLD (e.g. `microsoft.xyz`).
- **Escalation:** Any typosquat match forces `label = 1`, `confidence ≥ 85%` regardless of ML output.

#### Module B — Suspicious TLD
- Checks the URL's TLD against a curated blocklist of high-abuse extensions (`.tk`, `.ml`, `.ga`, `.cf`, `.xyz`, `.top`, `.club`, `.work`, `.click`, etc.).
- Fires a threat flag; if combined with one other flag, triggers `heuristic_override`.

#### Module C — Phishing Keywords
- Scans the full URL string for a vocabulary of 30+ phishing trigger words (`verify`, `secure`, `login`, `update`, `confirm`, `account`, `billing`, `password`, `recover`, etc.).
- Contributes to `threat_flags` count; does not escalate alone.

#### Module D — URL Complexity
- Flags URLs with excessive dot count (> 4) or extreme length (> 200 chars).
- Catches subdomain-chaining evasion: `legit.brand.attacker.evil.com`.

#### Module E — IPFS Gateway Detection
- Detects known IPFS gateway hostnames (`ipfs.io`, `dweb.link`, `cloudflare-ipfs.com`, etc.).
- Extracts and logs the CID for audit; flags as threat since IPFS content is immutable and uncensorable — a common phishing hosting technique.
- A small whitelist of known-legitimate IPFS gateways is maintained for research use.

#### Module F — URL Shortener Expansion
- Checks against `KNOWN_SHORTENERS` set (30+ domains) plus a structural heuristic (path ≤ 10 chars, no query string) for unknown shorteners.
- Issues `requests.HEAD` (5s timeout) → falls back to `requests.GET(stream=True)` on `405`.
- Follows the full redirect chain; the **final resolved URL** replaces the input for all downstream analysis (ML features, Modules A–E).
- Redirect chain is logged in `security_analysis.shortener_expansion` for audit.
- Dead links and redirect loops return the original URL safely (no crash, no false positive).

#### Infrastructure Abuse Patch (v3.5) — Pre-Trust-Protocol Check
- **Problem addressed:** `service-mitld.firebaseapp.com` → `_extract_domain()` returns `firebaseapp.com` → whitelisted as trusted → verdict: Safe (false negative).
- **Detection logic:** `check_infrastructure_abuse(url)` fires when:
  1. eTLD+1 is in `FREE_HOSTING_PROVIDERS` (`firebaseapp.com`, `vercel.app`, `github.io`, `web.app`, `netlify.app`, `pages.dev`, `workers.dev`, `ngrok.io`, `onrender.com`, `fly.dev`).
  2. A non-empty subdomain exists.
  3. Either: subdomain contains a sensitive keyword (`verify`, `service`, `secure`, `login`, `auth`, `billing`, `payment`, etc.), OR subdomain Shannon entropy > 3.5 bits (catches random phishing hash slugs like `a8f3k2p9m1z7qr.vercel.app`).
- **Integration:** Runs as Step 0a — before the whitelist check. Sets `infra_abuse_flag = True`, which blocks the whitelist bypass. The URL then flows through all heuristic modules and ML inference normally.

---

### Layer 3 — ML Inference (Random Forest)

#### v3.0 Model (current production)
- **Algorithm:** `RandomForestClassifier(n_estimators=100, class_weight='balanced')`
- **Features (10):** `url_length`, `num_special_chars`, `num_dots`, `num_hyphens`, `num_at`, `num_query_params`, `has_https`, `has_ip` (regex), `subdomain_depth`, `path_depth`
- **Performance:** Accuracy 91%, ROC-AUC 0.9648
- **Known failures:** Raw IP hosts (`http://213.190.128.11/paypal/`) — structural features score benign; brand-in-path (`/paypal/`) not captured.

#### v3.5 Model (pending retrain)
- **Algorithm:** `Pipeline(StandardScaler → RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight='balanced'))`
- **New features (2):**
  - `is_ip`: `int` — regex match on hostname for bare IPv4 (`^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$`). Directly fixes the raw-IP blind spot.
  - `path_brand`: `int` — scans URL path + query string for top-20 spoofed brand names. Catches `/paypal/`, `/microsoft/`, `/apple-id/` etc. on attacker-owned domains.
- **Total features (12):** all v3.0 features + `is_ip` + `path_brand`
- **Note on StandardScaler:** RF is invariant to monotonic transforms; scaler is included for pipeline consistency and future linear model ensemble compatibility. Saved inside the `Pipeline` object — inference receives raw feature DataFrames.

---

### Layer 4 — Escalation & Final Verdict

- If `label == 0` (ML says benign) AND `len(threat_flags) >= 2` → escalate to `label = 1`, `confidence = max(current, 70%)`.
- Typosquat match → force `label = 1`, `confidence = max(current, 85%)`.
- Infra abuse flag → URL bypasses whitelist; final label determined by ML + escalation logic above.
- Confidence is capped at 80% if the URL has query parameters and ML returns benign (query params introduce uncertainty).

---

## Evasion Vectors & How They Are Handled

| Evasion Technique | Example | Detection Method |
|---|---|---|
| URL shorteners | `bit.ly/xK9p` | Module F: expand → analyse final URL |
| Novel shorteners | `ab.cd/xy` | Module F: structural heuristic (path ≤ 10 chars) |
| IPFS hosting | `ipfs.io/Qm...` | Module E: gateway detection + CID logging |
| Subdomain chaining | `paypal.legit.evil.com` | Module D: dot count + Module A: typosquat |
| Infrastructure abuse | `verify.firebaseapp.com` | Infra patch: keyword + entropy on free-host subdomains |
| Hash slug evasion | `a8f3k2p9m1z7.vercel.app` | Infra patch: Shannon entropy > 3.5 bits |
| Raw IP + brand path | `http://1.2.3.4/paypal/` | v3.5: `is_ip` + `path_brand` features |
| Typosquatting | `paypa1.com` | Module A: Levenshtein ≤ 2 |
| Suspicious TLD | `secure-login.tk` | Module B: TLD blocklist |

---

## Exact Next Steps

1. **Run `retrain_model.py`** on the Kaggle dataset to produce `phishing_model_v3_5.joblib`. Verify `is_ip` ranks in the top 5 features by Gini importance.
2. **Apply the infra abuse patch** into `predictor.py` per the 6-step injection guide in `infra_abuse_patch.py`.
3. **Update `predictor.py` `MODEL_PATH`** to point to `phishing_model_v3_5.joblib`.
4. **Update `main.py` version** string to `3.5.0` and add `InfrastructureAbuse` to the `SecurityAnalysis` Pydantic model.
5. **Redeploy to Hugging Face Spaces** via `git push` (model is tracked via Git LFS).
6. Consider adding rate limiting (`slowapi`) and a Tranco top-1M whitelist pre-filter for production hardening.
- [x] Local v3.5 testing complete — feature mismatch resolved, FastAPI curl tests passing (2026-05-19 09:32)
-   [ x ]   L o c a l   v 3 . 5   t e s t i n g   c o m p l e t e      f e a t u r e   m i s m a t c h   r e s o l v e d ,   F a s t A P I   c u r l   t e s t s   p a s s i n g   ( 2 0 2 6 - 0 5 - 1 9   0 9 : 3 8 )  
 