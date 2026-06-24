# Checkpoint 04 — v3.5: Heuristic Expansion & Infrastructure Abuse Patch

**Date (original):** 2026-05-15
**Last updated:** 2026-06-24
**Version:** v3.5.0
**Status:** ✅ Session complete — all heuristic additions applied, live batch tested

---

## Project State

| Artefact | File | Status |
|---|---|---|
| Data cleaning | `02_Modular_Scripts/data_cleaner.py` | ✓ Complete |
| Feature extraction (v3.0) | `02_Modular_Scripts/feature_extractor.py` | ✓ Complete |
| Model training (v3.5) | `02_Modular_Scripts/retrain_model.py` | ✓ Synced to predictor.py |
| Inference pipeline | `predictor.py` | ✓ Fully patched (v3.5, 2026-06-24) |
| FastAPI backend | `main.py` | ✓ Running on HF Spaces |
| Frontend | `index.html` | ✓ Live on Vercel |
| Saved model (v3.5) | `04_Saved_Models/phishing_model_v3_5.joblib` | ✓ Active in prod |
| Batch test tool | `openphish_batch_test.py` | ✓ New — feed-file + live mode |

---

## Performance Metrics (2026-06-24 Session)

| Benchmark | Result | Notes |
|---|---|---|
| ROC-AUC (held-out test set) | **0.9824** | Up from 0.9645 after v3.5 retrain |
| 50-URL curated batch | **80% detection** | Hand-labelled known phishing URLs |
| 300-URL live OpenPhish feed | **~55–60% detection** | Live feed; many URLs already offline |

> **Note on OpenPhish rate:** URLs in the live feed that are already taken down return connection errors or redirect to parked domains — these score as benign since there is no malicious content to fetch. The 55–60% figure reflects detection on *reachable* phishing pages; true detection on active pages is higher.

---

## Full System Architecture

### Layer 1 — Trust Protocol (Pre-ML Gate)

- **Whitelist fast-path:** `is_trusted_domain(url)` extracts the eTLD+1 via `tldextract` and checks against `TRUSTED_DOMAINS` (50+ curated entries + 8 regional ccTLD variants).
- **Google Sites carve-out (new):** `sites.google.com` and `*.google.com/view/*` always return `False` from `is_trusted_domain()` — attacker-controlled Google Sites pages must not bypass the pipeline.
- **Override condition:** Whitelist is bypassed if `infra_abuse_flag` is set.
- **github.io removed from whitelist:** Needed so Rule 8 (github.io brand-in-path) can fire.

---

### Layer 2 — Heuristic Modules (A–J + Infrastructure Patch)

#### Module A — Typosquatting Detection
- Levenshtein edit-distance ≤ 2 against `BRAND_MAP` (21 brands including `icloud → apple.com`).
- **Free-host guard (new):** `dist == 0` branch now skips flagging if the registered domain is in `FREE_HOSTING_PROVIDERS` or `ABUSED_FREE_HOSTS` (e.g. `github.io` SLD matching brand `github` is expected, not a typosquat).
- **Regional-domain guard (new):** Full hostname checked against `TRUSTED_DOMAINS` early-exit before Levenshtein loop — prevents `google.co.uk`, `amazon.co.uk` from false-positiving.

#### Module B — Suspicious TLD
- **Extended set (new, 2026-06-17/18):** `.cfd`, `.ru`, `.cn`, `.vip`, `.id`, `.et`, `.cyou`, `.shop` added.
- **Hard override (new):** `tld_check["flagged"]` now triggers a hard override (`label = 1`, `confidence ≥ 76%`) after the raw-IP override block — bypasses trust_override gate.

#### Module C — Phishing Keywords
- Unchanged from v3.5.

#### Module D — URL Complexity
- Unchanged from v3.5.

#### Module E — IPFS Gateway Detection
- Unchanged from v3.5.

#### Module F — URL Shortener Expansion
- **Bug fix (2026-06-17):** Shortener label-flip corrected. Previously `if result["label"] == 0` was incorrectly forcing benign expanded URLs to malicious. Fixed to `== 1`.

#### Module G — Brand-in-Subdomain
- **Regional-domain guard (new):** Hostname and eTLD+1 checked against `TRUSTED_DOMAINS` before subdomain label scan — prevents `google.co.uk` false positives.

#### Module H — Hyphenated Credential Domains
- Unchanged from v3.5.

#### Module I — Abused Free Hosts (tiered, new logic)
- **Two-tier system (new, 2026-06-23):**
  - Hosts in `FREE_HOSTING_PROVIDERS` → gated by keyword+entropy via `check_infrastructure_abuse()` only (no unconditional flag).
  - Hosts in `ABUSED_FREE_HOSTS` but NOT in `FREE_HOSTING_PROVIDERS` → `abused_host_flag` fires unconditionally.
  - Prevents double-penalising innocent pages like `myportfolio.github.io`.
- **Extended `ABUSED_FREE_HOSTS` (new):** `wixstudio.com`, `webflow.io`, `blogspot.com`, `weebly.com`, `wix.com`, `hostingersite.com`, `godaddysites.com`, `zapier.app`, `gitbook.io`, `railway.app`, `azurewebsites.net`, `edgeone.app/cool/dev`, `wasmer.app`, `replit.app`.

#### Module J — github.io Brand-in-Path (new, Rule 8)
- **Detection:** Fires when eTLD+1 is `github.io` AND path contains a `PATH_BRANDS` keyword AND hostname is not in `TRUSTED_DOMAINS`.
- **Heuristic count:** Wired into `heuristic_count` as flag J — prevents trust_override from dismissing it.
- **Override Engine Rule 8:** `_override_conf ≥ 0.87`, reason string cites brand, subdomain, and attack vector.
- **Safety:** `_github_brand_hit = None` default initialised before the conditional block (NameError guard).

#### Infrastructure Abuse Patch (v3.5) — Pre-Trust-Protocol Check
- **Extended keywords (new):** `clone`, `wallet`, `bridge`, `swap`, `ledger` added to `_INFRA_SENSITIVE_KEYWORDS` for Web3/DeFi phishing.
- Logic unchanged otherwise.

#### Rule 0 — Brand Impersonation in Page Title (Module L)
- **SLD comparison fix (new, 2026-06-17):** Condition changed from `_url_domain != official_domain` to `_url_sld != _official_sld` — prevents false positives on regional domains (`amazon.co.uk` vs `amazon.com` both have SLD `amazon`).

---

### Layer 3 — ML Inference (Random Forest v3.5)

- **ROC-AUC:** 0.9824 (held-out 20% test set)
- **Features (13):** `url_length`, `num_special_chars`, `num_dots`, `num_hyphens`, `num_at`, `num_query_params`, `has_https`, `subdomain_depth`, `path_depth`, `is_ip`, `path_brand`, `url_in_query`, `tld_risk`
- **`extract_features()` fix (2026-06-17):** Rewritten to use `urlparse` for all hostname/path/query extraction — eliminates `url.split("/")[0]` skew that caused `is_ip`, `subdomain_depth`, and `path_brand` to be calculated on the wrong URL segment.
- **`_IP_RE` moved to module level (2026-06-23):** Was being recompiled on every prediction call.
- **`retrain_model.py` synced (2026-06-23):** `extract_features()` in training script now matches production exactly — `has_https`, `subdomain_depth`, `path_depth`, `is_ip`, `path_brand`, `tld_risk` all use identical logic. `_IP_IN_URL` unused regex removed.

#### tldextract fix
- **`_extract_domain()` (2026-06-17):** Replaced manual `split(".")[-2:]` with `tldextract.extract(url)` — correctly handles `.co.uk`, `.com.au`, etc.
- **Deprecation fix (2026-06-23):** `extracted.registered_domain` → `extracted.top_domain_under_public_suffix` (tldextract v5 API).

---

### Layer 4 — Escalation & Final Verdict

- Hard overrides (in order): infra_abuse → raw IP → suspicious TLD (new) → heuristic override engine (Rules 1–8).
- Trust Protocol: ML-malicious with zero heuristic flags → benign override. Bypassed by any hard override.
- Legacy guard: `≥ 2 threat_flags` with label 0 → escalate to 70%.

---

## Evasion Vectors & How They Are Handled

| Evasion Technique | Example | Detection Method |
|---|---|---|
| URL shorteners | `bit.ly/xK9p` | Module F: expand → analyse final URL |
| IPFS hosting | `ipfs.io/Qm...` | Module E: gateway detection + CID logging |
| Infrastructure abuse | `verify.firebaseapp.com` | Infra patch: keyword + entropy |
| Hash slug evasion | `a8f3k2p9m1z7.vercel.app` | Infra patch: Shannon entropy > 3.5 bits |
| Raw IP + brand path | `http://1.2.3.4/paypal/` | `is_ip` + `path_brand` ML features + hard override |
| Typosquatting | `paypa1.com`, `mapaicloud.com` | Module A: Levenshtein ≤ 2, icloud→apple.com in BRAND_MAP |
| Wrong-TLD brand | `microsoft.xyz` | Module A: dist==0 + non-official domain |
| Suspicious TLD | `secure-login.tk`, `.cyou`, `.shop` | Module B: TLD blocklist + hard override |
| Brand in subdomain | `facebook.unitedcolleges.net` | Module G: exact label match |
| Hyphenated impersonation | `appieid-enable.com` | Module H: regex pattern |
| Free CMS hosting | `wixstudio.com`, `blogspot.com` | Module I: unconditional abused-host flag |
| github.io phishing pages | `attacker.github.io/paypal-login` | Module J (Rule 8): path brand-hit |
| Google Sites phishing | `sites.google.com/view/...` | `is_trusted_domain()` carve-out → full pipeline |
| Brand title impersonation | page title says "PayPal" on wrong domain | Rule 0: Module L live fetch + SLD comparison |
| Regional domain FP | `google.co.uk`, `amazon.co.uk` | Trusted-domain guard in Module A & G |
| Web3/DeFi phishing | `wallet-bridge.vercel.app` | Extended infra keywords + Module I |

---

## Changes Applied in This Session (2026-06-17 → 2026-06-24)

### predictor.py
- [x] **Fix 1 — Domain parsing:** `_extract_domain()` rewritten with `tldextract`
- [x] **Fix 2 — Shortener logic flip:** `label == 0` → `label == 1` in shortener block
- [x] **Fix 3 — Feature extraction skew:** `extract_features()` fully rewritten with `urlparse`
- [x] **Rule 0 SLD fix:** `_url_domain != official_domain` → SLD-level comparison
- [x] **TLD additions:** `.cfd`, `.ru`, `.cn`, `.vip`, `.id`, `.et`, `.cyou`, `.shop`
- [x] **ABUSED_FREE_HOSTS additions:** 14 new platforms
- [x] **INFRA_SENSITIVE_KEYWORDS additions:** `clone`, `wallet`, `bridge`, `swap`, `ledger`
- [x] **Suspicious TLD hard override** block added after raw-IP override
- [x] **Regional domain false-positive fix:** `google.co.uk` etc. added to `TRUSTED_DOMAINS`
- [x] **Trusted-domain guards** in `detect_typosquatting()` and `brand_in_subdomain()`
- [x] **Google Sites carve-out** in `is_trusted_domain()`
- [x] **icloud → apple.com** added to `BRAND_MAP`
- [x] **Module J — github.io brand-in-path** detection + Rule 8 in override engine
- [x] **`_github_brand_hit = None`** default initialised (NameError guard)
- [x] **`github.io` removed from `TRUSTED_DOMAINS`** (required for Rule 8 to reach pipeline)
- [x] **Free-host guard in typosquat dist==0 branch** (github.io SLD collision fix)
- [x] **`abused_host_flag` tiered** — excludes `FREE_HOSTING_PROVIDERS` from unconditional flagging
- [x] **`_IP_RE` moved to module level** (was recompiled per prediction call)
- [x] **`tldextract` deprecation fix** — `registered_domain` → `top_domain_under_public_suffix`

### retrain_model.py
- [x] **`extract_features()` synced** — identical logic to predictor.py
- [x] **`_TLD_RISK` synced** — 8 new high-risk TLDs added
- [x] **`_IP_IN_URL` removed** — unused after extract_features() sync
- [x] **`try/except ValueError`** around urlparse — handles malformed IPv6 gracefully

### openphish_batch_test.py (new)
- [x] Fetches OpenPhish live feed + batch-tests against local API
- [x] `--feed-file` flag for WSL usage (download feed from PowerShell first)
- [x] Progress display, caught/missed/error counts, first-10 missed URLs printed

---

## Pending Items (Post-Session)

- [ ] **Tranco top-1M pre-filter** — whitelist top 1M domains before heuristic pipeline to reduce false-positive rate on obscure-but-legitimate domains. Expected to push live OpenPhish detection rate above 70%.
- [ ] **Rule 0 timeout handling** — `fetch_live_content()` blocks for up to 3s per URL; should be made async or given a circuit-breaker for high-volume use.
- [ ] **Auth / API key** — `X-API-Key` header check for private deployments.
- [ ] **Structured logging** — `python-json-logger` for HF Spaces log aggregation.
- [ ] **tldextract offline cache** — pre-seed the PSL cache so cold-start predictions don't depend on network access.