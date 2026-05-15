"""
predictor.py — Phishing URL Prediction Pipeline
=================================================
Loads the trained Random Forest model and provides a single-function
inference interface:  predict_url(url) -> dict

Pipeline order:
  1. Trusted-domain whitelist  (fast path — skips everything if matched)
  2. Cryptographic / vulnerability pre-screening
  3. ML feature extraction + Random Forest classification
  4. Threat-flag escalation (overrides ML if ≥ 2 red flags)
"""

import re
import math
import base64
import joblib
import os
import pandas as pd
from urllib.parse import urlparse, parse_qs

# ── Model Loading ────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "04_Saved_Models", "phishing_model_v3_5.joblib")
_model = joblib.load(MODEL_PATH)

SPECIAL_CHARS = "@-?=.#%+&_~/!"

FEATURES = [
    "url_length", "num_special_chars", "num_dots", "num_hyphens",
    "num_at", "num_query_params", "has_https", "subdomain_depth", "path_depth",
    "is_ip", "path_brand"  # <--- The two new v3.5 features
]


# ═══════════════════════════════════════════════════════════════════════════════
#  TRUSTED DOMAIN WHITELIST  (Fast Path)
# ═══════════════════════════════════════════════════════════════════════════════

# Top 50 trusted domains — curated for the MVP to eliminate false positives
# on well-known, reputable sites whose URL structure the ML model misclassifies.
TRUSTED_DOMAINS: set[str] = {
    # Search engines
    "google.com",       "bing.com",         "yahoo.com",
    "duckduckgo.com",   "baidu.com",        "yandex.ru",
    # Big tech / cloud
    "microsoft.com",    "apple.com",        "amazon.com",
    "meta.com",         "facebook.com",     "instagram.com",
    "twitter.com",      "x.com",            "linkedin.com",
    # Developer / code
    "github.com",       "gitlab.com",       "stackoverflow.com",
    "npmjs.com",        "pypi.org",         "docker.com",
    # Media & content
    "youtube.com",      "netflix.com",      "spotify.com",
    "reddit.com",       "wikipedia.org",    "medium.com",
    # Commerce & payments
    "paypal.com",       "stripe.com",       "shopify.com",
    "ebay.com",         "etsy.com",
    # Productivity & SaaS
    "zoom.us",          "slack.com",        "notion.so",
    "dropbox.com",      "adobe.com",        "salesforce.com",
    # Security / infra
    "cloudflare.com",   "godaddy.com",      "namecheap.com",
    "letsencrypt.org",  "digicert.com",
    # Email & comms
    "outlook.com",      "proton.me",        "whatsapp.com",
    # News / reference
    "bbc.com",          "cnn.com",          "nytimes.com",
    # Education
    "coursera.org",     "edx.org",          "khanacademy.org",
    # Custom / Personal
    "claude.ai",        "anthropic.com",    "ankitband.me",
    "vercel.app",       "github.com",
}


def _extract_domain(url: str) -> str:
    """
    Extract the registrable domain (eTLD+1) from a URL.

    Examples:
        https://mail.google.com/inbox  →  google.com
        http://docs.github.com:8080    →  github.com
        github.com/user/repo           →  github.com
    """
    # Ensure the URL has a scheme so urlparse doesn't treat it as a path
    normalized = url if "://" in url else f"https://{url}"
    hostname = urlparse(normalized).hostname or ""
    hostname = hostname.lower().rstrip(".")

    # Walk back to the registrable domain (last two labels, or last three
    # for two-part TLDs like .co.uk — MVP simplification).
    parts = hostname.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def is_trusted_domain(url: str) -> bool:
    """Return True if the URL's domain belongs to the trusted whitelist."""
    return _extract_domain(url) in TRUSTED_DOMAINS


# ═══════════════════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE ABUSE PATCH
# ═══════════════════════════════════════════════════════════════════════════════

# Platforms that offer free subdomain hosting — their eTLD+1 is legitimately
# whitelisted, but individual subdomains can be attacker-controlled.
FREE_HOSTING_PROVIDERS: set[str] = {
    "firebaseapp.com",
    "web.app",
    "vercel.app",
    "netlify.app",
    "github.io",
    "pages.dev",       # Cloudflare Pages
    "workers.dev",     # Cloudflare Workers
    "glitch.me",
    "repl.co",
    "ngrok.io",
    "onrender.com",
    "fly.dev",
}

# Sensitive keywords that are legitimate in a root domain (paypal.com)
# but deeply suspicious in an attacker-controlled subdomain.
_INFRA_SENSITIVE_KEYWORDS: list[str] = [
    "verify", "service", "secure", "login", "account", "update",
    "confirm", "support", "billing", "payment", "auth", "signin",
    "password", "recover", "unlock", "validate", "alert", "admin",
]

# Entropy threshold — random subdomain hashes score > 3.5
_SUBDOMAIN_ENTROPY_THRESHOLD = 3.5


def _subdomain_entropy(hostname: str) -> float:
    """Shannon entropy of the subdomain label (everything left of eTLD+1)."""
    parts = hostname.rstrip(".").split(".")
    # Strip the last two labels (eTLD+1) to isolate the subdomain
    subdomain = ".".join(parts[:-2]) if len(parts) > 2 else ""
    if not subdomain:
        return 0.0
    freq = {}
    for ch in subdomain:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(subdomain)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def check_infrastructure_abuse(url: str) -> dict:
    """
    Detect attackers abusing free-hosting subdomains of trusted platforms.

    Fires when ALL three conditions are true:
      1. The eTLD+1 belongs to FREE_HOSTING_PROVIDERS.
      2. There IS a non-empty subdomain (i.e. not the root domain itself).
      3. Either the subdomain contains a sensitive keyword  OR  its Shannon
         entropy exceeds _SUBDOMAIN_ENTROPY_THRESHOLD (flags random hashes).
    """
    normalized = url if "://" in url else f"https://{url}"
    hostname   = (urlparse(normalized).hostname or "").lower().rstrip(".")
    parts      = hostname.split(".")

    # Derive eTLD+1
    etld1 = ".".join(parts[-2:]) if len(parts) >= 2 else hostname

    if etld1 not in FREE_HOSTING_PROVIDERS:
        return {"is_abuse": False, "provider": None, "subdomain": None, "reason": ""}

    # Must have an actual subdomain (not just "firebaseapp.com" itself)
    if len(parts) <= 2:
        return {"is_abuse": False, "provider": etld1, "subdomain": None,
                "reason": "Root provider domain — no subdomain present."}

    subdomain = ".".join(parts[:-2])

    # Condition A: sensitive keyword present in subdomain
    keyword_hit = next(
        (kw for kw in _INFRA_SENSITIVE_KEYWORDS if kw in subdomain.lower()),
        None,
    )

    # Condition B: high entropy (randomised hash subdomain)
    entropy = _subdomain_entropy(hostname)
    high_entropy = entropy > _SUBDOMAIN_ENTROPY_THRESHOLD

    is_abuse = bool(keyword_hit or high_entropy)

    if not is_abuse:
        return {"is_abuse": False, "provider": etld1, "subdomain": subdomain,
                "reason": "Subdomain on free provider but no suspicious indicators."}

    reasons = []
    if keyword_hit:
        reasons.append(f"sensitive keyword '{keyword_hit}' in subdomain '{subdomain}'")
    if high_entropy:
        reasons.append(f"high subdomain entropy ({entropy:.2f} bits) suggests randomised phishing slug")

    return {
        "is_abuse":  True,
        "provider":  etld1,
        "subdomain": subdomain,
        "reason":    f"Infrastructure abuse on {etld1}: " + "; ".join(reasons) + ".",
    }


# ── Feature Extraction (mirrors feature_extractor.py) ────────────────────────


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAND FUZZY MATCHING  (Typosquatting Detection)
# ═══════════════════════════════════════════════════════════════════════════════

# Maps brand keywords (lowercase) to their official registrable domains.
# If a URL's domain is within edit-distance 1-2 of a brand keyword but
# does NOT belong to the official domain, it is flagged as typosquatting.
BRAND_MAP: dict[str, str] = {
    "microsoft":   "microsoft.com",
    "amazon":      "amazon.com",
    "apple":       "apple.com",
    "google":      "google.com",
    "facebook":    "facebook.com",
    "instagram":   "instagram.com",
    "netflix":     "netflix.com",
    "paypal":      "paypal.com",
    "linkedin":    "linkedin.com",
    "twitter":     "twitter.com",
    "dropbox":     "dropbox.com",
    "github":      "github.com",
    "spotify":     "spotify.com",
    "adobe":       "adobe.com",
    "yahoo":       "yahoo.com",
    "chase":       "chase.com",
    "wellsfargo":  "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
    "ebay":        "ebay.com",
    "walmart":     "walmart.com",
}


def _levenshtein(s: str, t: str) -> int:
    """Compute the Levenshtein edit distance between two strings."""
    if len(s) < len(t):
        return _levenshtein(t, s)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s):
        curr = [i + 1]
        for j, ct in enumerate(t):
            curr.append(min(
                prev[j + 1] + 1,       # deletion
                curr[j] + 1,           # insertion
                prev[j] + (cs != ct),  # substitution
            ))
        prev = curr
    return prev[-1]


def detect_typosquatting(url: str) -> dict:
    """
    Check if the URL's domain is a near-match (edit distance ≤ 2) for any
    known brand but is NOT the brand's official domain.

    Returns:
        {
            "is_typosquat": bool,
            "matched_brand": str | None,
            "edit_distance": int | None,
        }
    """
    domain = _extract_domain(url)
    # Strip TLD to compare just the brand-name portion (e.g. "amaz0n")
    domain_label = domain.split(".")[0].lower()

    for brand, official in BRAND_MAP.items():
        dist = _levenshtein(domain_label, brand)
        if 0 < dist <= 2 and domain != official:
            return {
                "is_typosquat":  True,
                "matched_brand": brand.title(),
                "edit_distance": dist,
            }
        # Exact brand name but wrong TLD (e.g. microsoft.xyz)
        if dist == 0 and domain != official:
            return {
                "is_typosquat":  True,
                "matched_brand": brand.title(),
                "edit_distance": 0,
            }

    return {"is_typosquat": False, "matched_brand": None, "edit_distance": None}

# Top-20 brand names most commonly spoofed in phishing URL paths
PATH_BRANDS: list[str] = [
    "paypal", "microsoft", "apple", "google", "amazon",
    "facebook", "netflix", "instagram", "linkedin", "twitter",
    "dropbox", "github", "spotify", "adobe", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "ebay", "walmart",
]

def extract_features(url: str) -> pd.DataFrame:
    """Extract the 11 ML features from a raw URL string for v3.5."""
    try:
        parsed   = urlparse(url if "://" in url else f"https://{url}")
        hostname = (parsed.hostname or "").lower()
        path     = (parsed.path or "").lower()
        query    = (parsed.query or "").lower()
    except Exception:
        hostname = path = query = ""

    is_ip      = bool(re.match(r"^(\d{1,3}\.){3}\d{1,3}$", hostname)) or bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", hostname))
    path_lower = path + "/" + query
    path_brand = any(brand in path_lower for brand in PATH_BRANDS)

    feats = {
        "url_length":        len(url),
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_at":            url.count("@"),
        "num_query_params":  url.count("?") + url.count("&"),
        "has_https":         int(url.startswith("https")),
        "subdomain_depth":   max(0, len(hostname.split(".")) - 2),
        "path_depth":        url.count("/"),
        "is_ip":             int(is_ip),
        "path_brand":        int(path_brand),
    }
    return pd.DataFrame([feats])[FEATURES]


# ═══════════════════════════════════════════════════════════════════════════════
#  CRYPTOGRAPHIC & VULNERABILITY ANALYSIS  (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Base64 / Hex Encoded Payload Detection ────────────────────────────────

# Regex for Base64 strings: 8+ chars from the Base64 alphabet ending with
# optional padding.  Lowered from 16 to catch short payloads like PHNjcmlwdD4=.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")

# Regex for hex-encoded byte strings (e.g. \x41\x42 or 0x4142 or %41%42)
_HEX_ESCAPE_RE  = re.compile(r"(\\x[0-9a-fA-F]{2}){4,}")
_HEX_PERCENT_RE = re.compile(r"(%[0-9a-fA-F]{2}){2,}")   # lowered from 4 to 2
_HEX_0X_RE      = re.compile(r"0x[0-9a-fA-F]{8,}")


def detect_encoded_payloads(url: str) -> dict:
    """
    Scan the URL path and query parameters for Base64-encoded strings and
    hex-encoded byte sequences that may conceal malicious payloads.

    Returns:
        dict with keys 'base64_found', 'hex_found', 'encoded_snippets'
    """
    parsed = urlparse(url)
    search_area = parsed.path + "?" + parsed.query if parsed.query else parsed.path

    base64_matches = _BASE64_RE.findall(search_area)
    hex_matches = (
        _HEX_ESCAPE_RE.findall(search_area)
        + _HEX_PERCENT_RE.findall(search_area)
        + _HEX_0X_RE.findall(search_area)
    )

    # Validate Base64 candidates by attempting decode
    verified_b64 = []
    for candidate in base64_matches:
        padded = candidate + "=" * (-len(candidate) % 4)
        decoded = base64.b64decode(padded, validate=True)
        # Only flag if decoded bytes look like printable ASCII (payload text)
        if decoded and sum(32 <= b < 127 for b in decoded) / len(decoded) > 0.7:
            verified_b64.append(candidate[:40] + ("…" if len(candidate) > 40 else ""))

    return {
        "base64_found":    len(verified_b64) > 0,
        "hex_found":       len(hex_matches) > 0,
        "encoded_snippets": verified_b64[:3],  # cap at 3 for readability
    }


# ── 2. Shannon Entropy Analysis ──────────────────────────────────────────────

def shannon_entropy(text: str) -> float:
    """
    Compute the Shannon entropy (bits per character) of a string.

    High entropy (> 4.0) in a query string suggests encrypted, compressed,
    or randomised data — a tactic used to smuggle payloads past filters.
    """
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


# Threshold: typical English text ≈ 3.5-4.0 bits; encrypted blobs > 4.5
ENTROPY_THRESHOLD = 4.0


def analyse_query_entropy(url: str) -> dict:
    """
    Measure Shannon entropy across the full query string.

    Returns:
        dict with 'query_entropy', 'high_entropy' flag, and per-param details.
    """
    parsed = urlparse(url)
    query = parsed.query

    overall_entropy = shannon_entropy(query)

    # Per-parameter breakdown
    param_details = {}
    for key, values in parse_qs(query).items():
        for val in values:
            param_details[key] = {
                "value_preview": val[:30] + ("…" if len(val) > 30 else ""),
                "entropy": round(shannon_entropy(val), 3),
            }

    return {
        "query_entropy":  round(overall_entropy, 3),
        "high_entropy":   overall_entropy > ENTROPY_THRESHOLD,
        "param_details":  param_details,
    }


# ── 3. Common Vulnerability Pattern Detection ────────────────────────────────

VULN_PATTERNS = {
    "directory_traversal": [
        re.compile(r"\.\./"),                          # ../
        re.compile(r"\.\.\%2[fF]"),                    # url-encoded ../
        re.compile(r"%2[eE]%2[eE]%2[fF]"),             # double url-encoded
        re.compile(r"\.\.\\"),                         # backslash variant
    ],
    "xss_tags": [
        re.compile(r"<\s*script", re.IGNORECASE),     # <script
        re.compile(r"javascript\s*:", re.IGNORECASE),  # javascript:
        re.compile(r"on\w+\s*=", re.IGNORECASE),       # onerror=, onload=
        re.compile(r"<\s*img[^>]+onerror", re.IGNORECASE),
        re.compile(r"<\s*svg[^>]+onload", re.IGNORECASE),
    ],
    "sql_injection": [
        re.compile(r"(\b(UNION|SELECT|DROP|INSERT|DELETE|UPDATE)\b.*\b(FROM|TABLE|INTO)\b)", re.IGNORECASE),
        re.compile(r"['\"];\s*--"),                    # SQL comment injection
        re.compile(r"1\s*=\s*1"),                      # tautology
        re.compile(r"'\s*OR\s+'", re.IGNORECASE),      # 'OR' injection
    ],
    "command_injection": [
        re.compile(r";\s*(ls|cat|wget|curl|bash|sh|cmd)\b", re.IGNORECASE),
        re.compile(r"\|\s*(ls|cat|id|whoami)\b", re.IGNORECASE),
    ],
}


def detect_vulnerability_patterns(url: str) -> dict:
    """
    Scan the URL for common attack-vector patterns.

    Returns:
        dict mapping each category to a list of matched pattern descriptions.
    """
    findings = {}
    for category, patterns in VULN_PATTERNS.items():
        matched = []
        for pat in patterns:
            if pat.search(url):
                matched.append(pat.pattern)
        if matched:
            findings[category] = matched
    return findings


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def predict_url(url: str) -> dict:
    """
    Full prediction pipeline:

    0a. Check for infrastructure abuse on trusted platforms.
    0b. Trusted-domain fast path (whitelist bypass).
    1. Run cryptographic / vulnerability pre-screening.
    2. Extract ML features and run the trained model.
    3. Merge all results into a single response dict.

    Returns:
        {
            "url":        str,
            "label":      0 | 1,
            "verdict":    "benign" | "malicious",
            "confidence": float (0-100),
            "reason":     str,
            "security_analysis": {
                "encoded_payloads": {...},
                "entropy_analysis": {...},
                "vulnerability_patterns": {...},
                "threat_flags": [str, ...],
                "infrastructure_abuse": {...}
            }
        }
    """
    # ── Step 0a: Infrastructure Abuse Pre-Check ────────────────────────
    _infra = check_infrastructure_abuse(url)
    infra_abuse_flag = _infra["is_abuse"]

    # ── Step 0b: Trusted-domain fast path ──────────────────────────────────
    # Note: Whitelist bypasses only if infra_abuse_flag is False
    if is_trusted_domain(url) and not infra_abuse_flag:
        domain = _extract_domain(url)
        return {
            "url":        url,
            "label":      0,
            "verdict":    "benign",
            "confidence": 100.0,
            "reason":     f"Trusted domain ({domain}) — bypassed via whitelist.",
            "security_analysis": "Bypassed via trusted domain whitelist",
        }

    # ── Pre-screening ────────────────────────────────────────────────────
    encoded  = detect_encoded_payloads(url)
    entropy  = analyse_query_entropy(url)
    vulns    = detect_vulnerability_patterns(url)
    typo     = detect_typosquatting(url)

    threat_flags = []
    if encoded["base64_found"]:
        threat_flags.append("Base64-encoded payload detected")
    if encoded["hex_found"]:
        threat_flags.append("Hex-encoded payload detected")
    if entropy["high_entropy"]:
        threat_flags.append(f"High query-string entropy ({entropy['query_entropy']} bits)")
    for category in vulns:
        threat_flags.append(f"Vulnerability pattern: {category.replace('_', ' ')}")
    if typo["is_typosquat"]:
        threat_flags.append(f"Typosquatting: domain resembles {typo['matched_brand']}")
    if infra_abuse_flag:
        threat_flags.append(_infra["reason"])

    # ── ML Inference ─────────────────────────────────────────────────────
    X = extract_features(url)
    label      = int(_model.predict(X)[0])
    proba      = _model.predict_proba(X)[0]
    confidence = float(proba[label])

    # If the pre-screening found threats, escalate a "benign" ML result
    if label == 0 and len(threat_flags) >= 2:
        label = 1
        confidence = max(confidence, 0.70)

    # Typosquatting always forces malicious — regardless of ML score
    if typo["is_typosquat"]:
        label = 1
        confidence = max(confidence, 0.85)
        
    # Infrastructure abuse forces malicious
    if infra_abuse_flag:
        label = 1
        confidence = max(confidence, 0.90)

    verdict = "benign" if label == 0 else "malicious"

    # ── Confidence cap: never say 100% benign with query parameters ──────
    parsed_url = urlparse(url)
    if label == 0 and parsed_url.query:
        confidence = min(confidence, 0.80)

    # ── Build detection reason ───────────────────────────────────────────
    reasons = []

    # Priorities in reasoning
    if typo["is_typosquat"]:
        reasons.append(
            f"Visual Deception: This domain is highly similar to "
            f"{typo['matched_brand']}, suggesting a typosquatting attack."
        )
    if infra_abuse_flag:
        reasons.append(f"Infrastructure Abuse: {_infra['reason']}")

    if encoded["base64_found"]:
        reasons.append("Encoded cryptographic payload detected in URL parameters.")
    if encoded["hex_found"]:
        reasons.append("Hex-encoded byte sequence found in URL.")
    if entropy["high_entropy"]:
        reasons.append(f"Abnormally high entropy ({entropy['query_entropy']} bits) in query string suggests obfuscated data.")
    for category in vulns:
        readable = category.replace("_", " ").title()
        reasons.append(f"Vulnerability pattern detected: {readable}.")

    if not reasons:
        if label == 1:
            reasons.append("Model detected structural patterns common in phishing sites.")
        else:
            reasons.append("No suspicious indicators found. URL structure appears benign.")

    reason = " ".join(reasons)

    return {
        "url":        url,
        "label":      label,
        "verdict":    verdict,
        "confidence": round(confidence * 100, 2),
        "reason":     reason,
        "security_analysis": {
            "encoded_payloads":       encoded,
            "entropy_analysis":       entropy,
            "vulnerability_patterns": vulns,
            "typosquatting":          typo,
            "infrastructure_abuse":   _infra,
            "threat_flags":           threat_flags,
        },
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Phishing URL Predictor")
    parser.add_argument("url", help="URL to analyse")
    args = parser.parse_args()

    result = predict_url(args.url)
    print(json.dumps(result, indent=2, ensure_ascii=False))