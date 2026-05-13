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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "04_Saved_Models", "phishing_model.joblib")
_model = joblib.load(MODEL_PATH)

SPECIAL_CHARS = "@-?=.#%+&_~/!"

FEATURES = [
    "url_length", "num_special_chars", "num_dots", "num_hyphens",
    "num_at", "num_query_params", "has_https", "has_ip",
    "subdomain_depth", "path_depth",
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


# ── Feature Extraction (mirrors feature_extractor.py) ────────────────────────

def extract_features(url: str) -> pd.DataFrame:
    """Extract the 10 ML features from a raw URL string."""
    feats = {
        "url_length":        len(url),
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_at":            url.count("@"),
        "num_query_params":  url.count("?") + url.count("&"),
        "has_https":         int(url.startswith("https")),
        "has_ip":            int(bool(re.search(r"(\d{1,3}\.){3}\d{1,3}", url))),
        "subdomain_depth":   max(0, len(url.split("/")[0].split(".")) - 2),
        "path_depth":        url.count("/"),
    }
    return pd.DataFrame([feats])[FEATURES]


# ═══════════════════════════════════════════════════════════════════════════════
#  CRYPTOGRAPHIC & VULNERABILITY ANALYSIS  (Task 3)
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1. Base64 / Hex Encoded Payload Detection ────────────────────────────────

# Regex for Base64 strings: 16+ chars from the Base64 alphabet ending with
# optional padding.  Minimum length avoids false-positives on short tokens.
_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

# Regex for hex-encoded byte strings (e.g. \x41\x42 or 0x4142 or %41%42)
_HEX_ESCAPE_RE  = re.compile(r"(\\x[0-9a-fA-F]{2}){4,}")
_HEX_PERCENT_RE = re.compile(r"(%[0-9a-fA-F]{2}){4,}")
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

    0. Trusted-domain fast path (whitelist bypass).
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
                "threat_flags": [str, ...]
            }
        }
    """
    # ── Step 0: Trusted-domain fast path ──────────────────────────────────
    if is_trusted_domain(url):
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

    threat_flags = []
    if encoded["base64_found"]:
        threat_flags.append("Base64-encoded payload detected")
    if encoded["hex_found"]:
        threat_flags.append("Hex-encoded payload detected")
    if entropy["high_entropy"]:
        threat_flags.append(f"High query-string entropy ({entropy['query_entropy']} bits)")
    for category in vulns:
        threat_flags.append(f"Vulnerability pattern: {category.replace('_', ' ')}")

    # ── ML Inference ─────────────────────────────────────────────────────
    X = extract_features(url)
    label      = int(_model.predict(X)[0])
    proba      = _model.predict_proba(X)[0]
    confidence = float(proba[label])

    # If the pre-screening found threats, escalate a "benign" ML result
    if label == 0 and len(threat_flags) >= 2:
        label = 1
        confidence = max(confidence, 0.70)

    verdict = "benign" if label == 0 else "malicious"

    # ── Build detection reason ───────────────────────────────────────────
    reasons = []
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
