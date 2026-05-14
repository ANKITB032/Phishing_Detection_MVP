"""
predictor.py — Phishing URL Prediction Pipeline
=================================================
Loads the trained Random Forest model and provides a single-function
inference interface:  predict_url(url) -> dict

Pipeline order:
  0. Trusted-domain whitelist  (fast path — skips everything if matched)
  F. URL Shortener Expansion   (follow redirects → analyse the final URL)
  1. Cryptographic / vulnerability pre-screening
  2. ML feature extraction + Random Forest classification
  3. Threat-flag escalation (overrides ML if ≥ 2 red flags)
"""

import re
import math
import base64
import joblib
import os
import pandas as pd
import requests
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
#  MODULE F: URL SHORTENER EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════

# Known shortener domains — caught before HTTP expansion as a fast pre-check.
# Expansion via HTTP HEAD is also attempted for any unknown short domain
# (path with ≤ 10 chars, no query string) to catch novel shorteners.
KNOWN_SHORTENERS: set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "dlvr.it", "ift.tt", "lnk.ink", "lnkd.in", "rb.gy", "cutt.ly",
    "short.io", "tiny.cc", "is.gd", "v.gd", "bl.ink", "rebrand.ly",
    "shorte.st", "adf.ly", "bc.vc", "clk.sh", "mcaf.ee", "qr.ae",
    "su.pr", "tr.im", "u.to", "x.co", "zi.ma", "snip.ly",
    # Infrastructure / CDN abuse shorteners
    "firebaseapp.com", "web.app", "ipfs.io", "dweb.link",
    "pages.dev", "workers.dev", "netlify.app", "vercel.app",
    "github.io", "glitch.me", "repl.co", "ngrok.io",
}

_EXPAND_TIMEOUT  = 5      # seconds — HEAD request hard timeout
_EXPAND_MAX_HOPS = 10     # maximum redirect hops before giving up


def _is_likely_shortener(url: str) -> bool:
    """
    Return True if the URL is a known shortener OR looks like one
    (registered domain only, path ≤ 10 chars, no query string).
    """
    domain = _extract_domain(url)
    if domain in KNOWN_SHORTENERS:
        return True
    # Heuristic for unknown shorteners: very short path, no query
    parsed = urlparse(url if "://" in url else f"https://{url}")
    path   = parsed.path.strip("/")
    return len(path) <= 10 and not parsed.query and len(path) > 0


def expand_url(url: str) -> tuple[str, list[str]]:
    """
    Follow HTTP redirects and return the final destination URL.

    Uses HEAD (falls back to GET on 405) with a tight timeout.
    Captures the full redirect chain for audit purposes.

    Args:
        url: The raw (possibly shortened) URL to expand.

    Returns:
        (final_url, redirect_chain)
        - final_url:      The resolved destination, or the original on failure.
        - redirect_chain: Ordered list of intermediate URLs traversed.
    """
    normalized = url if "://" in url else f"https://{url}"
    chain: list[str] = []

    try:
        resp = requests.head(
            normalized,
            allow_redirects=True,
            timeout=_EXPAND_TIMEOUT,
            headers={"User-Agent": "PhishGuard-Scanner/2.0"},
        )
    except requests.exceptions.TooManyRedirects:
        return url, chain          # redirect loop — analyse the original
    except requests.exceptions.ConnectionError:
        return url, chain          # dead link — analyse the original
    except requests.RequestException:
        # Fall back to GET (some servers reject HEAD)
        try:
            resp = requests.get(
                normalized,
                allow_redirects=True,
                timeout=_EXPAND_TIMEOUT,
                stream=True,       # don't download the body
                headers={"User-Agent": "PhishGuard-Scanner/2.0"},
            )
            resp.close()
        except requests.RequestException:
            return url, chain

    # Build the redirect chain from response history
    for r in resp.history:
        if r.headers.get("Location"):
            chain.append(r.headers["Location"])

    final = resp.url
    return (final if final != normalized else url), chain


# ═══════════════════════════════════════════════════════════════════════════════
#  TRUSTED DOMAIN WHITELIST  (Fast Path)
# ═══════════════════════════════════════════════════════════════════════════════

TRUSTED_DOMAINS: set[str] = {
    "google.com",       "bing.com",         "yahoo.com",
    "duckduckgo.com",   "baidu.com",        "yandex.ru",
    "microsoft.com",    "apple.com",        "amazon.com",
    "meta.com",         "facebook.com",     "instagram.com",
    "twitter.com",      "x.com",            "linkedin.com",
    "github.com",       "gitlab.com",       "stackoverflow.com",
    "npmjs.com",        "pypi.org",         "docker.com",
    "youtube.com",      "netflix.com",      "spotify.com",
    "reddit.com",       "wikipedia.org",    "medium.com",
    "paypal.com",       "stripe.com",       "shopify.com",
    "ebay.com",         "etsy.com",
    "zoom.us",          "slack.com",        "notion.so",
    "dropbox.com",      "adobe.com",        "salesforce.com",
    "cloudflare.com",   "godaddy.com",      "namecheap.com",
    "letsencrypt.org",  "digicert.com",
    "outlook.com",      "proton.me",        "whatsapp.com",
    "bbc.com",          "cnn.com",          "nytimes.com",
    "coursera.org",     "edx.org",          "khanacademy.org",
}


def _extract_domain(url: str) -> str:
    normalized = url if "://" in url else f"https://{url}"
    hostname   = urlparse(normalized).hostname or ""
    hostname   = hostname.lower().rstrip(".")
    parts      = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def is_trusted_domain(url: str) -> bool:
    return _extract_domain(url) in TRUSTED_DOMAINS


# ═══════════════════════════════════════════════════════════════════════════════
#  BRAND FUZZY MATCHING  (Typosquatting Detection)
# ═══════════════════════════════════════════════════════════════════════════════

BRAND_MAP: dict[str, str] = {
    "microsoft":     "microsoft.com",
    "amazon":        "amazon.com",
    "apple":         "apple.com",
    "google":        "google.com",
    "facebook":      "facebook.com",
    "instagram":     "instagram.com",
    "netflix":       "netflix.com",
    "paypal":        "paypal.com",
    "linkedin":      "linkedin.com",
    "twitter":       "twitter.com",
    "dropbox":       "dropbox.com",
    "github":        "github.com",
    "spotify":       "spotify.com",
    "adobe":         "adobe.com",
    "yahoo":         "yahoo.com",
    "chase":         "chase.com",
    "wellsfargo":    "wellsfargo.com",
    "bankofamerica": "bankofamerica.com",
    "ebay":          "ebay.com",
    "walmart":       "walmart.com",
}


def _levenshtein(s: str, t: str) -> int:
    if len(s) < len(t):
        return _levenshtein(t, s)
    if not t:
        return len(s)
    prev = list(range(len(t) + 1))
    for i, cs in enumerate(s):
        curr = [i + 1]
        for j, ct in enumerate(t):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (cs != ct)))
        prev = curr
    return prev[-1]


def detect_typosquatting(url: str) -> dict:
    domain       = _extract_domain(url)
    domain_label = domain.split(".")[0].lower()
    for brand, official in BRAND_MAP.items():
        dist = _levenshtein(domain_label, brand)
        if 0 < dist <= 2 and domain != official:
            return {"is_typosquat": True, "matched_brand": brand.title(), "edit_distance": dist}
        if dist == 0 and domain != official:
            return {"is_typosquat": True, "matched_brand": brand.title(), "edit_distance": 0}
    return {"is_typosquat": False, "matched_brand": None, "edit_distance": None}


# ── Feature Extraction ───────────────────────────────────────────────────────

def extract_features(url: str) -> pd.DataFrame:
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
#  CRYPTOGRAPHIC & VULNERABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

_BASE64_RE      = re.compile(r"[A-Za-z0-9+/]{8,}={0,2}")
_HEX_ESCAPE_RE  = re.compile(r"(\\x[0-9a-fA-F]{2}){4,}")
_HEX_PERCENT_RE = re.compile(r"(%[0-9a-fA-F]{2}){2,}")
_HEX_0X_RE      = re.compile(r"0x[0-9a-fA-F]{8,}")


def detect_encoded_payloads(url: str) -> dict:
    parsed      = urlparse(url)
    search_area = parsed.path + "?" + parsed.query if parsed.query else parsed.path
    base64_matches = _BASE64_RE.findall(search_area)
    hex_matches    = (
        _HEX_ESCAPE_RE.findall(search_area)
        + _HEX_PERCENT_RE.findall(search_area)
        + _HEX_0X_RE.findall(search_area)
    )
    verified_b64 = []
    for candidate in base64_matches:
        try:
            padded  = candidate + "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(padded, validate=True)
            if decoded and sum(32 <= b < 127 for b in decoded) / len(decoded) > 0.7:
                verified_b64.append(candidate[:40] + ("…" if len(candidate) > 40 else ""))
        except Exception:
            pass
    return {
        "base64_found":    len(verified_b64) > 0,
        "hex_found":       len(hex_matches) > 0,
        "encoded_snippets": verified_b64[:3],
    }


ENTROPY_THRESHOLD = 4.0


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq   = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def analyse_query_entropy(url: str) -> dict:
    parsed  = urlparse(url)
    query   = parsed.query
    overall = shannon_entropy(query)
    details = {}
    for key, values in parse_qs(query).items():
        for val in values:
            details[key] = {
                "value_preview": val[:30] + ("…" if len(val) > 30 else ""),
                "entropy": round(shannon_entropy(val), 3),
            }
    return {
        "query_entropy": round(overall, 3),
        "high_entropy":  overall > ENTROPY_THRESHOLD,
        "param_details": details,
    }


VULN_PATTERNS = {
    "directory_traversal": [
        re.compile(r"\.\./"),
        re.compile(r"\.\.\%2[fF]"),
        re.compile(r"%2[eE]%2[eE]%2[fF]"),
        re.compile(r"\.\.\\"),
    ],
    "xss_tags": [
        re.compile(r"<\s*script", re.IGNORECASE),
        re.compile(r"javascript\s*:", re.IGNORECASE),
        re.compile(r"on\w+\s*=", re.IGNORECASE),
        re.compile(r"<\s*img[^>]+onerror", re.IGNORECASE),
        re.compile(r"<\s*svg[^>]+onload", re.IGNORECASE),
    ],
    "sql_injection": [
        re.compile(r"(\b(UNION|SELECT|DROP|INSERT|DELETE|UPDATE)\b.*\b(FROM|TABLE|INTO)\b)", re.IGNORECASE),
        re.compile(r"['\"];\s*--"),
        re.compile(r"1\s*=\s*1"),
        re.compile(r"'\s*OR\s+'", re.IGNORECASE),
    ],
    "command_injection": [
        re.compile(r";\s*(ls|cat|wget|curl|bash|sh|cmd)\b", re.IGNORECASE),
        re.compile(r"\|\s*(ls|cat|id|whoami)\b", re.IGNORECASE),
    ],
}


def detect_vulnerability_patterns(url: str) -> dict:
    findings = {}
    for category, patterns in VULN_PATTERNS.items():
        matched = [pat.pattern for pat in patterns if pat.search(url)]
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
    F. URL shortener expansion (follow redirects, analyse final URL).
    1. Run cryptographic / vulnerability pre-screening.
    2. Extract ML features and run the trained model.
    3. Merge all results into a single response dict.
    """
    # ── Step 0: Trusted-domain fast path ──────────────────────────────────
    if is_trusted_domain(url):
        domain = _extract_domain(url)
        return {
            "url":        url,
            "label":      0,
            "verdict":    "Safe",
            "confidence": 100.0,
            "reason":     f"Trusted domain ({domain}) — bypassed via whitelist.",
            "security_analysis": "Bypassed via trusted domain whitelist",
        }

    # ── Step F: URL Shortener Expansion ───────────────────────────────────
    original_url    = url
    redirect_chain  = []
    was_expanded    = False

    if _is_likely_shortener(url):
        expanded, redirect_chain = expand_url(url)
        if expanded != url:
            url          = expanded
            was_expanded = True

    # ── Pre-screening (on final URL) ────────────────────────────────────
    encoded  = detect_encoded_payloads(url)
    entropy  = analyse_query_entropy(url)
    vulns    = detect_vulnerability_patterns(url)
    typo     = detect_typosquatting(url)

    threat_flags = []
    if was_expanded:
        threat_flags.append(f"URL shortener resolved: {original_url} → {url}")
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

    # ── ML Inference (on final URL) ────────────────────────────────────
    X          = extract_features(url)
    label      = int(_model.predict(X)[0])
    proba      = _model.predict_proba(X)[0]
    confidence = float(proba[label])

    if label == 0 and len(threat_flags) >= 2:
        label      = 1
        confidence = max(confidence, 0.70)

    if typo["is_typosquat"]:
        label      = 1
        confidence = max(confidence, 0.85)

    verdict = "Safe" if label == 0 else "Phishing"

    parsed_url = urlparse(url)
    if label == 0 and parsed_url.query:
        confidence = min(confidence, 0.80)

    # ── Build detection reason ───────────────────────────────────────────
    reasons = []

    if was_expanded:
        reasons.append(
            f"Shortened URL resolved to: {url}."
            + (f" Traversed {len(redirect_chain)} redirect(s)." if redirect_chain else "")
        )
    if typo["is_typosquat"]:
        reasons.append(
            f"Visual Deception: This domain closely resembles "
            f"{typo['matched_brand']}, suggesting a typosquatting attack."
        )
    if encoded["base64_found"]:
        reasons.append("Encoded cryptographic payload detected in URL parameters.")
    if encoded["hex_found"]:
        reasons.append("Hex-encoded byte sequence found in URL.")
    if entropy["high_entropy"]:
        reasons.append(
            f"Abnormally high entropy ({entropy['query_entropy']} bits) in "
            f"query string suggests obfuscated data."
        )
    for category in vulns:
        reasons.append(f"Vulnerability pattern detected: {category.replace('_', ' ').title()}.")

    if not reasons:
        if label == 1:
            reasons.append("Model detected structural patterns common in phishing sites.")
        else:
            reasons.append("No suspicious indicators found. URL structure appears benign.")

    return {
        "url":          url,
        "original_url": original_url if was_expanded else None,
        "label":        label,
        "verdict":      verdict,
        "confidence":   round(confidence * 100, 2),
        "reason":       " ".join(reasons),
        "security_analysis": {
            "encoded_payloads":       encoded,
            "entropy_analysis":       entropy,
            "vulnerability_patterns": vulns,
            "typosquatting":          typo,
            "threat_flags":           threat_flags,
            "shortener_expansion": {
                "was_expanded":    was_expanded,
                "original_url":    original_url if was_expanded else None,
                "redirect_chain":  redirect_chain,
            },
        },
    }


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Phishing URL Predictor")
    parser.add_argument("url", help="URL to analyse")
    args   = parser.parse_args()

    result = predict_url(args.url)
    print(json.dumps(result, indent=2, ensure_ascii=False))