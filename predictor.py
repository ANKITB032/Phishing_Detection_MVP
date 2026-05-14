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
import binascii
import joblib
import os
import logging
import pandas as pd
import requests
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)

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
    # AI / research
    "claude.ai",        "anthropic.com",
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
    # Hosting & deploy
    "vercel.app",
    # Email & comms
    "outlook.com",      "proton.me",        "whatsapp.com",
    # News / reference
    "bbc.com",          "cnn.com",          "nytimes.com",
    # Education
    "coursera.org",     "edx.org",          "khanacademy.org",
    # Personal / portfolio
    "ankitband.me",
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
        try:
            padded = candidate + "=" * (-len(candidate) % 4)
            decoded = base64.b64decode(padded)
            # Only flag if decoded bytes look like printable ASCII (payload text)
            if decoded and sum(32 <= b < 127 for b in decoded) / len(decoded) > 0.7:
                verified_b64.append(candidate[:40] + ("…" if len(candidate) > 40 else ""))
        except (binascii.Error, ValueError):
            # Not valid Base64 — skip silently
            continue

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
#  MODULE A — SUSPICIOUS TLD CHECK
# ═══════════════════════════════════════════════════════════════════════════════

# TLDs disproportionately used in phishing campaigns (cheap/free registration).
SUSPICIOUS_TLDS: set[str] = {
    ".top", ".xyz", ".tk", ".ml", ".ga", ".cf", ".gq",
    ".buzz", ".club", ".work", ".info", ".icu", ".cam",
    ".rest", ".surf", ".monster", ".sbs",
}


def check_suspicious_tld(url: str) -> dict:
    """
    Flag the URL if its TLD belongs to the known-abused set.

    Returns:
        {"flagged": bool, "tld": str}
    """
    domain = _extract_domain(url)
    tld = "." + domain.split(".")[-1].lower() if "." in domain else ""
    return {"flagged": tld in SUSPICIOUS_TLDS, "tld": tld}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE B — URL COMPLEXITY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

MAX_SAFE_DOTS   = 5     # more than 5 dots → suspicious subdomain chaining
MAX_SAFE_LENGTH = 100   # URLs > 100 chars often hide payloads in path/query


def check_url_complexity(url: str) -> dict:
    """
    Flag URLs that are structurally complex — long length or excessive
    dot-separated labels — which correlates with phishing infrastructure.

    Returns:
        {"is_complex": bool, "dot_count": int, "url_length": int, "details": list[str]}
    """
    dot_count  = url.count(".")
    url_length = len(url)
    details    = []

    if dot_count > MAX_SAFE_DOTS:
        details.append(f"Excessive dot count ({dot_count}) suggests subdomain chaining.")
    if url_length > MAX_SAFE_LENGTH:
        details.append(f"URL length ({url_length} chars) exceeds safe threshold.")

    return {
        "is_complex": len(details) > 0,
        "dot_count":  dot_count,
        "url_length": url_length,
        "details":    details,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE C — PHISHING KEYWORD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

# Keywords commonly found in credential-harvesting paths.
PHISHING_KEYWORDS: set[str] = {
    "login", "signin", "sign-in", "log-in",
    "verify", "verification", "validate",
    "update", "upgrade",
    "account", "myaccount", "my-account",
    "banking", "secure", "security",
    "password", "passwd", "credential",
    "confirm", "suspend", "alert",
    "wallet", "recover", "unlock",
}


def check_phishing_keywords(url: str) -> dict:
    """
    Scan the URL path (everything after the domain) for credential-harvesting
    keywords that are strongly associated with phishing landing pages.

    Returns:
        {"found": bool, "matched_keywords": list[str]}
    """
    parsed = urlparse(url if "://" in url else f"https://{url}")
    # Examine the path + query (lowercased) for keyword presence
    search_area = (parsed.path + "?" + parsed.query).lower() if parsed.query \
        else parsed.path.lower()

    matched = [kw for kw in PHISHING_KEYWORDS if kw in search_area]
    return {"found": len(matched) > 0, "matched_keywords": sorted(set(matched))}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE E — IPFS GATEWAY CHECK
# ═══════════════════════════════════════════════════════════════════════════════

# Known public IPFS gateways used to serve decentralised content.
# Phishing kits are increasingly hosted on IPFS because the content
# is immutable and cannot be taken down by a single provider.
IPFS_GATEWAYS: set[str] = {
    "ipfs.io",
    "cloudflare-ipfs.com",
    "pinata.cloud",
    "gateway.pinata.cloud",
    "dweb.link",
    "w3s.link",
    "nftstorage.link",
}

# CIDv0 = 46-char base58 starting with Qm; CIDv1 = 59+ char base32/base36
_IPFS_CID_RE = re.compile(
    r"(?:/ipfs/|/ipns/)"
    r"(?P<cid>[A-Za-z0-9]{46,})",
)

# Optional: hashes that have been manually verified as safe.
WHITELISTED_IPFS_HASHES: set[str] = set()


def check_ipfs_gateway(url: str) -> dict:
    """
    Detect whether the URL routes through a known IPFS gateway and
    contains a long content-addressed hash (CID).

    Returns:
        {
            "is_ipfs":   bool,
            "gateway":   str | None,
            "cid":       str | None,
            "whitelisted": bool,
        }
    """
    parsed = urlparse(url if "://" in url else f"https://{url}")
    hostname = (parsed.hostname or "").lower().rstrip(".")

    # Check if the hostname matches or is a subdomain of a known gateway
    gateway_match = None
    for gw in IPFS_GATEWAYS:
        if hostname == gw or hostname.endswith(f".{gw}"):
            gateway_match = gw
            break

    if not gateway_match:
        return {"is_ipfs": False, "gateway": None, "cid": None, "whitelisted": False}

    # Look for a CID in the path
    cid_match = _IPFS_CID_RE.search(parsed.path)
    if not cid_match:
        # Some gateways use subdomain-style: <cid>.ipfs.<gateway>
        # e.g. bafybei...ipfs.dweb.link
        parts = hostname.split(".")
        if len(parts) >= 3 and len(parts[0]) >= 46:
            cid = parts[0]
            is_safe = cid in WHITELISTED_IPFS_HASHES
            return {"is_ipfs": True, "gateway": gateway_match, "cid": cid, "whitelisted": is_safe}
        return {"is_ipfs": False, "gateway": gateway_match, "cid": None, "whitelisted": False}

    cid = cid_match.group("cid")
    is_safe = cid in WHITELISTED_IPFS_HASHES
    return {"is_ipfs": True, "gateway": gateway_match, "cid": cid, "whitelisted": is_safe}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE F — URL SHORTENER EXPANSION
# ═══════════════════════════════════════════════════════════════════════════════

SHORTENER_DOMAINS: set[str] = {
    "bit.ly", "t.co", "tinyurl.com", "lnk.ink",
    "cutt.ly", "t.ly", "goo.gl", "ow.ly",
    "is.gd", "buff.ly", "rb.gy", "short.io",
}

_EXPAND_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def is_shortener(url: str) -> bool:
    """Return True if the URL belongs to a known shortener domain."""
    return _extract_domain(url) in SHORTENER_DOMAINS


def expand_short_url(url: str) -> dict:
    """
    Follow redirects on a shortened URL to discover the final destination.

    Uses a HEAD request with a 2-second timeout to avoid hanging on
    unresponsive hosts.

    Returns:
        {
            "is_short":     bool,
            "original_url": str,
            "expanded_url": str | None,
            "error":        str | None,
        }
    """
    if not is_shortener(url):
        return {"is_short": False, "original_url": url, "expanded_url": None, "error": None}

    normalized = url if "://" in url else f"https://{url}"
    try:
        resp = requests.head(
            normalized,
            allow_redirects=True,
            timeout=2,
            headers=_EXPAND_HEADERS,
        )
        final = resp.url
        # Guard against redirect loops back to the same shortener
        if _extract_domain(final) in SHORTENER_DOMAINS:
            return {"is_short": True, "original_url": url, "expanded_url": final, "error": "Redirect loop"}
        return {"is_short": True, "original_url": url, "expanded_url": final, "error": None}
    except requests.RequestException as exc:
        logger.warning("Shortener expansion failed for %s: %s", url, exc)
        return {"is_short": True, "original_url": url, "expanded_url": None, "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIFIED PREDICTION INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def _analyse_url(url: str) -> dict:
    """
    Core analysis engine — runs the full heuristic + ML pipeline on a
    single URL.  Called directly for normal URLs and recursively for
    the expanded destination of shortened URLs.
    """
    # ── Step 0: Trusted-domain fast path ──────────────────────────────────
    if is_trusted_domain(url):
        return {
            "url":        url,
            "label":      0,
            "verdict":    "benign",
            "confidence": 100.0,
            "reason":     "Verified Source: This domain is on the global trusted whitelist.",
            "security_analysis": "Bypassed via trusted domain whitelist",
        }

    # ── Pre-screening ────────────────────────────────────────────────────
    encoded    = detect_encoded_payloads(url)
    entropy    = analyse_query_entropy(url)
    vulns      = detect_vulnerability_patterns(url)
    typo       = detect_typosquatting(url)
    tld_check  = check_suspicious_tld(url)
    complexity = check_url_complexity(url)
    keywords   = check_phishing_keywords(url)
    ipfs       = check_ipfs_gateway(url)

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
    if tld_check["flagged"]:
        threat_flags.append(f"Suspicious TLD: {tld_check['tld']}")
    if complexity["is_complex"]:
        threat_flags.append("URL structural complexity exceeded safe thresholds")
    if keywords["found"]:
        threat_flags.append(f"Phishing keywords in path: {', '.join(keywords['matched_keywords'])}")
    if ipfs["is_ipfs"] and not ipfs["whitelisted"]:
        threat_flags.append(f"IPFS gateway detected: {ipfs['gateway']}")

    # ── ML Inference ─────────────────────────────────────────────────────
    X = extract_features(url)
    label      = int(_model.predict(X)[0])
    proba      = _model.predict_proba(X)[0]
    confidence = float(proba[label])

    # ── Heuristic flag counter (A-F modules) ─────────────────────────────
    heuristic_count = sum([
        typo["is_typosquat"],                          # A — Typosquatting
        tld_check["flagged"],                          # B — Suspicious TLD
        keywords["found"],                             # C — Phishing keywords
        complexity["is_complex"],                      # D — URL complexity
        ipfs["is_ipfs"] and not ipfs["whitelisted"],   # E — IPFS gateway
    ])

    # ── Trust Protocol: ML-malicious but zero heuristic flags → benign ──
    trust_override = False
    if label == 1 and heuristic_count == 0:
        label = 0
        trust_override = True
        confidence = float(proba[0])

    # ── Heuristic escalation (only if NOT trust-overridden) ─────────────
    if not trust_override:
        if label == 0 and len(threat_flags) >= 2:
            label = 1
            confidence = max(confidence, 0.70)

        if typo["is_typosquat"]:
            label = 1
            confidence = max(confidence, 0.85)

        if tld_check["flagged"] and len(threat_flags) >= 2:
            label = 1
            confidence = max(confidence, 0.75)

        if keywords["found"] and label == 0 and len(threat_flags) >= 1:
            label = 1
            confidence = max(confidence, 0.72)

    # ── IPFS escalation: overrides Trust Protocol — always malicious ────
    if ipfs["is_ipfs"] and not ipfs["whitelisted"]:
        label = 1
        trust_override = False
        confidence = min(confidence, 0.65)

    verdict = "benign" if label == 0 else "malicious"

    # ── Confidence cap: never say 100% benign with query parameters ──────
    parsed_url = urlparse(url)
    if label == 0 and parsed_url.query:
        confidence = min(confidence, 0.80)

    # ══════════════════════════════════════════════════════════════════════
    #  REASONING HIERARCHY
    #  Trust-override → Heuristics (A-E) → Crypto/vuln → Model fallback
    # ══════════════════════════════════════════════════════════════════════
    reasons = []

    if trust_override:
        reasons.append(
            "Matches clean structural profiles. "
            "No deceptive keywords or hidden payloads detected."
        )

    if not trust_override:
        # A — Typosquatting
        if typo["is_typosquat"]:
            reasons.append(
                f"Visual Deception: This domain is highly similar to "
                f"{typo['matched_brand']}, suggesting a typosquatting attack."
            )

        # B — Suspicious TLD
        if tld_check["flagged"]:
            reasons.append(
                f"Risky TLD: The domain uses '{tld_check['tld']}', a top-level "
                f"domain frequently associated with phishing campaigns."
            )

        # C — Phishing keywords (enriched for OpenPhish)
        if keywords["found"]:
            kw_list = ", ".join(keywords["matched_keywords"])
            parsed_tmp = urlparse(url if "://" in url else f"https://{url}")
            sub_depth  = max(0, len((parsed_tmp.hostname or "").split(".")) - 2)
            path_seg   = parsed_tmp.path.strip("/").count("/") + 1 if parsed_tmp.path.strip("/") else 0
            reasons.append(
                f"Credential-harvesting keywords ({kw_list}) found across "
                f"{path_seg} path segment(s) at subdomain depth {sub_depth}."
            )

        # D — URL complexity (enriched with subdomain depth)
        if complexity["is_complex"]:
            parsed_tmp = urlparse(url if "://" in url else f"https://{url}")
            sub_depth  = max(0, len((parsed_tmp.hostname or "").split(".")) - 2)
            for detail in complexity["details"]:
                reasons.append(f"{detail} (subdomain depth: {sub_depth})")

        # E — IPFS Gateway
        if ipfs["is_ipfs"] and not ipfs["whitelisted"]:
            cid_preview = f" CID: {ipfs['cid'][:16]}..." if ipfs["cid"] and len(ipfs["cid"]) > 16 else ""
            reasons.append(
                f"IPFS Gateway detected ({ipfs['gateway']}): "
                f"Decentralised hosting is frequently used to serve phishing "
                f"content that bypasses traditional domain filters.{cid_preview}"
            )

    # ── SECONDARY: Crypto / vuln details ────────────────────────────────
    if not trust_override:
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
            readable = category.replace("_", " ").title()
            reasons.append(f"Vulnerability pattern detected: {readable}.")

    # ── FALLBACK: Model patterns ────────────────────────────────────────
    if not reasons:
        if label == 1:
            reasons.append(
                "Model Patterns: Structural analysis detected statistical "
                "anomalies consistent with known phishing infrastructure."
            )
        else:
            reasons.append(
                "No suspicious indicators found. URL structure appears benign."
            )

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
            "suspicious_tld":         tld_check,
            "url_complexity":         complexity,
            "phishing_keywords":      keywords,
            "threat_flags":           threat_flags,
            "trust_override":         trust_override,
            "ipfs_gateway":           ipfs,
        },
    }


def predict_url(url: str) -> dict:
    """
    Public prediction interface with URL-shortener expansion (Module F).

    If the URL belongs to a known shortener, it is expanded via HTTP HEAD
    and the *expanded* URL is recursively analysed through the full pipeline.
    The result includes both the original and expanded URLs along with the
    analysis of the final destination.
    """
    shortener = expand_short_url(url)

    if shortener["is_short"] and shortener["expanded_url"] and not shortener["error"]:
        # Analyse the expanded destination
        expanded = shortener["expanded_url"]
        result = _analyse_url(expanded)

        # Cap shortener confidence at 70 %
        if result["confidence"] > 70.0:
            result["confidence"] = 70.0
        # Force malicious if destination was also flagged
        if result["label"] == 0:
            result["label"] = 1
            result["verdict"] = "malicious"
            result["confidence"] = max(result["confidence"], 65.0)

        # Prepend shortener context to reason
        result["reason"] = (
            f"URL Shortener ({_extract_domain(url)}): "
            f"Expanded to {expanded}. " + result["reason"]
        )
        # Attach shortener metadata
        result["url"] = url
        if isinstance(result["security_analysis"], dict):
            result["security_analysis"]["shortener_expansion"] = shortener
        return result

    if shortener["is_short"] and shortener["error"]:
        # Could not expand — flag as suspicious shortener
        result = _analyse_url(url)
        result["label"] = 1
        result["verdict"] = "malicious"
        result["confidence"] = max(result["confidence"], 65.0)
        result["reason"] = (
            f"URL Shortener ({_extract_domain(url)}): "
            f"Expansion failed ({shortener['error']}). Treating as suspicious. "
            + result["reason"]
        )
        if isinstance(result["security_analysis"], dict):
            result["security_analysis"]["shortener_expansion"] = shortener
        return result

    # Not a shortener — normal analysis
    result = _analyse_url(url)
    if isinstance(result["security_analysis"], dict):
        result["security_analysis"]["shortener_expansion"] = shortener
    return result


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, json

    parser = argparse.ArgumentParser(description="Phishing URL Predictor")
    parser.add_argument("url", help="URL to analyse")
    args = parser.parse_args()

    result = predict_url(args.url)
    print(json.dumps(result, indent=2, ensure_ascii=False))
