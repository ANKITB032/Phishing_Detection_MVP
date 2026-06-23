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
import tldextract
from urllib.parse import urlparse, parse_qs

# ── Optional: beautifulsoup4 for live HTML title extraction ──────────────────
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "beautifulsoup4 not installed — live title checks disabled. "
        "Run: pip install beautifulsoup4"
    )

logger = logging.getLogger(__name__)

# ── Model Loading ────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "04_Saved_Models", "phishing_model_v3_5.joblib")
_model = joblib.load(MODEL_PATH)

SPECIAL_CHARS = "@-?=.#%+&_~/!"

FEATURES = [
    "url_length", "num_special_chars", "num_dots", "num_hyphens",
    "num_at", "num_query_params", "has_https",
    "subdomain_depth", "path_depth",
    "is_ip", "path_brand",
    "url_in_query", "tld_risk",    # v3.5 FN patch
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
    # Hosting & deploy (root domains — subdomain abuse handled by infra patch)
    # github.io removed: Rule 8 brand-in-path check requires it to reach _analyse_url()
    "vercel.app",
    # Email & comms
    "outlook.com",      "proton.me",        "whatsapp.com",
    # News / reference
    "bbc.com",          "cnn.com",          "nytimes.com",
    # Education
    "coursera.org",     "edx.org",          "khanacademy.org",
    # Personal / portfolio
    "ankitband.me",
    # Regional variants — prevent false positives on ccTLD domains
    "google.co.uk",   "google.com.au",  "google.co.in",
    "amazon.co.uk",   "amazon.com.au",  "amazon.co.in",
    "bbc.co.uk",      "microsoft.co.uk",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  INFRASTRUCTURE ABUSE PATCH (v3.5)
#  Detects attackers abusing free-hosting subdomains of trusted platforms.
#  Must run BEFORE the whitelist check — see Step 0a in _analyse_url().
# ═══════════════════════════════════════════════════════════════════════════════

FREE_HOSTING_PROVIDERS: set[str] = {
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app",
    "github.io", "pages.dev", "workers.dev", "glitch.me",
    "repl.co", "ngrok.io", "onrender.com", "fly.dev",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE CONTENT ANALYSIS  (Module L — Brand Impersonation in HTML Title)
# ═══════════════════════════════════════════════════════════════════════════════

# Maps brand keyword (lowercase) → official registrable domain.
# If a fetched page <title> contains the brand keyword but the URL's domain
# does NOT match the official domain, it is a brand-impersonation-in-title hit.
TARGET_BRANDS: dict[str, str] = {
    "paypal":         "paypal.com",
    "netflix":        "netflix.com",
    "amazon":         "amazon.com",
    "apple":          "apple.com",
    "bank of america": "bankofamerica.com",
    "microsoft":      "microsoft.com",
    "google":         "google.com",
    "facebook":       "facebook.com",
    "instagram":      "instagram.com",
    "linkedin":       "linkedin.com",
    "dropbox":        "dropbox.com",
    "spotify":        "spotify.com",
    "chase":          "chase.com",
    "wellsfargo":     "wellsfargo.com",
    "ebay":           "ebay.com",
}

# Free / abused hosting platforms where brand-title impersonation is
# especially dangerous — domains on these hosts are attacker-controlled.
BRAND_TITLE_FREE_HOSTS: set[str] = {
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app",
    "github.io", "pages.dev", "workers.dev", "glitch.me",
    "repl.co", "ngrok.io", "onrender.com", "fly.dev",
    "wixsite.com", "wix.com", "weebly.com", "squarespace.com",
    "000webhostapp.com", "byethost.com", "infinityfree.net",
}

_INFRA_SENSITIVE_KEYWORDS: list[str] = [
    "verify", "service", "secure", "login", "account", "update",
    "confirm", "support", "billing", "payment", "auth", "signin",
    "password", "recover", "unlock", "validate", "alert", "admin",
    "clone", "wallet", "bridge", "swap", "ledger",
]

_SUBDOMAIN_ENTROPY_THRESHOLD = 3.5


def _subdomain_entropy(hostname: str) -> float:
    """Shannon entropy of the subdomain portion (everything left of eTLD+1)."""
    parts = hostname.rstrip(".").split(".")
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
    e.g. service-mitld.firebaseapp.com  →  is_abuse=True

    Fires when:
      1. eTLD+1 is in FREE_HOSTING_PROVIDERS.
      2. A non-empty subdomain exists.
      3. Subdomain contains a sensitive keyword OR Shannon entropy > 3.5 bits.
    """
    normalized = url if "://" in url else f"https://{url}"
    hostname   = (urlparse(normalized).hostname or "").lower().rstrip(".")
    parts      = hostname.split(".")
    etld1      = ".".join(parts[-2:]) if len(parts) >= 2 else hostname

    if etld1 not in FREE_HOSTING_PROVIDERS:
        return {"is_abuse": False, "provider": None, "subdomain": None, "reason": ""}

    if len(parts) <= 2:
        return {"is_abuse": False, "provider": etld1, "subdomain": None,
                "reason": "Root provider domain — no subdomain present."}

    subdomain    = ".".join(parts[:-2])
    keyword_hit  = next((kw for kw in _INFRA_SENSITIVE_KEYWORDS if kw in subdomain.lower()), None)
    entropy      = _subdomain_entropy(hostname)
    high_entropy = entropy > _SUBDOMAIN_ENTROPY_THRESHOLD
    is_abuse     = bool(keyword_hit or high_entropy)

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


PATH_BRANDS: list[str] = [
    "paypal", "microsoft", "apple", "google", "amazon",
    "facebook", "netflix", "instagram", "linkedin", "twitter",
    "dropbox", "github", "spotify", "adobe", "yahoo",
    "chase", "wellsfargo", "bankofamerica", "ebay", "walmart",
]

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
_FETCH_TIMEOUT = 3  # seconds — strict so the API never hangs


def fetch_live_content(url: str) -> dict:
    """
    Attempt a live GET request and extract the page <title> via BeautifulSoup.

    Returns:
        {
            "title":   str | None  — stripped <title> text, or None on failure,
            "fetched": bool        — True if request succeeded,
            "error":   str | None  — short exception description on failure,
        }

    All failure modes (no bs4, DNS error, timeout, HTTP 4xx/5xx, etc.) are
    caught and returned as title=None so the caller always falls back to ML.
    """
    if not _BS4_AVAILABLE:
        return {"title": None, "fetched": False, "error": "bs4 not installed"}

    normalized = url if "://" in url else f"https://{url}"
    try:
        resp = requests.get(
            normalized,
            headers=_FETCH_HEADERS,
            timeout=_FETCH_TIMEOUT,
            allow_redirects=True,
            stream=False,
        )
        resp.raise_for_status()
        content = resp.content[:51200]  # cap at 50 KB — avoid hanging on huge pages
        soup  = BeautifulSoup(content, "html.parser")
        tag   = soup.find("title")
        title = tag.get_text(strip=True) if tag else None
        return {"title": title, "fetched": True, "error": None}
    except Exception as exc:
        logger.debug("fetch_live_content failed for %s: %s", url, exc)
        return {"title": None, "fetched": False, "error": str(exc)[:120]}


def _extract_domain(url: str) -> str:
    """
    Extract the registrable domain (eTLD+1) from a URL.

    Uses tldextract for accurate parsing of multi-part TLDs (e.g. .co.uk,
    .com.au) — the previous naive split(".")[-2:] approach incorrectly
    returned 'co.uk' instead of 'example.co.uk' for such domains.

    Examples:
        https://mail.google.com/inbox    →  google.com
        http://docs.github.com:8080      →  github.com
        https://bbc.co.uk/news           →  bbc.co.uk
        http://192.168.1.1/admin         →  192.168.1.1  (IPv4 fallback)
        github.com/user/repo             →  github.com
    """
    extracted = tldextract.extract(url)
    # registered_domain = SLD + TLD (e.g. "google.com", "bbc.co.uk")
    # Fall back to ipv4 for bare-IP URLs, then to the raw host label.
    return extracted.registered_domain or extracted.ipv4 or extracted.host


def is_trusted_domain(url: str) -> bool:
    """
    Return True if the URL's domain belongs to the trusted whitelist.

    Special carve-out: sites.google.com and any *.google.com/view/* URL are
    attacker-controlled Google Sites pages — they must NOT be trusted even
    though google.com is on the whitelist.
    """
    normalized = url if "://" in url else f"https://{url}"
    parsed     = urlparse(normalized)
    hostname   = (parsed.hostname or "").lower().rstrip(".")
    path       = parsed.path.lower()

    # Carve out Google Sites — user-generated content on google.com infra
    if hostname == "sites.google.com":
        return False
    if hostname.endswith(".google.com") and path.startswith("/view/"):
        return False

    return _extract_domain(url) in TRUSTED_DOMAINS

# ── FN Patch constants (v3.5 error-analysis) ─────────────────────────────────

# Detects a second embedded URL in the query string (open redirect bait).
_URL_IN_QUERY_RE = re.compile(
    r"(https?://|//|www\.)[a-z0-9\-]+\.[a-z]{2,}",
    re.IGNORECASE,
)

# Matches a bare IPv4 address hostname (e.g. 192.168.1.1)
_IP_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")

# TLD ordinal risk score: 0=low, 1=medium-risk, 2=high-abuse
_TLD_RISK: dict[str, int] = {
    ".info": 1, ".biz": 1, ".name": 1, ".mobi": 1, ".pro": 1,
    ".top": 2, ".xyz": 2, ".tk": 2, ".ml": 2, ".ga": 2,
    ".cf": 2, ".gq": 2, ".buzz": 2, ".club": 2, ".work": 2,
    ".icu": 2, ".cam": 2, ".rest": 2, ".surf": 2, ".monster": 2, ".sbs": 2,
}


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
    "icloud":      "apple.com",
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
    # If the full hostname belongs to the trusted whitelist, it is a legitimate
    # regional or subdomain of a known brand — never a typosquat.
    normalized = url if "://" in url else f"https://{url}"
    _full_host = (urlparse(normalized).hostname or "").lower().rstrip(".")
    if _full_host in TRUSTED_DOMAINS or _extract_domain(url) in TRUSTED_DOMAINS:
        return {"is_typosquat": False, "matched_brand": None, "edit_distance": None}

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
            # Skip if the registered domain is itself a free/abused hosting
            # platform — e.g. github.io: the SLD "github" matching the brand
            # "github" is expected, not a typosquat.
            if domain in FREE_HOSTING_PROVIDERS or domain in ABUSED_FREE_HOSTS:
                continue
            return {
                "is_typosquat":  True,
                "matched_brand": brand.title(),
                "edit_distance": 0,
            }

    return {"is_typosquat": False, "matched_brand": None, "edit_distance": None}

def extract_features(url: str) -> pd.DataFrame:
    """
    Extract ML features from a raw URL string.

    All hostname/path/query values are derived from urllib.parse.urlparse so
    the logic matches the feature engineering used during training and avoids
    the raw url.split('/')[0] skew that caused is_ip, subdomain_depth, and
    path_brand to be calculated on the wrong URL segment in production.
    """
    # Normalise: ensure a scheme is present so urlparse works correctly
    normalized = url if "://" in url else f"https://{url}"
    parsed     = urlparse(normalized)
    hostname   = (parsed.hostname or "").lower().rstrip(".")
    path       = parsed.path
    query      = parsed.query

    # subdomain_depth: number of labels beyond the registrable domain (eTLD+1)
    # e.g. mail.google.com → depth 1;  a.b.evil.co.uk → depth 2
    host_labels     = hostname.split(".") if hostname else []
    subdomain_depth = max(0, len(host_labels) - 2)

    # is_ip: True when the hostname is a bare IPv4 address
    is_ip = int(bool(_IP_RE.match(hostname)))

    # path_brand: brand keyword found anywhere in path or query
    path_and_query = (path + "?" + query).lower() if query else path.lower()
    path_brand     = int(any(b in path_and_query for b in PATH_BRANDS))

    feats = {
        "url_length":        len(url),
        "num_special_chars": sum(url.count(c) for c in SPECIAL_CHARS),
        "num_dots":          url.count("."),
        "num_hyphens":       url.count("-"),
        "num_at":            url.count("@"),
        "num_query_params":  url.count("?") + url.count("&"),
        "has_https":         int(parsed.scheme == "https"),
        "subdomain_depth":   subdomain_depth,
        "path_depth":        len([s for s in path.split("/") if s]),
        "is_ip":             is_ip,
        "path_brand":        path_brand,
        # v3.5 FN patch
        "url_in_query": int(bool(_URL_IN_QUERY_RE.search(query))),
        "tld_risk":     _TLD_RISK.get("." + hostname.split(".")[-1] if hostname else "", 0),
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
    # Extended — confirmed phishing abuse (HFN analysis 2026-05-19)
    ".tech", ".live", ".online", ".site", ".space", ".fun",
    ".click", ".link", ".pw", ".cc", ".ws",
    # Additional high-abuse TLDs (2026-06-17)
    ".cfd", ".ru", ".cn", ".vip", ".id", ".et",
    # Further additions (2026-06-18)
    ".cyou", ".shop",
}

# Free / abused hosting platforms where any subdomain is attacker-controlled.
# Distinct from FREE_HOSTING_PROVIDERS (which requires keyword/entropy check);
# these are flagged unconditionally when combined with a brand/cred signal.
ABUSED_FREE_HOSTS: set[str] = {
    "000webhostapp.com", "byethost.com", "infinityfree.net",
    "freehosting.com", "awardspace.com", "biz.nf", "ezyro.com",
    "hostfree.pw", "webhostapp.com", "atwebpages.com",
    # Already in FREE_HOSTING_PROVIDERS — duplicate here for unconditional check
    "firebaseapp.com", "web.app", "vercel.app", "netlify.app",
    "github.io", "pages.dev", "workers.dev", "glitch.me",
    "repl.co", "onrender.com", "fly.dev", "ngrok.io",
    # Extended — web builder / CMS platforms abused for phishing (2026-06-17)
    "wixstudio.com", "webflow.io", "blogspot.com", "weebly.com",
    "wix.com", "hostingersite.com",
    # No-code / automation platforms abused for phishing (2026-06-17)
    "godaddysites.com", "zapier.app",
    # Cloud / PaaS platforms abused for phishing (2026-06-18)
    "gitbook.io", "railway.app", "azurewebsites.net", "edgeone.app",
    "edgeone.cool", "edgeone.dev", "wasmer.app", "replit.app",
}

# Regex: hyphenated credential / brand-impersonation patterns in hostname labels
# Matches both directions:
#   brand-first:  paypal-secure-login, apple-id-verify, appieid-enable
#   cred-first:   secure-paypal, login-microsoft, verify-amazon
_HYPHEN_CRED_RE = re.compile(
    r"(?:"
    # brand-first (including common typos: appieid, googel, etc.)
    r"(apple|appl[ei]|appleid|appieid|google|googl[ei]|paypal|paypall|microsoft|amazon|amaz0n"
    r"|facebook|instagram|netflix|linkedin|twitter|dropbox|github|spotify|adobe|yahoo"
    r"|chase|ebay|wellsfargo|bankofamerica|walmart|icloud|i-cloud)"
    r"[\-][a-z0-9\-]{2,}"
    r"|"
    # cred-first: credential word hyphenated to anything >=3 chars
    r"(signin|sign-in|secure|login|log-in|verify|account|update|confirm|support|billing|recovery|enable|activate)"
    r"[\-][a-z0-9\-]{3,}"
    r")",
    re.IGNORECASE,
)


def brand_in_subdomain(url: str) -> dict:

    """
    Detect brand names planted as subdomain labels on an untrusted domain.
    e.g. facebook.unitedcolleges.net  →  brand='facebook', host_domain='unitedcolleges.net'

    Only fires when the brand label is NOT the registrable domain itself
    (i.e., it is not facebook.com).
    """
    normalized = url if "://" in url else f"https://{url}"
    hostname   = (urlparse(normalized).hostname or "").lower().rstrip(".")
    parts      = hostname.split(".")
    etld1      = ".".join(parts[-2:]) if len(parts) >= 2 else hostname

    # If the full hostname is trusted, it is a legitimate regional/subdomain —
    # not an attacker planting a brand label on an untrusted host.
    if hostname in TRUSTED_DOMAINS or etld1 in TRUSTED_DOMAINS:
        return {"found": False, "brand": None, "host_domain": etld1}

    if len(parts) < 3:
        return {"found": False, "brand": None, "host_domain": etld1}

    # Check every non-eTLD label for an exact brand match
    subdomain_labels = parts[:-2]
    for label in subdomain_labels:
        for brand, official in BRAND_MAP.items():
            if label == brand and etld1 != official:
                return {"found": True, "brand": brand.title(), "host_domain": etld1}
    return {"found": False, "brand": None, "host_domain": etld1}


def check_hyphenated_creds(url: str) -> dict:
    """
    Detect credential-harvesting hostnames that use hyphens to mimic brands.
    e.g.  appieid-enable.com   drive-google-com.xyz   paypal-secure-login.net

    Scans only the hostname (not path) to avoid false positives on
    legitimate dashed product URLs.
    """
    normalized = url if "://" in url else f"https://{url}"
    hostname   = (urlparse(normalized).hostname or "").lower().rstrip(".")
    # Examine labels left of the eTLD
    parts  = hostname.split(".")
    labels = ".".join(parts[:-1]) if len(parts) > 1 else hostname
    match  = _HYPHEN_CRED_RE.search(labels)
    return {
        "found":   bool(match),
        "matched": match.group(0) if match else None,
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
    # ── Step 0a: Infrastructure Abuse Pre-Check ───────────────────────────
    # Must run BEFORE the whitelist — free-hosting subdomains (e.g.
    # service-mitld.firebaseapp.com) resolve to a trusted eTLD+1 and would
    # otherwise bypass all analysis.
    _infra          = check_infrastructure_abuse(url)
    infra_abuse_flag = _infra["is_abuse"]

    # ── Step 0: Trusted-domain fast path (bypassed if infra abuse detected) ──
    if is_trusted_domain(url) and not infra_abuse_flag:
        return {
            "url":        url,
            "label":      0,
            "verdict":    "benign",
            "confidence": 100.0,
            "reason":     "Verified Source: This domain is on the global trusted whitelist.",
            "security_analysis": "Bypassed via trusted domain whitelist",
        }

    # ── Module L: Live HTML Title Fetch ──────────────────────────────────
    # Performed early so the result can short-circuit the rest of the pipeline.
    live_content = fetch_live_content(url)
    _url_domain  = _extract_domain(url)

    # Rule 0 — Brand Impersonation in Page Title (highest-priority override)
    # Fires when: fetched title contains a TARGET_BRAND keyword AND the URL's
    # domain is NOT the brand's official domain OR is hosted on a free host.
    brand_title_impersonation: dict = {
        "fired": False, "brand": None, "title": live_content["title"], "flag": None
    }
    if live_content["title"]:
        title_lower = live_content["title"].lower()
        for brand_kw, official_domain in TARGET_BRANDS.items():
            _url_sld = _url_domain.split(".")[0]
            _official_sld = official_domain.split(".")[0]
            if brand_kw in title_lower and _url_sld != _official_sld:
                _host_parts_l = _url_domain.split(".")
                _etld1_l      = ".".join(_host_parts_l[-2:]) if len(_host_parts_l) >= 2 else _url_domain
                on_free_host  = _etld1_l in BRAND_TITLE_FREE_HOSTS
                flag_msg = (
                    f"Brand Impersonation: Page title claims to be "
                    f"'{live_content['title']}' (contains '{brand_kw}') "
                    f"but domain '{_url_domain}' does not match official "
                    f"'{official_domain}'"
                    + (f" and is hosted on free platform '{_etld1_l}'" if on_free_host else "")
                    + "."
                )
                brand_title_impersonation = {
                    "fired":   True,
                    "brand":   brand_kw.title(),
                    "title":   live_content["title"],
                    "flag":    flag_msg,
                }
                # Immediate return — no further analysis needed
                return {
                    "url":        url,
                    "label":      1,
                    "verdict":    "malicious",
                    "confidence": 99.0,
                    "reason":     flag_msg,
                    "security_analysis": {
                        "live_content":              live_content,
                        "brand_title_impersonation": brand_title_impersonation,
                        "threat_flags":              [flag_msg],
                        "note": "Short-circuited at Rule 0 — brand impersonation in live page title.",
                    },
                }
                break

    # ── Pre-screening ────────────────────────────────────────────────────
    encoded    = detect_encoded_payloads(url)
    entropy    = analyse_query_entropy(url)
    vulns      = detect_vulnerability_patterns(url)
    typo       = detect_typosquatting(url)
    tld_check  = check_suspicious_tld(url)
    complexity = check_url_complexity(url)
    keywords   = check_phishing_keywords(url)
    ipfs       = check_ipfs_gateway(url)
    brand_sub  = brand_in_subdomain(url)
    hyph_creds = check_hyphenated_creds(url)

    # Abused free-host check — fires only for platforms NOT in FREE_HOSTING_PROVIDERS.
    # Hosts in FREE_HOSTING_PROVIDERS already go through the keyword+entropy gate in
    # check_infrastructure_abuse(), so unconditionally flagging them here would
    # double-penalise innocent pages like myportfolio.github.io.
    _norm_host  = (urlparse(url if "://" in url else f"https://{url}").hostname or "").lower().rstrip(".")
    _host_parts = _norm_host.split(".")
    _etld1      = ".".join(_host_parts[-2:]) if len(_host_parts) >= 2 else _norm_host
    abused_host_flag = (
        len(_host_parts) > 2
        and _etld1 in ABUSED_FREE_HOSTS
        and _etld1 not in FREE_HOSTING_PROVIDERS
    )

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
    if infra_abuse_flag:
        threat_flags.append(_infra["reason"])
    if brand_sub["found"]:
        threat_flags.append(
            f"Brand-in-subdomain: '{brand_sub['brand']}' planted as subdomain on "
            f"'{brand_sub['host_domain']}' — impersonation attack."
        )
    if hyph_creds["found"]:
        threat_flags.append(
            f"Hyphenated credential domain: '{hyph_creds['matched']}' "
            f"mimics a brand via hyphen obfuscation."
        )
    if abused_host_flag:
        threat_flags.append(
            f"Abused free host: subdomain on '{_etld1}' — attacker-controlled hosting."
        )

    # github.io brand-in-path check — user controls path, not the domain label
    # Fires when: eTLD+1 is github.io AND path contains a known brand AND
    # the full hostname (user subdomain) is not itself in TRUSTED_DOMAINS.
    _github_brand_hit = None
    _github_io_flag = False
    if _etld1 == "github.io" and _norm_host not in TRUSTED_DOMAINS:
        _parsed_path = urlparse(url if "://" in url else f"https://{url}").path.lower()
        _github_brand_hit = next((b for b in PATH_BRANDS if b in _parsed_path), None)
        if _github_brand_hit:
            _github_io_flag = True
            threat_flags.append(
                f"github.io brand impersonation: path contains '{_github_brand_hit}' "
                f"on attacker-controlled subdomain '{_norm_host}'."
            )

    # ── ML Inference ─────────────────────────────────────────────────────
    X = extract_features(url)
    label      = int(_model.predict(X)[0])
    proba      = _model.predict_proba(X)[0]
    confidence = float(proba[label])

    # ── Heuristic flag counter (A-F modules + new signals) ───────────────
    heuristic_count = sum([
        typo["is_typosquat"],                          # A — Typosquatting
        tld_check["flagged"],                          # B — Suspicious TLD
        keywords["found"],                             # C — Phishing keywords
        complexity["is_complex"],                      # D — URL complexity
        ipfs["is_ipfs"] and not ipfs["whitelisted"],   # E — IPFS gateway
        brand_sub["found"],                            # G — Brand-in-subdomain
        hyph_creds["found"],                           # H — Hyphenated creds
        abused_host_flag,                              # I — Abused free host
        _github_io_flag,                               # J — github.io brand-in-path
    ])

    # ── Trust Protocol: ML-malicious but zero heuristic flags → benign ──
    trust_override = False
    if label == 1 and heuristic_count == 0:
        label = 0
        trust_override = True
        confidence = float(proba[0])

    # ── Hard overrides — always fire regardless of trust_override ────────

    # Bug fix: infra abuse must override trust protocol — an attacker-controlled
    # subdomain on a free host is never safe regardless of ML score.
    if infra_abuse_flag:
        label = 1
        trust_override = False
        confidence = max(confidence, 0.85)

    # Raw IP host — legitimate services don't use bare IPs as public URLs.
    _parsed_host = (urlparse(url if "://" in url else f"https://{url}").hostname or "")
    if re.match(r"^(\d{1,3}\.){3}\d{1,3}$", _parsed_host):
        label = 1
        trust_override = False
        confidence = max(confidence, 0.80)
        if "Raw IP host detected" not in threat_flags:
            threat_flags.append("Raw IP host detected — legitimate services don't use bare IP addresses")

    # Suspicious TLD hard override — bypasses trust_override gate
    if tld_check["flagged"]:
        label = 1
        trust_override = False
        confidence = max(confidence, 0.76)

    # ══════════════════════════════════════════════════════════════════════
    #  HEURISTIC OVERRIDE ENGINE  (v3.5 FN hotfix — 2026-05-19)
    #  Fires on ML-benign verdicts when deterministic attack signals are
    #  present that the RF model structurally cannot score above 0.5.
    #  Each rule is independent; the highest confidence wins.
    # ══════════════════════════════════════════════════════════════════════
    _override_conf = 0.0
    _override_reasons: list[str] = []

    # Rule 1 — Typosquatting (edit-distance ≤ 2 to a known brand)
    if typo["is_typosquat"]:
        _override_conf = max(_override_conf, 0.88)
        _override_reasons.append(
            f"Visual Deception: domain closely mimics {typo['matched_brand']} "
            f"(edit-distance {typo['edit_distance']})."
        )

    # Rule 2 — Brand planted as subdomain label on untrusted domain
    #           facebook.unitedcolleges.net, apple.some-phish.tk, etc.
    if brand_sub["found"]:
        _override_conf = max(_override_conf, 0.90)
        _override_reasons.append(
            f"Brand Subdomain Hijack: '{brand_sub['brand']}' is used as a "
            f"subdomain on untrusted domain '{brand_sub['host_domain']}' — "
            f"designed to deceive users scanning only the subdomain."
        )

    # Rule 3 — Hyphenated credential / brand-impersonation in hostname
    #           appieid-enable.com, drive-google-com.xyz, paypal-secure.net
    if hyph_creds["found"]:
        _override_conf = max(_override_conf, 0.87)
        _override_reasons.append(
            f"Hyphenated Impersonation: hostname segment "
            f"'{hyph_creds['matched']}' mimics a trusted brand via "
            f"hyphen-obfuscation — a classic credential-harvesting pattern."
        )

    # Rule 4 — Subdomain on known abused free-hosting platform
    if abused_host_flag:
        _override_conf = max(_override_conf, 0.85)
        _override_reasons.append(
            f"Abused Free Host: subdomain on '{_etld1}' where all subdomains "
            f"are attacker-controlled — high-risk hosting vector."
        )

    # Rule 5 — Suspicious TLD  +  any phishing keyword  →  compound signal
    if tld_check["flagged"] and keywords["found"]:
        _override_conf = max(_override_conf, 0.83)
        _override_reasons.append(
            f"Compound signal: high-abuse TLD '{tld_check['tld']}' combined "
            f"with credential-harvesting keywords — strong phishing indicator."
        )

    # Rule 6 — Any single suspicious TLD (standalone escalation)
    if tld_check["flagged"] and not keywords["found"]:
        _override_conf = max(_override_conf, 0.76)
        _override_reasons.append(
            f"Risky TLD: '{tld_check['tld']}' is disproportionately "
            f"associated with phishing campaigns."
        )

    # Rule 7 — Phishing keywords present + at least one other flag
    if keywords["found"] and len(threat_flags) >= 2:
        _override_conf = max(_override_conf, 0.78)
        kw_list = ", ".join(keywords["matched_keywords"])
        _override_reasons.append(
            f"Credential-harvesting keywords ({kw_list}) combined with "
            f"{len(threat_flags) - 1} additional threat signal(s)."
        )

    # Rule 8 — github.io brand impersonation via path
    if _github_io_flag:
        _override_conf = max(_override_conf, 0.87)
        _override_reasons.append(
            f"github.io Brand Impersonation: path contains brand keyword "
            f"'{_github_brand_hit}' on attacker-controlled subdomain '{_norm_host}' — "
            f"a known phishing delivery vector abusing GitHub's trusted domain."
        )

    # Apply override: flip any ML-benign prediction when engine fires
    if _override_conf > 0.0:
        label = 1
        trust_override = False
        confidence = max(confidence, _override_conf)

    # ── Legacy escalation guard: ≥2 flags with no specific rule matched ──
    if not trust_override and label == 0 and len(threat_flags) >= 2:
        label = 1
        confidence = max(confidence, 0.70)

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

    # ── PRIORITY 0: Override engine reasons (highest specificity) ──────────
    # These are prepended so the most specific attack description leads the output.
    if _override_reasons and not trust_override:
        reasons.extend(_override_reasons)

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

        # G — Brand-in-subdomain (only if not already in override reasons)
        if brand_sub["found"] and not _override_reasons:
            reasons.append(
                f"Brand Subdomain Hijack: '{brand_sub['brand']}' planted as a "
                f"subdomain on '{brand_sub['host_domain']}'."
            )

        # H — Hyphenated credential hostname (only if not already in override reasons)
        if hyph_creds["found"] and not _override_reasons:
            reasons.append(
                f"Hyphenated Impersonation: '{hyph_creds['matched']}' in hostname."
            )

        # I — Abused free host (only if not already in override reasons)
        if abused_host_flag and not _override_reasons:
            reasons.append(
                f"Abused Free Host: subdomain on '{_etld1}'."
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
            "live_content":              live_content,
            "brand_title_impersonation": brand_title_impersonation,
            "encoded_payloads":          encoded,
            "entropy_analysis":          entropy,
            "vulnerability_patterns":    vulns,
            "typosquatting":             typo,
            "suspicious_tld":            tld_check,
            "url_complexity":            complexity,
            "phishing_keywords":         keywords,
            "threat_flags":              threat_flags,
            "trust_override":            trust_override,
            "ipfs_gateway":              ipfs,
            "infrastructure_abuse":      _infra,
            "brand_in_subdomain":        brand_sub,
            "hyphenated_creds":          hyph_creds,
            "abused_free_host":          {"flagged": abused_host_flag, "host": _etld1 if abused_host_flag else None},
            "override_engine_fired":     _override_conf > 0.0,
            "override_confidence":       round(_override_conf * 100, 2),
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
        # Force malicious only when the expanded destination was flagged malicious.
        # Previously used label == 0 which incorrectly flipped benign destinations.
        if result["label"] == 1:
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