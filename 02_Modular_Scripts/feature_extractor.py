import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

# ── Optional dependency: beautifulsoup4 ──────────────────────────────────────
try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    logger.warning(
        "beautifulsoup4 is not installed. fetch_live_content() will always "
        "return None. Run: pip install beautifulsoup4"
    )

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

INPUT  = "../01_Datasets/cleaned_data.csv"
OUTPUT = "../01_Datasets/features.csv"

SPECIAL_CHARS = "@-?=.#%+&_~/!"

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
    Attempt a live GET request to the URL and extract the page <title>.

    Returns:
        {
            "title":   str | None  — page <title> text, or None on failure,
            "fetched": bool        — True if the request succeeded,
            "error":   str | None  — exception message on failure,
        }

    Failure modes handled gracefully (all return title=None):
      - bs4 / requests not installed
      - Connection error / DNS failure
      - Timeout (> 3 s)
      - HTTP error (4xx / 5xx)
      - Redirect to a different domain that also fails
      - Any unexpected exception
    """
    if not _BS4_AVAILABLE or not _REQUESTS_AVAILABLE:
        return {"title": None, "fetched": False, "error": "Missing dependency (bs4/requests)"}

    normalized = url if "://" in url else f"https://{url}"
    try:
        resp = _requests.get(
            normalized,
            headers=_FETCH_HEADERS,
            timeout=_FETCH_TIMEOUT,
            allow_redirects=True,
            stream=False,
        )
        resp.raise_for_status()

        # Limit parsing to first 50 KB — avoid hanging on huge pages
        content = resp.content[:51200]
        soup  = BeautifulSoup(content, "html.parser")
        tag   = soup.find("title")
        title = tag.get_text(strip=True) if tag else None
        return {"title": title, "fetched": True, "error": None}

    except Exception as exc:
        logger.debug("fetch_live_content failed for %s: %s", url, exc)
        return {"title": None, "fetched": False, "error": str(exc)[:120]}


def extract_features(url: str) -> dict:
    return {
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


if __name__ == "__main__":
    df = pd.read_csv(INPUT)

    features_df = pd.DataFrame(df["url"].map(extract_features).tolist())
    out = pd.concat([features_df, df["label"]], axis=1)

    out.to_csv(OUTPUT, index=False)
    print(f"Saved {len(out):,} rows × {len(out.columns)} cols → {OUTPUT}")
    print(out.drop(columns="label").describe().round(2).to_string())
