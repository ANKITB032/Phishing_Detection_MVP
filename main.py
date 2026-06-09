"""
main.py -- FastAPI backend for PhishGuard MVP
=============================================
Serves the trained phishing-detection model and the
unified heuristic pipeline (Modules A-F) via a REST API.

Endpoints:
    POST /predict   -- full prediction + security analysis
    POST /report    -- submit a user correction for data feedback loop
    GET  /health    -- readiness probe
"""

import csv
import os
import threading
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
# SlowAPIMiddleware intentionally NOT used — it resolves IP from raw socket
# (always the HF Spaces proxy), bypassing our key function. Decorator-based
# limiting with app.state.limiter is used instead.

from predictor import predict_url

# -- Rate Limiter -------------------------------------------------------------
# Priority order for client IP behind reverse proxies (HF Spaces, Cloudflare):
#   1. X-Real-IP      — set by nginx / Cloudflare with the true client IP
#   2. X-Forwarded-For first entry — set by most load balancers
#   3. raw socket IP  — fallback (will be proxy IP on HF Spaces)
def _get_client_ip(request: Request) -> str:
    real_ip = request.headers.get("X-Real-IP", "").strip()
    if real_ip:
        return real_ip
    forwarded_for = request.headers.get("X-Forwarded-For", "").strip()
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)

limiter = Limiter(key_func=_get_client_ip, default_limits=["200/day"])

# -- App Setup ----------------------------------------------------------------

app = FastAPI(
    title="PhishGuard API",
    description="Phishing URL detection with cryptographic payload analysis",
    version="3.5.0",
)

# Wire rate-limiter — decorator-based approach (no SlowAPIMiddleware).
# The exception handler converts RateLimitExceeded into a 429 JSON response.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Request / Response Models ------------------------------------------------

class URLRequest(BaseModel):
    url: str


# Single flexible SecurityAnalysis — extra="allow" absorbs any shape the
# predictor returns (full pipeline, trusted-domain string, or Rule 0 dict).
# All heuristic fields are Optional[Any] so a short-circuit response that
# omits them never triggers a ResponseValidationError.
class SecurityAnalysis(BaseModel):
    model_config = {"extra": "allow"}

    # Core fields — present on all code paths
    threat_flags:           List[str]      = []
    trust_override:         Optional[bool] = None

    # Heuristic modules — absent on short-circuit responses
    encoded_payloads:       Optional[Any]  = None
    entropy_analysis:       Optional[Any]  = None
    vulnerability_patterns: Optional[Any]  = None
    typosquatting:          Optional[Any]  = None
    suspicious_tld:         Optional[Any]  = None
    url_complexity:         Optional[Any]  = None
    phishing_keywords:      Optional[Any]  = None
    ipfs_gateway:           Optional[Any]  = None
    shortener_expansion:    Optional[Any]  = None
    infrastructure_abuse:   Optional[Any]  = None
    # Live content (Module L)
    live_content:                Optional[Any]  = None
    brand_title_impersonation:   Optional[Any]  = None


class PredictionResponse(BaseModel):
    url:               str
    original_url:      Optional[str]  = None   # set when shortener was expanded
    label:             int
    verdict:           str
    confidence:        float
    reason:            str
    security_analysis: Any = None              # Any — accepts dict, str, or None


class ReportRequest(BaseModel):
    url:             str
    correct_verdict: Literal["benign", "malicious"]
    comments:        Optional[str] = None


# -- Report CSV path & write-lock ---------------------------------------------

_REPORT_CSV  = os.path.join(os.path.dirname(__file__), "01_Datasets", "user_reported_corrections.csv")
_REPORT_LOCK = threading.Lock()
_REPORT_COLS = ["timestamp", "url", "correct_verdict", "comments"]

# -- Endpoints ----------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
#@limiter.limit("5/minute")
def predict(request: Request, req: URLRequest):
    """
    Analyse a URL for phishing indicators.

    Rate limit: 5 requests / minute per IP.
    Input constraints: max 2048 chars, no null bytes, must be non-empty.

    Pipeline (Modules A-F + Override Engine):
      0. Trusted-domain whitelist fast path
      1. A: Typosquatting  B: Suspicious TLD  C: Phishing keywords
         D: URL complexity  E: IPFS gateway  F: Shortener expansion
         G: Brand-in-subdomain  H: Hyphenated creds  I: Abused free host
      2. ML feature extraction -> Random Forest classification
      3. Heuristic Override Engine (7 deterministic rules)
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="URL must not be empty.")
    if len(url) > 2048:
        raise HTTPException(status_code=422, detail=f"URL exceeds maximum length of 2048 characters ({len(url)} received).")
    if "\x00" in url or "%00" in url:
        raise HTTPException(status_code=422, detail="URL contains null bytes.")
    return predict_url(url)


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    """Readiness / liveness probe. Rate limit: 60/min."""
    return {"status": "ok", "model": "Random Forest (AUC 0.9645)"}


@app.post("/report", status_code=201)
@limiter.limit("10/minute")
def report(request: Request, req: ReportRequest):
    """
    Accept a user correction and append it to the feedback CSV.

    Creates 01_Datasets/user_reported_corrections.csv with headers on
    first write. Thread-safe via a module-level threading.Lock.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=422, detail="url must not be empty.")
    if req.correct_verdict not in ("benign", "malicious"):
        raise HTTPException(status_code=422, detail="correct_verdict must be 'benign' or 'malicious'.")

    row = {
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "url":             url,
        "correct_verdict": req.correct_verdict,
        "comments":        (req.comments or "").strip(),
    }

    try:
        with _REPORT_LOCK:
            file_exists = os.path.isfile(_REPORT_CSV)
            os.makedirs(os.path.dirname(_REPORT_CSV), exist_ok=True)
            with open(_REPORT_CSV, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=_REPORT_COLS)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(row)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write report: {exc}") from exc

    return {"status": "accepted", "message": "Thank you — your correction has been recorded.", "data": row}

@app.get("/")
def read_root():
    """Redirect root access to the Swagger documentation."""
    return RedirectResponse(url="/docs")

