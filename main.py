"""
main.py -- FastAPI backend for PhishGuard MVP
=============================================
Serves the trained phishing-detection model and the
unified heuristic pipeline (Modules A-F) via a REST API.

Endpoints:
    POST /predict   -- full prediction + security analysis
    GET  /health    -- readiness probe
"""

from typing import Any, List, Optional, Union

from fastapi import FastAPI, Request
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


class EncodedPayloads(BaseModel):
    base64_found: bool
    hex_found: bool
    encoded_snippets: list[str]


class EntropyAnalysis(BaseModel):
    query_entropy: float
    high_entropy: bool
    param_details: dict


class TyposquattingResult(BaseModel):
    is_typosquat: bool
    matched_brand: Optional[str]
    edit_distance: Optional[int]


class SuspiciousTLD(BaseModel):
    flagged: bool
    tld: str


class URLComplexity(BaseModel):
    is_complex: bool
    dot_count: int
    url_length: int
    details: list[str]


class PhishingKeywords(BaseModel):
    found: bool
    matched_keywords: list[str]


class IPFSGateway(BaseModel):
    is_ipfs: bool
    gateway: Optional[str]
    cid: Optional[str]
    whitelisted: bool


class ShortenerExpansion(BaseModel):
    is_short: bool
    original_url: str
    expanded_url: Optional[str]
    error: Optional[str]


class SecurityAnalysis(BaseModel):
    encoded_payloads: EncodedPayloads
    entropy_analysis: EntropyAnalysis
    vulnerability_patterns: dict
    typosquatting: TyposquattingResult
    suspicious_tld: SuspiciousTLD
    url_complexity: URLComplexity
    phishing_keywords: PhishingKeywords
    threat_flags: list[str]
    trust_override: bool
    ipfs_gateway: IPFSGateway
    shortener_expansion: ShortenerExpansion


class PredictionResponse(BaseModel):
    url:                str
    original_url:       Optional[str]  = None   # set when shortener was expanded
    label:              int
    verdict:            str
    confidence:         float
    reason:             str
    security_analysis:  Union[SecurityAnalysis, str]

class InfrastructureAbuse(BaseModel):
    is_abuse: bool
    provider: Optional[str]
    subdomain: Optional[str]
    reason: str

class SecurityAnalysis(BaseModel):
    model_config = {"extra": "allow"}   # absorbs any future fields silently

    # Core fields — always present
    threat_flags:            List[str]        = []
    trust_override:          Optional[bool]   = None

    # Heuristic modules — Optional so a crashed module never 500s
    encoded_payloads:        Optional[Any]    = None
    entropy_analysis:        Optional[Any]    = None
    vulnerability_patterns:  Optional[Any]    = None
    typosquatting:           Optional[Any]    = None
    suspicious_tld:          Optional[Any]    = None
    url_complexity:          Optional[Any]    = None
    phishing_keywords:       Optional[Any]    = None
    ipfs_gateway:            Optional[Any]    = None
    shortener_expansion:     Optional[Any]    = None
    infrastructure_abuse:    Optional[Any]    = None  # v3.5

# -- Endpoints ----------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("5/minute")
def predict(request: Request, req: URLRequest):
    """
    Analyse a URL for phishing indicators.

    Rate limit: 10 requests / minute per IP.

    Pipeline (Modules A-F):
      0. Trusted-domain whitelist fast path
      1. A: Typosquatting  B: Suspicious TLD  C: Phishing keywords
         D: URL complexity  E: IPFS gateway  F: Shortener expansion
      2. ML feature extraction -> Random Forest classification
      3. Trust Protocol override / heuristic escalation / Override Engine
    """
    return predict_url(req.url)


@app.get("/health")
@limiter.limit("60/minute")
def health(request: Request):
    """Readiness / liveness probe. Rate limit: 60/min."""
    return {"status": "ok", "model": "Random Forest (AUC 0.9645)"}

@app.get("/")
def read_root():
    """Redirect root access to the Swagger documentation."""
    return RedirectResponse(url="/docs")

