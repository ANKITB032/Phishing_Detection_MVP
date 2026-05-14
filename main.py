"""
main.py -- FastAPI backend for PhishGuard MVP
=============================================
Serves the trained phishing-detection model and the
unified heuristic pipeline (Modules A-F) via a REST API.

Endpoints:
    POST /predict   -- full prediction + security analysis
    GET  /health    -- readiness probe
"""

from typing import Optional, Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from predictor import predict_url

# -- App Setup ----------------------------------------------------------------

app = FastAPI(
    title="PhishGuard API",
    description="Phishing URL detection with cryptographic payload analysis",
    version="3.0.0",
)

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
    url: str
    label: int
    verdict: str
    confidence: float
    reason: str
    security_analysis: Union[SecurityAnalysis, str]


# -- Endpoints ----------------------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    """
    Analyse a URL for phishing indicators.

    Pipeline (Modules A-F):
      0. Trusted-domain whitelist fast path
      1. A: Typosquatting  B: Suspicious TLD  C: Phishing keywords
         D: URL complexity  E: IPFS gateway  F: Shortener expansion
      2. ML feature extraction -> Random Forest classification
      3. Trust Protocol override / heuristic escalation
    """
    return predict_url(req.url)


@app.get("/health")
def health():
    """Readiness / liveness probe."""
    return {"status": "ok", "model": "Random Forest (AUC 0.9648)"}

