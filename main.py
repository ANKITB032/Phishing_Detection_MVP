"""
main.py — FastAPI backend for PhishGuard MVP
=============================================
Serves the trained phishing-detection model and the
cryptographic / vulnerability pre-screening pipeline
via a REST API.

Endpoints:
    POST /predict   — full prediction + security analysis
    GET  /health    — readiness probe
"""

from typing import Union

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from predictor import predict_url

# ── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PhishGuard API",
    description="Phishing URL detection with cryptographic payload analysis",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows your Vercel domain to connect
    allow_credentials=True,
    allow_methods=["*"],  # CRITICAL: This tells it to accept the OPTIONS preflight
    allow_headers=["*"],
)


# ── Request / Response Models ────────────────────────────────────────────────

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


class SecurityAnalysis(BaseModel):
    encoded_payloads: EncodedPayloads
    entropy_analysis: EntropyAnalysis
    vulnerability_patterns: dict
    threat_flags: list[str]


class PredictionResponse(BaseModel):
    url: str
    label: int
    verdict: str
    confidence: float
    reason: str
    security_analysis: Union[SecurityAnalysis, str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
def predict(req: URLRequest):
    """
    Analyse a URL for phishing indicators.

    Pipeline:
      0. Trusted-domain whitelist fast path (immediate benign if matched)
      1. Cryptographic pre-screening (Base64, hex, entropy, vuln patterns)
      2. ML feature extraction → Random Forest classification
      3. Threat-flag escalation if multiple red flags detected
    """
    return predict_url(req.url)


@app.get("/health")
def health():
    """Readiness / liveness probe."""
    return {"status": "ok", "model": "Random Forest (AUC 0.9648)"}
