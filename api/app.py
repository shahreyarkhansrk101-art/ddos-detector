"""
FastAPI service exposing the trained DDoS detector.

Endpoints:
  GET  /health        - liveness check
  POST /predict        - score a single network flow
  POST /predict/batch  - score a list of flows
  GET  /metrics         - saved model evaluation metrics (results/metrics.json)
  GET  /stream/next      - generate + score one simulated "live" flow (for dashboards)

Run locally:  uvicorn api.app:app --reload --port 8000
Docker:       see Dockerfile / docker-compose.yml
"""
import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.generate_traffic_data import CATEGORY_GENERATORS, generate_flow, generate_random_flow
from src.detector import ALERT_THRESHOLD, DDoSDetector
from src.preprocessing import ALL_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
METRICS_PATH = ROOT / "results" / "metrics.json"

app = FastAPI(
    title="DDoS Detector API",
    description="ML-based network flow classifier for DDoS traffic detection.",
    version="1.0.0",
)

# Dashboard runs on a different origin (e.g. localhost:3000) in dev, so allow
# cross-origin calls. In production this is fronted by the same nginx host,
# but CORS stays permissive here since this is a demo/portfolio service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_detector: DDoSDetector | None = None
_stream_rng = np.random.default_rng()


def get_detector() -> DDoSDetector:
    global _detector
    if _detector is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run `python main.py` to train and save models first.",
            )
        _detector = DDoSDetector(MODEL_PATH)
    return _detector


class FlowFeatures(BaseModel):
    flow_duration_s: float = Field(..., gt=0, json_schema_extra={"example": 0.12})
    total_fwd_packets: int = Field(..., ge=0, json_schema_extra={"example": 350})
    total_bwd_packets: int = Field(..., ge=0, json_schema_extra={"example": 1})
    total_fwd_bytes: float = Field(..., ge=0, json_schema_extra={"example": 19500.0})
    total_bwd_bytes: float = Field(..., ge=0, json_schema_extra={"example": 60.0})
    packet_rate: float = Field(..., ge=0, json_schema_extra={"example": 2900.0})
    byte_rate: float = Field(..., ge=0, json_schema_extra={"example": 160000.0})
    avg_packet_size: float = Field(..., ge=0, json_schema_extra={"example": 55.0})
    fwd_bwd_ratio: float = Field(..., ge=0, json_schema_extra={"example": 175.0})
    syn_flag_ratio: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.95})
    ack_flag_ratio: float = Field(..., ge=0, le=1, json_schema_extra={"example": 0.02})
    unique_src_ips: int = Field(..., ge=1, json_schema_extra={"example": 340})
    protocol: Literal["TCP", "UDP", "ICMP"] = "TCP"


class PredictionResponse(BaseModel):
    is_ddos: bool
    confidence: float
    threshold: float


class BatchPredictionRequest(BaseModel):
    flows: list[FlowFeatures]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(flow: FlowFeatures):
    detector = get_detector()
    result = detector.score_flow(flow.model_dump())
    return PredictionResponse(**result, threshold=ALERT_THRESHOLD)


@app.post("/predict/batch")
def predict_batch(request: BatchPredictionRequest):
    detector = get_detector()
    df = pd.DataFrame([f.model_dump() for f in request.flows])
    scored = detector.score_batch(df)
    return scored[["ddos_confidence", "is_ddos"]].to_dict(orient="records")


@app.get("/metrics")
def metrics():
    """Return the saved model evaluation metrics for the dashboard."""
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Metrics not found. Run `python main.py` to train models first.",
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/stream/next")
def stream_next():
    """
    Generate one simulated flow and score it — used by the dashboard's
    automatic live feed to simulate traffic without a real packet capture.
    """
    detector = get_detector()
    flow = generate_random_flow(_stream_rng)
    result = detector.score_flow({k: flow[k] for k in ALL_FEATURES})
    return {
        "timestamp": time.time(),
        "protocol": flow["protocol"],
        "packet_rate": round(float(flow["packet_rate"]), 2),
        "unique_src_ips": int(flow["unique_src_ips"]),
        "syn_flag_ratio": round(float(flow["syn_flag_ratio"]), 3),
        "true_label": flow["true_label"],
        "category": flow["category"],
        **result,
        "threshold": ALERT_THRESHOLD,
    }


SimCategory = Literal["benign", "flash_crowd", "syn_flood", "udp_flood", "http_flood"]


@app.get("/simulate/{category}")
def simulate(category: SimCategory):
    """
    Manually trigger one flow of a specific category, on demand — lets the
    dashboard inject a chosen attack type (or benign traffic) independent
    of the automatic random feed.
    """
    if category not in CATEGORY_GENERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown category: {category}")

    detector = get_detector()
    flow = generate_flow(category, _stream_rng)
    result = detector.score_flow({k: flow[k] for k in ALL_FEATURES})
    return {
        "timestamp": time.time(),
        "protocol": flow["protocol"],
        "packet_rate": round(float(flow["packet_rate"]), 2),
        "unique_src_ips": int(flow["unique_src_ips"]),
        "syn_flag_ratio": round(float(flow["syn_flag_ratio"]), 3),
        "true_label": flow["true_label"],
        "category": flow["category"],
        **result,
        "threshold": ALERT_THRESHOLD,
    }
