"""FastAPI scoring application for Real-Time Fraud Scoring Service (Stage 10)."""

import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.fraud.serving.auth import get_api_key
from src.fraud.serving.db import PredictionLog, SessionLocal, log_prediction_to_db
from src.fraud.serving.metrics import (
    DECISION_COUNT,
    PREDICTION_SCORE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.fraud.serving.predictor import FraudPredictor
from src.fraud.serving.ratelimit import check_rate_limit
from src.fraud.serving.schemas import (
    BatchScoreRequest,
    BatchScoreResponse,
    ModelInfoResponse,
    ScoreResponse,
    Transaction,
)

predictor: FraudPredictor | None = None


def get_predictor() -> FraudPredictor:
    """Get loaded predictor instance or attempt lazy initialization."""
    global predictor
    if predictor is None:
        try:
            predictor = FraudPredictor()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model predictor not ready: {e}",
            )
    return predictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model predictor instance at startup."""
    global predictor
    try:
        predictor = FraudPredictor()
        print("FraudPredictor loaded successfully.")
    except Exception as e:
        print(f"Warning: FraudPredictor initial load deferred/failed: {e}")
    yield


app = FastAPI(
    title="Real-Time Fraud Scoring Service",
    description="Production-grade fraud detection API for card transactions.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe."""
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
def readiness_check():
    """Readiness probe checking predictor status."""
    pred = get_predictor()
    return {"status": "ready", "model_version": pred.model_version}


@app.get("/metrics", tags=["Monitoring"])
def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/model/info", response_model=ModelInfoResponse, tags=["Model Info"])
def get_model_info(api_key: str = Depends(get_api_key)):
    """Return model version, thresholds, feature count and headline metrics."""
    pred = get_predictor()
    return ModelInfoResponse(
        model_version=pred.model_version,
        training_date="2026-10-01",
        thresholds={"t_review": pred.t_review, "t_block": pred.t_block},
        num_features=len(pred.feature_builder.final_feature_names),
        headline_metrics={
            "expected_roc_auc": "0.90-0.95",
            "expected_pr_auc": "0.55-0.75",
        },
    )


@app.post("/v1/score", response_model=ScoreResponse, tags=["Scoring"])
def score_transaction(
    tx: Transaction,
    explain: bool = True,
    api_key: str = Depends(get_api_key),
):
    """Score a single transaction in real time."""
    check_rate_limit(api_key)
    pred = get_predictor()

    start_time = time.perf_counter()
    payload = tx.model_dump()

    res = pred.score_dict(payload, explain=explain)

    # Record Prometheus metrics
    elapsed = time.perf_counter() - start_time
    REQUEST_COUNT.labels(endpoint="/v1/score", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/v1/score").observe(elapsed)
    PREDICTION_SCORE.observe(res["fraud_probability"])
    DECISION_COUNT.labels(decision=res["decision"]).inc()

    # Log prediction to DB asynchronously / safely
    log_prediction_to_db(res)

    return ScoreResponse(**res)


@app.post("/v1/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
def score_batch_transactions(
    req: BatchScoreRequest,
    explain: bool = False,
    api_key: str = Depends(get_api_key),
):
    """Score a batch of transactions (up to 1000)."""
    check_rate_limit(api_key)
    pred = get_predictor()

    start_time = time.perf_counter()
    results = []

    for tx in req.transactions:
        payload = tx.model_dump()
        res = pred.score_dict(payload, explain=explain)
        results.append(ScoreResponse(**res))

    batch_latency = round((time.perf_counter() - start_time) * 1000, 2)
    REQUEST_COUNT.labels(endpoint="/v1/score/batch", status="200").inc()

    return BatchScoreResponse(
        results=results,
        total_processed=len(results),
        batch_latency_ms=batch_latency,
    )


@app.get("/v1/predictions/{transaction_id}", tags=["Audit Log"])
def get_prediction_by_id(transaction_id: str, api_key: str = Depends(get_api_key)):
    """Retrieve historical prediction log from database by transaction_id."""
    db = SessionLocal()
    entry = db.query(PredictionLog).filter(PredictionLog.transaction_id == transaction_id).first()
    db.close()

    if not entry:
        raise HTTPException(status_code=404, detail="Prediction ID not found.")

    return {
        "transaction_id": entry.transaction_id,
        "timestamp": entry.timestamp,
        "fraud_probability": entry.fraud_probability,
        "decision": entry.decision,
        "risk_band": entry.risk_band,
        "model_version": entry.model_version,
        "latency_ms": entry.latency_ms,
    }
