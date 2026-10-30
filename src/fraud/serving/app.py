import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath("."))


# Ensure single-threaded BLAS to prevent thread context allocation failures under concurrency
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from src.fraud.evaluation.streaming_drift import get_drift_monitor
from src.fraud.serving.auth import get_api_key
from src.fraud.serving.db import PredictionLog, SessionLocal, init_db, log_prediction_to_db
from src.fraud.serving.metrics import (
    DECISION_COUNT,
    FEATURE_PSI,
    MODEL_HEALTH_STATUS,
    PREDICTION_SCORE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
    SHADOW_EVALUATIONS,
    SHADOW_SCORE_DIVERGENCE,
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

logger = logging.getLogger(__name__)
predictor: FraudPredictor | None = None
challenger_predictor: FraudPredictor | None = None


def get_predictor() -> FraudPredictor:
    """Get loaded champion predictor instance or attempt lazy initialization."""
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


def get_challenger_predictor() -> FraudPredictor | None:
    """Retrieve candidate challenger predictor if models_candidate exists."""
    global challenger_predictor
    if challenger_predictor is None:
        candidate_dir = Path("models_candidate")
        if candidate_dir.exists() and (candidate_dir / "model.txt").exists():
            try:
                challenger_predictor = FraudPredictor(models_dir=candidate_dir)
                logger.info("Challenger shadow model loaded from %s", candidate_dir)
            except Exception as e:
                logger.warning("Could not load challenger shadow model: %s", e)
                challenger_predictor = None
    return challenger_predictor



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database tables and load model predictor instance at startup."""
    global predictor
    try:
        init_db()
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.warning("Database initialization deferred/failed: %s", e)

    try:
        predictor = FraudPredictor()
        logger.info("FraudPredictor loaded successfully.")
    except Exception as e:
        logger.warning("FraudPredictor initial load deferred/failed: %s", e)

    try:
        import anyio
        limiter = anyio.to_thread.current_default_thread_limiter()
        initial_tokens = limiter.total_tokens
        anyio_threads = int(os.getenv("ANYIO_MAX_THREADS", "40"))
        limiter.total_tokens = anyio_threads
        app.state.anyio_limiter_tokens = limiter.total_tokens
        logger.info(
            "AnyIO default thread limiter tokens: initial=%d, configured=%d",
            initial_tokens,
            limiter.total_tokens,
        )
    except Exception as e:
        logger.warning("Could not configure anyio thread limiter: %s", e)

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
@app.get("/healthz", tags=["Health"])
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
    """Prometheus metrics endpoint with real-time drift gauges."""
    drift_mon = get_drift_monitor()
    psi_dict = drift_mon.compute_all_psi()
    for feat, psi_val in psi_dict.items():
        FEATURE_PSI.labels(feature=feat).set(psi_val)

    summary = drift_mon.get_summary()
    status_map = {"HEALTHY": 1.0, "MODERATE_DRIFT": 2.0, "CRITICAL_DRIFT": 3.0}
    MODEL_HEALTH_STATUS.set(status_map.get(summary["status"], 1.0))

    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def evaluate_shadow_model(payload: dict[str, Any], champion_score: float) -> None:
    """Evaluate challenger model in shadow mode asynchronously without affecting client latency."""
    challenger = get_challenger_predictor()
    if challenger is None:
        return
    try:
        challenger_res = challenger.score_dict(payload, explain=False)
        challenger_score = challenger_res["fraud_probability"]
        divergence = abs(champion_score - challenger_score)

        SHADOW_EVALUATIONS.labels(status="success").inc()
        SHADOW_SCORE_DIVERGENCE.observe(divergence)
    except Exception as e:
        logger.warning("Shadow model evaluation error: %s", e)
        SHADOW_EVALUATIONS.labels(status="error").inc()


@app.get("/v1/model/shadow", tags=["Monitoring"])
def get_shadow_status(api_key: str = Depends(get_api_key)):
    """Return status and comparison specs of Champion vs Challenger shadow model."""
    champ = get_predictor()
    challenger = get_challenger_predictor()
    return {
        "shadow_mode_active": challenger is not None,
        "champion": {
            "version": champ.model_version,
            "thresholds": {"t_review": champ.t_review, "t_block": champ.t_block},
            "num_features": len(champ.feature_builder.final_feature_names),
        },
        "challenger": {
            "version": challenger.model_version if challenger else None,
            "thresholds": {"t_review": challenger.t_review, "t_block": challenger.t_block} if challenger else None,
            "num_features": len(challenger.feature_builder.final_feature_names) if challenger else None,
        } if challenger else None,
    }


@app.get("/v1/model/drift", tags=["Monitoring"])
def get_model_drift(api_key: str = Depends(get_api_key)):
    """Return real-time PSI drift summary and feature degradation status."""
    drift_mon = get_drift_monitor()
    return drift_mon.get_summary()



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
async def score_transaction(
    request: Request,
    background_tasks: BackgroundTasks,
    explain: bool = True,
):
    """Score a single transaction in real time with comprehensive stage timing."""
    handler_t0 = time.perf_counter()

    # 1. Auth stage
    t_auth_0 = time.perf_counter()
    api_key_header = request.headers.get("X-API-Key")
    api_key = get_api_key(api_key_header)
    t_auth_ms = (time.perf_counter() - t_auth_0) * 1000.0

    # 2. Ratelimit stage
    t_rate_0 = time.perf_counter()
    check_rate_limit(api_key)
    t_ratelimit_ms = (time.perf_counter() - t_rate_0) * 1000.0

    # 3. Request validation (Pydantic) stage
    t_val_0 = time.perf_counter()
    try:
        raw_json = await request.json()
        tx = Transaction.model_validate(raw_json)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    t_val_ms = (time.perf_counter() - t_val_0) * 1000.0

    pred = get_predictor()
    payload = tx.model_dump()

    res, timings = pred.score_dict(payload, explain=explain, return_timings=True)

    # 4. Audit write & velocity store ingestion (scheduled via BackgroundTasks)
    t_audit_0 = time.perf_counter()
    background_tasks.add_task(log_prediction_to_db, res)

    # Ingest event into online velocity feature store asynchronously
    card_id = payload.get("card1")
    if card_id is not None:
        from src.fraud.features.velocity import get_velocity_store
        vstore = get_velocity_store()
        amt = float(payload.get("TransactionAmt", 0.0))
        tx_dt = float(payload.get("TransactionDT", time.time()))
        background_tasks.add_task(vstore.record_transaction, str(card_id), amt, tx_dt)

    # Ingest event into streaming drift monitor asynchronously
    drift_mon = get_drift_monitor()
    background_tasks.add_task(drift_mon.record_observation, payload, res["fraud_probability"])

    # Evaluate challenger model in shadow mode asynchronously
    background_tasks.add_task(evaluate_shadow_model, payload, res["fraud_probability"])

    t_audit_ms = (time.perf_counter() - t_audit_0) * 1000.0

    # 5. Serialization stage
    t_ser_0 = time.perf_counter()
    score_resp = ScoreResponse(**res)
    resp_body = score_resp.model_dump_json()
    t_ser_ms = (time.perf_counter() - t_ser_0) * 1000.0

    total_handler_ms = (time.perf_counter() - handler_t0) * 1000.0

    # Record Prometheus metrics
    REQUEST_COUNT.labels(endpoint="/v1/score", status="200").inc()
    REQUEST_LATENCY.labels(endpoint="/v1/score").observe(total_handler_ms / 1000.0)
    PREDICTION_SCORE.observe(res["fraud_probability"])
    DECISION_COUNT.labels(decision=res["decision"]).inc()

    server_timing = (
        f"req_val;dur={t_val_ms:.2f}, "
        f"auth;dur={t_auth_ms:.2f}, "
        f"rate;dur={t_ratelimit_ms:.2f}, "
        f"feat;dur={timings['feat_build']:.2f}, "
        f"pred;dur={timings['predict']:.2f}, "
        f"cal;dur={timings['calibrate']:.2f}, "
        f"shap;dur={timings['shap']:.2f}, "
        f"reasons;dur={timings['reasons_build']:.2f}, "
        f"audit;dur={t_audit_ms:.2f}, "
        f"ser;dur={t_ser_ms:.2f}, "
        f"total_handler;dur={total_handler_ms:.2f}"
    )

    return Response(
        content=resp_body,
        media_type="application/json",
        headers={"Server-Timing": server_timing},
    )


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
    try:
        entry = db.query(PredictionLog).filter(PredictionLog.transaction_id == transaction_id).first()
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
    finally:
        db.close()
