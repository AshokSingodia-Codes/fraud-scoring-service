"""Integration tests for FastAPI endpoints (Stage 10)."""

import pytest
from fastapi.testclient import TestClient
from src.fraud.serving.app import app

client = TestClient(app)
VALID_API_KEY = "demo-api-key-12345"
HEADERS = {"X-API-Key": VALID_API_KEY}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_endpoint():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_anyio_thread_limiter_size(monkeypatch):
    monkeypatch.setenv("ANYIO_MAX_THREADS", "123")
    with TestClient(app):
        assert app.state.anyio_limiter_tokens == 123




def test_auth_failure():
    # Missing API key
    res1 = client.post("/v1/score", json={})
    assert res1.status_code == 401

    # Invalid API key
    res2 = client.post("/v1/score", headers={"X-API-Key": "invalid-key"}, json={})
    assert res2.status_code == 403


def test_auth_env_scenarios(monkeypatch):
    from src.fraud.serving.auth import get_allowed_api_keys

    # Scenario A: ALLOWED_API_KEYS is explicitly set
    monkeypatch.setenv("ALLOWED_API_KEYS", "custom-key-1,custom-key-2")
    monkeypatch.setenv("APP_ENV", "prod")
    keys = get_allowed_api_keys()
    assert keys == {"custom-key-1", "custom-key-2"}
    assert "demo-api-key-12345" not in keys

    # Scenario B: ALLOWED_API_KEYS is empty and APP_ENV is dev
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    monkeypatch.setenv("APP_ENV", "dev")
    keys_dev = get_allowed_api_keys()
    assert "demo-api-key-12345" in keys_dev
    assert "prod-secret-key-67890" in keys_dev

    # Scenario C: ALLOWED_API_KEYS is empty and APP_ENV is prod
    monkeypatch.delenv("ALLOWED_API_KEYS", raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    keys_prod = get_allowed_api_keys()
    assert len(keys_prod) == 0


def test_score_single_transaction():
    payload = {
        "transaction_id": "tx_api_001",
        "TransactionDT": 86400,
        "TransactionAmt": 250.00,
        "ProductCD": "W",
        "card1": 1000,
        "P_emaildomain": "yahoo.com",
    }
    response = client.post("/v1/score", headers=HEADERS, json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["transaction_id"] == "tx_api_001"
    assert "fraud_probability" in data
    assert data["decision"] in ["approve", "review", "block"]
    assert len(data["reasons"]) > 0


def test_score_batch_transactions():
    payload = {
        "transactions": [
            {
                "transaction_id": "batch_001",
                "TransactionDT": 86400,
                "TransactionAmt": 50.0,
            },
            {
                "transaction_id": "batch_002",
                "TransactionDT": 86500,
                "TransactionAmt": 500.0,
            },
        ]
    }
    response = client.post("/v1/score/batch", headers=HEADERS, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2


def test_get_prediction_by_id():
    payload = {
        "transaction_id": "tx_audit_999",
        "TransactionDT": 86400,
        "TransactionAmt": 75.0,
    }
    # Score transaction first to write audit log
    client.post("/v1/score", headers=HEADERS, json=payload)

    # Query prediction log
    response = client.get("/v1/predictions/tx_audit_999", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "tx_audit_999"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "fraud_requests_total" in response.text


def test_model_info_endpoint():
    response = client.get("/v1/model/info", headers=HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "thresholds" in data
    assert "num_features" in data
    assert data["num_features"] > 0


def test_invalid_payload():
    # TransactionAmt <= 0 should fail validation (422)
    bad_payload = {
        "transaction_id": "bad_tx_1",
        "TransactionDT": 86400,
        "TransactionAmt": -10.0,
    }
    response = client.post("/v1/score", headers=HEADERS, json=bad_payload)
    assert response.status_code == 422


def test_training_serving_parity():
    tx_payload = {
        "transaction_id": "parity_tx",
        "TransactionDT": 86400,
        "TransactionAmt": 150.0,
        "ProductCD": "W",
        "card1": 1000,
        "addr1": 150,
        "P_emaildomain": "gmail.com",
    }
    # Single score
    res_single = client.post("/v1/score?explain=false", headers=HEADERS, json=tx_payload)
    assert res_single.status_code == 200
    prob_single = res_single.json()["fraud_probability"]

    # Batch score with single item
    res_batch = client.post("/v1/score/batch?explain=false", headers=HEADERS, json={"transactions": [tx_payload]})
    assert res_batch.status_code == 200
    prob_batch = res_batch.json()["results"][0]["fraud_probability"]

    assert prob_single == prob_batch


@pytest.mark.slow
def test_http_level_parity_on_500_test_rows(monkeypatch):
    """Post 500 real test partition transactions as JSON over HTTP vs batch, assert max diff <= 1e-6 and 0 tier flips."""
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from src.fraud.data.split import time_split_data
    from src.fraud.models.threshold import make_decision
    from src.fraud.serving.predictor import FraudPredictor

    parquet_file = Path("data/interim/train_merged.parquet")
    if not parquet_file.exists():
        pytest.skip("Test parquet data/interim/train_merged.parquet absent")

    monkeypatch.setenv("APP_ENV", "dev")
    df = pd.read_parquet(parquet_file)
    _, _, test_df = time_split_data(df)
    sample_500 = test_df.head(500)

    predictor = FraudPredictor()
    mat = predictor.feature_builder.transform(sample_500)
    raw = predictor.booster.predict(mat, num_threads=1)
    cal = predictor.calibrator.calibrate(raw)
    served_batch = np.round(cal, 4)
    batch_decisions = [make_decision(p, predictor.t_review, predictor.t_block) for p in served_batch]

    http_client = TestClient(app)
    headers = {"X-API-Key": "demo-api-key-12345"}
    records = sample_500.to_dict(orient="records")

    http_probs = []
    http_decisions = []
    for r in records:
        clean_r = {k: v for k, v in r.items() if pd.notna(v)}
        clean_r["transaction_id"] = str(clean_r.get("TransactionID", "tx_test"))
        res = http_client.post("/v1/score?explain=false", headers=headers, json=clean_r)
        assert res.status_code == 200, res.text
        data = res.json()
        http_probs.append(data["fraud_probability"])
        http_decisions.append(data["decision"])

    http_probs = np.array(http_probs)
    diffs = np.abs(http_probs - served_batch)
    max_diff = float(np.max(diffs))
    flips = sum(d1 != d2 for d1, d2 in zip(http_decisions, batch_decisions))

    assert max_diff <= 1e-6, f"HTTP vs batch max diff {max_diff} exceeds 1e-6"
    assert flips == 0, f"HTTP vs batch had {flips} tier flips"




