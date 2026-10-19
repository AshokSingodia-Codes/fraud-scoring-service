"""End-to-End production integration tests verifying frontend-backend contracts."""

import os

from fastapi.testclient import TestClient

os.environ["APP_ENV"] = "dev"
from src.fraud.serving.app import app  # noqa: E402

client = TestClient(app)
VALID_API_KEY = "demo-api-key-12345"
HEADERS = {"X-API-Key": VALID_API_KEY}


def test_e2e_health_and_ready():
    """Verify backend health and readiness probes consumed by frontend status pill."""
    r_health = client.get("/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json()["status"] == "ready"


def test_e2e_model_info_contract():
    """Verify /v1/model/info satisfies frontend ModelInfoPage contract."""
    res = client.get("/v1/model/info", headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "model_version" in data
    assert "training_date" in data
    assert "thresholds" in data
    assert "t_review" in data["thresholds"]
    assert "t_block" in data["thresholds"]
    assert data["num_features"] > 0


def test_e2e_model_drift_contract():
    """Verify /v1/model/drift endpoint satisfies frontend drift dashboard contract."""
    res = client.get("/v1/model/drift", headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "window_size" in data
    assert "current_samples" in data
    assert "status" in data
    assert data["status"] in ["HEALTHY", "MODERATE_DRIFT", "CRITICAL_DRIFT"]
    assert "feature_psi" in data


def test_e2e_score_single_transaction_contract():
    """Verify /v1/score single transaction returns all fields required by CheckOnePage."""
    payload = {
        "transaction_id": "tx_e2e_001",
        "TransactionDT": 86400,
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card1": 1000,
        "card2": 555.0,
        "P_emaildomain": "gmail.com",
    }
    res = client.post("/v1/score?explain=true", headers=HEADERS, json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["transaction_id"] == "tx_e2e_001"
    assert "fraud_probability" in data
    assert data["decision"] in ["approve", "review", "block"]
    assert "reasons" in data
    assert len(data["reasons"]) > 0
    # Confirm rolling velocity features are present in response
    assert "velocity_features" in data
    assert "card1_tx_cnt_10m" in data["velocity_features"]
    assert "card1_amt_sum_10m" in data["velocity_features"]


def test_e2e_score_batch_transactions_contract():
    """Verify /v1/score/batch returns required fields for CheckBulkPage."""
    batch_payload = {
        "transactions": [
            {
                "transaction_id": "tx_batch_001",
                "TransactionDT": 86400,
                "TransactionAmt": 45.0,
                "ProductCD": "W",
                "card1": 1000,
            },
            {
                "transaction_id": "tx_batch_002",
                "TransactionDT": 86410,
                "TransactionAmt": 999.0,
                "ProductCD": "C",
                "card1": 9999,
            },
        ]
    }
    res = client.post("/v1/score/batch?explain=false", headers=HEADERS, json=batch_payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_processed"] == 2
    assert len(data["results"]) == 2
    assert "batch_latency_ms" in data


def test_e2e_model_shadow_contract():
    """Verify /v1/model/shadow endpoint returns champion and candidate telemetry."""
    res = client.get("/v1/model/shadow", headers=HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert "shadow_mode_active" in data
    assert "champion" in data
    assert "version" in data["champion"]
    if data["shadow_mode_active"]:
        assert "challenger" in data
        assert "version" in data["challenger"]

