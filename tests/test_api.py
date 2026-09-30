"""Integration tests for FastAPI endpoints (Stage 10)."""

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


def test_auth_failure():
    # Missing API key
    res1 = client.post("/v1/score", json={})
    assert res1.status_code == 401

    # Invalid API key
    res2 = client.post("/v1/score", headers={"X-API-Key": "invalid-key"}, json={})
    assert res2.status_code == 403


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
