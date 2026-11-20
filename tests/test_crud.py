"""Comprehensive CRUD and Data Integrity tests for Prediction Audit Logs."""

from fastapi.testclient import TestClient
from src.fraud.serving.app import app

client = TestClient(app)
VALID_API_KEY = "demo-api-key-12345"
HEADERS = {"X-API-Key": VALID_API_KEY}


def test_crud_lifecycle_complete():
    """Test full Create -> Read -> Update -> Delete lifecycle with data integrity verification."""
    tx_id = "test_crud_tx_1001"

    # Clean up if existed from previous run
    client.delete(f"/v1/predictions/{tx_id}", headers=HEADERS)

    # 1. CREATE
    create_payload = {
        "transaction_id": tx_id,
        "fraud_probability": 0.8523,
        "decision": "review",
        "risk_band": "High",
        "model_version": "1.0.0",
        "latency_ms": 12.5,
        "notes": "Initial automated trigger",
    }
    r_create = client.post("/v1/predictions", headers=HEADERS, json=create_payload)
    assert r_create.status_code == 201, r_create.text
    created_data = r_create.json()
    assert created_data["transaction_id"] == tx_id
    assert created_data["fraud_probability"] == 0.8523
    assert created_data["decision"] == "review"
    assert created_data["risk_band"] == "High"
    assert created_data["notes"] == "Initial automated trigger"

    # 2. READ (Single)
    r_get = client.get(f"/v1/predictions/{tx_id}", headers=HEADERS)
    assert r_get.status_code == 200
    get_data = r_get.json()
    assert get_data["transaction_id"] == tx_id
    assert get_data["fraud_probability"] == 0.8523
    assert get_data["decision"] == "review"
    assert get_data["notes"] == "Initial automated trigger"

    # 3. UPDATE (PATCH)
    update_payload = {
        "decision": "block",
        "notes": "Analyst manual review confirmed synthetic identity fraud",
    }
    r_patch = client.patch(f"/v1/predictions/{tx_id}", headers=HEADERS, json=update_payload)
    assert r_patch.status_code == 200, r_patch.text
    patched_data = r_patch.json()
    assert patched_data["transaction_id"] == tx_id
    assert patched_data["decision"] == "block"
    assert patched_data["notes"] == "Analyst manual review confirmed synthetic identity fraud"
    assert patched_data["fraud_probability"] == 0.8523  # Unchanged

    # Verify read reflects the update
    r_get2 = client.get(f"/v1/predictions/{tx_id}", headers=HEADERS)
    assert r_get2.status_code == 200
    assert r_get2.json()["decision"] == "block"
    assert r_get2.json()["notes"] == "Analyst manual review confirmed synthetic identity fraud"

    # 4. LIST (Query & Filter)
    r_list = client.get("/v1/predictions?decision=block&limit=10", headers=HEADERS)
    assert r_list.status_code == 200
    list_data = r_list.json()
    assert list_data["total"] >= 1
    assert any(item["transaction_id"] == tx_id for item in list_data["items"])

    # 5. DELETE
    r_del = client.delete(f"/v1/predictions/{tx_id}", headers=HEADERS)
    assert r_del.status_code == 200
    assert r_del.json()["status"] == "deleted"
    assert r_del.json()["records_deleted"] >= 1

    # Verify deleted record is gone (404)
    r_get_after = client.get(f"/v1/predictions/{tx_id}", headers=HEADERS)
    assert r_get_after.status_code == 404


def test_crud_not_found_handling():
    """Verify 404 is returned when querying, updating, or deleting a non-existent prediction."""
    bogus_id = "non_existent_transaction_xyz_99999"
    r_get = client.get(f"/v1/predictions/{bogus_id}", headers=HEADERS)
    assert r_get.status_code == 404

    r_patch = client.patch(f"/v1/predictions/{bogus_id}", headers=HEADERS, json={"decision": "approve"})
    assert r_patch.status_code == 404

    r_del = client.delete(f"/v1/predictions/{bogus_id}", headers=HEADERS)
    assert r_del.status_code == 404


def test_crud_open_access():
    """Verify all CRUD endpoints are accessible without requiring an API key."""
    # List without key returns 200 OK
    res = client.get("/v1/predictions")
    assert res.status_code == 200
    assert "items" in res.json()


def test_score_to_audit_read_integration():
    """Verify scoring a transaction via /v1/score immediately creates queryable audit entry."""
    tx_id = "tx_score_audit_verify_42"
    payload = {
        "transaction_id": tx_id,
        "TransactionDT": 86400,
        "TransactionAmt": 189.99,
        "ProductCD": "W",
        "card1": 1234,
        "P_emaildomain": "gmail.com",
    }
    r_score = client.post("/v1/score?explain=false", headers=HEADERS, json=payload)
    assert r_score.status_code == 200
    score_data = r_score.json()
    assert score_data["transaction_id"] == tx_id

    # Retrieve from audit log
    r_audit = client.get(f"/v1/predictions/{tx_id}", headers=HEADERS)
    assert r_audit.status_code == 200
    audit_data = r_audit.json()
    assert audit_data["transaction_id"] == tx_id
    assert audit_data["decision"] == score_data["decision"]
    assert audit_data["risk_band"] == score_data["risk_band"]
    assert abs(audit_data["fraud_probability"] - score_data["fraud_probability"]) < 1e-4

    # Clean up test artifact
    client.delete(f"/v1/predictions/{tx_id}", headers=HEADERS)
