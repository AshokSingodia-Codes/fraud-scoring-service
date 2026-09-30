"""Unit tests for FraudPredictor class (Stage 10)."""

from src.fraud.serving.predictor import FraudPredictor


def test_predictor_initialization():
    predictor = FraudPredictor()
    assert predictor.feature_builder is not None
    assert predictor.booster is not None
    assert predictor.t_review > 0
    assert predictor.t_block > predictor.t_review


def test_predictor_score_dict():
    predictor = FraudPredictor()
    sample_payload = {
        "transaction_id": "tx_test_1001",
        "TransactionDT": 86400,
        "TransactionAmt": 125.50,
        "ProductCD": "W",
        "card1": 1000,
        "addr1": 150,
        "P_emaildomain": "gmail.com",
    }

    res = predictor.score_dict(sample_payload, explain=True)

    assert res["transaction_id"] == "tx_test_1001"
    assert 0.0 <= res["fraud_probability"] <= 1.0
    assert res["decision"] in ["approve", "review", "block"]
    assert res["risk_band"] in ["Low", "Medium", "High"]
    assert len(res["reasons"]) > 0
    assert res["latency_ms"] > 0
