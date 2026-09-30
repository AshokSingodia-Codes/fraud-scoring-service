"""Unit tests for explainability and drift calculation (Stages 8 & 9)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.fraud.evaluation.drift import analyze_drift, compute_psi
from src.fraud.explain.reasons import explain_prediction


def test_compute_psi_identical_distribution():
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(0, 1, 1000)

    psi = compute_psi(data1, data2)
    # Identical distributions should have very small PSI (< 0.05)
    assert psi < 0.05


def test_compute_psi_shifted_distribution():
    np.random.seed(42)
    data1 = np.random.normal(0, 1, 1000)
    data2 = np.random.normal(3, 1, 1000)

    psi = compute_psi(data1, data2)
    # Large distribution shift should produce high PSI (> 0.25)
    assert psi > 0.25


def test_compute_psi_empty_data():
    psi = compute_psi([], [])
    assert psi == 0.0


def test_analyze_drift():
    train_df = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0], "feat2": [10.0, 20.0, 30.0, 40.0]})
    test_df = pd.DataFrame({"feat1": [1.1, 2.1, 3.1, 4.1], "feat2": [100.0, 200.0, 300.0, 400.0]})

    res = analyze_drift(train_df, test_df, ["feat1", "feat2"])
    assert "feat1" in res
    assert "feat2" in res
    assert res["feat2"] > res["feat1"]


def test_explain_prediction_sanity():
    # Train tiny dummy LightGBM booster
    X = pd.DataFrame({
        "TransactionAmt": [10.0, 200.0, 50.0, 500.0],
        "card1_fq": [0.1, 0.9, 0.3, 0.8],
    })
    y = np.array([0, 1, 0, 1])

    ds = lgb.Dataset(X, label=y)
    booster = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 1}, ds, num_boost_round=5)

    sample = X.iloc[[0]]
    exp = explain_prediction(booster, sample, top_k=2)

    assert "base_value" in exp
    assert "margin_score" in exp
    assert "reasons" in exp
    assert len(exp["reasons"]) == 2

    raw_margin = float(booster.predict(sample, raw_score=True)[0])
    assert np.isclose(raw_margin, exp["margin_score"], atol=1e-3)
