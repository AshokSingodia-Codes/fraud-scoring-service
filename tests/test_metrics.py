"""Unit tests for metrics calculation."""

import numpy as np
import pytest
from src.fraud.evaluation.metrics import (
    compute_brier_score,
    compute_pr_auc,
    compute_roc_auc,
    evaluate_all_metrics,
    expected_calibration_error,
    expected_cost,
    recall_at_precision,
    recall_at_top_pct,
)


def test_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    assert compute_pr_auc(y_true, y_prob) == 1.0
    assert compute_roc_auc(y_true, y_prob) == 1.0
    assert recall_at_precision(y_true, y_prob, target_precision=0.8) == pytest.approx(1.0)
    assert recall_at_top_pct(y_true, y_prob, top_pct=0.5) == 1.0
    assert compute_brier_score(y_true, y_prob) < 0.05
    assert expected_calibration_error(y_true, y_prob) < 0.2


def test_dummy_constant_predictions():
    y_true = np.array([0, 0, 0, 1])
    y_prob = np.array([0.25, 0.25, 0.25, 0.25])

    # PR-AUC for constant prediction equals baseline fraud rate (0.25)
    assert compute_pr_auc(y_true, y_prob) == 0.25
    assert compute_roc_auc(y_true, y_prob) == 0.5


def test_expected_cost():
    y_true = np.array([0, 1])
    y_prob = np.array([0.9, 0.1])  # FP for row 0, FN for row 1
    amounts = np.array([50.0, 200.0])

    # threshold = 0.5
    # Row 0: y=0, prob=0.9 -> FP (cost = 10.0)
    # Row 1: y=1, prob=0.1 -> FN (cost = 200.0)
    # Total cost = 210.0
    cost = expected_cost(y_true, y_prob, amounts, threshold=0.5, fp_cost=10.0)
    assert cost == 210.0


def test_evaluate_all_metrics():
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    amounts = np.array([10.0, 20.0, 30.0, 100.0, 150.0])

    metrics = evaluate_all_metrics(y_true, y_prob, amounts)

    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "brier_score" in metrics
    assert "ece" in metrics
    assert "expected_cost" in metrics
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
