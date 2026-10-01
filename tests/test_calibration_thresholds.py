"""Unit tests for calibration and threshold optimization (Stage 7)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from src.fraud.models.calibrate import ProbabilityCalibrator, evaluate_calibration
from src.fraud.models.threshold import make_decision, optimize_thresholds


def test_probability_calibrator_fit_and_calibrate():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    raw_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

    calibrator = ProbabilityCalibrator()
    assert not calibrator.is_fitted

    with pytest.raises(ValueError, match="must be fit"):
        calibrator.calibrate(raw_probs)

    calibrator.fit(y_true, raw_probs)
    assert calibrator.is_fitted

    cal_probs = calibrator.calibrate(raw_probs)
    assert len(cal_probs) == len(raw_probs)
    assert np.all(cal_probs >= 0.0)
    assert np.all(cal_probs <= 1.0)


def test_calibrator_save_load():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1])
    raw_probs = np.array([0.2, 0.3, 0.7, 0.8])

    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_true, raw_probs)

    with tempfile.TemporaryDirectory() as tmpdir:
        cal_path = Path(tmpdir) / "calibrator.joblib"
        calibrator.save(cal_path)

        loaded = ProbabilityCalibrator.load(cal_path)
        assert loaded.is_fitted
        np.testing.assert_allclose(loaded.calibrate(raw_probs), calibrator.calibrate(raw_probs))


def test_evaluate_calibration():
    y_true = np.array([0, 0, 1, 1])
    raw_probs = np.array([0.3, 0.4, 0.6, 0.7])
    cal_probs = np.array([0.1, 0.1, 0.9, 0.9])

    results = evaluate_calibration(y_true, raw_probs, cal_probs)
    assert "raw_brier_score" in results
    assert "calibrated_brier_score" in results
    assert "raw_ece" in results
    assert "calibrated_ece" in results
    assert results["calibrated_brier_score"] < results["raw_brier_score"]


def test_optimize_thresholds():
    np.random.seed(42)
    y_val = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    val_probs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.7, 0.8, 0.85, 0.9, 0.95])
    val_amounts = np.array([100.0] * 10)

    res = optimize_thresholds(y_val, val_probs, val_amounts, fp_review_cost=10.0, target_block_precision=0.80)
    assert "t_review" in res
    assert "t_block" in res
    assert res["t_review"] <= res["t_block"]
    assert "cost_assumptions" in res


def test_make_decision():
    t_review = 0.20
    t_block = 0.80

    assert make_decision(0.10, t_review, t_block) == "approve"
    assert make_decision(0.50, t_review, t_block) == "review"
    assert make_decision(0.90, t_review, t_block) == "block"


def test_fit_split_not_test():
    import json
    thresholds_path = Path("models/thresholds.json")
    if thresholds_path.exists():
        with thresholds_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            assert data.get("fit_split") != "test", "Thresholds were fit on the test split!"
            assert data.get("fit_split") == "validation"

    calibrator_path = Path("models/calibrator.joblib")
    if calibrator_path.exists():
        cal = ProbabilityCalibrator.load(calibrator_path)
        assert getattr(cal, "fit_split", None) != "test", "Calibrator was fit on the test split!"
        assert getattr(cal, "fit_split", None) == "validation"

