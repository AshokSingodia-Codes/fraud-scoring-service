"""Calibration module for fraud probability scores (Stage 7).

Fits Isotonic Regression calibrator on validation predictions and evaluates
Brier score and ECE before and after calibration. Saves models/calibrator.joblib.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from src.fraud.evaluation.metrics import compute_brier_score, compute_ece


class ProbabilityCalibrator:
    """Isotonic regression calibrator wrapper for fraud probability scores."""

    def __init__(self):
        self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.is_fitted = False

    def fit(self, y_val: np.ndarray | pd.Series, val_probs: np.ndarray) -> "ProbabilityCalibrator":
        """Fit calibrator on validation probabilities."""
        self.calibrator.fit(val_probs, y_val)
        self.is_fitted = True
        return self

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """Transform raw probabilities to calibrated probabilities."""
        if not self.is_fitted:
            raise ValueError("Calibrator must be fit before calibrate can be called.")
        calibrated = self.calibrator.transform(probs)
        return np.clip(calibrated, 0.0, 1.0)

    def save(self, file_path: str | Path) -> None:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path: str | Path) -> "ProbabilityCalibrator":
        return joblib.load(file_path)


def evaluate_calibration(
    y_true: np.ndarray | pd.Series,
    raw_probs: np.ndarray,
    calibrated_probs: np.ndarray,
) -> dict:
    """Return dictionary comparing calibration metrics before and after."""
    return {
        "raw_brier_score": compute_brier_score(y_true, raw_probs),
        "raw_ece": compute_ece(y_true, raw_probs),
        "calibrated_brier_score": compute_brier_score(y_true, calibrated_probs),
        "calibrated_ece": compute_ece(y_true, calibrated_probs),
    }
