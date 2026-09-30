"""Predictor service class loading artifacts once at startup (Stage 10)."""

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from src.fraud.config import load_config
from src.fraud.explain.reasons import explain_prediction
from src.fraud.features.builder import FeatureBuilder
from src.fraud.models.calibrate import ProbabilityCalibrator
from src.fraud.models.threshold import make_decision


class FraudPredictor:
    """Thread-safe fraud prediction service."""

    def __init__(self, models_dir: str | Path | None = None, config_path: str | Path | None = None):
        self.config = load_config(config_path) if config_path else load_config()
        base_models_dir = Path(models_dir) if models_dir else Path(self.config["paths"]["models_dir"])

        fb_path = base_models_dir / "feature_builder.joblib"
        model_path = base_models_dir / "model.txt"
        cal_path = base_models_dir / "calibrator.joblib"
        thresh_path = base_models_dir / "thresholds.json"

        if not (fb_path.exists() and model_path.exists()):
            raise FileNotFoundError(f"Model artifacts missing in {base_models_dir}. Run training pipeline first.")

        self.feature_builder = FeatureBuilder.load(fb_path)
        self.booster = lgb.Booster(model_file=str(model_path))

        if cal_path.exists():
            self.calibrator = ProbabilityCalibrator.load(cal_path)
        else:
            self.calibrator = None

        if thresh_path.exists():
            with thresh_path.open("r", encoding="utf-8") as f:
                t_data = json.load(f)
                self.t_review = float(t_data.get("t_review", 0.10))
                self.t_block = float(t_data.get("t_block", 0.50))
        else:
            self.t_review = 0.10
            self.t_block = 0.50

        self.model_version = self.config["project"].get("version", "1.0.0")

    def score_dict(self, payload: dict[str, Any], explain: bool = True) -> dict[str, Any]:
        """Score a single transaction dictionary."""
        start_time = time.perf_counter()

        tx_id = str(payload.get("transaction_id", "unknown"))
        raw_df = pd.DataFrame([payload])

        # Feature transformation
        features_df = self.feature_builder.transform(raw_df)

        # Raw score & calibration
        raw_prob = float(self.booster.predict(features_df)[0])
        if self.calibrator and self.calibrator.is_fitted:
            cal_prob = float(self.calibrator.calibrate(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob

        # Decision & Risk Band
        decision = make_decision(cal_prob, self.t_review, self.t_block)
        if decision == "block":
            risk_band = "High"
        elif decision == "review":
            risk_band = "Medium"
        else:
            risk_band = "Low"

        # Reasons
        reasons = []
        if explain:
            exp = explain_prediction(self.booster, features_df, top_k=5)
            reasons = exp["reasons"]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "transaction_id": tx_id,
            "fraud_probability": round(cal_prob, 4),
            "decision": decision,
            "risk_band": risk_band,
            "reasons": reasons,
            "model_version": self.model_version,
            "thresholds": {"t_review": self.t_review, "t_block": self.t_block},
            "latency_ms": latency_ms,
        }
