"""Predictor service class loading artifacts once at startup (Stage 10)."""

import json
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
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
        self.feature_names = self.booster.feature_name()
        self._row_buffer = np.full(
            (1, len(self.feature_builder.final_feature_names)), np.nan, dtype=np.float32
        )

    def score_dict(
        self,
        payload: dict[str, Any],
        explain: bool = True,
        return_timings: bool = False,
    ) -> dict[str, Any] | tuple[dict[str, Any], dict[str, float]]:
        """Score a single transaction dictionary with preallocated numpy buffer and stage timing."""
        start_time = time.perf_counter()

        tx_id = str(payload.get("transaction_id", "unknown"))

        # 1. Feature build
        t0 = time.perf_counter()
        features_mat = self.feature_builder.transform_dict_to_numpy(payload, out=self._row_buffer)
        t_feat = (time.perf_counter() - t0) * 1000.0

        # 2. SHAP & Predict
        reasons = []
        t_shap = 0.0
        t_reasons_build = 0.0
        if explain:
            exp = explain_prediction(
                self.booster,
                features_mat,
                top_k=5,
                raw_payload=payload,
                feature_names=self.feature_names,
                num_threads=1,
            )
            t_shap = exp.get("timing", {}).get("shap_compute", 0.0)
            t_reasons_build = exp.get("timing", {}).get("reasons_build", 0.0)
            reasons = exp["reasons"]
            margin = exp["margin_score"]
            raw_prob = float(1.0 / (1.0 + np.exp(-margin)))
            t_pred = 0.0  # margin derived within explain_prediction
        else:
            t0 = time.perf_counter()
            raw_prob = float(self.booster.predict(features_mat, num_threads=1)[0])
            t_pred = (time.perf_counter() - t0) * 1000.0

        # 3. Calibrate
        t0 = time.perf_counter()
        if self.calibrator and self.calibrator.is_fitted:
            cal_prob = float(self.calibrator.calibrate(np.array([raw_prob]))[0])
        else:
            cal_prob = raw_prob
        t_cal = (time.perf_counter() - t0) * 1000.0

        # Decision & Risk Band
        decision = make_decision(cal_prob, self.t_review, self.t_block)
        if decision == "block":
            risk_band = "High"
        elif decision == "review":
            risk_band = "Medium"
        else:
            risk_band = "Low"

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        res_dict = {
            "transaction_id": tx_id,
            "fraud_probability": round(cal_prob, 4),
            "decision": decision,
            "risk_band": risk_band,
            "reasons": reasons,
            "model_version": self.model_version,
            "thresholds": {"t_review": self.t_review, "t_block": self.t_block},
            "latency_ms": latency_ms,
        }

        stage_timings = {
            "feat_build": t_feat,
            "predict": t_pred,
            "calibrate": t_cal,
            "shap": t_shap,
            "reasons_build": t_reasons_build,
        }

        if return_timings:
            return res_dict, stage_timings
        return res_dict
