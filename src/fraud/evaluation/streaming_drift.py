"""Streaming drift detection and model health monitoring engine.

Calculates Population Stability Index (PSI) and score distribution shifts
over streaming sliding-window buffers in production serving, exposing
real-time drift metrics to Prometheus.
"""

import json
import logging
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class StreamingDriftMonitor:
    """Thread-safe streaming drift monitor tracking feature and score distribution shifts."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        window_size: int = 1000,
        min_samples_to_evaluate: int = 50,
    ):
        self.window_size = window_size
        self.min_samples_to_evaluate = min_samples_to_evaluate
        self._lock = Lock()
        self.feature_buffers: dict[str, deque[float]] = {}
        self.score_buffer: deque[float] = deque(maxlen=window_size)
        self.reference_distributions: dict[str, dict[str, Any]] = {}

        if reference_path:
            self.load_reference_distributions(reference_path)

    def load_reference_distributions(self, reference_path: str | Path) -> None:
        """Load precomputed baseline decile edges and expected percentage distributions."""
        p = Path(reference_path)
        if not p.exists():
            logger.warning("Reference distribution file not found at: %s", p)
            return

        with open(p, encoding="utf-8") as f:
            self.reference_distributions = json.load(f)

        for feat in self.reference_distributions.keys():
            self.feature_buffers[feat] = deque(maxlen=self.window_size)

    def record_observation(
        self,
        features: dict[str, Any],
        fraud_probability: float,
    ) -> None:
        """Ingest a single scoring event into rolling observation buffers."""
        with self._lock:
            self.score_buffer.append(float(fraud_probability))
            for feat, val in features.items():
                if feat in self.reference_distributions and val is not None:
                    try:
                        self.feature_buffers[feat].append(float(val))
                    except (ValueError, TypeError):
                        pass

    def compute_feature_psi(self, feature_name: str) -> float | None:
        """Compute PSI for a monitored feature against reference baseline."""
        with self._lock:
            if feature_name not in self.reference_distributions:
                return None
            buffer = self.feature_buffers.get(feature_name)
            if not buffer or len(buffer) < self.min_samples_to_evaluate:
                return None
            samples = list(buffer)

        ref = self.reference_distributions[feature_name]
        edges = np.array(ref["edges"])
        expected_pct = np.array(ref["expected_pct"])

        if len(edges) < 2:
            return 0.0

        actual_counts, _ = np.histogram(samples, bins=edges)
        actual_pct = actual_counts / max(len(samples), 1)

        # Numerical stability clip
        expected_pct = np.clip(expected_pct, 1e-6, None)
        actual_pct = np.clip(actual_pct, 1e-6, None)

        psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        return float(psi_val)

    def compute_all_psi(self) -> dict[str, float]:
        """Compute current PSI across all monitored features."""
        results = {}
        for feat in self.reference_distributions.keys():
            psi = self.compute_feature_psi(feat)
            if psi is not None:
                results[feat] = round(psi, 4)
        return results

    def get_summary(self) -> dict[str, Any]:
        """Retrieve current drift health status."""
        with self._lock:
            sample_count = len(self.score_buffer)
            mean_score = float(np.mean(self.score_buffer)) if sample_count > 0 else 0.0

        psi_metrics = self.compute_all_psi()
        # Categorize drift status: <0.10: Stable, 0.10-0.25: Moderate Drift, >0.25: Significant Drift
        drift_alerts = {feat: val for feat, val in psi_metrics.items() if val >= 0.25}
        moderate_drifts = {feat: val for feat, val in psi_metrics.items() if 0.10 <= val < 0.25}

        return {
            "window_size": self.window_size,
            "current_samples": sample_count,
            "mean_fraud_probability": round(mean_score, 4),
            "feature_psi": psi_metrics,
            "status": "CRITICAL_DRIFT" if drift_alerts else ("MODERATE_DRIFT" if moderate_drifts else "HEALTHY"),
            "critical_features": list(drift_alerts.keys()),
        }


# Global Singleton
_global_drift_monitor: StreamingDriftMonitor | None = None


def get_drift_monitor(
    reference_path: str = "models/reference_distributions.json",
    window_size: int = 1000,
) -> StreamingDriftMonitor:
    """Singleton getter for StreamingDriftMonitor."""
    global _global_drift_monitor
    if _global_drift_monitor is None:
        _global_drift_monitor = StreamingDriftMonitor(
            reference_path=reference_path,
            window_size=window_size,
        )
    return _global_drift_monitor
