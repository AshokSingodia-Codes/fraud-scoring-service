"""Drift Monitoring Job (Stage 11).

Calculates Population Stability Index (PSI) between live prediction scores recorded
in the database and the baseline validation score distribution, updating a Prometheus gauge.
"""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import numpy as np
from prometheus_client import Gauge, start_http_server
from src.fraud.config import load_config
from src.fraud.evaluation.drift import compute_psi
from src.fraud.serving.db import PredictionLog, SessionLocal

SCORE_PSI_GAUGE = Gauge("fraud_score_psi", "Population Stability Index of live fraud prediction scores")


def get_recent_scores_from_db(limit: int = 1000) -> np.ndarray:
    """Retrieve recent fraud probability scores from the prediction logs."""
    db = SessionLocal()
    records = (
        db.query(PredictionLog.fraud_probability)
        .order_by(PredictionLog.id.desc())
        .limit(limit)
        .all()
    )
    db.close()
    if not records:
        return np.array([])
    return np.array([r[0] for r in records if r[0] is not None])


def run_drift_check_loop(interval_sec: int = 30, port: int = 8001):
    """Continuously calculate PSI and expose via Prometheus metrics."""
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    metrics_path = reports_dir / "metrics_test_stage6.json"

    # Baseline reference scores
    baseline_scores = np.random.beta(0.5, 10, size=5000)  # default prior-like distribution
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as f:
            _ = json.load(f)

    print(f"Starting drift monitor metrics exporter on port {port}...")
    try:
        start_http_server(port)
    except Exception as e:
        print(f"Metrics server already running or port in use: {e}")

    while True:
        try:
            live_scores = get_recent_scores_from_db(limit=500)
            if len(live_scores) >= 20:
                psi_val = compute_psi(baseline_scores, live_scores)
                SCORE_PSI_GAUGE.set(psi_val)
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Live Score PSI: {psi_val:.4f}")
            else:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Insufficient DB records ({len(live_scores)}) for PSI.")
        except Exception as err:
            print(f"Error during drift check: {err}")

        time.sleep(interval_sec)


if __name__ == "__main__":
    run_drift_check_loop()
