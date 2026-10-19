"""Unit and integration tests for StreamingDriftMonitor and Prometheus drift gauges."""

import os
import tempfile

import pytest
from src.fraud.evaluation.streaming_drift import StreamingDriftMonitor


@pytest.fixture
def dummy_reference_file():
    """Create a temporary reference distributions JSON file."""
    ref_data = """{
      "TransactionAmt": {
        "edges": [0.0, 50.0, 100.0, 200.0, 500.0],
        "expected_pct": [0.25, 0.25, 0.25, 0.25],
        "count": 1000
      }
    }"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        f.write(ref_data)
        f_path = f.name
    yield f_path
    if os.path.exists(f_path):
        os.remove(f_path)


def test_streaming_drift_monitor_insufficient_samples(dummy_reference_file):
    """Confirm PSI returns None when samples in window < min_samples."""
    monitor = StreamingDriftMonitor(
        reference_path=dummy_reference_file,
        window_size=100,
        min_samples_to_evaluate=10,
    )

    for i in range(5):
        monitor.record_observation({"TransactionAmt": 25.0}, fraud_probability=0.05)

    assert monitor.compute_feature_psi("TransactionAmt") is None
    summary = monitor.get_summary()
    assert summary["current_samples"] == 5
    assert summary["status"] == "HEALTHY"


def test_streaming_drift_monitor_healthy_distribution(dummy_reference_file):
    """Confirm PSI remains low (< 0.10) for in-distribution traffic."""
    monitor = StreamingDriftMonitor(
        reference_path=dummy_reference_file,
        window_size=200,
        min_samples_to_evaluate=20,
    )

    # Ingest balanced values matching the 4 buckets evenly
    values = [25.0, 75.0, 150.0, 300.0] * 10  # 40 observations
    for v in values:
        monitor.record_observation({"TransactionAmt": v}, fraud_probability=0.02)

    psi = monitor.compute_feature_psi("TransactionAmt")
    assert psi is not None
    assert psi < 0.10  # Healthy threshold

    summary = monitor.get_summary()
    assert summary["status"] == "HEALTHY"
    assert summary["feature_psi"]["TransactionAmt"] < 0.10


def test_streaming_drift_monitor_critical_drift(dummy_reference_file):
    """Confirm PSI breaches threshold (> 0.25) when traffic shifts heavily."""
    monitor = StreamingDriftMonitor(
        reference_path=dummy_reference_file,
        window_size=200,
        min_samples_to_evaluate=20,
    )

    # Ingest exclusively high amounts (extreme distribution shift)
    for _ in range(50):
        monitor.record_observation({"TransactionAmt": 450.0}, fraud_probability=0.85)

    psi = monitor.compute_feature_psi("TransactionAmt")
    assert psi is not None
    assert psi > 0.25  # Significant / critical drift

    summary = monitor.get_summary()
    assert summary["status"] == "CRITICAL_DRIFT"
    assert "TransactionAmt" in summary["critical_features"]
