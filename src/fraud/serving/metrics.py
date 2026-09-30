"""Prometheus metrics for API service (Stage 10)."""

from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter(
    "fraud_requests_total",
    "Total API Requests",
    ["endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "fraud_api_request_latency_seconds",
    "API Request Latency in Seconds",
    ["endpoint"],
)

PREDICTION_SCORE = Histogram(
    "fraud_prediction_score",
    "Predicted Fraud Probability Distribution",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
)

DECISION_COUNT = Counter(
    "fraud_decisions_total",
    "Count of decisions by action type",
    ["decision"],
)
