"""Prometheus metrics for API service (Stage 10)."""

from prometheus_client import Counter, Gauge, Histogram

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

FEATURE_PSI = Gauge(
    "fraud_feature_psi",
    "Population Stability Index (PSI) drift per feature against baseline",
    ["feature"],
)

MODEL_HEALTH_STATUS = Gauge(
    "fraud_model_health_status",
    "Model health status code (1=Healthy, 2=Moderate Drift, 3=Critical Drift)",
)

SHADOW_EVALUATIONS = Counter(
    "fraud_shadow_evaluations_total",
    "Total shadow model challenger evaluations",
    ["status"],
)

SHADOW_SCORE_DIVERGENCE = Histogram(
    "fraud_shadow_score_divergence",
    "Absolute difference between champion and challenger fraud probabilities",
    buckets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
)


