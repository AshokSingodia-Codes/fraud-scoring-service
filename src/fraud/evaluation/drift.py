"""Drift analysis module implementing Population Stability Index (PSI) (Stage 9)."""


import numpy as np
import pandas as pd


def compute_psi(
    expected: np.ndarray | pd.Series,
    actual: np.ndarray | pd.Series,
    bins: int = 10,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and current distribution."""
    expected_series = pd.Series(expected).dropna()
    actual_series = pd.Series(actual).dropna()

    if len(expected_series) == 0 or len(actual_series) == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(expected_series, quantiles)
    edges = np.unique(edges)

    if len(edges) < 2:
        return 0.0

    e_counts, _ = np.histogram(expected_series, edges)
    a_counts, _ = np.histogram(actual_series, edges)

    e_pct = e_counts / max(len(expected_series), 1)
    a_pct = a_counts / max(len(actual_series), 1)

    e_pct = np.clip(e_pct, 1e-6, None)
    a_pct = np.clip(a_pct, 1e-6, None)

    psi_val = np.sum((a_pct - e_pct) * np.log(a_pct / e_pct))
    return float(psi_val)


def analyze_drift(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_features: list[str],
) -> dict[str, float]:
    """Compute PSI for top features between train and test sets."""
    drift_results = {}
    for feat in top_features:
        if feat in train_df.columns and feat in test_df.columns:
            s_train = pd.to_numeric(train_df[feat], errors="coerce")
            s_test = pd.to_numeric(test_df[feat], errors="coerce")
            psi_val = compute_psi(s_train, s_test)
            drift_results[feat] = round(psi_val, 4)
    return drift_results
