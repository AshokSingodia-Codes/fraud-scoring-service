"""Per-prediction explanation module using LightGBM native contributions (Stage 8).

Computes SHAP contributions using booster.predict(X, pred_contrib=True),
returning readable top reason codes and directions for real-time API responses.
"""

import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

FEATURE_LABEL_MAP = {
    "TransactionAmt": "Transaction Amount",
    "log_TransactionAmt": "Log Transaction Amount",
    "amt_cents": "Amount Cents Part",
    "amt_is_round": "Round Amount Flag",
    "hour_of_day": "Hour of Day",
    "day_of_week": "Day of Week",
    "card1_fq": "Card1 Frequency",
    "card2_fq": "Card2 Frequency",
    "card1_addr1_fq": "Card & Address Frequency",
    "addr1_fq": "Address1 Frequency",
    "P_emaildomain_provider": "Purchaser Email Provider",
    "R_emaildomain_provider": "Recipient Email Provider",
    "email_domain_match": "Email Domain Match Flag",
    "has_identity": "Identity Provided Flag",
    "id_missing_count": "Missing Identity Fields Count",
}


def explain_prediction(
    booster: lgb.Booster,
    feature_row: pd.DataFrame | np.ndarray,
    top_k: int = 5,
    raw_payload: dict[str, Any] | None = None,
    feature_names: list[str] | None = None,
    contribs: np.ndarray | None = None,
    num_threads: int = 1,
) -> dict[str, Any]:
    """Compute per-feature SHAP contributions for a single feature row.

    Returns:
        Dict with top_reasons list, base_value, and margin_score.
    """
    t_shap_0 = time.perf_counter()
    if contribs is None:
        contribs = booster.predict(feature_row, pred_contrib=True, num_threads=num_threads)[0]
    elif contribs.ndim > 1:
        contribs = contribs[0]
    t_shap_compute = (time.perf_counter() - t_shap_0) * 1000.0

    t_reasons_0 = time.perf_counter()
    if feature_names is None:
        feature_names = booster.feature_name()

    feat_contribs = contribs[:-1]
    bias = float(contribs[-1])
    total_margin = float(np.sum(contribs))

    # Absolute contribution ranking
    abs_indices = np.argsort(-np.abs(feat_contribs))[:top_k]

    reasons = []
    for idx in abs_indices:
        fname = feature_names[idx]
        if raw_payload is not None and fname in raw_payload:
            val = raw_payload[fname]
        elif isinstance(feature_row, pd.DataFrame):
            val = feature_row[fname].iloc[0]
        else:
            val = feature_row[0, idx]
        contrib_val = float(feat_contribs[idx])
        readable_name = FEATURE_LABEL_MAP.get(fname, fname)

        reasons.append(
            {
                "feature": fname,
                "label": readable_name,
                "value": str(val) if pd.notna(val) else None,
                "contribution": round(contrib_val, 4),
                "direction": "increases_risk" if contrib_val > 0 else "lowers_risk",
            }
        )
    t_reasons_build = (time.perf_counter() - t_reasons_0) * 1000.0

    return {
        "base_value": round(bias, 4),
        "margin_score": round(total_margin, 4),
        "reasons": reasons,
        "timing": {
            "shap_compute": t_shap_compute,
            "reasons_build": t_reasons_build,
        },
    }
