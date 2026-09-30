"""Per-prediction explanation module using LightGBM native contributions (Stage 8).

Computes SHAP contributions using booster.predict(X, pred_contrib=True),
returning readable top reason codes and directions for real-time API responses.
"""

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
    "day_index": "Day Index",
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
    feature_row: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute per-feature SHAP contributions for a single feature row.

    Returns:
        Dict with top_reasons list, base_value, and margin_score.
    """
    # Compute contributions: returns array of shape (1, num_features + 1)
    contribs = booster.predict(feature_row, pred_contrib=True)[0]
    feature_names = booster.feature_name()

    feat_contribs = contribs[:-1]
    bias = float(contribs[-1])
    total_margin = float(np.sum(contribs))

    # Absolute contribution ranking
    abs_indices = np.argsort(-np.abs(feat_contribs))[:top_k]

    reasons = []
    for idx in abs_indices:
        fname = feature_names[idx]
        val = feature_row[fname].iloc[0]
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

    return {
        "base_value": round(bias, 4),
        "margin_score": round(total_margin, 4),
        "reasons": reasons,
    }
