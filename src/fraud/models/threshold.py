"""Threshold optimization and decision policy module (Stage 7).

Finds t_review (cost-optimal) and t_block (precision >= 0.90) on validation data.
"""

import os
import sys
from typing import Any

sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def optimize_thresholds(
    y_val: np.ndarray | pd.Series,
    val_probs: np.ndarray,
    val_amounts: np.ndarray | pd.Series,
    fp_review_cost: float = 10.0,
    target_block_precision: float = 0.90,
) -> dict[str, Any]:
    """Compute cost-optimal t_review and high-precision t_block on validation data.

    Returns:
        Dict with t_review, t_block, cost assumptions, and validation metrics at thresholds.
    """
    y_val = np.asarray(y_val)
    val_amounts = np.asarray(val_amounts)

    precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)

    # 1. Block Threshold (t_block): lowest threshold where precision >= target_block_precision
    valid_idx = np.where(precisions >= target_block_precision)[0]
    if len(valid_idx) > 0 and valid_idx[0] < len(thresholds):
        t_block = float(thresholds[valid_idx[0]])
    else:
        # Fallback to high quantile threshold if 0.90 precision is not strictly reached
        t_block = float(np.percentile(val_probs, 99))

    # 2. Review Threshold (t_review): search grid of thresholds to minimize total cost
    grid_thresholds = np.linspace(0.01, 0.99, 100)
    best_cost = float("inf")
    best_t_review = 0.5

    for t in grid_thresholds:
        flagged = val_probs >= t
        fn = (~flagged) & (y_val == 1)
        fp = flagged & (y_val == 0)

        fn_cost = np.sum(val_amounts[fn])
        fp_cost = np.sum(fp) * fp_review_cost
        total_cost = fn_cost + fp_cost

        if total_cost < best_cost:
            best_cost = total_cost
            best_t_review = float(t)

    # Ensure t_review <= t_block
    if best_t_review > t_block:
        best_t_review = round(t_block * 0.8, 4)

    return {
        "t_review": round(best_t_review, 4),
        "t_block": round(t_block, 4),
        "cost_assumptions": {
            "fp_review_cost": fp_review_cost,
            "fn_cost_definition": "transaction amount",
            "target_block_precision": target_block_precision,
        },
    }


def make_decision(prob: float, t_review: float, t_block: float) -> str:
    """Map calibrated fraud probability to action decision."""
    if prob >= t_block:
        return "block"
    elif prob >= t_review:
        return "review"
    else:
        return "approve"
