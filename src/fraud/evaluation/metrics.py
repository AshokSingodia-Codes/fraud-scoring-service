"""Model evaluation metrics including PR-AUC, ROC-AUC, ECE, Brier score, and expected cost."""


import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def compute_pr_auc(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Compute Precision-Recall Area Under Curve (Average Precision)."""
    return float(average_precision_score(y_true, y_prob))


def compute_roc_auc(y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series) -> float:
    """Compute Receiver Operating Characteristic Area Under Curve."""
    return float(roc_auc_score(y_true, y_prob))


def recall_at_precision(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    target_precision: float = 0.5,
) -> float:
    """Compute maximum recall achievable at or above a target precision threshold."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    sorted_indices = np.argsort(-y_prob_arr)
    y_true_sorted = y_true_arr[sorted_indices]

    cum_tp = np.cumsum(y_true_sorted)
    cum_fp = np.cumsum(1 - y_true_sorted)

    precisions = cum_tp / (cum_tp + cum_fp)
    recalls = cum_tp / (y_true_arr.sum() + 1e-12)

    valid_mask = precisions >= target_precision
    if not np.any(valid_mask):
        return 0.0

    return float(np.max(recalls[valid_mask]))


def recall_at_top_pct(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    top_pct: float = 0.01,
) -> float:
    """Compute recall when flagging the top K% highest probability predictions."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    n_top = int(np.ceil(len(y_true_arr) * top_pct))
    if n_top == 0:
        return 0.0

    top_indices = np.argsort(-y_prob_arr)[:n_top]
    total_positives = np.sum(y_true_arr)
    if total_positives == 0:
        return 0.0

    return float(np.sum(y_true_arr[top_indices]) / total_positives)


def compute_brier_score(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series
) -> float:
    """Compute Brier Score Loss (Mean Squared Error between prob and true binary label)."""
    return float(brier_score_loss(y_true, y_prob))


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) across equal-width probability bins."""
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        if i == n_bins - 1:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr <= bin_upper)
        else:
            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper)

        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_prob_in_bin = np.mean(y_prob_arr[in_bin])
            accuracy_in_bin = np.mean(y_true_arr[in_bin])
            ece += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


compute_ece = expected_calibration_error


def expected_cost(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
    fp_cost: float = 10.0,
) -> float:
    """Calculate total cost given predictions and thresholds.

    FN cost = transaction amount (or 100.0 if not provided).
    FP cost = fixed manual review cost (default 10.0).
    """
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    preds = (y_prob_arr >= threshold).astype(int)

    if amounts is None:
        amounts_arr = np.full_like(y_true_arr, fill_value=100.0, dtype=float)
    else:
        amounts_arr = np.asarray(amounts, dtype=float)

    # False Negatives: actual fraud (y=1) predicted clean (pred=0)
    fn_mask = (y_true_arr == 1) & (preds == 0)
    fn_loss = float(np.sum(amounts_arr[fn_mask]))

    # False Positives: clean txn (y=0) predicted fraud (pred=1)
    fp_mask = (y_true_arr == 0) & (preds == 1)
    fp_loss = float(np.sum(fp_mask) * fp_cost)

    return fn_loss + fp_loss


def evaluate_all_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    amounts: np.ndarray | pd.Series | None = None,
    threshold: float = 0.5,
    fp_cost: float = 10.0,
) -> dict[str, float]:
    """Evaluate comprehensive metric suite for a set of predictions.

    Returns:
        Dict containing PR-AUC, ROC-AUC, Recall@Prec(0.5, 0.8), Recall@Top(1%, 5%),
        Brier Score, ECE, and Expected Cost.
    """
    return {
        "pr_auc": round(compute_pr_auc(y_true, y_prob), 6),
        "roc_auc": round(compute_roc_auc(y_true, y_prob), 6),
        "recall_at_p50": round(recall_at_precision(y_true, y_prob, 0.5), 6),
        "recall_at_p80": round(recall_at_precision(y_true, y_prob, 0.8), 6),
        "recall_at_top1pct": round(recall_at_top_pct(y_true, y_prob, 0.01), 6),
        "recall_at_top5pct": round(recall_at_top_pct(y_true, y_prob, 0.05), 6),
        "brier_score": round(compute_brier_score(y_true, y_prob), 6),
        "ece": round(expected_calibration_error(y_true, y_prob), 6),
        "expected_cost": round(expected_cost(y_true, y_prob, amounts, threshold, fp_cost), 2),
    }
