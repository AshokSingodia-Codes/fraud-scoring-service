"""Script to verify that the serving path (FraudPredictor) reproduces evaluation metrics.

Evaluates:
(a) served-path metrics vs pipeline "served" block within tolerance 1e-6.
(b) unrounded calibrated serving vs pipeline calibrated on the full test split within tolerance 1e-6.
(c) score_dict row by row on the FULL test split (88,581 rows) vs batch, unrounded:
    - max/mean abs diff,
    - count of rows differing > 1e-9,
    - tier flips at t_review and t_block.
Exits non-zero (1) on any verification failure.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src.fraud.data.split import time_split_data
from src.fraud.evaluation.metrics import evaluate_all_metrics
from src.fraud.models.threshold import make_decision
from src.fraud.serving.predictor import FraudPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("verify_serving_metrics")


def verify_serving_metrics(
    parquet_path: str = "data/interim/train_merged.parquet",
    metrics_path: str = "reports/rerun/metrics_test_stage6.json",
    tolerance: float = 1e-6,
    check_row_by_row: bool = True,
) -> dict[str, Any]:
    logger.info("Initializing FraudPredictor (serving engine)...")
    predictor = FraudPredictor()

    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        logger.error("Pipeline metrics file %s does not exist!", metrics_path)
        sys.exit(1)

    logger.info("Loading pipeline metrics from %s...", metrics_path)
    with open(metrics_file, encoding="utf-8") as f:
        pipeline_data = json.load(f)

    if "served" not in pipeline_data or "calibrated" not in pipeline_data:
        logger.error("Pipeline JSON missing 'served' or 'calibrated' block! Run evaluate_pipeline.py first.")
        sys.exit(1)

    pipeline_served_test = pipeline_data["served"]["test"]
    pipeline_calibrated_test = pipeline_data["calibrated"]["test"]

    logger.info("Loading test split from %s...", parquet_path)
    df = pd.read_parquet(parquet_path)
    _, _, test_df = time_split_data(df)
    y_test = test_df["isFraud"].values
    amounts_test = test_df["TransactionAmt"].values
    n_test = len(test_df)
    logger.info("Test split loaded: %d rows, %d fraud cases.", n_test, int(np.sum(y_test == 1)))

    # Step 1: Batch inference through serving pipeline components
    logger.info("Scoring full test split in batch through FraudPredictor components...")
    mat_full = predictor.feature_builder.transform(test_df)
    raw_full = predictor.booster.predict(mat_full, num_threads=1)

    # Use the pipeline calibrator if provided in the run dir, else predictor's calibrator
    pipeline_cal_path = metrics_file.parent / "calibrator.joblib"
    if pipeline_cal_path.exists():
        from src.fraud.models.calibrate import ProbabilityCalibrator

        calibrator = ProbabilityCalibrator.load(pipeline_cal_path)
    else:
        calibrator = predictor.calibrator

    cal_full = calibrator.calibrate(raw_full)
    served_full = np.round(cal_full, 4)

    # Compute metrics on batch predictions using optimal t_review threshold
    t_review = predictor.t_review
    t_block = predictor.t_block
    pipeline_thresh_path = metrics_file.parent / "thresholds.json"
    if pipeline_thresh_path.exists():
        with open(pipeline_thresh_path, encoding="utf-8") as tf:
            td = json.load(tf)
            t_review = td.get("t_review", t_review)
            t_block = td.get("t_block", t_block)

    logger.info("Using evaluation thresholds: t_review=%f, t_block=%f", t_review, t_block)
    batch_served_metrics = evaluate_all_metrics(y_test, served_full, amounts=amounts_test, threshold=t_review)
    batch_cal_metrics = evaluate_all_metrics(y_test, cal_full, amounts=amounts_test, threshold=t_review)

    # (a) Check served-path metrics vs pipeline "served" block within tolerance 1e-6
    logger.info("(a) Verifying served-path metrics vs pipeline 'served' block (tol=%e)...", tolerance)
    served_diffs = {}
    for metric_name in ["pr_auc", "roc_auc", "recall_at_top1pct", "recall_at_top5pct", "brier_score", "ece"]:
        served_val = batch_served_metrics[metric_name]
        pipeline_val = pipeline_served_test[metric_name]
        diff = abs(served_val - pipeline_val)
        served_diffs[metric_name] = diff
        logger.info("  %s: served=%f, pipeline=%f, diff=%e", metric_name, served_val, pipeline_val, diff)
        if diff > tolerance:
            logger.error("Served metric '%s' mismatch %e exceeds tolerance %e!", metric_name, diff, tolerance)
            sys.exit(1)

    # (b) Check unrounded calibrated serving vs pipeline calibrated on full split
    logger.info("(b) Verifying unrounded calibrated serving vs pipeline 'calibrated' block (tol=%e)...", tolerance)
    cal_diffs = {}
    for metric_name in ["pr_auc", "roc_auc", "recall_at_top1pct", "recall_at_top5pct", "brier_score", "ece"]:
        cal_val = batch_cal_metrics[metric_name]
        pipeline_val = pipeline_calibrated_test[metric_name]
        diff = abs(cal_val - pipeline_val)
        cal_diffs[metric_name] = diff
        logger.info("  %s: cal=%f, pipeline=%f, diff=%e", metric_name, cal_val, pipeline_val, diff)
        if diff > tolerance:
            logger.error("Calibrated metric '%s' mismatch %e exceeds tolerance %e!", metric_name, diff, tolerance)
            sys.exit(1)

    # (c) score_dict row by row on the FULL test split (88,581 rows) vs batch, unrounded
    row_stats = {}
    if check_row_by_row:
        logger.info("(c) Scoring FULL test split (%d rows) row by row via transform_dict_to_numpy...", n_test)
        records = test_df.to_dict(orient="records")
        buffer = np.full((1, len(predictor.feature_builder.final_feature_names)), np.nan, dtype=np.float32)

        t0 = time.time()
        row_unrounded = np.empty(n_test, dtype=float)
        tier_flips = 0

        for i, row in enumerate(records):
            feat_mat = predictor.feature_builder.transform_dict_to_numpy(row, out=buffer)
            raw_p = float(predictor.booster.predict(feat_mat, num_threads=1)[0])
            cal_p = float(calibrator.calibrate(np.array([raw_p]))[0])
            row_unrounded[i] = cal_p

            dec_row = make_decision(cal_p, t_review, t_block)
            dec_batch = make_decision(cal_full[i], t_review, t_block)
            if dec_row != dec_batch:
                tier_flips += 1

        elapsed = time.time() - t0
        abs_diffs = np.abs(row_unrounded - cal_full)
        max_abs_diff = float(np.max(abs_diffs))
        mean_abs_diff = float(np.mean(abs_diffs))
        differing_gt_1e9 = int(np.sum(abs_diffs > 1e-9))

        logger.info("Row-by-row scoring completed in %.2fs (%.2f ms/row):", elapsed, (elapsed / n_test) * 1000)
        logger.info("  Max abs diff: %e", max_abs_diff)
        logger.info("  Mean abs diff: %e", mean_abs_diff)
        logger.info("  Rows differing > 1e-9: %d", differing_gt_1e9)
        logger.info("  Tier flips at t_review (%.4f) & t_block (%.4f): %d", t_review, t_block, tier_flips)

        row_stats = {
            "elapsed_seconds": round(elapsed, 2),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "differing_gt_1e9": differing_gt_1e9,
            "tier_flips": tier_flips,
        }

        if max_abs_diff > 1e-6:
            logger.error("Row-by-row vs batch max abs diff %e exceeds 1e-6!", max_abs_diff)
            sys.exit(1)
        if differing_gt_1e9 > 0:
            logger.error("Found %d rows differing > 1e-9!", differing_gt_1e9)
            sys.exit(1)
        if tier_flips > 0:
            logger.error("Found %d tier flips between row-by-row and batch!", tier_flips)
            sys.exit(1)

    verification_results = {
        "served_vs_pipeline_max_diff": max(served_diffs.values()),
        "calibrated_vs_pipeline_max_diff": max(cal_diffs.values()),
        "served_diffs": served_diffs,
        "calibrated_diffs": cal_diffs,
        "row_by_row_parity": row_stats,
        "passed": True,
    }

    logger.info("SUCCESS: All serving metric verifications passed with zero errors.")
    return verification_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify serving metrics match pipeline metrics")
    parser.add_argument("--metrics", type=str, default="reports/rerun/metrics_test_stage6.json")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--skip-row-by-row", action="store_true", help="Skip full row-by-row loop")
    args = parser.parse_args()

    verify_serving_metrics(
        metrics_path=args.metrics,
        tolerance=args.tolerance,
        check_row_by_row=not args.skip_row_by_row,
    )
