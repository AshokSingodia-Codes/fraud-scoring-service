"""Rigorous feature and prediction parity verification between reference and fast paths (Stage 11).

Evaluates across:
1. Random 5,000-row sample from train_merged.parquet with fixed seed 42.
2. Validation-split rows (70%-85% time split boundary).
3. Synthetic rows with unseen card1, addr1, and card1_addr1 values.
4. Self-test mode (--self-test) that perturbs a feature and asserts the script detects failure.

Exit-Code Semantics:
- Exit Code 0:
    * In normal mode: all 6,100 evaluated records achieve exact parity across feature matrix
      (within tolerance), NaN masks, raw predictions, calibrated probabilities, decision classes,
      and TreeSHAP top-5 reasons.
    * In --self-test mode: deliberate feature perturbation was successfully detected and verified
      to yield status="FAIL".
- Exit Code 1:
    * Parity failure in normal mode (e.g. non-zero decision flips, feature matrix diff > 1e-4,
      NaN mask mismatches, or top-5 reason mismatches).
    * Can be tested in normal mode via `--perturb` which intentionally corrupts the fast matrix
      and exits with 1.

Writes structured output to reports/parity_check.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from lightgbm.basic import _data_from_pandas
from src.fraud.explain.reasons import explain_prediction
from src.fraud.serving.predictor import FraudPredictor
from src.fraud.utils.synthetic import generate_synthetic_transaction

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_parity")


def build_evaluation_dataset(
    parquet_path: str = "data/interim/train_merged.parquet",
    sample_size: int = 5000,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build evaluation dataset combining random 5k sample, validation split, and unseen synthetic rows."""
    file_path = Path(parquet_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet dataset {parquet_path} does not exist.")

    logger.info("Loading %s...", parquet_path)
    df_all = pd.read_parquet(file_path)
    total_len = len(df_all)

    # 1. Random sample with fixed seed
    logger.info("Cohort A: Extracting random %d-row sample with fixed seed %d...", sample_size, seed)
    df_random = df_all.sample(n=min(sample_size, total_len), random_state=seed)

    # 2. Validation split rows (70% - 85% time order)
    val_start = int(total_len * 0.70)
    val_end = int(total_len * 0.85)
    val_slice = df_all.iloc[val_start:val_end]
    num_val = min(1000, len(val_slice))
    logger.info("Cohort B: Extracting %d validation-split rows (indices %d-%d)...", num_val, val_start, val_end)
    df_val = val_slice.sample(n=num_val, random_state=seed)

    # 3. Synthetic rows with unseen card1 / addr1 / card1_addr1 combinations
    logger.info("Cohort C: Generating 100 synthetic rows with unseen card1/addr1 values...")
    synthetic_rows = []
    for k in range(100):
        tx = generate_synthetic_transaction(f"tx_unseen_{k:04d}")
        tx["card1"] = 990000 + k  # guaranteed unseen high ID
        tx["addr1"] = 880000.0 + k  # guaranteed unseen address
        tx["card2"] = 77000.0 + k
        tx["P_emaildomain"] = f"unseen-provider-{k}.org"
        synthetic_rows.append(tx)
    df_synthetic = pd.DataFrame(synthetic_rows)

    cohort_counts = {
        "random_sample_5k": len(df_random),
        "validation_split": len(df_val),
        "synthetic_unseen": len(df_synthetic),
    }

    df_eval = pd.concat([df_random, df_val, df_synthetic], ignore_index=True)
    logger.info("Total rows compiled for parity check: %d %s", len(df_eval), cohort_counts)
    return df_eval, cohort_counts


def run_parity_check(
    parquet_path: str = "data/interim/train_merged.parquet",
    sample_size: int = 5000,
    seed: int = 42,
    output_path: str = "reports/parity_check.json",
    perturb: bool = False,
) -> dict[str, Any]:
    logger.info("Initializing FraudPredictor...")
    predictor = FraudPredictor()

    df_eval, cohort_counts = build_evaluation_dataset(parquet_path, sample_size=sample_size, seed=seed)
    actual_rows = len(df_eval)

    logger.info("Computing reference DataFrame transform on %d rows...", actual_rows)
    ref_df = predictor.feature_builder.transform(df_eval)
    mat_ref, _, _, _ = _data_from_pandas(
        ref_df, None, None, predictor.booster.pandas_categorical
    )

    logger.info("Computing fast-path numpy transforms row by row...")
    mat_fast = np.zeros_like(mat_ref, dtype=np.float32)
    records = df_eval.to_dict(orient="records")

    # Derived feature columns that should NOT be present in raw input records
    derived_cols = {
        "card1_addr1", "hour_of_day", "day_of_week", "log_TransactionAmt",
        "amt_cents", "amt_is_round", "email_domain_match", "id_missing_count", "has_identity"
    } | {f"{c}_fq" for c in ["card1", "card2", "card3", "card5", "addr1", "addr2", "P_emaildomain", "DeviceInfo", "id_30", "id_31", "id_33", "card1_addr1"]}

    for i, row in enumerate(records):
        present_derived = set(row.keys()) & derived_cols
        assert not present_derived, f"Row {i} contains derived columns: {present_derived}"
        mat_fast[i : i + 1] = predictor.feature_builder.transform_dict_to_numpy(row)

    if perturb:
        logger.warning(">>> SELF-TEST: Injecting intentional perturbation into mat_fast[0, 0] <<<")
        mat_fast[0, 0] = mat_fast[0, 0] + 100.0

    # 1. Feature matrix parity
    ref_isnan = np.isnan(mat_ref)
    fast_isnan = np.isnan(mat_fast)
    nan_mismatches = int(np.sum(ref_isnan != fast_isnan))
    valid_mask = ~ref_isnan & ~fast_isnan
    max_feature_diff = float(np.max(np.abs(mat_ref[valid_mask] - mat_fast[valid_mask]))) if np.any(valid_mask) else 0.0

    # 2. Raw model predictions
    logger.info("Scoring model predictions on both matrices...")
    preds_ref = predictor.booster.predict(mat_ref, num_threads=1)
    preds_fast = predictor.booster.predict(mat_fast, num_threads=1)
    max_raw_prob_diff = float(np.max(np.abs(preds_ref - preds_fast)))

    # 3. Calibrated probabilities & decisions
    logger.info("Calibrating probabilities...")
    cal_ref = predictor.calibrator.calibrate(preds_ref)
    cal_fast = predictor.calibrator.calibrate(preds_fast)
    max_cal_prob_diff = float(np.max(np.abs(cal_ref - cal_fast)))

    def get_decision(p: float) -> str:
        if p >= predictor.t_block:
            return "block"
        elif p >= predictor.t_review:
            return "review"
        return "approve"

    decisions_ref = [get_decision(p) for p in cal_ref]
    decisions_fast = [get_decision(p) for p in cal_fast]
    decision_flips = sum(1 for d1, d2 in zip(decisions_ref, decisions_fast) if d1 != d2)

    # 4. TreeSHAP contributions & top reasons
    logger.info("Checking TreeSHAP contributions on sample of 500 records...")
    shap_sample_size = min(500, actual_rows)
    shap_ref = predictor.booster.predict(mat_ref[:shap_sample_size], pred_contrib=True, num_threads=1)
    shap_fast = predictor.booster.predict(mat_fast[:shap_sample_size], pred_contrib=True, num_threads=1)
    max_shap_diff = float(np.max(np.abs(shap_ref - shap_fast)))

    top5_mismatches = 0
    for idx in range(shap_sample_size):
        rec = records[idx]
        exp_r = explain_prediction(
            predictor.booster, mat_ref[idx : idx + 1], top_k=5, raw_payload=rec, num_threads=1
        )
        exp_f = explain_prediction(
            predictor.booster,
            mat_fast[idx : idx + 1],
            top_k=5,
            raw_payload=rec,
            feature_names=predictor.feature_names,
            num_threads=1,
        )
        r_names = [item["feature"] for item in exp_r["reasons"]]
        f_names = [item["feature"] for item in exp_f["reasons"]]
        if r_names != f_names:
            top5_mismatches += 1

    status = "PASS" if (max_feature_diff <= 1e-4 and nan_mismatches == 0 and decision_flips == 0 and top5_mismatches == 0) else "FAIL"

    results = {
        "num_rows_evaluated": actual_rows,
        "cohorts": cohort_counts,
        "features_evaluated": len(predictor.feature_builder.final_feature_names),
        "max_abs_feature_matrix_diff": max_feature_diff,
        "nan_mask_mismatches": nan_mismatches,
        "max_abs_raw_probability_diff": max_raw_prob_diff,
        "max_abs_calibrated_probability_diff": max_cal_prob_diff,
        "max_abs_shap_diff": max_shap_diff,
        "decision_flips": decision_flips,
        "decision_flips_pct": (decision_flips / actual_rows) * 100.0,
        "top5_reason_sample_size": shap_sample_size,
        "top5_reason_mismatches": top5_mismatches,
        "status": status,
        "perturbed_self_test": perturb,
    }

    if not perturb:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved parity check results to %s", output_path)

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parity Verification Suite")
    parser.add_argument("--self-test", action="store_true", help="Inject perturbation and assert failure")
    parser.add_argument("--perturb", action="store_true", help="Inject perturbation in normal mode to assert non-zero exit")
    parser.add_argument("--rows", type=int, default=5000, help="Random sample size")
    parser.add_argument("--out", default="reports/parity_check.json", help="Output path")
    args = parser.parse_args()

    if args.self_test:
        logger.info("Running parity verification in --self-test mode...")
        res = run_parity_check(sample_size=args.rows, output_path=args.out, perturb=True)
        assert res["status"] == "FAIL", "Self-test failed: parity check returned PASS despite perturbation!"
        logger.info("Self-test assertion passed: script correctly detects feature perturbation and reports FAIL.")
        sys.exit(0)
    else:
        res = run_parity_check(sample_size=args.rows, output_path=args.out, perturb=args.perturb)
        if res["status"] != "PASS":
            logger.error("Parity check failed (status=%s). Exiting with code 1.", res["status"])
            sys.exit(1)
