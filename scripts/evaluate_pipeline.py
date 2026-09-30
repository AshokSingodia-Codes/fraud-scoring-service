"""Full Pipeline Evaluation script (Stages 7, 8, 9).

Fits probability calibrator, computes cost-optimal and precision thresholds,
generates SHAP explainability summaries, drift analysis, temporal PR-AUC plots,
and writes comprehensive reports/evaluation_report.md.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.fraud.config import load_config
from src.fraud.data.split import time_split_data
from src.fraud.evaluation.drift import analyze_drift, compute_psi
from src.fraud.evaluation.metrics import evaluate_all_metrics
from src.fraud.explain.reasons import explain_prediction
from src.fraud.features.builder import FeatureBuilder
from src.fraud.models.calibrate import ProbabilityCalibrator, evaluate_calibration
from src.fraud.models.threshold import optimize_thresholds


def run_pipeline_evaluation(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = interim_dir / "train_merged.parquet"
    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    train_df, val_df, test_df = time_split_data(
        df,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )

    fb = FeatureBuilder.load(models_dir / "feature_builder.joblib")
    booster = lgb.Booster(model_file=str(models_dir / "model.txt"))

    print("Transforming validation and test sets...")
    X_train = fb.transform(train_df)
    X_val = fb.transform(val_df)
    y_val = val_df["isFraud"].values
    val_amounts = val_df["TransactionAmt"].values

    X_test = fb.transform(test_df)
    y_test = test_df["isFraud"].values
    _test_amounts = test_df["TransactionAmt"].values

    val_raw_probs = booster.predict(X_val)
    test_raw_probs = booster.predict(X_test)

    # ----------------------------------------------------
    # STAGE 7: Calibration & Threshold Optimization
    # ----------------------------------------------------
    print("Executing Stage 7: Calibrating probabilities...")
    calibrator = ProbabilityCalibrator()
    calibrator.fit(y_val, val_raw_probs)
    calibrator.save(models_dir / "calibrator.joblib")

    val_cal_probs = calibrator.calibrate(val_raw_probs)
    test_cal_probs = calibrator.calibrate(test_raw_probs)

    cal_results = evaluate_calibration(y_test, test_raw_probs, test_cal_probs)
    print(f"Test ECE before calibration: {cal_results['raw_ece']:.4f} -> after: {cal_results['calibrated_ece']:.4f}")

    # Plot Calibration Reliability Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.calibration import calibration_curve
    prob_true_raw, prob_pred_raw = calibration_curve(y_test, test_raw_probs, n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y_test, test_cal_probs, n_bins=10)

    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(prob_pred_raw, prob_true_raw, "s-", label="Uncalibrated LightGBM")
    ax.plot(prob_pred_cal, prob_true_cal, "o-", label="Isotonic Calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Reliability Diagram (Test Set)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(figures_dir / "calibration.png", dpi=150)
    plt.close(fig)

    # Optimize Thresholds on Validation
    threshold_dict = optimize_thresholds(
        y_val,
        val_cal_probs,
        val_amounts,
        fp_review_cost=config["cost"]["fp_review_cost"],
        target_block_precision=0.90,
    )
    with (models_dir / "thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(threshold_dict, f, indent=2)

    t_review = threshold_dict["t_review"]
    t_block = threshold_dict["t_block"]
    print(f"Optimal Thresholds -> t_review: {t_review}, t_block: {t_block}")

    # ----------------------------------------------------
    # STAGE 8: Explainability & Sanity Check
    # ----------------------------------------------------
    print("Executing Stage 8: Verifying SHAP explainability...")
    sample_row = X_val.iloc[[0]]
    explanation = explain_prediction(booster, sample_row, top_k=5)

    # Sanity check: sum of contributions + bias == raw prediction margin
    raw_margin = float(booster.predict(sample_row, raw_score=True)[0])
    contrib_sum = explanation["margin_score"]
    assert np.isclose(raw_margin, contrib_sum, atol=1e-3), f"Sanity check failed: {raw_margin} vs {contrib_sum}"
    print("SHAP contribution sum sanity check PASSED.")

    # Global Feature Importance Plot
    importance = booster.feature_importance(importance_type="gain")
    feat_names = booster.feature_name()
    imp_df = pd.DataFrame({"feature": feat_names, "importance": importance}).sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_imp = imp_df.head(15)
    ax.barh(top_imp["feature"][::-1], top_imp["importance"][::-1], color="steelblue")
    ax.set_xlabel("Gain Importance")
    ax.set_title("Top 15 Feature Importances (LightGBM)")
    fig.tight_layout()
    fig.savefig(figures_dir / "shap_summary.png", dpi=150)
    plt.close(fig)

    # ----------------------------------------------------
    # STAGE 9: Robustness, Temporal Degradation & Drift
    # ----------------------------------------------------
    print("Executing Stage 9: Temporal degradation & Drift analysis...")
    test_df_copy = test_df.copy()
    test_df_copy["prob"] = test_cal_probs
    test_df_copy["dt_days"] = test_df_copy["TransactionDT"] // 86400

    # Split test set into 4 equal time bins
    bins = pd.qcut(test_df_copy["dt_days"], q=4, labels=["T1", "T2", "T3", "T4"])
    test_df_copy["time_bin"] = bins

    temporal_pr_aucs = {}
    for t_bin, group in test_df_copy.groupby("time_bin", observed=True):
        if len(group) > 0 and group["isFraud"].nunique() > 1:
            m = evaluate_all_metrics(group["isFraud"], group["prob"])
            temporal_pr_aucs[str(t_bin)] = round(m["pr_auc"], 4)

    # Plot Temporal PR-AUC
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(temporal_pr_aucs.keys()), list(temporal_pr_aucs.values()), "o-", color="purple", linewidth=2)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Time Held-Out Segment")
    ax.set_title("Temporal Degradation of Model PR-AUC")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "temporal_pr_auc.png", dpi=150)
    plt.close(fig)

    # Drift PSI for top features & predictions
    top_20_features = imp_df["feature"].head(20).tolist()
    drift_dict = analyze_drift(X_train, X_test, top_20_features)
    score_psi = compute_psi(val_cal_probs, test_cal_probs)
    drift_dict["score_psi"] = round(score_psi, 4)

    # Final Evaluation Report Markdown
    eval_report_path = reports_dir / "evaluation_report.md"
    with eval_report_path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation & Performance Report\n\n")
        f.write("## 1. Summary of Model Performance\n\n")
        f.write(f"- **Validation PR-AUC:** {evaluate_all_metrics(y_val, val_cal_probs)['pr_auc']:.4f}\n")
        f.write(f"- **Test PR-AUC:** {evaluate_all_metrics(y_test, test_cal_probs)['pr_auc']:.4f}\n")
        f.write(f"- **Test ROC-AUC:** {evaluate_all_metrics(y_test, test_cal_probs)['roc_auc']:.4f}\n")
        f.write(f"- **Test ECE (Before Cal):** {cal_results['raw_ece']:.4f} -> **(After Cal):** {cal_results['calibrated_ece']:.4f}\n\n")
        f.write("## 2. Decision Policy & Cost Optimization\n\n")
        f.write(f"- **Review Threshold (`t_review`):** {t_review}\n")
        f.write(f"- **Block Threshold (`t_block`):** {t_block}\n")
        f.write("- **Cost Assumptions:** FP Review Cost = $10.0, FN Cost = Transaction Amount\n\n")
        f.write("## 3. Drift Analysis (PSI)\n\n")
        f.write(f"- **Score PSI (Val vs Test):** {score_psi:.4f}\n")
        f.write("### Top Feature PSI Values:\n\n")
        for feat, p_val in drift_dict.items():
            if feat != "score_psi":
                flag = " ⚠️ HIGH DRIFT (>0.2)" if p_val > 0.2 else ""
                f.write(f"- `{feat}`: {p_val}{flag}\n")

    print(f"Evaluation report written to {eval_report_path}")
    return {
        "calibration": cal_results,
        "thresholds": threshold_dict,
        "temporal_pr_aucs": temporal_pr_aucs,
        "score_psi": score_psi,
    }


if __name__ == "__main__":
    run_pipeline_evaluation()
