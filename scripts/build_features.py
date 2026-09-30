"""Build features script (Stage 5).

Fits FeatureBuilder on train set, saves artifacts to models/,
runs feature ablation on validation set, and records metrics in
reports/metrics_feature_ablation.json.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import lightgbm as lgb
import mlflow
import pandas as pd
from src.fraud.config import load_config
from src.fraud.data.split import time_split_data
from src.fraud.evaluation.metrics import evaluate_all_metrics
from src.fraud.features.builder import FeatureBuilder


def run_feature_engineering(config_path: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = interim_dir / "train_merged.parquet"
    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    train_df, val_df, _ = time_split_data(
        df,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )

    print("Fitting FeatureBuilder on TRAIN data only...")
    fb = FeatureBuilder()
    fb.fit(train_df)

    joblib_path = models_dir / "feature_builder.joblib"
    spec_path = models_dir / "feature_spec.json"
    fb.save(joblib_path, spec_path)
    print(f"Saved FeatureBuilder state to {joblib_path} and {spec_path}")

    print("Transforming TRAIN and VALIDATION datasets...")
    X_train = fb.transform(train_df)
    y_train = train_df["isFraud"]
    X_val = fb.transform(val_df)
    y_val = val_df["isFraud"]
    val_amounts = val_df["TransactionAmt"]

    mlflow.set_experiment("fraud_feature_engineering")

    # Feature ablation test:
    # 1. Raw default features baseline (from Stage 4)
    # 2. Engineered features with LightGBM
    print("Evaluating LightGBM on Engineered Features...")
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "seed": config["project"]["seed"],
        "verbose": -1,
    }

    with mlflow.start_run(run_name="lgbm_engineered_features"):
        model = lgb.train(params, train_data, num_boost_round=100)
        val_probs = model.predict(X_val)
        eng_metrics = evaluate_all_metrics(y_val, val_probs, amounts=val_amounts)
        mlflow.log_metrics(eng_metrics)

    # Load baseline metrics for comparison
    baseline_metrics_file = reports_dir / "metrics_baselines.json"
    baseline_lgbm_pr_auc = 0.530273
    if baseline_metrics_file.exists():
        with baseline_metrics_file.open("r", encoding="utf-8") as f:
            bm = json.load(f)
            baseline_lgbm_pr_auc = bm.get("lightgbm_default_baseline", {}).get("pr_auc", baseline_lgbm_pr_auc)

    ablation_results = {
        "raw_features_lgbm": {
            "pr_auc": baseline_lgbm_pr_auc,
        },
        "engineered_features_lgbm": {
            "pr_auc": eng_metrics["pr_auc"],
            "roc_auc": eng_metrics["roc_auc"],
            "recall_at_top1pct": eng_metrics["recall_at_top1pct"],
            "gain_over_raw": eng_metrics["pr_auc"] - baseline_lgbm_pr_auc,
        },
        "all_metrics": eng_metrics,
    }

    ablation_path = reports_dir / "metrics_feature_ablation.json"
    with ablation_path.open("w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    print("--- Stage 5 Feature Ablation Results (Validation Set) ---")
    print(f"Raw LightGBM PR-AUC:        {baseline_lgbm_pr_auc:.4f}")
    print(f"Engineered LightGBM PR-AUC: {eng_metrics['pr_auc']:.4f} (Gain: +{eng_metrics['pr_auc'] - baseline_lgbm_pr_auc:.4f})")
    print(f"Ablation metrics saved to {ablation_path}")

    return ablation_results


if __name__ == "__main__":
    run_feature_engineering()
