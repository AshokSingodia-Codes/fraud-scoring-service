"""Baseline models implementation (Dummy, Logistic Regression, Default LightGBM).

Fits baseline models on TRAIN data only, evaluates on VALIDATION data,
and logs metrics to MLflow and reports/metrics_baselines.json.
"""

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.fraud.config import load_config
from src.fraud.data.split import time_split_data
from src.fraud.evaluation.metrics import evaluate_all_metrics


def train_dummy_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> np.ndarray:
    """Predict prior target probability from training distribution."""
    clf = DummyClassifier(strategy="prior")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_val)[:, 1]
    return probs


def train_logistic_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> np.ndarray:
    """Train Logistic Regression baseline with median imputation and scaling."""
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)),
        ]
    )

    pipeline.fit(X_train[numeric_cols], y_train)
    probs = pipeline.predict_proba(X_val[numeric_cols])[:, 1]
    return probs


def train_lgbm_baseline(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame
) -> np.ndarray:
    """Train LightGBM default baseline on raw features."""
    cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()

    X_train_c = X_train.copy()
    X_val_c = X_val.copy()

    for col in cat_cols:
        X_train_c[col] = X_train_c[col].astype("category")
        X_val_c[col] = X_val_c[col].astype("category")

    train_data = lgb.Dataset(X_train_c, label=y_train)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "seed": 42,
        "verbose": -1,
    }

    model = lgb.train(params, train_data, num_boost_round=100)
    probs = model.predict(X_val_c)
    return probs


def run_baselines(config_path: str | Path | None = None) -> dict[str, Any]:
    """Execute baseline models training and evaluation.

    Returns:
        Dict containing baseline metrics for all models.
    """
    config = load_config(config_path) if config_path else load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = interim_dir / "train_merged.parquet"
    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    print("Splitting dataset into train (70%), validation (15%), test (15%)...")
    train_df, val_df, _ = time_split_data(
        df,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )

    drop_cols = ["TransactionID", "TransactionDT", "isFraud"]
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols]
    y_train = train_df["isFraud"]
    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"]
    val_amounts = val_df["TransactionAmt"]

    mlflow.set_experiment("fraud_baseline_models")

    # 1. Dummy Baseline
    print("Evaluating Dummy Prior Baseline...")
    with mlflow.start_run(run_name="dummy_baseline"):
        dummy_probs = train_dummy_baseline(X_train, y_train, X_val)
        dummy_metrics = evaluate_all_metrics(y_val, dummy_probs, amounts=val_amounts)
        mlflow.log_metrics(dummy_metrics)

    # 2. Logistic Regression Baseline
    print("Evaluating Logistic Regression Baseline...")
    with mlflow.start_run(run_name="logistic_regression_baseline"):
        logreg_probs = train_logistic_baseline(X_train, y_train, X_val)
        logreg_metrics = evaluate_all_metrics(y_val, logreg_probs, amounts=val_amounts)
        mlflow.log_metrics(logreg_metrics)

    # 3. LightGBM Default Baseline
    print("Evaluating LightGBM Default Baseline...")
    with mlflow.start_run(run_name="lightgbm_default_baseline"):
        lgbm_probs = train_lgbm_baseline(X_train, y_train, X_val)
        lgbm_metrics = evaluate_all_metrics(y_val, lgbm_probs, amounts=val_amounts)
        mlflow.log_metrics(lgbm_metrics)

    results = {
        "dummy_baseline": dummy_metrics,
        "logistic_regression_baseline": logreg_metrics,
        "lightgbm_default_baseline": lgbm_metrics,
    }

    metrics_path = reports_dir / "metrics_baselines.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("--- Baseline Evaluation Results (Validation Set) ---")
    for model_name, metrics in results.items():
        print(f"[{model_name}] ROC-AUC: {metrics['roc_auc']:.4f}, PR-AUC: {metrics['pr_auc']:.4f}, ECE: {metrics['ece']:.4f}")

    print(f"Baseline metrics saved to {metrics_path}")
    return results


if __name__ == "__main__":
    run_baselines()
