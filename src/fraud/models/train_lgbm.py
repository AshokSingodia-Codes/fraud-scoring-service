"""LightGBM training, tuning and tracking module (Stage 6).

Trains tuned LightGBM model on engineered features, early-stops on validation,
saves native model.txt, logs to MLflow, and evaluates held-out test set once.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath("."))

import lightgbm as lgb
import mlflow
import optuna
import pandas as pd
from src.fraud.config import load_config
from src.fraud.data.split import time_split_data
from src.fraud.evaluation.metrics import evaluate_all_metrics
from src.fraud.features.builder import FeatureBuilder

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any] | None = None,
    num_boost_round: int = 400,
    dtrain: lgb.Dataset | None = None,
    dval: lgb.Dataset | None = None,
) -> lgb.Booster:
    """Train LightGBM booster with early stopping on validation data."""
    default_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": 0.08,
        "num_leaves": 63,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": 42,
        "verbose": -1,
        "n_jobs": -1,
    }
    if params:
        default_params.update(params)

    if dtrain is None:
        dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False, params={"feature_pre_filter": False})
    if dval is None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False, params={"feature_pre_filter": False})

    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )
    return booster


def run_optuna_tuning(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 5,
) -> dict[str, Any]:
    """Run Optuna study to find optimal hyperparameter config."""
    print("Creating LightGBM training datasets...", flush=True)
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False, params={"feature_pre_filter": False})
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, free_raw_data=False, params={"feature_pre_filter": False})

    print(f"Running Optuna tuning for {n_trials} trials...", flush=True)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "boosting_type": "gbdt",
            "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.95),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 5.0, log=True),
            "feature_pre_filter": False,
            "seed": 42,
            "verbose": -1,
            "n_jobs": -1,
        }
        booster = train_lgbm(
            X_train, y_train, X_val, y_val,
            params=params,
            num_boost_round=200,
            dtrain=dtrain,
            dval=dval,
        )
        val_probs = booster.predict(X_val)
        metrics = evaluate_all_metrics(y_val, val_probs)
        pr_auc = metrics["pr_auc"]
        print(f"  Trial {trial.number}: PR-AUC = {pr_auc:.4f} (leaves={params['num_leaves']}, lr={params['learning_rate']:.3f})", flush=True)
        return pr_auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    print(f"Best trial PR-AUC: {study.best_value:.4f}", flush=True)
    return study.best_params


def train_and_evaluate(config_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path) if config_path else load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(config["paths"]["reports_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = interim_dir / "train_merged.parquet"
    print(f"Loading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    train_df, val_df, test_df = time_split_data(
        df,
        train_ratio=config["data"]["train_split"],
        val_ratio=config["data"]["val_split"],
        test_ratio=config["data"]["test_split"],
    )

    fb_path = models_dir / "feature_builder.joblib"
    if not fb_path.exists():
        print("FeatureBuilder artifact not found. Fitting a new instance...")
        fb = FeatureBuilder()
        fb.fit(train_df)
        fb.save(fb_path, models_dir / "feature_spec.json")
    else:
        print(f"Loading FeatureBuilder from {fb_path}...")
        fb = FeatureBuilder.load(fb_path)

    print("Transforming train, validation and test data...")
    X_train = fb.transform(train_df)
    y_train = train_df["isFraud"]
    X_val = fb.transform(val_df)
    y_val = val_df["isFraud"]
    val_amounts = val_df["TransactionAmt"]

    X_test = fb.transform(test_df)
    y_test = test_df["isFraud"]
    test_amounts = test_df["TransactionAmt"]

    # Tune hyperparameters
    best_params = run_optuna_tuning(X_train, y_train, X_val, y_val, n_trials=5)

    # Train final model with best params
    print("Training final LightGBM model with best parameters...", flush=True)
    full_params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "seed": config["project"]["seed"],
        "verbose": -1,
        "n_jobs": -1,
    }
    full_params.update(best_params)

    booster = train_lgbm(X_train, y_train, X_val, y_val, full_params, num_boost_round=400)

    # Save model
    model_txt_path = models_dir / "model.txt"
    booster.save_model(str(model_txt_path))
    print(f"Saved native LightGBM model to {model_txt_path}", flush=True)

    # Evaluate on Validation
    val_probs = booster.predict(X_val)
    val_metrics = evaluate_all_metrics(y_val, val_probs, amounts=val_amounts)

    # Evaluate on Test (once!)
    test_probs = booster.predict(X_test)
    test_metrics = evaluate_all_metrics(y_test, test_probs, amounts=test_amounts)

    mlflow.set_experiment("fraud_stage6_lgbm_training")
    with mlflow.start_run(run_name="final_lgbm_stage6"):
        mlflow.log_params(best_params)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_artifact(str(model_txt_path))

    results = {
        "best_params": best_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    test_metrics_path = reports_dir / "metrics_test_stage6.json"
    with test_metrics_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("--- Stage 6 Evaluation Results ---", flush=True)
    print(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f}", flush=True)
    print(f"Test PR-AUC:       {test_metrics['pr_auc']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}", flush=True)
    print(f"Test metrics saved to {test_metrics_path}", flush=True)

    return results


if __name__ == "__main__":
    train_and_evaluate()
