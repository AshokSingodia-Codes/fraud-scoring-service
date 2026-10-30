"""Script to verify artifact consistency and manifest integrity (Item 2).

Validates:
1. All SHA-256 hashes declared in models/manifest.json match files on disk.
2. Deriving thresholds from the shipped calibrator on validation probabilities
   reproduces models/thresholds.json within tolerance 1e-4.

Exits non-zero (1) on any failure.
"""

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))

import lightgbm as lgb
import pandas as pd
from src.fraud.config import load_config
from src.fraud.data.split import time_split_data
from src.fraud.features.builder import FeatureBuilder
from src.fraud.models.calibrate import ProbabilityCalibrator
from src.fraud.models.threshold import optimize_thresholds

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("check_artifact_consistency")


def check_artifact_consistency(models_dir: str = "models", tolerance: float = 1e-4) -> None:
    models_path = Path(models_dir)
    manifest_file = models_path / "manifest.json"
    if not manifest_file.exists():
        logger.error("Missing manifest file at %s", manifest_file)
        sys.exit(1)

    with manifest_file.open("r", encoding="utf-8") as mf:
        manifest_data = json.load(mf)

    # 1. Verify SHA-256 hashes
    logger.info("1. Verifying SHA-256 hashes declared in manifest...")
    declared_hashes = manifest_data.get("sha256", {})
    for fname, exp_hash in declared_hashes.items():
        fpath = models_path / fname
        if not fpath.exists():
            logger.error("Declared artifact %s does not exist on disk!", fpath)
            sys.exit(1)
        file_bytes = fpath.read_bytes()
        actual_hash = hashlib.sha256(file_bytes).hexdigest()
        valid_hashes = [exp_hash] if isinstance(exp_hash, str) else list(exp_hash)
        if actual_hash not in valid_hashes and any(fname.endswith(ext) for ext in [".txt", ".json", ".yaml", ".yml"]):
            norm_bytes = file_bytes.replace(b"\r\n", b"\n")
            norm_hash = hashlib.sha256(norm_bytes).hexdigest()
            if norm_hash in valid_hashes:
                actual_hash = norm_hash
            else:
                crlf_bytes = norm_bytes.replace(b"\n", b"\r\n")
                crlf_hash = hashlib.sha256(crlf_bytes).hexdigest()
                if crlf_hash in valid_hashes:
                    actual_hash = crlf_hash

        if actual_hash not in valid_hashes:
            logger.error("Hash mismatch for %s: expected %s, got %s", fname, valid_hashes, actual_hash)
            sys.exit(1)
        logger.info("  [OK] %s hash matches (%s)", fname, actual_hash[:16])

    # 2. Re-derive thresholds on validation with shipped calibrator
    logger.info("2. Re-deriving thresholds on validation split with shipped calibrator...")
    config = load_config()
    df = pd.read_parquet(Path(config["paths"]["interim_data_dir"]) / "train_merged.parquet")
    _, val_df, _ = time_split_data(df)
    y_val = val_df["isFraud"].values
    val_amounts = val_df["TransactionAmt"].values

    fb = FeatureBuilder.load(models_path / "feature_builder.joblib")
    booster = lgb.Booster(model_file=str(models_path / "model.txt"))
    X_val = fb.transform(val_df)
    val_raw = booster.predict(X_val)

    calibrator = ProbabilityCalibrator.load(models_path / "calibrator.joblib")
    val_cal = calibrator.calibrate(val_raw)

    rederived = optimize_thresholds(
        y_val,
        val_cal,
        val_amounts,
        fp_review_cost=config["cost"]["fp_review_cost"],
        target_block_precision=0.90,
        fit_split="validation",
    )

    with (models_path / "thresholds.json").open("r", encoding="utf-8") as tf:
        shipped = json.load(tf)

    diff_rev = abs(rederived["t_review"] - shipped["t_review"])
    diff_blk = abs(rederived["t_block"] - shipped["t_block"])
    logger.info("  t_review: rederived=%.4f, shipped=%.4f (diff=%e)", rederived["t_review"], shipped["t_review"], diff_rev)
    logger.info("  t_block : rederived=%.4f, shipped=%.4f (diff=%e)", rederived["t_block"], shipped["t_block"], diff_blk)

    if diff_rev > tolerance:
        logger.error("t_review difference %e exceeds tolerance %e!", diff_rev, tolerance)
        sys.exit(1)
    if diff_blk > tolerance:
        logger.error("t_block difference %e exceeds tolerance %e!", diff_blk, tolerance)
        sys.exit(1)

    logger.info("SUCCESS: Artifact consistency and threshold derivations fully verified.")


if __name__ == "__main__":
    check_artifact_consistency()
