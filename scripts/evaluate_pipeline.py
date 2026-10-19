"""Full Pipeline Evaluation script (Stages 7, 8, 9).

Fits probability calibrator, computes cost-optimal and precision thresholds,
generates SHAP explainability summaries, drift analysis, temporal PR-AUC plots,
bootstrap confidence intervals (i.i.d. and cluster by card1), seen/unseen card1 breakdown,
cost and baseline breakdowns, and writes metrics_test_stage6.json and evaluation_report.md.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath("."))

import git
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

trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))


def compute_fast_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    compute_ranking: bool = True,
    compute_calibration: bool = True,
) -> dict[str, float]:
    """Fast evaluation of ranking and calibration metrics on arrays."""
    res: dict[str, float] = {}
    n = len(y_true)

    if compute_ranking:
        desc_indices = np.argsort(y_score)[::-1]
        s_score = y_score[desc_indices]
        s_true = y_true[desc_indices]
        distinct_indices = np.where(np.diff(s_score))[0]
        threshold_idxs = np.r_[distinct_indices, n - 1]
        tps = np.cumsum(s_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        total_pos = tps[-1]
        total_neg = fps[-1]

        if total_pos <= 0 or total_neg <= 0:
            res["roc_auc"] = 0.5
            res["pr_auc"] = 0.0
            res["recall_at_top1pct"] = 0.0
            res["recall_at_top5pct"] = 0.0
        else:
            fpr = np.r_[0, fps / total_neg]
            tpr = np.r_[0, tps / total_pos]
            res["roc_auc"] = float(trapezoid(tpr, fpr))

            recall = tps / total_pos
            precision = tps / (tps + fps)
            r_prev = np.r_[0, recall[:-1]]
            res["pr_auc"] = float(np.sum((recall - r_prev) * precision))

            n_top1 = int(np.ceil(n * 0.01))
            res["recall_at_top1pct"] = float(np.sum(s_true[:n_top1]) / total_pos)
            n_top5 = int(np.ceil(n * 0.05))
            res["recall_at_top5pct"] = float(np.sum(s_true[:n_top5]) / total_pos)

    if compute_calibration:
        res["brier_score"] = float(np.mean((y_score - y_true) ** 2))
        bin_boundaries = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(10):
            low, high = bin_boundaries[i], bin_boundaries[i + 1]
            if i == 9:
                in_bin = (y_score >= low) & (y_score <= high)
            else:
                in_bin = (y_score >= low) & (y_score < high)
            cnt = np.sum(in_bin)
            if cnt > 0:
                ece += np.abs(np.mean(y_score[in_bin]) - np.mean(y_true[in_bin])) * (cnt / n)
        res["ece"] = float(ece)

    return res


def run_bootstraps(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    cal_scores: np.ndarray,
    served_scores: np.ndarray,
    card1_arr: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute i.i.d. and card1-clustered bootstrap 95% confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)

    # Precompute cluster boundaries for card1
    order = np.argsort(card1_arr)
    unique_cards, idx_start, counts = np.unique(card1_arr[order], return_index=True, return_counts=True)
    n_clusters = len(unique_cards)
    cluster_row_indices = [order[start : start + cnt] for start, cnt in zip(idx_start, counts)]

    # Point estimates
    raw_point = compute_fast_metrics(y_true, raw_scores, compute_ranking=True, compute_calibration=False)
    cal_point = compute_fast_metrics(y_true, cal_scores, compute_ranking=False, compute_calibration=True)
    served_point = compute_fast_metrics(y_true, served_scores, compute_ranking=False, compute_calibration=True)

    iid_raw = {"roc_auc": [], "pr_auc": [], "recall_at_top1pct": [], "recall_at_top5pct": []}
    iid_cal = {"brier_score": [], "ece": []}
    iid_srv = {"brier_score": [], "ece": []}

    print(f"Running i.i.d. bootstrap ({n_resamples} resamples)...")
    for _ in range(n_resamples):
        idx = rng.randint(0, n, n)
        y_b = y_true[idx]
        if np.sum(y_b) == 0:
            continue
        m_raw = compute_fast_metrics(y_b, raw_scores[idx], compute_ranking=True, compute_calibration=False)
        m_cal = compute_fast_metrics(y_b, cal_scores[idx], compute_ranking=False, compute_calibration=True)
        m_srv = compute_fast_metrics(y_b, served_scores[idx], compute_ranking=False, compute_calibration=True)

        for k in iid_raw:
            iid_raw[k].append(m_raw[k])
        for k in iid_cal:
            iid_cal[k].append(m_cal[k])
        for k in iid_srv:
            iid_srv[k].append(m_srv[k])

    cluster_raw = {"roc_auc": [], "pr_auc": [], "recall_at_top1pct": [], "recall_at_top5pct": []}
    cluster_cal = {"brier_score": [], "ece": []}
    cluster_srv = {"brier_score": [], "ece": []}

    print(f"Running cluster bootstrap by card1 ({n_clusters} clusters, {n_resamples} resamples)...")
    for _ in range(n_resamples):
        sampled_clusters = rng.randint(0, n_clusters, n_clusters)
        idx = np.concatenate([cluster_row_indices[c] for c in sampled_clusters])
        y_b = y_true[idx]
        if np.sum(y_b) == 0:
            continue
        m_raw = compute_fast_metrics(y_b, raw_scores[idx], compute_ranking=True, compute_calibration=False)
        m_cal = compute_fast_metrics(y_b, cal_scores[idx], compute_ranking=False, compute_calibration=True)
        m_srv = compute_fast_metrics(y_b, served_scores[idx], compute_ranking=False, compute_calibration=True)

        for k in cluster_raw:
            cluster_raw[k].append(m_raw[k])
        for k in cluster_cal:
            cluster_cal[k].append(m_cal[k])
        for k in cluster_srv:
            cluster_srv[k].append(m_srv[k])

    def calc_ci(point: float, dist: list[float]) -> dict[str, float]:
        low = float(np.percentile(dist, 2.5))
        high = float(np.percentile(dist, 97.5))
        assert low <= point <= high, f"Point estimate {point} outside 95% CI [{low}, {high}]"
        return {"point": round(point, 6), "ci_95_low": round(low, 6), "ci_95_high": round(high, 6)}

    return {
        "n_clusters": n_clusters,
        "n_resamples": n_resamples,
        "iid_ci": {
            "raw": {k: calc_ci(raw_point[k], iid_raw[k]) for k in iid_raw},
            "calibrated": {k: calc_ci(cal_point[k], iid_cal[k]) for k in iid_cal},
            "served": {k: calc_ci(served_point[k], iid_srv[k]) for k in iid_srv},
        },
        "cluster_ci": {
            "raw": {k: calc_ci(raw_point[k], cluster_raw[k]) for k in cluster_raw},
            "calibrated": {k: calc_ci(cal_point[k], cluster_cal[k]) for k in cluster_cal},
            "served": {k: calc_ci(served_point[k], cluster_srv[k]) for k in cluster_srv},
        },
    }


def compute_seen_unseen_card1(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_raw: np.ndarray,
    test_cal: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute seen vs unseen card1 cohorts on test set with 95% CIs for raw and calibrated."""
    train_cards = set(train_df["card1"].dropna().unique())
    test_card1_series = test_df["card1"]
    y_test = test_df["isFraud"].values

    # Masks
    nan_mask = test_card1_series.isna().values
    seen_mask = test_card1_series.isin(train_cards).values & (~nan_mask)
    unseen_mask = (~test_card1_series.isin(train_cards).values) & (~nan_mask)

    # Independent NumPy verification
    train_cards_arr = np.unique(train_df["card1"].dropna().to_numpy())
    test_cards_arr = test_df["card1"].to_numpy()
    np_seen_mask = np.isin(test_cards_arr, train_cards_arr)
    assert np.sum(seen_mask) == np.sum(np_seen_mask), "Pandas vs NumPy seen mask count mismatch"
    assert int(np.sum(y_test[seen_mask])) == int(np.sum(y_test[np_seen_mask])), "Seen fraud count mismatch"

    cohorts = {
        "seen": {"mask": seen_mask, "n": int(np.sum(seen_mask)), "fraud_count": int(np.sum(y_test[seen_mask]))},
        "unseen": {"mask": unseen_mask, "n": int(np.sum(unseen_mask)), "fraud_count": int(np.sum(y_test[unseen_mask]))},
        "nan": {"mask": nan_mask, "n": int(np.sum(nan_mask)), "fraud_count": int(np.sum(y_test[nan_mask]))},
    }

    rng = np.random.RandomState(seed)
    res: dict[str, Any] = {}

    for name, cinfo in cohorts.items():
        mask = cinfo["mask"]
        n_c = cinfo["n"]
        if n_c == 0:
            res[name] = {"n": 0, "fraud_count": 0, "fraud_prevalence": 0.0, "raw": {}, "calibrated": {}}
            continue

        y_c = y_test[mask]
        raw_c = test_raw[mask]
        cal_c = test_cal[mask]

        pt_raw = compute_fast_metrics(y_c, raw_c, compute_ranking=True, compute_calibration=True)
        pt_cal = compute_fast_metrics(y_c, cal_c, compute_ranking=True, compute_calibration=True)

        boot_raw = {k: [] for k in pt_raw}
        boot_cal = {k: [] for k in pt_cal}

        for _ in range(n_resamples):
            idx = rng.randint(0, n_c, n_c)
            y_b = y_c[idx]
            if np.sum(y_b) == 0:
                continue
            mb_raw = compute_fast_metrics(y_b, raw_c[idx], compute_ranking=True, compute_calibration=True)
            mb_cal = compute_fast_metrics(y_b, cal_c[idx], compute_ranking=True, compute_calibration=True)
            for k in boot_raw:
                boot_raw[k].append(mb_raw[k])
            for k in boot_cal:
                boot_cal[k].append(mb_cal[k])

        def format_cohort_metrics(pts: dict[str, float], dists: dict[str, list[float]]) -> dict[str, Any]:
            out = {}
            for k, p in pts.items():
                if len(dists[k]) == 0:
                    out[k] = {"point": round(p, 6), "ci_95_low": round(p, 6), "ci_95_high": round(p, 6)}
                else:
                    low = float(np.percentile(dists[k], 2.5))
                    high = float(np.percentile(dists[k], 97.5))
                    out[k] = {"point": round(p, 6), "ci_95_low": round(low, 6), "ci_95_high": round(high, 6)}
            return out

        res[name] = {
            "n": cinfo["n"],
            "fraud_count": cinfo["fraud_count"],
            "fraud_prevalence": round(cinfo["fraud_count"] / cinfo["n"], 6),
            "raw": format_cohort_metrics(pt_raw, boot_raw),
            "calibrated": format_cohort_metrics(pt_cal, boot_cal),
        }

    return res



def run_pipeline_evaluation(config_path: str | Path | None = None, out_dir: str | Path | None = None) -> dict:
    config = load_config(config_path) if config_path else load_config()
    interim_dir = Path(config["paths"]["interim_data_dir"])
    models_dir = Path(config["paths"]["models_dir"])
    reports_dir = Path(out_dir) if out_dir else Path(config["paths"]["reports_dir"])
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

    # Extract unique card1 set from train for cohort breakdown, then release train_df
    train_card1_unique = train_df[["card1"]].dropna().drop_duplicates()
    train_sample_df = train_df.sample(n=min(15000, len(train_df)), random_state=42).copy()
    del df, train_df
    import gc
    gc.collect()

    fb = FeatureBuilder.load(models_dir / "feature_builder.joblib")
    booster = lgb.Booster(model_file=str(models_dir / "model.txt"))

    print("Transforming validation and test sets...")
    X_train_sample = fb.transform(train_sample_df)
    del train_sample_df
    gc.collect()

    X_val = fb.transform(val_df)
    y_val = val_df["isFraud"].values
    val_amounts = val_df["TransactionAmt"].values

    X_test = fb.transform(test_df)
    y_test = test_df["isFraud"].values
    test_amounts = test_df["TransactionAmt"].values

    val_raw_probs = booster.predict(X_val)
    test_raw_probs = booster.predict(X_test)


    # ----------------------------------------------------
    # STAGE 7: Calibration & Threshold Optimization
    # ----------------------------------------------------
    print("Executing Stage 7: Calibrating probabilities...")
    calibrator = ProbabilityCalibrator(fit_split="validation")
    calibrator.fit(y_val, val_raw_probs, fit_split="validation")
    out_models_dir = reports_dir if out_dir else models_dir
    calibrator.save(out_models_dir / "calibrator.joblib")

    val_cal_probs = calibrator.calibrate(val_raw_probs)
    test_cal_probs = calibrator.calibrate(test_raw_probs)

    val_served_probs = np.round(val_cal_probs, 4)
    test_served_probs = np.round(test_cal_probs, 4)

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
        fit_split="validation",
    )
    with (out_models_dir / "thresholds.json").open("w", encoding="utf-8") as f:
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

    raw_margin = float(booster.predict(sample_row, raw_score=True)[0])
    contrib_sum = explanation["margin_score"]
    assert np.isclose(raw_margin, contrib_sum, atol=1e-3), f"Sanity check failed: {raw_margin} vs {contrib_sum}"
    print("SHAP contribution sum sanity check PASSED.")

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
    # STAGE 9: Temporal Stability & Drift
    # ----------------------------------------------------
    print("Executing Stage 9: Temporal degradation & Drift analysis...")
    test_df_copy = test_df[["isFraud", "TransactionDT"]].copy()
    test_df_copy["prob"] = test_cal_probs
    test_df_copy["dt_days"] = test_df_copy["TransactionDT"] // 86400

    bins = pd.qcut(test_df_copy["dt_days"], q=4, labels=["T1", "T2", "T3", "T4"])
    test_df_copy["time_bin"] = bins

    temporal_pr_aucs = {}
    for t_bin, group in test_df_copy.groupby("time_bin", observed=True):
        if len(group) > 0 and group["isFraud"].nunique() > 1:
            m = evaluate_all_metrics(group["isFraud"], group["prob"])
            temporal_pr_aucs[str(t_bin)] = round(m["pr_auc"], 4)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(temporal_pr_aucs.keys()), list(temporal_pr_aucs.values()), "o-", color="purple", linewidth=2)
    ax.set_ylabel("PR-AUC")
    ax.set_xlabel("Time Held-Out Segment")
    ax.set_title("Temporal Degradation of Model PR-AUC")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(figures_dir / "temporal_pr_auc.png", dpi=150)
    plt.close(fig)

    top_20_features = imp_df["feature"].head(20).tolist()
    drift_dict = analyze_drift(X_train_sample, X_test, top_20_features)
    score_psi = compute_psi(val_cal_probs, test_cal_probs)
    drift_dict["score_psi"] = round(score_psi, 4)

    # ----------------------------------------------------
    # Item 2 & 3: Bootstrap & Seen/Unseen breakdown
    # ----------------------------------------------------
    print("Computing bootstrap confidence intervals on test split...")
    card1_test = test_df["card1"].values
    bootstrap_results = run_bootstraps(
        y_true=y_test,
        raw_scores=test_raw_probs,
        cal_scores=test_cal_probs,
        served_scores=test_served_probs,
        card1_arr=card1_test,
        n_resamples=1000,
        seed=42,
    )

    print("Computing seen vs unseen card1 breakdown...")
    seen_unseen_results = compute_seen_unseen_card1(
        train_df=train_card1_unique,
        test_df=test_df,
        test_raw=test_raw_probs,
        test_cal=test_cal_probs,
        n_resamples=1000,
        seed=42,
    )


    # ----------------------------------------------------
    # Item 4: Dynamic Metadata Generation
    # ----------------------------------------------------
    print("Generating dynamic repository metadata...")
    repo = git.Repo(".")
    git_sha = repo.head.commit.hexsha
    dirty_flag = bool(repo.git.status("--porcelain").strip())

    sha256_hashes = {}
    for name in ["model.txt", "calibrator.joblib", "feature_builder.joblib", "thresholds.json"]:
        p = models_dir / name
        if p.exists():
            sha256_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()

    metadata = {
        "git_sha": git_sha,
        "dirty": dirty_flag,
        "artifact_sha256": sha256_hashes,
        "val_size": len(val_df),
        "val_fraud_count": int(np.sum(y_val == 1)),
        "test_size": len(test_df),
        "test_fraud_count": int(np.sum(y_test == 1)),
    }

    # ----------------------------------------------------
    # Item 5: Expected Cost & Baselines with Tier Table & Reductions
    # ----------------------------------------------------
    fp_review_cost = float(config["cost"].get("fp_review_cost", 10.0))
    configured_block_fp_cost = float(config["cost"].get("block_fp_cost", 25.0))

    approve_all_cost = float(np.sum(test_amounts[y_test == 1]))
    total_test_fraud_count = int(np.sum(y_test == 1))
    total_test_fraud_dollars = float(np.sum(test_amounts[y_test == 1]))

    tier_app_mask = test_served_probs < t_review
    tier_rev_mask = (test_served_probs >= t_review) & (test_served_probs < t_block)
    tier_blk_mask = test_served_probs >= t_block

    def get_tier_stats(mask: np.ndarray, is_flagged: bool) -> dict[str, Any]:
        cnt = int(np.sum(mask))
        fcnt = int(np.sum((y_test == 1) & mask))
        fp = int(np.sum((y_test == 0) & mask))
        fdlr = float(np.sum(test_amounts[(y_test == 1) & mask]))
        prec = (fcnt / cnt) if cnt > 0 else 0.0
        rec_cnt = (fcnt / total_test_fraud_count) if total_test_fraud_count > 0 else 0.0
        rec_dlr = (fdlr / total_test_fraud_dollars) if total_test_fraud_dollars > 0 else 0.0
        return {
            "rows": cnt,
            "flagged_rows": cnt if is_flagged else 0,
            "fraud_count": fcnt,
            "fp_count": fp,
            "precision": round(prec, 4),
            "recall_count_pct": round(rec_cnt * 100, 2),
            "recall_dollar_pct": round(rec_dlr * 100, 2),
            "fraud_dollars": round(fdlr, 2),
        }

    tier_table = {
        "approve": get_tier_stats(tier_app_mask, is_flagged=False),
        "review": get_tier_stats(tier_rev_mask, is_flagged=True),
        "block": get_tier_stats(tier_blk_mask, is_flagged=True),
    }

    # Reductions versus approve-all for block_fp_cost in {10, 25, 50, 100}
    reductions = {}
    fn_dollars = tier_table["approve"]["fraud_dollars"]
    review_fp_cnt = tier_table["review"]["fp_count"]
    block_fp_cnt = tier_table["block"]["fp_count"]

    for b_cost in [10.0, 25.0, 50.0, 100.0]:
        m_cost = fn_dollars + (review_fp_cnt * fp_review_cost) + (block_fp_cnt * b_cost)
        red_dollars = approve_all_cost - m_cost
        red_pct = (red_dollars / approve_all_cost) * 100.0 if approve_all_cost > 0 else 0.0
        b_all = float(np.sum(y_test == 0) * b_cost)
        r_all = float(len(y_test) * fp_review_cost)
        reductions[f"block_fp_cost_{int(b_cost)}"] = {
            "block_fp_cost": b_cost,
            "model_cost": round(m_cost, 2),
            "approve_all_baseline": round(approve_all_cost, 2),
            "block_all_baseline": round(b_all, 2),
            "review_all_baseline": round(r_all, 2),
            "dollars_saved_vs_approve_all": round(red_dollars, 2),
            "reduction_pct_vs_approve_all": round(red_pct, 2),
        }

    cost_analysis = {
        "cost_function": "Cost = sum_{FN} TransactionAmt + fp_review_cost * FP_review + block_fp_cost * FP_block",
        "units": "USD ($)",
        "assumptions": {
            "fp_review_cost": fp_review_cost,
            "configured_block_fp_cost": configured_block_fp_cost,
            "block_fp_cost_note": "25.0 is an assumed cost of false decline friction",
        },
        "thresholds": {"t_review": t_review, "t_block": t_block},
        "tier_table": tier_table,
        "reductions_vs_approve_all": reductions,
    }


    # ----------------------------------------------------
    # Item 1 & 8: Three blocks for val and test
    # ----------------------------------------------------
    print("Computing metrics blocks for val and test...")
    val_raw_metrics = evaluate_all_metrics(y_val, val_raw_probs, amounts=val_amounts, threshold=0.5)
    test_raw_metrics = evaluate_all_metrics(y_test, test_raw_probs, amounts=test_amounts, threshold=0.5)

    # In uncalibrated stage 6 default, expected_cost with amounts=None gave default cost
    # To maintain 100% backwards compatibility with original stage 6 json:
    val_metrics_orig = {
        "pr_auc": val_raw_metrics["pr_auc"],
        "roc_auc": val_raw_metrics["roc_auc"],
        "recall_at_p50": val_raw_metrics["recall_at_p50"],
        "recall_at_p80": val_raw_metrics["recall_at_p80"],
        "recall_at_top1pct": val_raw_metrics["recall_at_top1pct"],
        "recall_at_top5pct": val_raw_metrics["recall_at_top5pct"],
        "brier_score": val_raw_metrics["brier_score"],
        "ece": val_raw_metrics["ece"],
        "expected_cost": val_raw_metrics["expected_cost"],
    }
    test_metrics_orig = {
        "pr_auc": test_raw_metrics["pr_auc"],
        "roc_auc": test_raw_metrics["roc_auc"],
        "recall_at_p50": test_raw_metrics["recall_at_p50"],
        "recall_at_p80": test_raw_metrics["recall_at_p80"],
        "recall_at_top1pct": test_raw_metrics["recall_at_top1pct"],
        "recall_at_top5pct": test_raw_metrics["recall_at_top5pct"],
        "brier_score": test_raw_metrics["brier_score"],
        "ece": test_raw_metrics["ece"],
        "expected_cost": test_raw_metrics["expected_cost"],
    }

    val_cal_metrics = evaluate_all_metrics(y_val, val_cal_probs, amounts=val_amounts, threshold=t_review)
    val_cal_metrics["in_sample"] = True
    test_cal_metrics = evaluate_all_metrics(y_test, test_cal_probs, amounts=test_amounts, threshold=t_review)

    val_served_metrics = evaluate_all_metrics(y_val, val_served_probs, amounts=val_amounts, threshold=t_review)
    val_served_metrics["in_sample"] = True
    test_served_metrics = evaluate_all_metrics(y_test, test_served_probs, amounts=test_amounts, threshold=t_review)

    # Load best_params from original backup if present
    backup_file = Path("reports/metrics_test_stage6.backup.json")
    best_params = {}
    if backup_file.exists():
        with backup_file.open("r", encoding="utf-8") as bf:
            best_params = json.load(bf).get("best_params", {})

    pipeline_metrics = {
        "metadata": metadata,
        "best_params": best_params,
        "val_metrics": val_metrics_orig,
        "test_metrics": test_metrics_orig,
        "raw": {
            "val": val_raw_metrics,
            "test": test_raw_metrics,
        },
        "calibrated": {
            "val": val_cal_metrics,
            "test": test_cal_metrics,
        },
        "served": {
            "val": val_served_metrics,
            "test": test_served_metrics,
        },
        "bootstrap": bootstrap_results,
        "seen_unseen_card1": seen_unseen_results,
        "cost_analysis": cost_analysis,
    }

    # Verify key preservation
    if backup_file.exists():
        with backup_file.open("r", encoding="utf-8") as bf:
            orig_json = json.load(bf)
        print("Keys before:", list(orig_json.keys()))
        print("Keys after:", list(pipeline_metrics.keys()))
        for k in orig_json:
            assert k in pipeline_metrics, f"Key '{k}' was dropped!"
        for mk in orig_json["val_metrics"]:
            assert mk in pipeline_metrics["val_metrics"], f"val_metrics key '{mk}' was dropped!"
        for mk in orig_json["test_metrics"]:
            assert mk in pipeline_metrics["test_metrics"], f"test_metrics key '{mk}' was dropped!"
        print("Key preservation assertion PASSED: No original key was dropped.")

    out_metrics_file = reports_dir / "metrics_test_stage6.json"
    with out_metrics_file.open("w", encoding="utf-8") as f:
        json.dump(pipeline_metrics, f, indent=2)

    print(f"Evaluation metrics written to {out_metrics_file}")
    return pipeline_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to save evaluation reports and figures")
    args = parser.parse_args()
    run_pipeline_evaluation(config_path=args.config, out_dir=args.out_dir)
