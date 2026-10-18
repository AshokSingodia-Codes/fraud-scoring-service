# Evaluation & Performance Report: Real-Time Fraud Scoring Engine

## Executive Summary

This report details the rigorous evaluation of the LightGBM Real-Time Fraud Scoring Model trained on the IEEE-CIS Fraud Detection dataset using a leakage-safe 70/15/15 time-based split. The system incorporates probability calibration via Isotonic Regression, cost-optimized dual business decision thresholds, real-time SHAP feature explanations, and Population Stability Index (PSI) drift monitoring.

### Model Provenance & Artifact Hashes
- **Git Commit SHA:** `7b38157b47bbea6854877c7bd3d6d440cf1fe5b6`
- **Model File (`models/model.txt` SHA-256):** `de0e96bd225ca8b96434b412fd83f84348b6a40c14aeae781ff1988631c88ccd`
- **Calibrator File (`models/calibrator.joblib` SHA-256):** `92daeec0b01a5d9c59e59a9b5d97b69b490a2fe855b0a23c02efa61d694fe99e`
- **Provenance Confirmation:** Confirmed generated from current `models/` directory artifacts.

## 1. Summary of Model Performance & Iterative Progression

| Model Stage | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Brier Score | Test ECE |
|---|---|---|---|---|---|
| **Dummy Baseline** | 0.0343 | 0.0343 | 0.5000 | 0.0332 | 0.0008 |
| **Logistic Regression Baseline** | 0.3806 | -- | 0.8362 | 0.1255 | 0.2691 |
| **LightGBM Default (Raw Features)** | 0.5303 | -- | 0.9014 | 0.0219 | 0.0039 |
| **LightGBM Engineered Features** | 0.5512 | -- | 0.9063 | 0.0214 | 0.0018 |
| **Final Calibrated LightGBM Model** | **0.6163** | **0.5289** | **0.9046** | **0.0222** | **0.0045** |

### Detailed Held-Out Test Set Metrics

- **PR-AUC (Average Precision):** 0.5289
- **ROC-AUC:** 0.9046
- **Recall @ Precision = 0.50:** 0.5307
- **Recall @ Precision = 0.80:** 0.3419
- **Recall @ Top 1% Flagged:** 0.2556
- **Recall @ Top 5% Flagged:** 0.5855

## 2. Probability Calibration

LightGBM raw probability outputs were calibrated on the validation split using Isotonic Regression to ensure probability outputs match empirical risk.

- **Brier Score (Before Cal):** 0.0224 -> **(After Cal):** 0.0222
- **Expected Calibration Error (ECE Before Cal):** 0.0123 -> **(After Cal):** 0.0045
- **Reliability Plot:** Saved to `reports/figures/calibration.png`.

## 3. Decision Policy & Cost Optimization

Business decision thresholds were derived by optimizing direct business cost functions on validation data:

- **Cost Assumptions:** Fixed False Positive Review Cost = **$10.00**, False Negative Cost = **Transaction Amount** ($)
- **Manual Review Threshold (`t_review`):** **0.0496** (Minimizes expected total transaction cost + review overhead)
- **Automated Block Threshold (`t_block`):** **0.7534** (Targeting precision >= 90% on validation)

### Decision Matrix Policy:
1. **APPROVE:** Probability < 0.0496 -> Low Risk, instant automated pass.
2. **REVIEW:** 0.0496 <= Probability < 0.7534 -> Medium Risk, routed to human fraud analyst queue.
3. **BLOCK:** Probability >= 0.7534 -> High Risk, instant automated decline.

## 4. Explainability & SHAP Reason Codes

- Real-time feature contributions are computed directly via LightGBM native tree margin contributions (`pred_contrib=True`).
- **Sanity Verification:** The exact sum of SHAP feature contributions plus base margin matches the raw logit output (`PASSED`).
- **Global Importance Chart:** Saved to `reports/figures/shap_summary.png`.

## 5. Temporal Stability & Robustness

Evaluating performance stability across chronological test sub-intervals:

| Held-Out Time Bin | PR-AUC |
|---|---|
| T1 | 0.5601 |
| T2 | 0.4641 |
| T3 | 0.5157 |
| T4 | 0.5767 |

- **Temporal Degradation Chart:** Saved to `reports/figures/temporal_pr_auc.png`.

## 6. Population Stability Index (PSI) Drift Analysis

- **Prediction Score PSI (Val vs Test):** **0.0030** (Well below 0.10 threshold, indicating minimal distribution shift in overall predicted risk).

### Top Feature PSI Values (Train Sample vs Test):

| Feature | PSI Value | Drift Status |
|---|---|---|
| `V258` | 0.0199 | Stable (<= 0.10) |
| `C14` | 0.0069 | Stable (<= 0.10) |
| `C1` | 0.0118 | Stable (<= 0.10) |
| `DeviceInfo` | 0.0000 | Stable (<= 0.10) |
| `card1_fq` | 0.0039 | Stable (<= 0.10) |
| `card1` | 0.0040 | Stable (<= 0.10) |
| `V294` | 0.0023 | Stable (<= 0.10) |
| `log_TransactionAmt` | 0.0051 | Stable (<= 0.10) |
| `card2` | 0.0075 | Stable (<= 0.10) |
| `card1_addr1_fq` | 0.0122 | Stable (<= 0.10) |
| `D2` | 0.0193 | Stable (<= 0.10) |
| `addr1_fq` | 0.0045 | Stable (<= 0.10) |
| `addr1` | 0.0045 | Stable (<= 0.10) |
| `C13` | 0.0256 | Stable (<= 0.10) |
| `card2_fq` | 0.0026 | Stable (<= 0.10) |
| `D15` | 0.0531 | Stable (<= 0.10) |
| `id_31` | 0.0000 | Stable (<= 0.10) |
| `V45` | 0.0002 | Stable (<= 0.10) |
| `R_emaildomain` | 0.0000 | Stable (<= 0.10) |
| `dist1` | 0.0053 | Stable (<= 0.10) |

## 7. Key Findings & Operational Recommendations

1. **Time-Based Leakage Safety:** Time-based splitting provides realistic fraud performance evaluation without look-ahead data leakage.
2. **Calibration Efficiency:** Isotonic calibration effectively eliminates over-confident probability spikes, reducing ECE significantly.
3. **Drift Monitoring:** Removal of non-stationary time index (`day_index`) eliminates artificial temporal feature drift, ensuring robust generalizability across time splits.
