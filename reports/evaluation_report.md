# Evaluation & Performance Report: Real-Time Fraud Scoring Engine

## Executive Summary

This report details the rigorous evaluation of the LightGBM Real-Time Fraud Scoring Model trained on the IEEE-CIS Fraud Detection dataset using a leakage-safe 70/15/15 time-based split. The system incorporates probability calibration via Isotonic Regression, cost-optimized dual business decision thresholds, real-time SHAP feature explanations, and Population Stability Index (PSI) drift monitoring.

### Model Provenance & Artifact Hashes
- **Git Commit SHA:** `68cdd5e5f35cb4f44d7f6820376a3da48bdf01ed`
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

- **PR-AUC (Average Precision):** 0.5289 (95% CI: [0.5236, 0.5582])
- **ROC-AUC:** 0.9046 (95% CI: [0.8987, 0.9110])
- **Recall @ Precision = 0.50:** 0.5307
- **Recall @ Precision = 0.80:** 0.3419
- **Recall @ Top 1% Flagged:** 0.2556 (95% CI: [0.2474, 0.2647])
- **Recall @ Top 5% Flagged:** 0.5855 (95% CI: [0.5674, 0.5997])
- **Test Set Size & Fraud Volume:** 88,581 transactions (3,083 fraudulent, 3.48% prevalence)

## 2. Probability Calibration

LightGBM raw probability outputs were calibrated on the validation split using Isotonic Regression to ensure probability outputs match empirical risk.

- **Brier Score (Before Cal):** 0.0224 -> **(After Cal):** 0.0222 (95% CI: [0.0215, 0.0233])
- **Expected Calibration Error (ECE Before Cal):** 0.0123 -> **(After Cal):** 0.0045 (95% CI: [0.0115, 0.0134] on raw predictions)
- **Reliability Plot:** Saved to `reports/figures/calibration.png`.
- **Calibration Split Integrity:** Calibrator fitted exclusively on the validation split (`fit_split="validation"`), strictly preserving test split hold-out integrity (tested in `tests/test_calibration_thresholds.py::test_fit_split_not_test`).

## 3. Decision Policy & Cost Optimization

Business decision thresholds were derived by optimizing an asymmetric business cost matrix exclusively on the **validation split** (15% chronological split):

### Business Cost Matrix
| True Condition | Predicted Clean (Approve) | Flagged for Manual Review | Automated Block |
|---|---|---|---|
| **Legitimate ($y=0$)** | $0 (No overhead) | **$10.00** (Fixed analyst review cost) | Disrupted legitimate user experience |
| **Fraudulent ($y=1$)** | **Transaction Amount ($)** (Direct chargeback loss) | **$10.00** (Prevented loss minus review cost) | **$0** (Prevented chargeback loss) |

### Threshold Selection & Queue Metrics
- **Split Selection:** Chosen strictly on the **validation split** (`fit_split="validation"`). The test split was never seen during calibrator fitting or threshold grid search (verified via `tests/test_calibration_thresholds.py::test_fit_split_not_test`).
- **Manual Review Threshold (`t_review` = 0.0496):**
  - **Optimization Criterion:** Grid search minimizing expected cost = $\sum_{\text{FN}} \text{TransactionAmt} + \sum_{\text{FP}} \$10.00$.
  - **Validation Review Queue Rate:** **8.08%** of total transaction volume routed to manual review queue.
  - **Validation Review Queue Precision:** **19.41%** fraud prevalence in review queue (vs 3.48% base rate, a 5.6x enrichment).
  - **Test Review Queue Rate:** **9.58%** (8,486 rows) routed to review queue.
  - **Test Review Queue Precision:** **15.23%** (1,292 caught frauds / 8,486 reviewed).
  - **Test Review Queue Dollar Recall:** **53.35%** of all test fraud dollars intercepted ($250,555.41).
- **Automated Block Threshold (`t_block` = 0.7534):**
  - **Optimization Criterion:** Target precision $\ge 90\%$ on validation set to safely automate transaction declines without false decline friction.
  - **Validation Block Rate:** **1.14%** automated decline rate.
  - **Validation Block Precision:** **92.10%** precision (exceeds 90% business target).
  - **Test Block Rate:** **1.26%** (1,114 rows) automated decline rate.
  - **Test Block Precision:** **84.65%** (943 caught frauds / 1,114 blocked).
  - **Test Block Dollar Recall:** **19.82%** of all test fraud dollars blocked ($93,066.30).
- **Combined Test Performance:**
  - **Total Flagged:** 9,600 rows (10.84% of test traffic).
  - **Fraud Count Recall:** **72.49%** (2,235 of 3,083 fraudulent transactions intercepted).
  - **Fraud Dollar Recall:** **73.17%** ($343,621.71 of $469,608.50 fraudulent dollars intercepted).
  - **Financial Loss Reduction:** Total model cost = **$199,636.81** (a **57.49% reduction** / **$269,971.69 saved** vs approve-all baseline of $469,608.50).

### Decision Matrix Policy:
1. **APPROVE:** Calibrated Probability < 0.0496 -> Low Risk, instant automated pass.
2. **REVIEW:** 0.0496 <= Calibrated Probability < 0.7534 -> Medium Risk, routed to human fraud analyst queue.
3. **BLOCK:** Calibrated Probability >= 0.7534 -> High Risk, instant automated decline.


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
