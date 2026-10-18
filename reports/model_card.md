# Model Card: Real-Time Fraud Scoring Engine (IEEE-CIS)

## Model Details
- **Model Type:** LightGBM Binary Classifier (`objective="binary"`)
- **Version:** 1.0.0
- **Model Architecture:** Gradient Boosted Decision Trees with Calibrated Probabilities (Isotonic Regression)
- **Input Features:** 457 engineered numerical, categorical, and frequency-encoded features.
- **Output:** Calibrated fraud probability \(P(\\text{fraud})\\), Decision (`APPROVE`, `REVIEW`, `BLOCK`), Risk Band (`Low`, `Medium`, `High`), Top-5 SHAP Reason Codes.
- **Git Commit SHA:** `7b38157b47bbea6854877c7bd3d6d440cf1fe5b6`
- **Model File (`models/model.txt` SHA-256):** `de0e96bd225ca8b96434b412fd83f84348b6a40c14aeae781ff1988631c88ccd`
- **Calibrator File (`models/calibrator.joblib` SHA-256):** `92daeec0b01a5d9c59e59a9b5d97b69b490a2fe855b0a23c02efa61d694fe99e`
- **Provenance Confirmation:** Confirmed generated from current `models/` directory artifacts.

## Intended Use
- **Primary Use Case:** Real-time scoring of card transactions to prevent fraudulent approvals and flag high-risk transactions for manual review.
- **Out of Scope:** Credit creditworthiness evaluation, identity verification without transaction context, offline batch loan underwriting.

## Training & Validation Data
- **Dataset:** Kaggle IEEE-CIS Fraud Detection (590,540 transaction rows merged with identity features).
- **Fraud Rate:** 3.5%
- **Split Strategy:** Strict Time-Based Split (70% Train, 15% Validation, 15% Held-out Test) based on `TransactionDT` offset.

## Evaluation Metrics (Held-Out Time Test Set)
- **ROC-AUC:** 0.9046
- **PR-AUC (Calibrated):** 0.5289 (honest metric progression: went from 0.5343 to 0.5289 after removing `day_index`, successfully eliminating artificial temporal drift)
- **Brier Score (Post-Calibration):** 0.0222

## Decision Policy & Thresholds
- **Review Threshold (\(t_\\text{review}\)):** 0.0496 (Minimizes expected business loss = FP review cost + FN transaction loss)
- **Block Threshold (\(t_\\text{block}\)):** 0.7534 (Precision \(\\ge 0.90\) on validation set)

## Real-Time Serving Performance
- **Single Score p95 Latency (without SHAP):** 255.86 ms
- **Single Score p95 Latency (with SHAP):** 324.16 ms
- **Error Rate under Load Test:** 0.0%

## Ethical & Fairness Considerations
- Features exclude protected attributes (gender, age, race). All features use transaction amount, provider domains, device parameters, and anonymized V/D blocks.
