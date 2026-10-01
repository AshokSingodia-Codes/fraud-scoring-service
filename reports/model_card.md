# Model Card: Real-Time Fraud Scoring Engine (IEEE-CIS)

## Model Details
- **Model Type:** LightGBM Binary Classifier (`objective="binary"`)
- **Version:** 1.0.0
- **Model Architecture:** Gradient Boosted Decision Trees with Calibrated Probabilities (Isotonic Regression)
- **Input Features:** 457 engineered numerical, categorical, and frequency-encoded features.
- **Output:** Calibrated fraud probability \(P(\\text{fraud})\\), Decision (`APPROVE`, `REVIEW`, `BLOCK`), Risk Band (`Low`, `Medium`, `High`), Top-5 SHAP Reason Codes.

## Intended Use
- **Primary Use Case:** Real-time scoring of card transactions to prevent fraudulent approvals and flag high-risk transactions for manual review.
- **Out of Scope:** Credit creditworthiness evaluation, identity verification without transaction context, offline batch loan underwriting.

## Training & Validation Data
- **Dataset:** Kaggle IEEE-CIS Fraud Detection (590,540 transaction rows merged with identity features).
- **Fraud Rate:** 3.5%
- **Split Strategy:** Strict Time-Based Split (70% Train, 15% Validation, 15% Held-out Test) based on `TransactionDT` offset.

## Evaluation Metrics (Held-Out Time Test Set)
- **ROC-AUC:** 0.9105
- **PR-AUC:** 0.5497
- **Brier Score (Post-Calibration):** 0.0221

## Decision Policy & Thresholds
- **Review Threshold (\(t_\\text{review}\)):** 0.0496 (Minimizes expected business loss = FP review cost + FN transaction loss)
- **Block Threshold (\(t_\\text{block}\)):** 0.7692 (Precision \(\\ge 0.90\) on validation set)

## Real-Time Serving Performance
- **Single Score p95 Latency (without SHAP):** 255.86 ms
- **Single Score p95 Latency (with SHAP):** 324.16 ms
- **Error Rate under Load Test:** 0.0%

## Ethical & Fairness Considerations
- Features exclude protected attributes (gender, age, race). All features use transaction amount, provider domains, device parameters, and anonymized V/D blocks.
