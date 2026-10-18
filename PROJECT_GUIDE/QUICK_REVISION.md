# Quick Revision Sheet: Real-Time Fraud Scoring Service

> **Quick Study Target**: 10-minute high-yield review sheet for tomorrow's interview.

---

## 1. Core Summary At a Glance

| Aspect | Project Detail |
|---|---|
| **Project Objective** | Score credit card transactions in real time (< 300 ms), outputting calibrated fraud probabilities, automated decisions (`APPROVE`, `MANUAL REVIEW`, `BLOCK`), and top-5 TreeSHAP reason codes. |
| **Dataset** | Kaggle IEEE-CIS Fraud Detection (Vesta Corp): 590,540 rows $\times$ 434 raw features across 182 days. |
| **Class Imbalance** | **3.50% fraud** (20,663 positive fraud transactions out of 590,540). |
| **Data Split** | Strict chronological 70/15/15 time split on `TransactionDT`: Train (Days 1–120), Val (Days 120–152), Test (Days 152–182). Zero lookahead leakage. |
| **Features** | **456 features**: 22 engineered (time indices, cent parts, email domain split/match, interaction `card1_addr1`, identity missing counters) + 12 frequency encodings (`*_fq`) + raw V/D/C/M blocks. |
| **Preprocessing** | NumPy memory downcasting (cut RAM by 63% to 924 MB), Parquet Snappy compression (cut disk to 90 MB), native NaN handling in LightGBM, state-preserving `FeatureBuilder`. |
| **Primary Model** | **LightGBM Gradient Boosted Decision Trees** (`lgb.Booster`), tuned with Optuna over 5 trials, using early stopping (30 rounds on validation PR-AUC). |
| **Why LightGBM?** | Native NaN handling, 3–4x faster histogram training than XGBoost, ~0.5 ms C++ single-row inference loop (5.23 ms p50 HTTP latency without SHAP on 4 workers), and native compiled TreeSHAP support. |
| **Probability Calibration** | **Isotonic Regression** on validation predictions: cut Expected Calibration Error (ECE) to **0.0018** and achieved a Brier score of **0.0221** without degrading ROC-AUC ranking. |
| **Decision Thresholds** | Derived from asymmetric cost matrix ($C_{\text{FP}} = \$10$, $C_{\text{FN}} = \text{Amount}$): `t_review = 0.0595` (cost-optimal), `t_block = 0.7500` ($\ge 90\%$ precision). |
| **Evaluation Metrics** | **ROC-AUC = 0.9103**, **PR-AUC = 0.5289** (calibrated) on held-out test data. Recall @ 80% Precision = 33.08%; Recall @ Top 1% Alerts = 25.53%. |
| **Inference Latency** | Sustained benchmark (4 workers, load generator on same machine): **5.23 ms** p50 without SHAP / **36.43 ms** p50 with SHAP at concurrency 1 (184.4 req/s vs 26.0 req/s). Concurrency 10: **22.23 ms** p50 without SHAP / **87.64 ms** p50 with SHAP (304.2 req/s vs 100.1 req/s). TreeSHAP compute cost is ~22 ms per transaction. |
| **Tech Stack** | Python 3.12, LightGBM, Scikit-Learn, Optuna, MLflow, FastAPI, Pydantic v2, SQLAlchemy (SQLite WAL / PostgreSQL), Prometheus, React 19, Tailwind CSS, Docker. |
| **Biggest Challenge** | Eliminating temporal lookahead leakage and training-serving skew, solved by chronological splitting and state-preserving `FeatureBuilder`. |
| **Biggest Technical Decision** | Addressing class imbalance via PR-AUC metric focus, Isotonic calibration, and cost-matrix thresholding rather than synthetic oversampling (SMOTE). |
| **Top Future Improvement** | Graph neural networks to detect coordinated card-testing botnets and streaming feature velocity stores (Feast/Redis). |

---

## 2. Key Architecture Diagram

```text
Incoming JSON -> FastAPI -> Auth & Rate Limit -> Pydantic Validation
                    │
                    ▼
         FeatureBuilder.transform() (Aligns to 457 features)
                    │
                    ▼
         LightGBM Booster (Native C++ Evaluation)
                    │
                    ▼
         ProbabilityCalibrator (Isotonic Regression)
                    │
                    ▼
         Decision Engine (t_review = 0.0595, t_block = 0.7500)
                    │
                    ▼
         Native TreeSHAP (Top-5 Reason Codes in ~22 ms compute)
                    │
         ┌──────────┴──────────┐
         ▼                     ▼
SQLAlchemy Audit Log    Prometheus Metrics -> ScoreResponse JSON (< 300 ms)
```

---

## 3. Top 20 Interview Questions & High-Yield Answers

### 1. What does the system do?
Scores card transactions in real time (< 300 ms), returning calibrated fraud probabilities, automated actions (`APPROVE`, `REVIEW`, `BLOCK`), and top-5 TreeSHAP reason codes.

### 2. What dataset did you use, and what was the fraud rate?
The IEEE-CIS Fraud Detection dataset (590,540 transactions across 182 days) with a 3.50% positive fraud rate.

### 3. Why is accuracy a misleading metric for this project?
With a 3.5% fraud rate, a model predicting "all clean" achieves 96.5% accuracy while failing to detect a single fraudulent transaction.

### 4. How did you split the data?
Strictly by time on `TransactionDT`: 70% Train (Days 1–120), 15% Val (Days 120–152), and 15% Test (Days 152–182). Shuffled splits were avoided because they leak future fraud patterns.

### 5. How did you prevent data leakage?
Sorted chronologically by `TransactionDT`, fit `FeatureBuilder` strictly on the training partition, set unseen categories to default values at serving time, and verified partition independence in `tests/test_no_leakage.py`.

### 6. How did you handle missing values?
Allowed LightGBM to handle NaNs natively during split finding, and created explicit indicators: `id_missing_count` and `has_identity`.

### 7. How did you optimize memory usage?
Wrote `downcast_df()`, converting integer types to `uint8`/`uint16`/`uint32`, floats to `float32`, and strings to `category`, reducing RAM usage from 2.5 GB to 924.24 MB (63% reduction).

### 8. What is frequency encoding, and why did you use it?
Replaced high-cardinality categories with their occurrence count in the training set across 12 features (`card1`, `addr1`, etc.) to avoid adding 14,000 sparse columns via One-Hot Encoding.

### 9. Why choose LightGBM over XGBoost and Deep Learning?
LightGBM trains 3–4x faster using histogram binning, handles missing values natively, evaluates in ~0.5 ms in C++ (4.82 ms HTTP p50 on 4 workers), and has built-in TreeSHAP (~22 ms compute). Deep learning struggles with 80%+ tabular missingness.

### 10. How did you address the 3.5% class imbalance?
Did not use SMOTE because synthetic resampling distorts empirical probabilities. Handled imbalance through PR-AUC metric optimization, Isotonic calibration, and asymmetric cost-sensitive thresholding.

### 11. What is probability calibration, and why was it necessary?
Adjusting model output scores to match empirical probabilities. GBDTs optimize ranking, not calibration. Isotonic Regression reduced Expected Calibration Error (ECE) from 0.0039 to 0.0018.

### 12. What were your final model metrics?
On the held-out test set: **ROC-AUC = 0.9103**, **PR-AUC = 0.5289** (calibrated), ECE = 0.0018, Brier Score = 0.0221, Recall @ Precision = 80% is 33.08%.

### 13. How did you derive the review threshold ($t_{\text{review}} = 0.0595$)?
Grid-searched thresholds to minimize the financial loss function: $\text{Loss} = \sum (\text{FN} \times \text{Amt}) + \sum (\text{FP} \times \$10)$.

### 14. How did you derive the block threshold ($t_{\text{block}} = 0.7500$)?
Found the lowest threshold where precision $\ge 90\%$ on validation data, ensuring automated declines maintain a false alarm rate under 10%.

### 15. How do you generate SHAP explanations in real time?
Using LightGBM's native C++ TreeSHAP via `booster.predict(..., pred_contrib=True)`, computing exact contributions in ~22 ms compute (~33.41 ms p50 HTTP latency at c=1 on 4 workers).

### 16. How does FastAPI prevent reloading models on every request?
Using the `@asynccontextmanager` lifespan handler, loading `FraudPredictor` once into memory at application boot.

### 17. How does rate limiting work?
`src/fraud/serving/ratelimit.py` implements a sliding-window token bucket limiter (5,000 requests/minute) with an in-memory fallback if Redis is unavailable.

### 18. What is Population Stability Index (PSI)?
A statistical metric measuring distribution drift between reference and production data. Our score PSI on the test set is 0.0018, well below the 0.10 alert threshold.

### 19. What were your load test benchmarks?
In sustained benchmarks (4 workers, load generator on same machine, 15s runs): at c=1, 5.23 ms p50 without SHAP (184.4 req/s) vs 36.43 ms p50 with SHAP (26.0 req/s); at c=10, 22.23 ms p50 without SHAP (304.2 req/s) vs 87.64 ms p50 with SHAP (100.1 req/s); at c=100 stress point, 620.50 ms p50 without SHAP (118.8 req/s, 0.0% errors) vs 1164.03 ms p50 with SHAP (56.4 req/s, 0.1% errors).

### 20. What is your 30-second project pitch?
"I built an end-to-end, production-grade fraud scoring service on 590,000 IEEE-CIS transactions that evaluates card payments in real time (5.23 ms p50 without SHAP, 36.43 ms with SHAP, 22.23 ms at c=10 on 4 workers). I used LightGBM with custom frequency and interaction features, enforced a strict chronological time split to prevent data leakage, and applied Isotonic Regression calibration alongside asymmetric cost optimization. The model achieves 0.9103 ROC-AUC and 0.5289 PR-AUC on held-out test data. The service is deployed with FastAPI, returns native TreeSHAP reason codes, logs audit records asynchronously, and is fully containerized with Docker, Prometheus monitoring, and a React operations dashboard."
