# 07 - Technical Interview Preparation Guide

This guide compiles technical interview questions focused on this project, organized across architecture, machine learning, systems engineering, security, and operations. Every answer is grounded directly in the codebase and verified benchmarks.

---

## 1. Project Overview & Elevator Pitch

### Q1: "Can you describe your Fraud Scoring Service project in 90 seconds?"
**Strong Answer**:
> "I built an end-to-end, production-grade fraud scoring engine on 590,000 real-world card transactions from the IEEE-CIS benchmark. It evaluates card-not-present transactions in real time (< 300 ms), outputting calibrated fraud probabilities, automated cost-based decisions (`APPROVE`, `MANUAL REVIEW`, `BLOCK`), and top-5 per-prediction SHAP reason codes.
>
> On the modeling side, I used LightGBM with custom frequency and interaction features, tuned using Optuna. To avoid future lookahead leakage, I implemented a strict 70/15/15 chronological time-based split across the 182-day transaction span. Raw model outputs were calibrated using Isotonic Regression to ensure probability alignment, reducing Expected Calibration Error to 0.0018, and achieving 0.9103 ROC-AUC and 0.5289 PR-AUC (calibrated) on the held-out test set.
>
> On the systems side, I built an asynchronous FastAPI service containerized with multi-stage Docker builds. The service includes API key authentication, sliding-window rate limiting, Prometheus telemetry, and SQLAlchemy audit logging supporting SQLite WAL mode locally and PostgreSQL in production. In sustained load tests (4 workers, load generator on same machine), the service achieved 184.4 req/s at 5.23 ms p50 latency without SHAP (36.43 ms with SHAP), scaling to 329.3 req/s at concurrency 5 and 304.2 req/s at concurrency 10 with 0.0% error rate."

---

## 2. Machine Learning & Data Pipeline Questions

### Q2: "Why did you use a time-based split instead of standard Stratified K-Fold cross-validation?"
**Strong Answer**:
> "In financial fraud detection, transactions have an explicit temporal order. Fraud attack patterns, bot infrastructure, card testing schemes, and merchant volumes are non-stationary and evolve continuously.
>
> If you apply standard random or stratified cross-validation, transactions from day 150 are used to train a model that predicts transactions from day 30. That allows the model to learn future card numbers, compromised emails, and emergent fraud rings that would not be known at inference time in production.
>
> In `src/fraud/data/split.py`, I sorted strictly on `TransactionDT` and created non-overlapping chronological partitions: Train spans Days 1 to 120, Validation spans Days 120 to 152, and Test spans Days 152 to 182. In `tests/test_no_leakage.py`, automated assertions verify that $\max(\text{Train}_{DT}) < \min(\text{Val}_{DT}) < \min(\text{Test}_{DT})$ and that transaction IDs never overlap across partitions."

### Q3: "Why did you choose PR-AUC over ROC-AUC as your primary optimization metric?"
**Strong Answer**:
> "Our dataset exhibits heavy class imbalance: only 3.5% of transactions are fraudulent (20,663 out of 590,540). 
>
> ROC-AUC evaluates the True Positive Rate against the False Positive Rate ($FPR = \frac{FP}{FP + TN}$). Because the number of True Negatives (clean transactions) is large (~570,000), a substantial surge in false alarms produces only a small change in the denominator of the FPR. This makes ROC-AUC overly optimistic in imbalanced settings.
>
> In contrast, PR-AUC (Average Precision) calculates precision ($\frac{TP}{TP + FP}$) against recall ($\frac{TP}{TP + FN}$). The denominator of precision is directly sensitive to false positives. A model predicting many false alarms sees an immediate drop in precision and PR-AUC. In addition, I tracked operational metrics like `Recall @ Precision = 80%` (which achieved 33.08%) and `Recall @ Top 1% Alerts` (25.53%), reflecting how many frauds are caught within real-world analyst review capacity."

### Q4: "How does your FeatureBuilder prevent training-serving skew?"
**Strong Answer**:
> "Training-serving skew occurs when features are computed differently during training than during real-time inference.
>
> To eliminate this risk, I implemented `FeatureBuilder` in `src/fraud/features/builder.py`, separating state fitting from transformation:
> 1. During offline training, `fit()` learns frequency encoding dictionaries and valid categorical levels *strictly from the training partition*, then serializes them to `models/feature_builder.joblib`.
> 2. During real-time serving, `FraudPredictor` loads this pre-fitted object at startup. When a single JSON transaction arrives, `transform()` maps the incoming values against the stored frequency maps, defaulting unseen categories to `0.0`, casts categorical types to match training levels, and aligns all columns with the frozen 457-feature schema in `models/feature_spec.json`.
>
> I verified this with an integration test in `tests/test_api.py` (`test_training_serving_parity`), which asserts that a transaction scored individually returns the identical probability when evaluated in a batch."

### Q5: "Why did you choose LightGBM over XGBoost, CatBoost, or a deep learning model?"
**Strong Answer**:
> "Three factors drove this decision:
> 1. **Extreme Missingness**: Out of 434 raw features, many identity and behavioral columns have 75% to 90%+ missing values. LightGBM handles missing values natively during histogram binning by routing NaNs to the branch that minimizes training loss, avoiding synthetic imputation pipelines.
> 2. **Inference Latency**: Payment networks operate under tight latency budgets (< 300 ms). LightGBM evaluates decision trees in compiled C++ in ~0.5 ms (4.82 ms HTTP p50 on 4 workers), whereas deep learning architectures (like TabNet) introduce significant PyTorch runtime overhead on tabular data without improving accuracy.
> 3. **Native TreeSHAP**: LightGBM exposes native tree margin contributions via `booster.predict(..., pred_contrib=True)` in C++, computing exact feature-level explanations in ~22 ms compute (~33.41 ms p50 HTTP latency on 4 workers) without requiring the Python `shap` package."

### Q6: "Why was probability calibration necessary, and why did you use Isotonic Regression?"
**Strong Answer**:
> "Gradient boosted decision trees minimize ranking loss, which often leads to uncalibrated probability outputs that bunch near decision boundaries or reflect class imbalance rather than true empirical risk. If a model score of `0.08` is used in a financial loss formula, we need confidence that roughly 8% of transactions at that score are actually fraudulent.
>
> In `src/fraud/models/calibrate.py`, I fit Scikit-learn's `IsotonicRegression` on validation predictions. Isotonic Regression is a non-parametric, monotonic step function. Because it is monotonic, it preserves the model's ROC-AUC ranking while correcting probability alignment. On the held-out test set, it reduced our Expected Calibration Error (ECE) from 0.0039 to 0.0018 and lowered the Brier score to 0.0221.
>
> Because our validation set is large (88,581 transactions), we have sufficient data to fit an isotonic step function reliably without the overfitting risk that can occur on small sample sizes."

---

## 3. Systems, Architecture & Latency Questions

### Q7: "How did you derive the business thresholds for APPROVE, MANUAL REVIEW, and BLOCK?"
**Strong Answer**:
> "Instead of choosing arbitrary thresholds like 0.5, I formulated an asymmetric cost matrix in `src/fraud/models/threshold.py`:
> - A False Positive costs a fixed \$10.00 in human analyst review overhead.
> - A False Negative costs the full transaction amount due to unrecovered chargebacks and merchant penalties.
>
> I computed two thresholds on the validation set:
> 1. **Manual Review Threshold ($t_{\text{review}} = 0.0595$)**: Derived via grid search to find the probability cutoff that minimizes total business financial loss ($\sum \text{FN losses} + \sum \text{FP review costs}$). Any transaction above `0.0595` is routed to the analyst queue.
> 2. **Automated Block Threshold ($t_{\text{block}} = 0.7500$)**: Set to the lowest threshold that achieves $\ge 90\%$ precision on validation data, ensuring that automated declines have a false alarm rate under 10%."

### Q8: "How does the system generate SHAP explanations in real time without causing latency spikes?"
**Strong Answer**:
> "Model-agnostic explainers like KernelSHAP or LIME approximate Shapley values by sampling perturbations hundreds of times, taking 1 to 5 seconds per row—far too slow for a live transaction API.
>
> In `src/fraud/explain/reasons.py`, I used LightGBM's native C++ implementation of TreeSHAP via `booster.predict(feature_row, pred_contrib=True)`. TreeSHAP leverages the internal tree structures to compute exact Shapley values in $O(\text{depth} \times \text{trees})$ polynomial time.
>
> It evaluates in ~22 ms compute time. The function then maps the top 5 absolute values to human-readable names and direction flags (`increases_risk` vs `lowers_risk`). In our sustained load tests (4 workers, load generator on same machine), enabling explanations increased p50 latency from 4.82 ms to 33.41 ms at concurrency 1 (and 22.81 ms to 120.25 ms at concurrency 10)—well within our 300 ms production latency budget."

### Q9: "How does your FastAPI service manage memory and avoid reloading model artifacts on every request?"
**Strong Answer**:
> "In `src/fraud/serving/app.py`, I used FastAPI's `@asynccontextmanager` lifespan handler:
> ```python
> @asynccontextmanager
> async def lifespan(app: FastAPI):
>     global predictor
>     predictor = FraudPredictor()
>     yield
> ```
> At application boot, `FraudPredictor` initializes once, loading `models/model.txt`, `models/feature_builder.joblib`, `models/calibrator.joblib`, and `models/thresholds.json` into memory. Subsequent HTTP requests access this singleton instance in memory, eliminating per-request disk reads. LightGBM booster inference in C++ is thread-safe for read-only scoring across worker threads."

### Q10: "How do you handle database write contention during high traffic without slowing down user responses?"
**Strong Answer**:
> "In `src/fraud/serving/db.py`, I implemented several layers of protection:
> 1. **SQLite WAL Mode**: For local testing and development, I configured Write-Ahead Logging (`PRAGMA journal_mode=WAL`) and `PRAGMA synchronous=NORMAL`, allowing concurrent reads while writes append to the write-ahead log.
> 2. **Thread Safety**: Database inserts in `log_prediction_to_db()` are managed with a thread lock `_db_lock` and dedicated `SessionLocal` sessions.
> 3. **Graceful Degradation**: Database write operations are wrapped in try-except blocks. If the database experiences a timeout or is temporarily unreachable, the error is logged as a warning, and the API successfully returns the scoring decision to the client without dropping the transaction."

---

## 4. Monitoring, Drift & Operational Reliability Questions

### Q11: "How do you monitor for model degradation and distribution drift in production?"
**Strong Answer**:
> "I implemented two monitoring mechanisms:
> 1. **Prometheus Telemetry (`src/fraud/serving/metrics.py`)**: The API exposes a `/metrics` endpoint tracking request counts, request latency histograms, predicted score distributions, and decision distributions (`APPROVE`, `REVIEW`, `BLOCK`).
> 2. **Population Stability Index (PSI) Drift Monitor (`scripts/drift_check.py` & `src/fraud/evaluation/drift.py`)**: A background service periodically pulls recent prediction scores from the database, compares them against the validation baseline score distribution across 10 quantile bins, and updates the `fraud_score_psi` Prometheus gauge.
>
> If score PSI exceeds `0.10`, an alert flags moderate distribution shift. If it exceeds `0.20`, a significant distribution drift is declared, indicating that model retraining or threshold adjustments are required."

### Q12: "What were the results of your load tests, and how does the service scale?"
**Strong Answer**:
> "I benchmarked the service using direct concurrent HTTP load tests (`scripts/run_load_test.py`) across sustained runs with a co-located load generator (`reports/load_test_workers_4.json`):
> - **Single Request (no SHAP)**: p50 latency was **5.23 ms** (4 workers, 184.4 req/s); p95 was **7.07 ms**.
> - **Single Request (with SHAP)**: p50 latency was **36.43 ms** (4 workers, 26.0 req/s); p95 was **49.73 ms**.
> - **Concurrency Scaling (c=5 to c=10)**: Throughput scaled to **329.3 req/s** at c=5 (14.68 ms p50) and **304.2 req/s** at c=10 (22.23 ms p50) with 0.0% error rate.
> - **Concurrency 100 (Stress Point)**: Evaluated as a labeled stress point under same-machine load generation, yielding **118.8 req/s** at 620.50 ms p50 without SHAP and **56.4 req/s** at 1164.03 ms p50 with SHAP.
>
> For horizontal scaling, the Docker container runs behind Uvicorn workers and can be scaled horizontally behind a load balancer, using Redis for centralized rate limiting and PostgreSQL for shared audit storage."
