# 03 - Step-by-Step Learning Curriculum

This curriculum outlines the exact sequence for mastering this codebase, progressing from foundational data concepts to production MLOps deployment.

---

## Curriculum Overview

```text
Step 1: Problem Definition & Data Foundations
   ↓
Step 2: Time-Based Splitting & Preventing Data Leakage
   ↓
Step 3: State-Preserving Feature Engineering
   ↓
Step 4: Baseline Models & Evaluation Metrics (PR-AUC vs ROC-AUC)
   ↓
Step 5: Gradient Boosting with LightGBM & Hyperparameter Tuning
   ↓
Step 6: Probability Calibration & Business Cost Optimization
   ↓
Step 7: Real-Time Explainability with TreeSHAP
   ↓
Step 8: Production Inference Engine & Latency Optimization
   ↓
Step 9: FastAPI Microservice, Authentication & Rate Limiting
   ↓
Step 10: Database Audit Logging, Prometheus Monitoring & Drift Detection
   ↓
Step 11: Frontend Dashboards (React + Vite & Streamlit)
   ↓
Step 12: Automated Testing, Containerization & CI/CD
```

---

## Step 1: Problem Definition & Data Foundations

### What to Learn
- The nature of the IEEE-CIS Fraud Detection dataset (590k transactions, 434 raw features).
- The transaction table vs the identity table, joined on `TransactionID`.
- Memory optimization via NumPy type downcasting.

### Why It Is Needed
Raw CSV data exceeds 2.5 GB of RAM when parsed as strings and 64-bit floats. Memory optimization prevents Out-Of-Memory (OOM) crashes and enables fast iteration.

### Files to Study
- `configs/config.yaml`
- `src/fraud/config.py`
- `src/fraud/data/download.py`
- `src/fraud/data/ingest.py`
- `reports/data_profile.json`

### Key Concepts to Understand Before Moving Forward
- **Left Join**: Understanding why only 24.4% of transactions have identity information, and why non-matches produce missing values (`NaN`).
- **Memory Downcasting**: How integer ranges (e.g. `uint8` for values $\le 255$, `uint16` for values $\le 65535$) reduce DataFrame memory usage from 2.5 GB to 924 MB.
- **Parquet Format**: Columnar compression with PyArrow vs row-oriented CSV text files.

---

## Step 2: Time-Based Splitting & Preventing Data Leakage

### What to Learn
- The `TransactionDT` feature: an integer representing elapsed seconds from Day 1 to Day 182.
- Why random $K$-fold cross-validation causes severe data leakage in fraud detection.
- Splitting data strictly by time into Train (first 70%), Validation (middle 15%), and Test (final 15%).

### Why It Is Needed
Fraud patterns evolve over time. If a model trains on future data to predict past transactions, evaluation metrics will be artificially inflated, and the model will degrade when deployed to production.

### Files to Study
- `src/fraud/data/split.py`
- `reports/split_info.json`
- `tests/test_split.py`
- `tests/test_no_leakage.py`

### Key Concepts to Understand Before Moving Forward
- **Temporal Leakage**: Why shuffling time-series data leaks future fraud patterns.
- **Out-of-Time Validation**: Evaluating models on chronologically future data to measure true generalization.
- **Monotonic Sorting**: Verifying that $\max(\text{Train}_{DT}) < \min(\text{Val}_{DT}) < \min(\text{Test}_{DT})$.

---

## Step 3: State-Preserving Feature Engineering

### What to Learn
- The design of the `FeatureBuilder` class.
- The separation between `fit()` (learning transformation parameters on training data) and `transform()` (applying transformations to new data).
- Frequency encoding for high-cardinality categorical variables (`card1`, `addr1`, `card1_addr1`).
- Feature interaction engineering: domain parsing (`P_emaildomain_provider`), round amount indicators, and missingness counters.

### Why It Is Needed
A major failure mode in machine learning systems is **training-serving skew**—calculating features differently during model training versus real-time serving. `FeatureBuilder` guarantees that serving-time transformations match training transformations down to the exact data types and category levels.

### Files to Study
- `src/fraud/features/builder.py`
- `scripts/build_features.py`
- `models/feature_spec.json`
- `tests/test_features.py`

### Key Concepts to Understand Before Moving Forward
- **Information Leakage in Preprocessing**: Why calculating frequencies or mean encodings on the full dataset leaks validation and test signals into the training phase.
- **High Cardinality**: Why One-Hot Encoding fails on `card1` (13,553 unique categories), producing an unmanageably wide matrix, and why frequency encoding is a more effective alternative.
- **Feature Alignment**: Ensuring that single-row inference payloads align with the full 457-feature model schema, filling missing columns with `NaN`.

---

## Step 4: Baseline Models & Evaluation Metrics

### What to Learn
- Setting up baseline benchmarks:
  - Dummy Classifier (predicting class prior: PR-AUC = 0.0343).
  - Logistic Regression with imputation and scaling (PR-AUC = 0.3806).
  - Default LightGBM on raw features (PR-AUC = 0.5303).
- Selecting the right evaluation metrics for imbalanced classification:
  - Precision-Recall Area Under Curve (PR-AUC / Average Precision).
  - ROC-AUC.
  - Recall at target precision thresholds (Recall @ Prec = 50%, Recall @ Prec = 80%).
  - Top-$k$% alert capture rates (Recall @ Top 1%, Recall @ Top 5%).

### Why It Is Needed
With 3.5% positive prevalence, accuracy is misleading (a naive classifier achieves 96.5% accuracy while missing all fraud). ROC-AUC can also mask large volumes of false positives in imbalanced scenarios. PR-AUC and alert-rate recalls provide a clear picture of real-world operational performance.

### Files to Study
- `src/fraud/models/baseline.py`
- `src/fraud/evaluation/metrics.py`
- `reports/metrics_baselines.json`
- `tests/test_baseline.py`
- `tests/test_metrics.py`

### Key Concepts to Understand Before Moving Forward
- **Precision vs Recall**: $\text{Precision} = \frac{TP}{TP + FP}$, $\text{Recall} = \frac{TP}{TP + FN}$.
- **Average Precision (PR-AUC)**: The area under the precision-recall curve across all possible decision thresholds.
- **Operational Recall**: The percentage of fraudulent transactions captured when an analyst team has bandwidth to review only the top 1% or 5% of highest-risk transactions.

---

## Step 5: Gradient Boosting with LightGBM & Hyperparameter Tuning

### What to Learn
- Gradient Boosted Decision Tree (GBDT) mechanics and histogram-based binning.
- How LightGBM handles native missing values (`NaN`) and categorical features.
- Early stopping based on validation PR-AUC to prevent overfitting.
- Automated Bayesian hyperparameter search using Optuna (`learning_rate`, `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, regularization).
- Native model serialization to `models/model.txt`.

### Why It Is Needed
Default hyperparameters often underperform on complex, noisy tabular datasets. Optuna systematically explores hyperparameter combinations, while early stopping ensures training halts when validation performance plateaus.

### Files to Study
- `src/fraud/models/train_lgbm.py`
- `scripts/train.py`
- `reports/metrics_test_stage6.json`
- `reports/metrics_feature_ablation.json`

### Key Concepts to Understand Before Moving Forward
- **Leaf-wise (Best-First) Tree Growth**: Why LightGBM splits the leaf with maximum loss reduction, and how `num_leaves` controls model capacity.
- **Feature Fraction & Bagging**: Subsampling features and rows to decorrelate individual trees and prevent overfitting.
- **Early Stopping**: Halting boosting when validation metric fails to improve for 30 consecutive iterations.

---

## Step 6: Probability Calibration & Business Cost Optimization

### What to Learn
- Why gradient boosted decision trees output uncalibrated probabilities.
- Non-parametric Isotonic Regression and how it improves Expected Calibration Error (ECE) and Brier Score.
- Translating probabilities into business decisions using an asymmetric financial cost matrix:
  - False Positive cost: $C_{FP} = \$10.00$ (manual review agent cost).
  - False Negative cost: $C_{FN} = \text{Transaction Amount}$ (unrecovered chargeback loss).
- Deriving operational thresholds:
  - Review threshold ($t_{\text{review}} = 0.0595$).
  - High-precision block threshold ($t_{\text{block}} = 0.7500$).

### Why It Is Needed
A probability of `0.20` must mean that approximately 20 out of 100 transactions are actually fraudulent. Uncalibrated scores can produce distorted risk assessments. Furthermore, business teams need concrete actions (`APPROVE`, `MANUAL REVIEW`, `BLOCK`) rather than raw probabilities, balanced against financial loss.

### Files to Study
- `src/fraud/models/calibrate.py`
- `src/fraud/models/threshold.py`
- `models/thresholds.json`
- `reports/figures/calibration.png`
- `tests/test_calibration_thresholds.py`

### Key Concepts to Understand Before Moving Forward
- **Expected Calibration Error (ECE)**: Grouping predictions into 10 bins and calculating the weighted average difference between predicted confidence and empirical accuracy.
- **Brier Score**: Mean squared error between predicted probabilities and binary true labels.
- **Cost-Sensitive Thresholding**: Choosing a decision boundary that minimizes expected financial losses rather than using an arbitrary default threshold of 0.5.

---

## Step 7: Real-Time Explainability with TreeSHAP

### What to Learn
- The mathematics of SHAP (Shapley Additive exPlanations) values in tree models.
- Native LightGBM prediction with `pred_contrib=True`.
- Translating raw log-odds margins into top-5 human-readable reason codes with directionality flags (`increases_risk` vs `lowers_risk`).

### Why It Is Needed
In banking and financial compliance (e.g. GDPR, Fair Credit Reporting Act), automated rejection or review of a transaction must include adverse action reasons explaining *why* the transaction was flagged. TreeSHAP provides mathematically consistent, local explanations in ~22 ms compute time.

### Files to Study
- `src/fraud/explain/reasons.py`
- `reports/figures/shap_summary.png`
- `tests/test_explain_drift.py`

### Key Concepts to Understand Before Moving Forward
- **Efficiency Property**: The sum of all individual feature SHAP contributions plus the base margin equals the model's raw output margin:
  $$\sum_{i=1}^{M} \phi_i(x) + \phi_0 = f(x)$$
- **TreeSHAP Complexity**: How tree structure enables exact polynomial-time explanation computation in $O(TLD^2)$ time, compared to exponential $O(2^M)$ time for model-agnostic KernelSHAP.
- **Log-Odds vs Probability Margin**: SHAP values explain contributions in log-odds space, where positive values increase fraud risk and negative values decrease it.

---

## Step 8: Production Inference Engine & Latency Optimization

### What to Learn
- Building the `FraudPredictor` class: loading all model artifacts (`feature_builder.joblib`, `model.txt`, `calibrator.joblib`, `thresholds.json`) once at startup.
- Executing the end-to-end scoring pipeline on a single transaction dictionary.
- Performance profiling: benchmarking p50, p95, and p99 latencies, throughput, and memory consumption.

### Why It Is Needed
In production payment gateways, disk I/O cannot occur during request processing. All models and transformation states must reside in memory, and the inference pipeline must consistently return responses within strict latency budgets.

### Files to Study
- `src/fraud/serving/predictor.py`
- `scripts/run_load_test.py`
- `reports/load_test.json`
- `tests/test_predictor.py`

### Key Concepts to Understand Before Moving Forward
- **Thread Safety**: Ensuring that loaded model boosters and calibrators can safely handle concurrent scoring requests across multiple worker threads.
- **Latency Percentiles (p50, p95, p99)**: Why average latency is insufficient for monitoring SLA compliance, and why tail latency matters in real-time systems.
- **Explanation Overhead**: Measuring the latency difference between running pure inference (4.82 ms p50 at c=1 on 4 workers) versus inference with TreeSHAP explanations (33.41 ms p50 at c=1 on 4 workers; ~22 ms pure TreeSHAP compute).

---

## Step 9: FastAPI Microservice, Authentication & Rate Limiting

### What to Learn
- FastAPI structure: dependency injection, lifespan context manager, and CORS middleware.
- Request payload validation using Pydantic v2 schemas (`Transaction`, `ScoreResponse`).
- Header-based API key authentication (`X-API-Key`).
- Sliding window token bucket rate limiting to prevent abuse.
- Batch scoring endpoint (`/v1/score/batch`) supporting up to 1,000 transactions.

### Why It Is Needed
Wrapping the ML engine in a secure, well-documented, authenticated HTTP API allows external payment systems, frontends, and fraud analysts to access the scoring engine.

### Files to Study
- `src/fraud/serving/app.py`
- `src/fraud/serving/schemas.py`
- `src/fraud/serving/auth.py`
- `src/fraud/serving/ratelimit.py`
- `tests/test_api.py`

### Key Concepts to Understand Before Moving Forward
- **FastAPI Lifespan (`@asynccontextmanager`)**: Loading the predictor once when the application boots and keeping it available across all requests.
- **Pydantic Validation**: How schemas reject invalid requests (e.g. negative transaction amounts) with HTTP 422 before touching the model.
- **Training-Serving Parity Test**: Verifying that a transaction scored through the single-score endpoint produces the exact same probability when scored via the batch endpoint.

---

## Step 10: Database Audit Logging, Prometheus Monitoring & Drift Detection

### What to Learn
- Relational audit logging using SQLAlchemy (`PredictionLog`) with support for SQLite (WAL mode) and PostgreSQL.
- Exporting Prometheus metrics (`fraud_requests_total`, `fraud_api_request_latency_seconds`, `fraud_prediction_score`, `fraud_decisions_total`).
- Measuring Population Stability Index (PSI) drift between reference training distributions and incoming production scores.
- Running background drift monitoring jobs (`scripts/drift_check.py`).

### Why It Is Needed
Production ML systems require ongoing observability. Storing prediction logs allows compliance audits and model retraining, Prometheus tracks real-time system health, and PSI drift alerts engineers when incoming data distributions change.

### Files to Study
- `src/fraud/serving/db.py`
- `src/fraud/serving/metrics.py`
- `src/fraud/evaluation/drift.py`
- `scripts/drift_check.py`
- `monitoring/prometheus.yml`

### Key Concepts to Understand Before Moving Forward
- **Write-Ahead Logging (WAL)**: How SQLite WAL mode allows concurrent reads and writes without thread deadlocks.
- **Graceful Failure Handling**: Why logging failures should be caught and logged as warnings rather than crashing customer-facing payment responses.
- **Population Stability Index (PSI)**: Interpreting PSI values to detect distribution shift:
  - $\text{PSI} < 0.10$: Distribution is stable.
  - $0.10 \le \text{PSI} \le 0.20$: Moderate shift; monitor closely.
  - $\text{PSI} > 0.20$: Significant distribution drift; retrain model.

---

## Step 11: Frontend Dashboards (React + Vite & Streamlit)

### What to Learn
- Building the modern React 19 + Vite dashboard:
  - Component decomposition: `Navbar`, `ScoreForm`, `ScoreResults`, `BatchScorer`, `AuditLog`, `ModelInfo`.
  - Transaction presets (Low-Risk, Medium-Risk, High-Risk).
  - Visualizing SHAP reason codes using Recharts horizontal bar charts.
- Exploring the Streamlit prototype (`app_ui/streamlit_app.py`).

### Why It Is Needed
Fraud analysts and business stakeholders need clear, visual tools to inspect transactions, understand flagged risk factors, and search historical audit logs.

### Files to Study
- `frontend/src/App.jsx`
- `frontend/src/components/ScoreForm.jsx`
- `frontend/src/components/ScoreResults.jsx`
- `frontend/src/components/BatchScorer.jsx`
- `frontend/src/components/AuditLog.jsx`
- `frontend/src/components/ModelInfo.jsx`
- `frontend/src/services/api.js`
- `app_ui/streamlit_app.py`

### Key Concepts to Understand Before Moving Forward
- **Component State Management**: Managing active tabs, form values, and API responses in React.
- **Data Visualization**: Transforming SHAP contribution dictionaries into visual horizontal bar charts with distinct risk colors (red for increased risk, green for reduced risk).

---

## Step 12: Automated Testing, Containerization & CI/CD

### What to Learn
- Pytest test suite covering unit tests, integration tests, API tests, and temporal leakage prevention.
- Multi-stage Docker builds (`python:3.11-slim` builder $\rightarrow$ runner).
- Multi-service orchestration using `docker-compose.yml` (FastAPI, PostgreSQL, Redis, Prometheus, Grafana).
- GitHub Actions CI workflow (`.github/workflows/ci.yml`).

### Why It Is Needed
Automated tests prevent regressions, Docker ensures consistent execution across development and production environments, and CI/CD verifies code quality on every commit.

### Files to Study
- `tests/conftest.py`
- `Dockerfile`
- `docker-compose.yml`
- `.github/workflows/ci.yml`
- `Makefile`
- `render.yaml`
- `vercel.json`

### Key Concepts to Understand Before Moving Forward
- **Multi-Stage Docker Builds**: Why building wheels in a builder stage and copying them to a clean runner stage produces smaller, more secure container images.
- **Non-Root Container Users**: Running containers as `appuser:1000` rather than `root` to prevent privilege escalation vulnerabilities.
- **Continuous Integration Pipeline**: Automating code checks (`ruff check .`) and running the test suite (`pytest`) on every pull request.
