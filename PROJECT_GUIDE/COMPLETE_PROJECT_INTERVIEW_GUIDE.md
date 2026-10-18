# Complete Project Interview Guide: Real-Time Fraud Scoring Service

> **Interview Readiness Target**: Comprehensive, project-grounded mastery for tomorrow's technical interview.
> **Repository Context**: Real-Time Fraud Scoring Service (`fraud-scoring-service`), trained on 590,540 IEEE-CIS transactions, served via FastAPI, containerized with Docker, evaluated with strict temporal splits, and monitored with Prometheus.

---

# TABLE OF CONTENTS
1. [Project Overview](#1-project-overview)
2. [Data Pipeline — The Core ML Interview Foundation](#2-data-pipeline--the-core-ml-interview-foundation)
3. [Machine Learning Model](#3-machine-learning-model)
4. [Complete Code Flow & Architecture Execution](#4-complete-code-flow--architecture-execution)
5. [Technology Stack & Tradeoff Analysis](#5-technology-stack--tradeoff-analysis)
6. [The Two Most Important Interview Questions](#6-the-two-most-important-interview-questions)
7. [Rapid-Fire Technical Interview Questions (40 Questions & Answers)](#7-rapid-fire-technical-interview-questions)
8. [Real Project Challenges & Engineering Fixes](#8-real-project-challenges--engineering-fixes)
9. [Why Not Other Methods? (The Decision Matrix)](#9-why-not-other-methods-the-decision-matrix)
10. [Limitations & Future Improvements](#10-limitations--future-improvements)
11. [2-Minute Project Pitch (Spoken Script)](#11-2-minute-project-pitch-spoken-script)
12. [30-Second Elevator Pitch (Spoken Script)](#12-30-second-elevator-pitch-spoken-script)
13. [Deep Technical Explanation](#13-deep-technical-explanation)

---

# 1. PROJECT OVERVIEW

### What the Project Does
The **Real-Time Fraud Scoring Service** evaluates credit card and digital transactions in real time (< 300 ms). It takes raw transaction attributes (amounts, card issuer codes, recipient information, email domains, device parameters), transforms them through a state-preserving feature pipeline, scores them using a tuned **LightGBM** gradient boosted decision tree, calibrates the output probability via **Isotonic Regression**, maps the probability to an automated business action (`APPROVE`, `MANUAL REVIEW`, `BLOCK`), computes **Top-5 per-prediction TreeSHAP reason codes**, and logs the result asynchronously to a database while emitting Prometheus telemetry.

### What Real-World Problem It Solves
1. **Severe Imbalance & Asymmetric Loss**: Fraud accounts for only ~3.5% of transactions. Missing a fraudulent transaction (**False Negative**) results in a direct financial chargeback loss equal to the full transaction amount plus merchant penalty fees ($15–$50). Conversely, declining a legitimate customer (**False Positive**) causes customer churn and costs ~$10.00 in human analyst manual review overhead. Standard ML models using 0.5 probability thresholds fail financially.
2. **Temporal Non-Stationarity**: Fraudsters continuously evolve their tactics. Shuffled cross-validation leaks future information, causing models to collapse when deployed into live production.
3. **Strict Latency Budgets**: Payment gateways (Stripe, Adyen, Visa Direct) enforce 100–300 ms timeout windows. Complex deep neural nets or slow perturbation-based explainers (KernelSHAP) cannot be run on the synchronous payment authorization path.
4. **Regulatory Explainability**: Financial compliance (GDPR, Fair Credit Reporting Act) mandates adverse action reasons whenever a user's transaction is blocked or held.

### Why I Built It
I built this project to master the complete lifecycle of a production-grade machine learning system—moving beyond toy Jupyter notebooks to solve real-world ML challenges: leakage-free temporal data splitting, feature engineering parity between training and serving, post-hoc probability calibration, cost-sensitive threshold optimization, measured ~22 ms per-transaction TreeSHAP explainability, asynchronous REST API serving, and distribution drift monitoring.

### Who Would Use It
- **Payment Service Providers (PSPs) & Payment Gateways**: Automated pass/fail authorization in milliseconds.
- **Card-Issuing Banks & Fintech Platforms**: Flagging suspicious transfers for step-up multi-factor authentication (MFA) or human review.
- **Fraud Operations Teams & Risk Analysts**: Reviewing flagged transactions accompanied by human-readable SHAP reason codes.

### Input $\rightarrow$ Processing $\rightarrow$ Output
- **Input**: JSON payload containing `transaction_id`, `TransactionDT` (offset seconds), `TransactionAmt`, `ProductCD`, `card1`–`card6`, `addr1`–`addr2`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`.
- **Processing**:
  1. API key authentication & sliding-window token bucket rate limiting.
  2. Pydantic v2 payload validation (`TransactionAmt > 0`).
  3. Real-time feature transformation (derived time, cent portions, frequency encodings, domain matching, identity counts) aligning to a 457-feature schema.
  4. LightGBM booster inference in native C++.
  5. Isotonic Regression probability calibration.
  6. Cost-optimal dual-threshold policy evaluation.
  7. Native TreeSHAP top-5 contribution ranking.
  8. Relational audit logging (SQLAlchemy/PostgreSQL/SQLite WAL) and Prometheus metric recording.
- **Output**: JSON payload returning `fraud_probability` (0.0000 to 1.0000), `decision` (`approve`, `review`, `block`), `risk_band` (`Low`, `Medium`, `High`), `reasons` (top-5 SHAP contributions with feature labels, values, and risk direction flags), `thresholds`, and `latency_ms`.

### Complete Project Architecture
```text
[ React 19 + Vite Dashboard / External Gateway Client ]
                       │  HTTP POST /v1/score?explain=true
                       ▼
┌──────────────────────────────────────────────────────────┐
│                   FastAPI Serving Layer                  │
│  - X-API-Key Authentication (src/fraud/serving/auth.py)  │
│  - Sliding Window Rate Limiting (serving/ratelimit.py)   │
│  - Pydantic v2 Request Validation (serving/schemas.py)   │
└──────────────────────────┬───────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────┐
│           Inference Engine (FraudPredictor)              │
│  1. FeatureBuilder.transform() (features/builder.py)     │
│     - Frequencies, domains, time, amount interactions    │
│     - Schema alignment: 457 features                     │
│  2. LightGBM Booster (models/model.txt)                  │
│     - Raw logit / margin output                          │
│  3. ProbabilityCalibrator (models/calibrator.joblib)     │
│     - Isotonic Regression piecewise mapping              │
│  4. Cost Decision Policy (models/thresholds.json)        │
│     - t_review = 0.0595, t_block = 0.7500                │
│  5. Native TreeSHAP Explainer (explain/reasons.py)       │
│     - O(depth × trees) native C++ feature attribution    │
└──────────────────────────┬───────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│     Persistence Layer     │ │   Observability & Drift   │
│  - SQLAlchemy Engine      │ │  - Prometheus /metrics    │
│  - PostgreSQL / SQLite WAL│ │  - Grafana Dashboards     │
│  - PredictionLog ORM table│ │  - PSI Drift Daemon       │
└───────────────────────────┘ └───────────────────────────┘
```

---

# 2. DATA — THE CORE ML INTERVIEW FOUNDATION

### 1. Data Source & Raw Structure
- **Dataset**: Kaggle IEEE-CIS Fraud Detection Benchmark (collected by Vesta Corporation).
- **Files**:
  - `train_transaction.csv`: 590,540 rows $\times$ 394 columns.
  - `train_identity.csv`: 144,233 rows $\times$ 41 columns.
- **Relational Key**: Joined on `TransactionID` using a **left join**.
- **Resulting Joined Shape**: **590,540 rows $\times$ 434 columns** (excluding redundant ID keys).
- **Identity Share**: Only **24.42%** of transactions have associated identity records. This matches real-world e-commerce, where card-present or basic web checkouts collect no device fingerprint data.
- **Target Variable**: `isFraud` (binary integer: `0` = legitimate, `1` = fraudulent).
- **Class Imbalance**: **3.50% fraud** (20,663 positive rows out of 590,540).

---

### 2. Deep Dive: Every Preprocessing Step (The 4 Questions)

#### Step 1: Memory Optimization & Data Loading
- **WHAT DID I DO?**:
  In `src/fraud/data/ingest.py`, implemented `downcast_df()`. Evaluated numerical minimums and maximums across all 434 columns. Downcast integers to `uint8` ($0 \le x < 255$), `uint16` ($x < 65,535$), or `uint32`; downcast float64 to `float32`; cast object strings to Pandas `category`. Wrote the output to `data/interim/train_merged.parquet` using PyArrow.
- **WHY DID I DO IT?**:
  Loading raw CSVs consumes > 2.5 GB of RAM. In a 64-bit Python environment, training tree models across 590k rows would cause memory thrashing or OOM errors. Downcasting reduced in-memory usage to **924.24 MB (63% reduction)** and shrank disk footprint from 800 MB to **90 MB (nearly 9x smaller)**.
- **WHAT ALTERNATIVES EXISTED?**:
  Loading raw CSVs with `low_memory=False`, or using Feather/HDF5 formats.
- **WHY DID I NOT USE THEM?**:
  Raw CSVs are slow to parse on every run; Feather files do not provide the snappy columnar compression ratio of Parquet.

#### Step 2: Temporal Train / Validation / Test Splitting
- **WHAT DID I DO?**:
  In `src/fraud/data/split.py`, sorted the dataset chronologically by `TransactionDT` (seconds offset from Day 1 to Day 182) and split it strictly by time without shuffling:
  - **Train (70%)**: Day 1 to Day 120 (413,378 rows, 3.52% fraud).
  - **Validation (15%)**: Day 120 to Day 152 (88,581 rows, 3.43% fraud).
  - **Held-Out Test (15%)**: Day 152 to Day 182 (88,581 rows, 3.48% fraud).
- **WHY DID I DO IT?**:
  Fraud detection is an adversarial time-series problem. Shuffling transactions leaks future fraud schemes, card testing attacks, and compromised merchant patterns into the training phase. A random split produces inflated, unrealistic test metrics that fail when deployed.
- **WHAT ALTERNATIVES EXISTED?**:
  Standard randomized Stratified K-Fold cross-validation or GroupKFold on card identifiers.
- **WHY DID I NOT USE THEM?**:
  Stratified K-Fold causes severe lookahead data leakage. GroupKFold on card IDs isolates cards but destroys the temporal drift dynamics necessary to evaluate real-world decay.

#### Step 3: Missing Value Handling
- **WHAT DID I DO?**:
  Retained missing values as native `np.nan` and allowed LightGBM to handle them natively during tree training. In `src/fraud/features/builder.py`, also created explicit missingness signals:
  - `id_missing_count`: Count of missing identity columns out of 38 `id_*` fields.
  - `has_identity`: Binary flag indicating if any identity data was present.
- **WHY DID I DO IT?**:
  In fraud detection, **missingness is informative**. A transaction missing device data or secondary address data is structurally different from one where full device data was captured. LightGBM handles NaNs natively by evaluating split gain when NaNs are routed to the left vs right branch, assigning them to whichever side minimizes training loss.
- **WHAT ALTERNATIVES EXISTED?**:
  Mean/median imputation (`SimpleImputer`), K-Nearest Neighbors imputation, or dropping columns with > 50% missingness.
- **WHY DID I NOT USE THEM?**:
  Synthetic imputation destroys the predictive signal carried by missingness, distorts feature distributions, and adds computational overhead to real-time serving. Dropping sparse columns (e.g. `id_07`, `id_08`) discarded valuable signals—when those rare fields are present, their fraud rate is significantly higher than the baseline 3.5%.

#### Step 4: Duplicate & Outlier Handling
- **WHAT DID I DO?**:
  - *Duplicates*: Verified that `TransactionID` values are strictly unique primary keys (zero duplicate transactions exist).
  - *Outliers*: In `src/fraud/features/builder.py`, applied $\ln(1 + \text{TransactionAmt})$ to create `log_TransactionAmt` and preserved the raw amount.
- **WHY DID I DO IT?**:
  Transaction amounts range from \$0.25 to \$31,937.39 with an extreme right-skew. While tree models split on rank order and are robust to monotonic scaling, log-transformation stabilizes leaf value predictions and aids baseline linear models. Outlier transactions were **not removed** because extreme dollar amounts represent genuine, high-value fraud targets.
- **WHAT ALTERNATIVES EXISTED?**:
  Trimming outliers via IQR or Z-score filtering; Winsorization (capping values at the 99th percentile).
- **WHY DID I NOT USE THEM?**:
  Dropping high-dollar transactions would blind the model to the exact high-value attacks that merchants care most about preventing.

#### Step 5: Feature Engineering
- **WHAT DID I DO?**:
  Constructed 23 domain-specific features in `FeatureBuilder._transform_raw()`:
  1. *Temporal*: `day_index` ($\lfloor DT / 86400 \rfloor$), `hour_of_day` ($\lfloor DT / 3600 \rfloor \pmod{24}$), `day_of_week`.
  2. *Financial*: `log_TransactionAmt`, `amt_cents` ($\text{Amt} \pmod{1.0}$), `amt_is_round` ($\text{cents} == 0$).
  3. *Email Domain Analysis*: Parsed `P_emaildomain` and `R_emaildomain` into provider and suffix (e.g. `gmail` and `com`), and engineered `email_domain_match` ($P == R$).
  4. *Interaction*: Combined card issuer and billing location into `card1_addr1`.
  5. *Identity Aggregation*: `id_missing_count` and `has_identity`.
  6. *Frequency Encodings*: Computed training frequencies for 12 features (`*_fq`).
- **WHY DID I DO IT?**:
  - Fraud rates rise during overnight hours (captured by `hour_of_day`).
  - Automated card-testing scripts often use round numbers or specific micro-cent values (captured by `amt_cents` and `amt_is_round`).
  - Mismatched purchaser and recipient email domains indicate unauthorized gift card or escrow transfers.
  - Rare card/address combinations correlate strongly with stolen card usage.
- **WHAT ALTERNATIVES EXISTED?**:
  Automated feature generation libraries (Featuretools, tsfresh).
- **WHY DID I NOT USE THEM?**:
  Automated tools generate thousands of noisy, uninterpretable features that increase training time, degrade latency, and risk overfitting.

#### Step 6: Categorical Encoding
- **WHAT DID I DO?**:
  Applied a two-pronged strategy:
  1. *High-Cardinality Features* (`card1`: 13,553 unique; `addr1`: 332 unique; `P_emaildomain`: 59 unique): Applied **Frequency Encoding** learned strictly on the training partition (`*_fq`). Unseen categories at inference time default to `0.0`.
  2. *Low-Cardinality Features* (`ProductCD`, `card4`, `card6`, `DeviceType`, `M1`–`M9`): Kept as Pandas `category` dtypes, allowing LightGBM to apply its native integer-bin Fisher partitioning.
- **WHY DID I DO IT?**:
  One-Hot Encoding would expand the feature space by > 14,000 sparse columns, creating memory bottlenecks and slowing tree splitting. Target encoding risks target leakage and overfitting on rare categories. Frequency encoding represents entity volume cleanly in a single numerical column with $O(1)$ lookup time at serving.
- **WHAT ALTERNATIVES EXISTED?**:
  One-Hot Encoding, Out-of-Fold Target Encoding, CatBoost Target Statistics.
- **WHY DID I NOT USE THEM?**:
  One-Hot blows up dimensionality; Target Encoding leaks target information unless complex out-of-fold cross-validation is used, and it behaves unpredictably when rare categories appear in production.

#### Step 7: Feature Scaling & Normalization
- **WHAT DID I DO?**:
  **No scaling or normalization was applied to tree features.** For the Logistic Regression baseline in `src/fraud/models/baseline.py`, applied `StandardScaler()` within a Scikit-Learn `Pipeline`.
- **WHY DID I DO IT?**:
  Decision trees select split points based purely on rank order. Any monotonic transformation (scaling, min-max normalization, standardization) produces the exact same split point and loss reduction. Avoiding unnecessary scaling saves compute time during online serving.
- **WHAT ALTERNATIVES EXISTED?**:
  Applying `StandardScaler` or `MinMaxScaler` across all features.
- **WHY DID I NOT USE THEM?**:
  It adds unnecessary computational overhead to real-time serving without providing any accuracy benefit for tree ensembles.

#### Step 8: Class Imbalance Handling
- **WHAT DID I DO?**:
  Did **not** apply synthetic resampling (SMOTE). Handled imbalance through:
  1. *Metric Selection*: Optimized for **PR-AUC (Average Precision)** and operational alert recall rather than accuracy.
  2. *Probability Calibration*: Applied Isotonic Regression on validation predictions to align model outputs with true empirical risk.
  3. *Cost-Sensitive Decision Boundaries*: Tuned operational thresholds ($t_{\text{review}} = 0.0595$, $t_{\text{block}} = 0.7500$) based on an asymmetric dollar loss matrix ($C_{\text{FP}} = \$10.00$, $C_{\text{FN}} = \text{Amount}$).
- **WHY DID I DO IT?**:
  SMOTE creates synthetic points in high-dimensional mixed tabular space with heavy missingness, generating unrealistic feature combinations. Resampling also distorts empirical probabilities, breaking downstream calibration. Maintaining natural base rates and optimizing decision thresholds produces a more robust, cost-effective system.
- **WHAT ALTERNATIVES EXISTED?**:
  SMOTE, ADASYN, Random Undersampling, `scale_pos_weight` in LightGBM.
- **WHY DID I NOT USE THEM?**:
  SMOTE distorts probability calibration; undersampling discards > 90% of legitimate transaction data; `scale_pos_weight` shifts probability outputs away from true empirical risk.

#### Step 9: Preventing Data Leakage
- **WHAT DID I DO?**:
  Enforced four structural safeguards:
  1. *Time Splitting*: Sorted on `TransactionDT` (Day 1–120 Train, Day 120–152 Val, Day 152–182 Test).
  2. *State Isolation*: `FeatureBuilder.fit()` ran **exclusively on the training partition**. Frequency maps, category levels, and missingness metrics were learned without observing validation or test data.
  3. *Unseen Category Handling*: Categories not seen during training default to `0.0` or `<NA>` without mutating stored state.
  4. *Automated Testing*: `tests/test_no_leakage.py` runs assertions on timestamp ordering and ID exclusivity.

#### Step 10: Final Data Given to the Model
- **Format**: Pandas DataFrame with **457 features**.
- **Data Types**: All continuous and frequency features are `float32`; categorical features are Pandas `category` with categories fixed from the training fit.
- **Missing Values**: Handled natively by LightGBM as `np.nan`.

---

# 3. MACHINE LEARNING MODEL

### 1. Algorithms & Model Selection
Three models were evaluated across the pipeline:
1. **Dummy Baseline** (`DummyClassifier(strategy="prior")`): Predicts class prior probability (3.5%). PR-AUC = **0.0343**, ROC-AUC = **0.5000**.
2. **Logistic Regression Baseline** (`LogisticRegression(class_weight="balanced")` with median imputation and scaling): PR-AUC = **0.3806**, ROC-AUC = **0.8362**.
3. **Default LightGBM on Raw Features**: PR-AUC = **0.5303**, ROC-AUC = **0.9014**.
4. **Final Calibrated LightGBM with Engineered Features**: PR-AUC = **0.5343**, ROC-AUC = **0.9103**, Expected Calibration Error (ECE) = **0.0018**, Brier Score = **0.0221** (on the strict held-out test set).

### 2. Why LightGBM? (And Why Not Others?)
- **vs XGBoost**: LightGBM uses histogram-based binning to discretize continuous values into 256 bins, training 3–4x faster than exact-greedy XGBoost. LightGBM also includes native categorical support and a faster compiled prediction loop in C++.
- **vs CatBoost**: CatBoost provides excellent categorical handling via target statistics, but its single-row inference latency is higher (~15–25 ms vs LightGBM's ~0.5 ms booster prediction / 4.82 ms HTTP p50 on 4 workers), making LightGBM a better fit for strict latency budgets.
- **vs Deep Learning (TabNet / MLP)**: Neural networks struggle on tabular data with 80%+ missingness and require extensive imputation, normalization, and GPU compute without outperforming tree ensembles.
- **vs Random Forest**: Random Forest cannot handle native NaNs without imputation, lacks gradient boosting accuracy, and produces large model artifacts that evaluate slowly.

### 3. Architecture & Internal Mechanics
LightGBM grows trees **leaf-wise (best-first)** rather than depth-wise (level-wise). At each step, it splits the leaf that produces the maximum reduction in loss, resulting in lower loss compared to depth-wise algorithms at the same leaf count.

To avoid overfitting, leaf growth is bounded:
- `num_leaves`: Maximum leaves per tree (tuned between 31 and 127).
- `min_child_samples`: Minimum observations required in a leaf (tuned between 20 and 200).
- `feature_fraction`: Subsamples 70%–95% of features per tree to decorrelate individual trees.
- `bagging_fraction`: Subsamples 70%–95% of rows per iteration.

### 4. Loss Function
- **Objective**: Binary Log-Loss (`binary` / Cross-Entropy Loss):
  $$\mathcal{L}(y, p) = - \frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$
- **Optimization Metric**: Average Precision / PR-AUC (`average_precision`).

### 5. Hyperparameter Tuning with Optuna
In `src/fraud/models/train_lgbm.py`, tuned 5 trials of Bayesian optimization using Optuna's Tree-structured Parzen Estimator (TPE), maximizing validation PR-AUC:
- `learning_rate`: Sampled float [0.05, 0.15] (Final best: ~0.08).
- `num_leaves`: Sampled int [31, 127] (Final best: ~63).
- `min_child_samples`: Sampled int [20, 200].
- `feature_fraction`: Sampled float [0.70, 0.95] (Final best: ~0.80).
- `bagging_fraction`: Sampled float [0.70, 0.95] with `bagging_freq=1`.
- `lambda_l1` / `lambda_l2`: Log-uniform regularization [1e-4, 5.0].
- **Early Stopping**: Set `stopping_rounds=30` monitoring validation PR-AUC to prevent overfitting.

### 6. Probability Calibration (Isotonic Regression)
- **Why**: GBDTs optimize ranking, producing probabilities that bunch near decision boundaries. If a model outputs 0.08, we need confidence that roughly 8% of transactions at that score are actually fraudulent.
- **Method**: In `src/fraud/models/calibrate.py`, fit `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")` on validation predictions.
- **Results**: Reduced Expected Calibration Error (ECE) from **0.0039 down to 0.0018** and achieved a Brier score of **0.0221** on the held-out test set, with zero degradation to ROC-AUC.

### 7. Cost-Optimal Decision Policy
In `src/fraud/models/threshold.py`, calculated dual thresholds:
- **`t_review = 0.0595`**: Grid-searched to minimize total financial loss ($\sum \text{FN} \times \text{Amt} + \sum \text{FP} \times \$10$). Scores between `0.0595` and `0.7500` route to human analysts.
- **`t_block = 0.7500`**: Lowest threshold where precision $\ge 90\%$ on validation data. Scores $\ge 0.7500$ are declined automatically.
- **`APPROVE`**: Scores $< 0.0595$ pass through without review friction.

### 8. Exact Held-Out Test Set Results (Days 152–182)
| Metric | Final Model | Baseline LightGBM | Logistic Regression | Dummy Prior |
|---|---|---|---|---|
| **ROC-AUC** | **0.9103** | 0.9014 | 0.8362 | 0.5000 |
| **PR-AUC (Average Precision)** | **0.5343** | 0.5303 | 0.3806 | 0.0343 |
| **Recall @ Precision = 50%** | **54.23%** | 51.55% | 33.83% | 0.03% |
| **Recall @ Precision = 80%** | **33.08%** | 33.00% | 16.96% | 0.00% |
| **Recall @ Top 1% Alerts** | **25.53%** | 25.38% | 20.61% | 1.12% |
| **Recall @ Top 5% Alerts** | **59.91%** | 58.78% | 46.15% | 6.15% |
| **Brier Score (Calibration)** | **0.0221** | 0.0219 | 0.1255 | 0.0332 |
| **Expected Calibration Error (ECE)** | **0.0018** | 0.0039 | 0.2691 | 0.0008 |
| **Score PSI (Distribution Drift)** | **0.0018** | -- | -- | -- |

### 9. Overfitting & Underfitting Handling
- *Underfitting*: Overcome by expanding raw features with 23 engineered features and moving from linear baselines (0.3806 PR-AUC) to gradient boosting (0.5343 PR-AUC).
- *Overfitting*: Controlled via early stopping (30 rounds on validation PR-AUC), row/column subsampling (`feature_fraction=0.8`, `bagging_fraction=0.8`), minimum leaf constraints (`min_child_samples`), and L1/L2 penalties.

---

# 4. COMPLETE CODE FLOW & EXECUTION TRACE

```text
1. DATA INGESTION:
   File: src/fraud/data/ingest.py -> ingest_data()
   - Loads train_transaction.csv & train_identity.csv
   - pd.merge(..., on="TransactionID", how="left")
   - downcast_df() converts types to uint8/uint16/float32/category
   - Saves data/interim/train_merged.parquet (924 MB RAM)

2. TEMPORAL DATA SPLITTING:
   File: src/fraud/data/split.py -> time_split_data()
   - Sorts chronologically on TransactionDT
   - Splits: Train (70%, Days 1-120), Val (15%, Days 120-152), Test (15%, Days 152-182)
   - Outputs split metadata to reports/split_info.json

3. FEATURE ENGINEERING & FITTING:
   File: src/fraud/features/builder.py -> FeatureBuilder.fit(train_df)
   - Computes frequencies for card1, addr1, card1_addr1 strictly on train_df
   - Freezes 457 canonical feature schema into models/feature_spec.json
   - Serializes state to models/feature_builder.joblib

4. MODEL TRAINING & TUNING:
   File: src/fraud/models/train_lgbm.py -> train_and_evaluate()
   - Transforms Train, Val, Test using FeatureBuilder.transform()
   - run_optuna_tuning() evaluates 5 Bayesian hyperparameter trials
   - train_lgbm() trains with early stopping (30 rounds on Val PR-AUC)
   - booster.save_model("models/model.txt")
   - Logs metrics, parameters, and model to MLflow (mlflow.db)

5. CALIBRATION & THRESHOLD OPTIMIZATION:
   File: scripts/evaluate_pipeline.py -> run_pipeline_evaluation()
   - ProbabilityCalibrator.fit(y_val, val_raw_probs) (Isotonic Regression)
   - Saves models/calibrator.joblib
   - optimize_thresholds() grid-searches financial loss matrix
   - Computes t_review = 0.0595, t_block = 0.7500 -> models/thresholds.json
   - Evaluates held-out Test set once -> reports/evaluation_report.md

6. ONLINE SERVING INITIALIZATION:
   File: src/fraud/serving/app.py -> lifespan()
   - Application startup loads FraudPredictor singleton into memory:
     loads feature_builder.joblib, model.txt, calibrator.joblib, thresholds.json

7. REAL-TIME REQUEST PROCESSING:
   File: src/fraud/serving/app.py -> score_transaction()
   - Header check: auth.py -> get_api_key() verifies X-API-Key
   - Abuse check: ratelimit.py -> check_rate_limit() enforces sliding window
   - Validation: schemas.py -> Transaction(**payload) validates fields
   - Scoring: predictor.py -> FraudPredictor.score_dict()
     * FeatureBuilder.transform() aligns to 457 columns in memory
     * booster.predict() outputs raw probability
     * ProbabilityCalibrator.calibrate() maps to calibrated score
     * make_decision() assigns "approve", "review", or "block"
     * explain_prediction() runs native TreeSHAP (pred_contrib=True)
   - Telemetry: metrics.py records request count, latency, score histograms
   - Audit: db.py -> log_prediction_to_db() inserts PredictionLog asynchronously
   - Response: returns ScoreResponse JSON (< 300 ms)
```

---

# 5. TECHNOLOGY STACK & TRADEOFF ANALYSIS

| Technology | Where Used | Why Selected | What Alternatives Exist | Why Alternatives Were Rejected |
|---|---|---|---|---|
| **Python 3.11** | Entire project | Native support for typing, high performance, and broad ML library ecosystem | Python 3.8/3.9, Go, Java | Python 3.11 provides significant interpreter speedups while maintaining compatibility with LightGBM and FastAPI |
| **LightGBM** | `src/fraud/models/` | Native NaN handling, fast histogram training, sub-5 ms HTTP inference without SHAP (4.82 ms p50 on 4 workers), built-in TreeSHAP (~22 ms compute) | XGBoost, CatBoost, PyTorch TabNet | XGBoost trains slower on large datasets; CatBoost inference latency is higher (~20 ms); TabNet struggles with heavy missingness and requires GPUs |
| **Scikit-Learn** | `src/fraud/models/`, `evaluation/` | Standard implementation of Isotonic Regression, Pipeline, SimpleImputer, and metrics | Custom NumPy scripts, Statsmodels | Scikit-Learn provides stable, battle-tested implementations of calibration and evaluation routines |
| **Optuna** | `src/fraud/models/train_lgbm.py` | Bayesian optimization via Tree-structured Parzen Estimator (TPE), finding optimal parameters faster than grid search | GridSearchCV, RandomSearch, Ray Tune | Grid/Random search is inefficient; Ray Tune adds unnecessary distributed complexity for single-node training |
| **MLflow** | `src/fraud/models/train_lgbm.py`, `scripts/` | Local, lightweight experiment tracking logging to SQLite (`mlflow.db`) without cloud dependencies | Weights & Biases, Neptune, Comet.ml | W&B and Neptune require cloud accounts, API keys, and external internet connectivity |
| **Pandas & PyArrow** | `src/fraud/data/`, `features/` | Flexible tabular manipulation; fast Snappy columnar compression for Parquet | Polars, DuckDB, CSV | Pandas integrates cleanly with Scikit-Learn; Parquet reduces disk size by 9x and speeds up loading |
| **FastAPI & Uvicorn** | `src/fraud/serving/app.py` | High-throughput asynchronous routing, Pydantic validation, auto-generated Swagger UI (`/docs`) | Flask, Django REST Framework, Triton | Flask is synchronous; Django is heavy; Triton is over-engineered for tabular tree models |
| **Pydantic (v2)** | `src/fraud/serving/schemas.py` | Fast Rust-based request validation and automatic OpenAPI schema generation | Marshmallow, manual dict checks | Pydantic v2 parses requests 5–10x faster than v1 and validates constraints like `TransactionAmt > 0` cleanly |
| **SQLAlchemy (v2.0)** | `src/fraud/serving/db.py` | Database-agnostic ORM supporting local SQLite (WAL mode) and PostgreSQL / Neon without code changes | Raw psycopg3, MongoDB | Raw SQL lacks ORM abstractions; MongoDB adds document DB complexity for structured audit logs |
| **Prometheus Client** | `src/fraud/serving/metrics.py` | Pull-based metrics collection (`/metrics`) tracking latency, request counts, and score distributions | Datadog, StatsD, New Relic | Open-source standard, zero runtime billing overhead, and zero performance impact on request handling |
| **React 19 + Vite + Tailwind** | `frontend/` | Fast, modern analyst dashboard with interactive Recharts SHAP graphs and preset loaders | Next.js, Vue, Angular, Streamlit | Compiles to static assets deployable on edge CDNs (Vercel); Streamlit is included separately for data science prototyping |
| **Docker** | `Dockerfile`, `docker-compose.yml` | Multi-stage build (`python:3.11-slim`) providing portable, reproducible container execution with non-root security | Bare-metal VMs, Conda packs | Multi-stage Docker strips build tools and libraries, producing lean, secure production images |

---

# 6. THE TWO MOST IMPORTANT INTERVIEW QUESTIONS

## Q1. "What exactly did you do in this project?"
> *"I built a real-time fraud scoring service from scratch using the Kaggle IEEE-CIS dataset of 590,000 transactions.
>
> On the data and ML side, I started by analyzing the dataset and finding severe class imbalance (3.5% fraud) and high missingness across identity and behavioral features. I implemented memory optimization downcasting that cut RAM usage by 63%, and structured a strict 70/15/15 chronological split by `TransactionDT` across the 182-day period to prevent future data leakage. I built a `FeatureBuilder` pipeline that engineered 23 domain features—including diurnal time indices, transaction cent-fractions, email domain matches, and frequency encodings on high-cardinality cards—while learning transformation parameters strictly on the training partition.
>
> I established baseline benchmarks with Dummy and Logistic Regression models, then trained a LightGBM model with Optuna hyperparameter tuning and early stopping. Because tree models output uncalibrated probabilities, I fit an Isotonic Regression calibrator on the validation set, cutting Expected Calibration Error to 0.0018. To make the model actionable for the business, I derived optimal decision thresholds—`t_review = 0.0595` and `t_block = 0.7500`—by minimizing an asymmetric financial loss matrix where false negatives cost the full transaction amount and false positives cost a \$10 manual review fee. On the held-out test set, the final model achieved 0.9103 ROC-AUC and 0.5343 PR-AUC.
>
> On the engineering side, I built an asynchronous FastAPI microservice with API key authentication, sliding-window rate limiting, Pydantic validation, and thread-safe SQLAlchemy audit logging. For explainability, I tapped into LightGBM's native C++ TreeSHAP implementation to return top-5 reason codes in ~22 ms compute (33.41 ms p50 HTTP latency with SHAP on 4 workers). I set up Prometheus monitoring for latency, score distributions, and Population Stability Index (PSI) drift tracking, packaged the service into a multi-stage Docker container, and built a React dashboard with Tailwind CSS to visualize transaction scores and SHAP contribution charts."*

---

## Q2. "Why did you do it this way?"
> *"Every technical choice was driven by the specific constraints of financial fraud:
>
> 1. **Why a time split?** Shuffled cross-validation leaks future fraud rings into the training set, producing artificially inflated metrics. Splitting chronologically (Days 1–120 Train, 120–152 Val, 152–182 Test) mimics real production conditions where the model must generalize to unseen future time periods.
> 2. **Why LightGBM?** Payment networks require sub-300 ms response times. LightGBM evaluates single-row decision trees in native C++ in ~0.5 ms (4.82 ms p50 HTTP latency without SHAP on 4 workers), handles sparse missing values natively without synthetic imputation, and exposes native TreeSHAP margins for real-time explainability.
> 3. **Why PR-AUC instead of ROC-AUC or Accuracy?** With a 3.5% fraud rate, accuracy is meaningless—a model predicting 'all legitimate' achieves 96.5% accuracy while preventing \$0 in fraud. ROC-AUC is distorted by large true negative counts, whereas PR-AUC evaluates precision against recall, making it sensitive to false alarms.
> 4. **Why Isotonic Calibration?** Tree boosting minimizes ranking loss, not calibration. Raw probabilities bunch near decision boundaries. Isotonic Regression fits a non-parametric monotonic step function that aligns probabilities with true empirical frequencies without degrading ROC-AUC ranking.
> 5. **Why an asymmetric cost matrix for thresholds?** A default 0.5 threshold is arbitrary and financially damaging. A missed \$1,000 fraud (False Negative) costs \$1,000, while flagging a legitimate user (False Positive) costs \$10 in analyst review time. Optimizing thresholds against this cost matrix yielded $t_{\text{review}} = 0.0595$, catching fraud early while keeping automated declines at $\ge 90\%$ precision ($t_{\text{block}} = 0.7500$).
> 6. **Why native TreeSHAP over KernelSHAP?** KernelSHAP takes 1 to 5 seconds per transaction because it repeatedly perturbs inputs. LightGBM's native C++ TreeSHAP evaluates tree paths in $O(\text{depth} \times \text{trees})$, computing exact Shapley contributions in ~22 ms compute (~33.41 ms p50 HTTP latency on 4 workers) within our API response budget.
> 7. **Why FastAPI with lifespan management?** Disk I/O during request processing is too slow for production serving. FastAPI's `@asynccontextmanager` loads all models into memory once at startup, keeping scoring requests stateless, thread-safe, and fast."*

---

# 7. RAPID-FIRE TECHNICAL INTERVIEW QUESTIONS (40 Questions & Answers)

### Project Basics & Problem Understanding
1. **What is the business objective of this project?**
   To score credit card transactions in real time (< 300 ms), outputting calibrated fraud probabilities, automated decisions (`APPROVE`, `REVIEW`, `BLOCK`), and regulatory reason codes to minimize financial fraud losses and manual review costs.
2. **What dataset did you use?**
   The IEEE-CIS Fraud Detection dataset (590,540 transaction rows merged with 144,233 identity rows across 182 days).
3. **What is the fraud rate in the dataset?**
   3.50% (20,663 positive fraud transactions out of 590,540).
4. **Why is accuracy a misleading metric here?**
   A trivial baseline predicting "all clean" achieves 96.5% accuracy while failing to detect a single fraudulent transaction.

### Data & Preprocessing
5. **How did you join the transaction and identity tables?**
   Using a left join on `TransactionID` in `src/fraud/data/ingest.py`. Only 24.42% of transactions matched an identity record.
6. **How did you optimize memory during ingestion?**
   Wrote `downcast_df()`, converting integer types to `uint8`/`uint16`/`uint32`, floats to `float32`, and strings to `category`, reducing RAM usage from 2.5 GB to 924.24 MB.
7. **Why did you use Parquet instead of CSV?**
   Parquet uses Snappy columnar compression, reducing file size from 800 MB to 90 MB and speeding up load times significantly.
8. **How did you split the dataset, and why?**
   Split strictly by time on `TransactionDT`: 70% Train (Days 1–120), 15% Validation (Days 120–152), and 15% Test (Days 152–182) to prevent temporal data leakage.
9. **How did you verify there was no data leakage?**
   In `tests/test_no_leakage.py`, asserted that $\max(\text{Train}_{DT}) < \min(\text{Val}_{DT}) < \min(\text{Test}_{DT})$ and that `TransactionID` sets have zero overlap across partitions.
10. **How did you handle missing values?**
    Allowed LightGBM to handle NaNs natively during split finding, and created explicit indicators: `id_missing_count` and `has_identity`.
11. **Did you drop columns with > 90% missingness?**
    Evaluated dropping them, but kept them (`drop_high_missing=False`) because rare fields like `id_07` carry strong fraud signals when present.

### Feature Engineering
12. **What is frequency encoding, and where did you use it?**
    Replaced high-cardinality categories with their occurrence count in the training set across 12 features (`card1`, `addr1`, `P_emaildomain`, etc.).
13. **What happens if a new category appears during live serving?**
    The frequency lookup defaults to `0.0`, and the categorical type defaults to `<NA>`, avoiding runtime KeyErrors.
14. **Why did you extract the cent part of transaction amounts?**
    Card testing bots and micro-fraud often use whole-dollar amounts ($100.00) or specific cent values, making `amt_cents` and `amt_is_round` informative features.
15. **What email features did you create?**
    Parsed domains into provider and suffix (`gmail` and `com`) and engineered `email_domain_match` ($P == R$).
16. **How does FeatureBuilder prevent training-serving skew?**
    Learns frequency maps and categorical levels strictly during `fit()` on the training set, saves them to disk, and applies the identical `transform()` logic during serving.

### Modeling & Algorithms
17. **What algorithm did you use for the final model?**
    LightGBM Gradient Boosted Decision Trees (`LGBMClassifier` / `lgb.Booster`).
18. **Why LightGBM instead of XGBoost?**
    LightGBM trains 3–4x faster using histogram binning, handles missing values natively, evaluates in ~0.5 ms in C++ (4.82 ms HTTP p50 on 4 workers), and has built-in TreeSHAP (~22 ms compute).
19. **What loss function did you optimize?**
    Binary Log-Loss (Binary Cross-Entropy), evaluating with Average Precision (PR-AUC).
20. **What hyperparameters did you tune with Optuna?**
    Tuned `learning_rate` (0.05–0.15), `num_leaves` (31–127), `min_child_samples` (20–200), `feature_fraction` (0.70–0.95), and L1/L2 penalties over 5 trials.
21. **How did you prevent overfitting during training?**
    Used early stopping (30 rounds on validation PR-AUC), row/column subsampling (`feature_fraction=0.8`, `bagging_fraction=0.8`), and minimum leaf constraints.

### Calibration & Thresholding
22. **What is probability calibration, and why was it necessary?**
    Adjusting model output scores to match empirical probabilities. GBDTs optimize ranking, not calibration, so raw probabilities can be distorted.
23. **What calibration method did you use?**
    Isotonic Regression on the validation set, fitting a non-parametric monotonic step function.
24. **What is Expected Calibration Error (ECE)?**
    The binned weighted absolute difference between predicted confidence and empirical accuracy. Calibration reduced ECE from 0.0039 to 0.0018.
25. **What is Brier Score?**
    The mean squared error between predicted probabilities and actual binary labels. The calibrated model achieved a Brier score of 0.0221.
26. **How did you derive the review threshold ($t_{\text{review}} = 0.0595$)?**
    Grid-searched thresholds to minimize the financial loss function: $\text{Loss} = \sum (\text{FN} \times \text{Amt}) + \sum (\text{FP} \times \$10)$.
27. **How did you derive the block threshold ($t_{\text{block}} = 0.7500$)?**
    Found the lowest threshold where precision $\ge 90\%$ on validation data, ensuring automated declines maintain a false alarm rate under 10%.

### Explainability & Systems
28. **How do you generate SHAP explanations in real time?**
    Using LightGBM's native C++ TreeSHAP via `booster.predict(..., pred_contrib=True)`, computing exact contributions in ~22 ms compute (~33.41 ms p50 HTTP latency on 4 workers).
29. **What is the additive property of SHAP?**
    The sum of all feature contributions plus the base value equals the model's raw output margin score: $\sum \phi_i(x) + \phi_0 = f(x)$.
30. **What framework serves the API?**
    FastAPI with Uvicorn, using asynchronous routing and Pydantic v2 schemas.
31. **How is the model loaded in FastAPI?**
    Using the `@asynccontextmanager` lifespan handler, loading `FraudPredictor` once into memory at application boot.
32. **How does rate limiting work?**
    `src/fraud/serving/ratelimit.py` implements a sliding-window token bucket limiter (5,000 requests/minute) with an in-memory fallback if Redis is unavailable.
33. **How are predictions logged?**
    `src/fraud/serving/db.py` uses SQLAlchemy to insert records into `prediction_logs` asynchronously, using SQLite WAL mode locally and PostgreSQL in production.
34. **What Prometheus metrics are exposed?**
    `fraud_requests_total`, `fraud_api_request_latency_seconds`, `fraud_prediction_score`, `fraud_decisions_total`, and `fraud_score_psi`.
35. **What is Population Stability Index (PSI)?**
    A statistical metric measuring distribution drift between reference and production data. Our score PSI on the test set is 0.0018, well below the 0.10 alert threshold.

### Testing & Deployment
36. **What tests did you write?**
    12 Pytest files covering temporal leakage (`test_no_leakage.py`), feature parity (`test_features.py`), API contracts (`test_api.py`), calibration, and metrics.
37. **What is training-serving parity, and how did you test it?**
    Ensuring identical predictions whether scored individually or in batches. Tested in `test_training_serving_parity` via `tests/test_api.py`.
38. **How is the application containerized?**
    Using a multi-stage `Dockerfile` (`python:3.11-slim` builder $\rightarrow$ runner), copying installed wheels and running as non-root user `appuser:1000`.
39. **What were your load test results?**
    In sustained load tests (4 workers, load generator on same machine), achieved 184.4 req/s at 5.23 ms p50 latency for single-request scoring without SHAP, 26.0 req/s at 36.43 ms p50 with SHAP, sustaining 329.3 req/s at concurrency 5 and 304.2 req/s at concurrency 10 (0.0% error rate). Concurrency 100 is characterized as a labeled stress point (118.8 req/s at 620.50 ms p50 without SHAP; 56.4 req/s at 1164.03 ms p50 with SHAP).
40. **How is CI/CD configured?**
    GitHub Actions (`.github/workflows/ci.yml`) runs Ruff linting and the Pytest test suite on every push and pull request.

---

# 8. REAL PROJECT CHALLENGES & ENGINEERING FIXES

### Challenge 1: Memory Thrashing During Data Ingestion
- **Problem**: Loading raw CSVs consumed > 2.5 GB of RAM, causing memory pressure during tree training.
- **Why It Happened**: Pandas defaults to 64-bit integer, float, and string object types on CSV reads.
- **How Detected**: Process memory usage inspections during initial profiling.
- **Solution**: Implemented `downcast_df()` in `src/fraud/data/ingest.py`, downcasting integers to `uint8`/`uint16`/`uint32` and floats to `float32`. Saved the merged dataset to Parquet.
- **Why It Worked**: Cut in-memory RAM usage from 2.5 GB to 924.24 MB (63% reduction) and reduced disk footprint to 90 MB.
- **Future Improvement**: Evaluate Polars lazy evaluation if dataset scales to tens of millions of rows.

### Challenge 2: Risk of Future Lookahead Leakage
- **Problem**: Standard shuffled cross-validation produced unrealistically high PR-AUC scores (> 0.75) that would collapse in production.
- **Why It Happened**: Random shuffling allows future fraud patterns, card testing rings, and compromised card numbers to leak into training folds.
- **How Detected**: Analyzing the temporal ordering of `TransactionDT` and observing metric inflation under random splits.
- **Solution**: Implemented a strict 70/15/15 chronological split by `TransactionDT` in `src/fraud/data/split.py` and built automated verification assertions in `tests/test_no_leakage.py`.
- **Why It Worked**: Provided an honest out-of-time evaluation (0.5343 PR-AUC on Days 152–182) that accurately reflects production performance.
- **Future Improvement**: Add expanding-window rolling backtests across multiple sequential months.

### Challenge 3: Training-Serving Feature Skew
- **Problem**: Computing frequency encodings or missingness statistics dynamically during online serving would either cause lookup failures on new categories or produce mismatched distributions.
- **Why It Happened**: Training pipelines operate on bulk DataFrames, while serving APIs process single JSON records.
- **How Detected**: Integration tests failing when single-row inference inputs encountered unseen categorical levels.
- **Solution**: Built `FeatureBuilder` in `src/fraud/features/builder.py`, separating `fit()` (run strictly on training data) from `transform()`. Stored frequency maps and category levels in `models/feature_builder.joblib`. Unseen categories default to `0.0`.
- **Why It Worked**: Guaranteed feature alignment across 456 columns, verified by `test_training_serving_parity` in `tests/test_api.py`.
- **Future Improvement**: Add additive Laplace smoothing to frequency encodings for rare tail categories.

### Challenge 4: Real-Time Explainability Latency
- **Problem**: Generating feature-level explanations using model-agnostic tools (KernelSHAP, LIME) took 1 to 5 seconds per transaction—far too slow for payment gateway timeouts (< 300 ms).
- **Why It Happened**: Perturbation-based explainers evaluate hundreds of synthetic samples per prediction.
- **How Detected**: Latency profiling showing response times exceeding 2,000 ms.
- **Solution**: Switched to LightGBM's native C++ TreeSHAP implementation via `booster.predict(..., pred_contrib=True)` in `src/fraud/explain/reasons.py`.
- **Why It Worked**: Computed exact Shapley contributions in ~22 ms (with num_threads=1), with sustained benchmark latency at 36.43 ms p50 at c=1, 87.64 ms p50 at c=10 on 4 workers (vs 5.23 ms and 22.23 ms without SHAP; load generator on same machine).
- **Future Improvement**: Pre-filter low-importance features to further optimize explanation computation.

### Challenge 5: Database Lock Contention Under Concurrent Load
- **Problem**: In concurrent load tests, multiple worker threads writing prediction logs to SQLite produced database lock errors (`OperationalError: database is locked`).
- **Why It Happened**: SQLite defaults to rollback journal mode, locking the entire file during write transactions.
- **How Detected**: Load tests at concurrency = 50 showing non-zero error rates.
- **Solution**: Configured SQLite Write-Ahead Logging (`PRAGMA journal_mode=WAL`), added a thread lock `_db_lock`, and wrapped logging in graceful degradation blocks in `src/fraud/serving/db.py`.
- **Why It Worked**: WAL mode allows concurrent readers while writes append to the write-ahead log, maintaining 0.0% error rate across concurrent load profiles.
- **Future Improvement**: Route write events through a distributed message broker (RabbitMQ/Kafka) in high-throughput deployments.

---

# 9. WHY NOT OTHER METHODS? (THE DECISION MATRIX)

| My Choice | Alternative | Why I Used My Choice | When Alternative Would Be Better |
|---|---|---|---|
| **LightGBM** | XGBoost | 3–4x faster histogram training; native NaN handling; compiled sub-millisecond C++ inference (~0.5 ms single-row predict; 5.23 ms HTTP p50); built-in TreeSHAP | When the deployment pipeline requires specialized GPU serving via NVIDIA Triton |
| **LightGBM** | CatBoost | Significantly lower inference latency on single rows (~0.5 ms vs 15–25 ms); lower memory usage | When dataset consists primarily of complex categorical interactions without dense numerical structures |
| **LightGBM** | Deep Learning (TabNet) | Better performance on tabular data with heavy missingness; no GPU requirement; faster training | When data includes unstructured modalities (free-form fraud notes, images, audio) |
| **Isotonic Regression** | Platt Scaling (Logistic) | Non-parametric; fits empirical probability distortions without assuming a rigid sigmoidal shape | When calibration dataset is small (< 1,000 samples) where isotonic regression risks overfitting |
| **Time-Series Split** | Stratified K-Fold | Eliminates future lookahead data leakage in time-series fraud data | In stationary domains without temporal ordering (e.g. medical image diagnosis) |
| **Frequency Encoding** | One-Hot Encoding | Keeps dimensionality compact; avoids adding 14,000 sparse columns; $O(1)$ lookup at serving | For low-cardinality features (< 10 categories) where explicit orthogonal vectors are useful |
| **Native TreeSHAP** | KernelSHAP / LIME | Evaluates in ~22 ms via compiled C++ tree paths (single-threaded OpenMP); satisfies real-time API latency budgets | For black-box models (neural networks, arbitrary ensembles) where tree structure is unavailable |
| **Asymmetric Cost Matrix** | Default 0.5 Threshold | Reflects real business losses ($C_{\text{FN}} = \text{Amount}$, $C_{\text{FP}} = \$10$); saves significant operational costs | When false positive and false negative errors carry equal operational consequences |
| **FastAPI** | Flask | Asynchronous concurrency; native Pydantic validation; auto-generated OpenAPI Swagger documentation | For simple legacy Python 2/3 maintenance projects |
| **SQLite WAL / Postgres** | MongoDB | Strict relational integrity for financial audit logs; structured time-range querying | For unstructured, polymorphic document storage |

---

# 10. LIMITATIONS & FUTURE IMPROVEMENTS

### Current Limitations
1. **Feature Transformation, Fast Path & TreeSHAP Latency:** Single-request scoring uses an optimized fast path (preallocated NumPy row buffers, cached booster feature names, single-threaded OpenMP `num_threads=1`, and direct margin sigmoid derivation). In sustained benchmarks (load generator running on the same machine), 4-worker HTTP latency is 4.82 ms p50 without SHAP (197.8 req/s) vs 33.41 ms p50 with SHAP (26.6 req/s) at concurrency 1. Single-row TreeSHAP compute alone is ~22 ms per transaction (evaluating 457 features across 500 decision trees), adding ~28 ms of median server latency at concurrency 1 and rising to 120.25 ms p50 at concurrency 10. Multi-record batch operations still involve Pandas DataFrame memory allocations.
2. **Non-Lua Redis Limiter:** The Redis sliding-window limiter executes check (`ZREMRANGEBYSCORE` + `ZCARD`) and insert (`ZADD` + `EXPIRE`) as separate calls rather than an atomic server-side Redis Lua script. Under high concurrency race conditions from the same key, requests could interleave between check and insert.
3. **Stationary Frequency Maps**: Frequency encodings are fixed to the training window. If a new card issuer launches and gains high volume, its frequency score remains 0.0 until the model is retrained.
4. **Simplified Cost Matrix**: Assumes a fixed \$10.00 review cost across all transactions, whereas actual review costs vary by merchant tier and investigation complexity.
5. **Absence of Graph Features**: The current pipeline evaluates individual transaction records without modeling cross-entity relationships (e.g. shared device IDs across different card numbers).
6. **Single-Node Rate Limiting**: The in-memory sliding window rate limiter runs per Uvicorn worker process unless connected to an external Redis instance.

### What I Would Improve With More Time
1. **Graph Neural Networks / Network Analysis**: Build a bipartite graph linking cards, IP addresses, and devices to detect organized fraud rings and card-testing botnets.
2. **Streaming Feature Store**: Integrate Feast or Redis to compute real-time sliding-window velocity aggregations (e.g. "number of transactions on this card in the last 10 minutes").
3. **Automated Continuous Retraining**: Deploy an automated pipeline that monitors Population Stability Index (PSI) and triggers retraining on an isolated Neon database branch when drift exceeds 0.20.
4. **Dynamic Cost Optimization**: Integrate merchant-specific fee structures and interchange tiers into the decision threshold optimization matrix.

---

# 11. 2-MINUTE PROJECT PITCH (Spoken Script)

> *"In payment processing, fraud detection involves a difficult tradeoff: missing a fraudulent transaction leads to expensive chargebacks, while falsely declining a good customer causes churn and costs review time.
>
> To solve this, I built an end-to-end Real-Time Fraud Scoring Service on 590,000 transactions from the Kaggle IEEE-CIS dataset. The service evaluates incoming card transactions and returns calibrated fraud probabilities, automated decisions (`APPROVE`, `MANUAL REVIEW`, `BLOCK`), and regulatory reason codes in under 300 ms.
>
> On the data side, only 3.5% of transactions were fraudulent, and identity features had up to 90% missing values. I implemented memory downcasting to reduce RAM usage by 63% and enforced a strict chronological 70/15/15 time split across the 182-day period to prevent future data leakage. I built a state-preserving `FeatureBuilder` that engineered 22 features—including diurnal time indices, cent-fractions, email domain matches, and frequency encodings on high-cardinality card numbers (removing the leaking day_index feature)—fitting all transformation state strictly on the training partition.
>
> For modeling, I started with Dummy and Logistic Regression baselines, then trained a LightGBM model with Optuna hyperparameter tuning and early stopping. I applied Isotonic Regression on validation predictions to calibrate probabilities, reducing Expected Calibration Error to 0.0018, and optimized business decision thresholds against an asymmetric cost matrix where false negatives cost the full amount and false positives cost \$10. On the held-out test set, the final model achieved 0.9103 ROC-AUC and 0.5289 PR-AUC (calibrated).
>
> On the engineering side, I served the model via an asynchronous FastAPI microservice with API key authentication, sliding-window rate limiting, and SQLAlchemy audit logging. To meet explainability requirements, I used LightGBM's native C++ TreeSHAP implementation to return top-5 reason codes in ~22 ms compute (33.41 ms p50 HTTP latency with SHAP on 4 workers). I added Prometheus monitoring for latency and Population Stability Index drift, containerized the system with a multi-stage Docker build, and built a React dashboard to visualize predictions and SHAP charts. In sustained load tests (4 workers, 20s runs, load generator on same machine), the service achieved: at c=1, 4.82 ms p50 without SHAP (197.8 req/s) vs 33.41 ms p50 with SHAP (26.6 req/s); at c=10, 22.81 ms p50 without SHAP (239.0 req/s) vs 120.25 ms p50 with SHAP (77.2 req/s); at c=100, 893.78 ms p50 without SHAP (66.1 req/s, 0.1% errors) vs 2432.72 ms p50 with SHAP (30.1 req/s, 0.1% errors). On 1 worker: c=1 is 7.55 ms without SHAP and 48.86 ms with SHAP."*

---

# 12. 30-SECOND ELEVATOR PITCH (Spoken Script)

> *"I built an end-to-end, production-grade fraud scoring service on 590,000 IEEE-CIS transactions that evaluates card payments in real time (4.82 ms p50 without SHAP, 33.41 ms with SHAP, 22.81 ms at c=10 on 4 workers).
>
> I used LightGBM with custom frequency and interaction features, enforced a strict chronological time split to prevent data leakage, and applied Isotonic Regression calibration alongside asymmetric cost optimization. The model achieves 0.9103 ROC-AUC and 0.5289 PR-AUC on held-out test data.
>
> The service is deployed with FastAPI, returns native TreeSHAP reason codes in ~22 ms compute (33.41 ms p50 HTTP latency on 4 workers), logs audit records asynchronously, and is fully containerized with Docker, Prometheus monitoring, and a React operations dashboard."*

---

# 13. DEEP TECHNICAL EXPLANATION

### 1. Mathematics of TreeSHAP Native Attribution
TreeSHAP decomposes the model's raw output margin $f(x)$ (in log-odds space) into additive feature attributions:
$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$
Where $\phi_0 = \mathbb{E}[f(x)]$ is the expected baseline value across the training dataset, and $\phi_j(x)$ is the Shapley contribution of feature $j$.

In `src/fraud/explain/reasons.py`, calling `booster.predict(feature_row, pred_contrib=True)` invokes LightGBM's compiled C++ implementation:
- It traverses decision paths across all boosting trees in $O(TLD^2)$ time (where $T$ is the number of trees, $L$ is max leaves, and $D$ is max depth).
- It returns an array of length $458$ (457 feature attributions + 1 base bias term).
- Features with positive attributions ($\phi_j > 0$) push the log-odds toward fraud (`increases_risk`), while negative attributions ($\phi_j < 0$) push toward legitimate (`lowers_risk`).
- The top 5 absolute values are sorted and mapped to human-readable labels from `FEATURE_LABEL_MAP`.

### 2. Mathematics of Isotonic Regression Calibration
Tree ensembles optimize binary cross-entropy or ranking loss, often yielding uncalibrated probabilities that cluster near decision boundaries.

Isotonic Regression fits a non-parametric, monotonic step function $m(f)$ that minimizes squared error:
$$\min_{m} \sum_{i=1}^{N} \left( y_i - m(f_i) \right)^2 \quad \text{subject to } m(f_i) \le m(f_j) \text{ whenever } f_i \le f_j$$
- Solved using the **Pool Adjacent Violators Algorithm (PAVA)** in $O(N)$ time.
- Because $m$ is monotonically non-decreasing, the ranking of instances is strictly preserved, leaving ROC-AUC unchanged while aligning probabilities with true empirical risk.

### 3. Asymmetric Business Cost Optimization
Given transaction amount $A_i$ and probability threshold $t$:
$$\text{Cost}(t) = \sum_{i \in \text{FN}(t)} A_i + \sum_{j \in \text{FP}(t)} \$10.00$$
Where:
- $\text{FN}(t) = \{i \mid y_i = 1 \land p_i < t\}$ (fraud transactions approved $\rightarrow$ merchant loses amount $A_i$).
- $\text{FP}(t) = \{j \mid j = 0 \land p_j \ge t\}$ (clean transactions flagged $\rightarrow$ merchant incurs \$10 review overhead).

Evaluating this across a 100-point grid on validation data identified the cost-minimizing review threshold at **$t_{\text{review}} = 0.0595$**.

### 4. Population Stability Index (PSI) Formula
Monitors distribution shift between reference validation scores and live production scores:
$$\text{PSI} = \sum_{b=1}^{B} \left( A_b - E_b \right) \times \ln\left(\frac{A_b}{E_b}\right)$$
Where $E_b$ is the proportion of validation records in quantile bin $b$, and $A_b$ is the proportion of live production records in bin $b$. In `src/fraud/evaluation/drift.py`, $B=10$ quantile bins are used. Our held-out test score PSI of **0.0018** confirms stable model output distributions over time.
