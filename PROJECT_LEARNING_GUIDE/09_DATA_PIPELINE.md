# 09 - End-to-End Data Pipeline

This document explains the complete data pipeline of the fraud scoring engine—from raw Kaggle CSV ingestion to final feature vectors passed to LightGBM. It details the rationale behind every preprocessing technique, evaluates alternatives, and explains how data leakage was avoided.

---

## 1. Complete Data Flow Diagram

```text
Raw Data (Kaggle IEEE-CIS)
├── train_transaction.csv (394 cols)
└── train_identity.csv (41 cols)
       │
       ▼ [src/fraud/data/download.py & ingest.py]
Data Loading & Memory Optimization
├── Ingestion via pd.read_csv()
├── Left-join on TransactionID (590,540 rows × 434 cols)
├── Type Downcasting (downcast_df: uint8, uint16, float32, category)
└── Save to data/interim/train_merged.parquet (924 MB RAM, 90 MB disk)
       │
       ▼ [scripts/run_eda.py & notebooks/01_eda.ipynb]
Exploratory Data Analysis (EDA)
├── Severe imbalance inspection (3.50% fraud)
├── Temporal structure analysis (TransactionDT spans Days 1–182)
├── Missingness auditing (up to 75–90%+ on V/M/identity blocks)
├── High-cardinality profiling (card1: 13,553 unique categories)
└── Distribution shift auditing (Month 1 vs Month 6)
       │
       ▼ [src/fraud/data/split.py]
Leakage-Safe Temporal Train / Validation / Test Split
├── Sort strictly by TransactionDT
├── Train Partition (70%): Days 1–120 (413,378 rows, 3.52% fraud)
├── Validation Partition (15%): Days 120–152 (88,581 rows, 3.43% fraud)
└── Held-Out Test Partition (15%): Days 152–182 (88,581 rows, 3.48% fraud)
       │
       ▼ [src/fraud/features/builder.py: FeatureBuilder.fit(train_df)]
State Fitting (TRAIN SET ONLY)
├── Learn frequency counts on 12 key features (card1, addr1, etc.)
├── Learn valid categorical category sets
└── Freeze 457 canonical feature schema (models/feature_spec.json)
       │
       ▼ [src/fraud/features/builder.py: FeatureBuilder.transform()]
Feature Engineering & Transformation
├── Temporal Features (day_index, hour_of_day, day_of_week)
├── Financial Features (log_TransactionAmt, amt_cents, amt_is_round)
├── Email Features (provider/suffix extraction, domain match flag)
├── Frequency Encodings (*_fq mapped from stored dictionaries)
├── Interaction Terms (card1_addr1 combined key)
├── Identity Missingness Flags (id_missing_count, has_identity)
└── Native Categorical Enforcement (pd.Categorical levels)
       │
       ▼
Final Model Input Matrix (X_train, X_val, X_test)
└── 457 columns, float32 and categorical dtypes → LightGBM Booster
```

---

## 2. Data Sources & Raw Format

- **Source**: Kaggle IEEE-CIS Fraud Detection Competition, provided by Vesta Corporation (a payment solutions provider).
- **Format**: Two relational CSV files joined by `TransactionID`:
  1. `train_transaction.csv` (394 columns): Contains core financial attributes, card parameters, recipient address indicators, email domains, and anonymized feature blocks (`C1`–`C14`, `D1`–`D15`, `M1`–`M9`, `V1`–`V339`).
  2. `train_identity.csv` (41 columns): Contains network and device attributes (`DeviceType`, `DeviceInfo`, `id_01`–`id_38`).
- **Join Cardinality**: Left join on `TransactionID`. Only 24.42% of transactions contain identity records, reflecting the real-world pattern where identity data is unavailable on many card-present or basic payment gateway flows.

---

## 3. Data Ingestion & Memory Optimization

### The Problem
Loading raw CSVs into Pandas creates `int64`, `float64`, and Python string `object` dtypes. On 590,540 rows, this consumes over 2.5 GB of RAM, causing memory pressure during multithreaded boosting or cross-validation.

### Implementation (`src/fraud/data/ingest.py`)
The `downcast_df(df)` function inspects column data types and applies lossless compression:
- **String/Object columns**: Converted to Pandas `category` dtype, replacing repeated string allocations with small integer pointers.
- **Float64 columns**: Downcast to `float32`.
- **Int64 columns**: Downcast based on observed value ranges:
  - If $0 \le x < 255$: `uint8` (1 byte instead of 8 bytes).
  - If $0 \le x < 65,535$: `uint16` (2 bytes instead of 8 bytes).
  - If $0 \le x < 4,294,967,295$: `uint32` (4 bytes instead of 8 bytes).

```python
# From src/fraud/data/ingest.py
if col_type == object or isinstance(col_type, pd.StringDtype):
    df[col] = df[col].astype("category")
elif np.issubdtype(col_type, np.floating):
    df[col] = df[col].astype(np.float32)
elif np.issubdtype(col_type, np.integer):
    c_min, c_max = df[col].min(), df[col].max()
    if c_min >= 0:
        if c_max < 255: df[col] = df[col].astype(np.uint8)
        elif c_max < 65535: df[col] = df[col].astype(np.uint16)
        ...
```

The cleaned DataFrame is saved to `data/interim/train_merged.parquet` using PyArrow with Snappy compression, reducing disk usage from ~800 MB to ~90 MB and in-memory footprint to 924.24 MB.

---

## 4. Exploratory Data Analysis (EDA) Findings

The EDA script in `scripts/run_eda.py` generated visualizations saved to `reports/figures/` and documented in `reports/eda_summary.md`:

1. **Severe Class Imbalance**: Overall fraud rate is **3.50%** (20,663 positive instances out of 590,540 rows). Metric choices must prioritize PR-AUC and recall at low alert rates over accuracy.
2. **Temporal Variation**: Fraud rates spike during low-volume nighttime hours and specific calendar weeks (`01_fraud_rate_time.png`). Random cross-validation would leak temporal signals.
3. **High Missingness**: `id_01`–`id_38` and `V*` / `M*` blocks exhibit missingness rates between 75% and 90%+ (`03_missingness_heatmap.png`).
4. **Transaction Amount Log-Skewness**: Amounts range from \$0.25 to \$31,937.39 with an extreme right-skew. Fraudulent transactions show distinct cent-fraction patterns (`amt % 1.0`), motivating the creation of `log1p(TransactionAmt)` and `amt_cents` features.
5. **High Cardinality**: Features like `card1` (13,553 unique categories) and `addr1` (332 unique categories) cannot be one-hot encoded without creating an impractically wide matrix.
6. **Feature Drift Over Time**: Count features (`C1`–`C14`) and accumulators (`D1`–`D15`) show significant mean shifts between Month 1 and Month 6 (`07_distribution_shift.png`), requiring temporal stability checks.

---

## 5. Cleaning & Preprocessing Steps

| Step | Technique Applied | Code Location | Why Applied | Alternatives Considered & Why Rejected |
|---|---|---|---|---|
| **Duplicates** | Verification on `TransactionID` | `src/fraud/data/ingest.py` | Each row represents a unique transaction; duplicate IDs would cause data corruption. | Dropping random rows: unnecessary since IDs are strictly unique. |
| **Missing Values** | **Native Tree Routing** | `src/fraud/models/train_lgbm.py` | LightGBM routes missing values to the branch that minimizes loss at each split, preserving missingness as a risk signal. | Mean/Median Imputation: distorts natural missingness signals and increases serving compute; MICE/KNN: too slow for large tabular data. |
| **Outliers** | Log1p transformation & clipping | `src/fraud/features/builder.py` | `log1p(TransactionAmt)` stabilizes variance and compresses extreme dollar amounts. | Truncating/dropping high amounts: rejected because large transactions are often genuine high-value fraud targets. |
| **Feature Scaling** | **None for LightGBM** (StandardScaler used only for Logistic Regression baseline) | `src/fraud/models/baseline.py` | Decision tree split points depend solely on rank order, making tree boosting invariant to monotonic feature scaling. | MinMax / RobustScaler: adds unnecessary computation without improving tree performance. |

---

## 6. Feature Engineering & Transformations

`FeatureBuilder` in `src/fraud/features/builder.py` derives the following features:

### 1. Base Time Features
- `day_index = TransactionDT // 86400`: Calendar day index from Day 1 to Day 182.
- `hour_of_day = (TransactionDT // 3600) % 24`: Diurnal hour (fraud rates rise during overnight hours).
- `day_of_week = (TransactionDT // 86400) % 7`: Day-of-week indicator.

### 2. Transaction Amount Features
- `log_TransactionAmt = np.log1p(np.maximum(amt, 0))`: Log-transformed transaction amount.
- `amt_cents = amt % 1.0`: Fractional cent portion (fraudulent card testing often uses round amounts like \$100.00 or micro-amounts).
- `amt_is_round = ((amt % 1.0) == 0)`: Boolean flag for round-dollar transactions.

### 3. Email Domain Features
- `P_emaildomain_provider` & `P_emaildomain_suffix`: Splits email domains (e.g. `gmail.com` $\rightarrow$ `gmail`, `com`).
- `R_emaildomain_provider` & `R_emaildomain_suffix`: Splits recipient email domains.
- `email_domain_match = (P_emaildomain == R_emaildomain)`: Flag indicating matching purchaser and recipient domains.

### 4. High-Cardinality Frequency Encodings
Computed for 12 features: `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, and the interaction `card1_addr1`.
- Replaces each category with its count in the training set:
  $$\text{freq}(x) = \text{Count}_{\text{train}}(x)$$
- Gives the model a measure of card and merchant popularity. Unseen categories at inference time default to `0.0`.

### 5. Identity Missingness Features
- `id_missing_count`: Total number of missing fields among identity columns `id_01` through `id_38`.
- `has_identity`: Boolean flag indicating whether any identity records were provided.

---

## 7. Preventing Data Leakage

Data leakage was systematically addressed across four dimensions:

1. **Chronological Splitting**:
   Train, Validation, and Test sets were split along time boundaries via `src/fraud/data/split.py`. No future transactions appear in the training partition.
2. **Feature Builder Fit Scope**:
   `FeatureBuilder.fit()` was executed **exclusively on the training partition**. Frequency maps, category lists, and missingness statistics were learned without observing validation or test data.
3. **Serving-Time Fallbacks**:
   Categories encountered at serving time that did not appear during training are mapped to `0.0` (frequency) or `<NA>` (category) rather than raising errors or mutating the stored state.
4. **Automated Leakage Verification**:
   The test suite in `tests/test_no_leakage.py` runs assertions on timestamp ordering, ID exclusivity, and partition independence.

---

## 8. Class Imbalance Handling Strategy

Rather than using synthetic resampling techniques, this project addressed the 3.5% class imbalance through metric selection and cost-sensitive thresholding:

- **Why Synthetic Oversampling (SMOTE / ADASYN) Was Not Used**:
  1. Interpolating synthetic points in 457-dimensional mixed categorical/numerical space with high missingness creates unrealistic combinations.
  2. Resampling alters the empirical risk distribution, breaking downstream probability calibration.
  3. Oversampling 400k rows increases training memory and compute overhead significantly.
- **The Chosen Approach**:
  1. **Metric Focus**: Optimizing for PR-AUC (Average Precision) and recall at low alert rates rather than accuracy.
  2. **Probability Calibration**: Applying Isotonic Regression on the validation set to align predicted scores with true empirical frequencies.
  3. **Cost-Sensitive Decision Boundaries**: Deriving operational thresholds ($t_{\text{review}} = 0.0595$, $t_{\text{block}} = 0.7500$) based on an asymmetric dollar loss matrix ($C_{\text{FP}} = \$10.00$, $C_{\text{FN}} = \text{Amount}$).

---

## 9. Final Model Input Format

The output of `FeatureBuilder.transform()` is a Pandas DataFrame with:
- **Rows**: 1 (during real-time serving), up to 1,000 (batch endpoint), or 413,378 (training partition).
- **Columns**: Exactly **457 features**, ordered and typed to match `models/feature_spec.json`.
- **Data Types**: Numeric features stored as `float32`; categorical features stored as Pandas `category` with categories fixed from the training fit.
- **Missing Values**: Handled natively by LightGBM as `np.nan` without requiring synthetic imputation.
