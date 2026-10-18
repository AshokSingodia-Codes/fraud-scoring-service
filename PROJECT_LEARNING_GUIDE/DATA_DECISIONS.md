# Data Decisions: Rationale, Alternatives & Future Improvements

This document records the architectural and statistical decisions made throughout the data pipeline, detailing why each approach was chosen, which alternatives were considered and rejected, and potential areas for future enhancement.

---

## Decision 1: Parquet Columnar Storage & NumPy Type Downcasting

### Context & Problem
Loading raw CSV files (`train_transaction.csv` and `train_identity.csv`) into Pandas creates 64-bit numerical types (`int64`, `float64`) and Python string object references. This consumes over 2.5 GB of RAM, causing memory pressure and slow disk I/O.

### Decision Made
Implement `downcast_df()` in `src/fraud/data/ingest.py` to downcast integers to `uint8`/`uint16`/`uint32`, floats to `float32`, and strings to Pandas `category`. Serialize the joined dataset to `data/interim/train_merged.parquet` using PyArrow with Snappy compression.

### Why This Was Chosen
- Reduces in-memory RAM usage from 2.5 GB to 924.24 MB (a 63% reduction), enabling fast multi-threaded tree training.
- Reduces disk storage from ~800 MB to ~90 MB (nearly 9x smaller), speeding up cold-start loading.

### Alternatives Considered & Rejected
- *Raw CSV reading with `low_memory=False`*: High memory overhead; re-parsing types on every script run is slow.
- *Feather / HDF5*: Faster write speeds, but higher disk footprint than Snappy-compressed Parquet.

### Possible Improvements
- Evaluate streaming lazy evaluations using Polars or DuckDB if the dataset grows beyond 20 million rows.

---

## Decision 2: Chronological Time-Based 70/15/15 Partitioning

### Context & Problem
Standard machine learning pipelines often apply randomized K-Fold or Stratified K-Fold cross-validation. In temporal domains like transaction fraud, random splitting allows the model to train on future transactions to predict the past.

### Decision Made
Implement strict chronological splitting in `src/fraud/data/split.py` based on `TransactionDT`:
- **Train (70%)**: Day 1 to Day 120 (413,378 rows, 3.52% fraud).
- **Validation (15%)**: Day 120 to Day 152 (88,581 rows, 3.43% fraud).
- **Held-Out Test (15%)**: Day 152 to Day 182 (88,581 rows, 3.48% fraud).

### Why This Was Chosen
- Accurately mirrors real-world deployment: models train on historical data and predict future transactions.
- Eliminates lookahead data leakage.

### Alternatives Considered & Rejected
- *Stratified Random Split*: Inflates evaluation metrics artificially because future card testing rings and compromised merchant IDs leak into the training set.
- *Purged / Embargoed GroupKFold on Card IDs*: Grouping by card number prevents identity leakage but discards the natural temporal drift required to evaluate out-of-time stability.

### Possible Improvements
- Implement rolling-window backtesting across monthly retraining cycles to evaluate model degradation dynamics over time.

---

## Decision 3: Frequency Encoding for High-Cardinality Categoricals

### Context & Problem
Categorical variables such as `card1` (13,553 unique categories), `addr1` (332 unique categories), and `P_emaildomain` (59 categories) have high cardinality.

### Decision Made
Implement training-set frequency encoding in `src/fraud/features/builder.py` across 12 key features, generating `*_fq` columns. Frequency maps are learned strictly during training; unseen categories at serving time default to `0.0`.

### Why This Was Chosen
- Projects high-cardinality categories into a single informative numerical column representing entity volume/popularity.
- Avoids dimensional explosion and sparsity issues associated with One-Hot Encoding.
- Stable at inference time: simple dictionary lookup in $O(1)$ time.

### Alternatives Considered & Rejected
- *One-Hot Encoding*: Would expand the feature space by over 14,000 sparse columns, creating memory issues and slowing inference.
- *Target Encoding (Mean Out-of-Fold)*: High risk of target leakage and overfitting on rare categories, especially in the presence of concept drift.

### Possible Improvements
- Add empirical Bayes smoothing to frequency encodings to better handle low-sample tail categories.

---

## Decision 4: Native Missing Value Routing Instead of Synthetic Imputation

### Context & Problem
Multiple feature blocks (including identity metrics `id_01`–`id_38`, `dist2`, and `V*` features) have missingness rates exceeding 75% to 90%.

### Decision Made
Rely on LightGBM's native missing value handling rather than applying synthetic imputation (mean, median, or KNN).

### Why This Was Chosen
- **Informative Missingness**: In transaction fraud, missing fields are often a risk indicator (e.g. transactions without identity fields behave differently from verified desktop checkouts). Imputing synthetic values removes this signal.
- **Serving Performance**: LightGBM branches missing values directly in C++ during tree evaluation, eliminating the need to maintain imputation pipelines at serving time.

### Alternatives Considered & Rejected
- *SimpleImputer (Mean / Median)*: Artificially concentrates variance and distorts empirical distributions.
- *Iterative Imputer (MICE) / KNN*: Impractically slow for 590,000 rows $\times$ 434 features.

### Possible Improvements
- Add explicit binary missingness indicator flags for key numerical blocks (`V*`, `D*`) to test whether additional split gain can be achieved.

---

## Decision 5: Retaining High-Missingness Feature Blocks

### Context & Problem
Several features (`dist2`, `D7`, `id_07`, `id_08`, `id_18`, `id_21`–`id_27`) have missingness rates exceeding 90%. Conventional guidelines often recommend dropping columns with > 90% missingness.

### Decision Made
Evaluate column dropping via `drop_high_missing=False` in `FeatureBuilder`, keeping these features in the model.

### Why This Was Chosen
- In fraud detection, rare events and rare fields often carry strong risk signals. For example, `id_07` and `id_08` are only populated when specific browser plugins or mobile security checks run. When present, their fraud rate is significantly higher than the baseline 3.5%.
- LightGBM's histogram binning handles sparse columns without performance degradation.

### Alternatives Considered & Rejected
- *Dropping all columns with > 90% missingness*: Ablation experiments showed that dropping these 12 columns caused a minor drop in PR-AUC on the validation set.

### Possible Improvements
- Group sparse identity metrics into a single composite risk indicator to simplify feature count.

---

## Decision 6: Addressing Class Imbalance via Metric Choice and Cost Matrix

### Context & Problem
With only 3.50% positive instances, naive optimization leads to models biased toward predicting legitimate transactions.

### Decision Made
Address class imbalance through evaluation metrics and decision thresholding rather than synthetic resampling (such as SMOTE):
1. **Metric Focus**: Optimize for PR-AUC (Average Precision) and recall at low alert rates.
2. **Probability Calibration**: Use Isotonic Regression on validation predictions.
3. **Cost-Sensitive Thresholds**: Derive $t_{\text{review}} = 0.0595$ by minimizing an asymmetric financial loss matrix ($C_{\text{FP}} = \$10.00$, $C_{\text{FN}} = \text{Amount}$).

### Why This Was Chosen
- Synthetic oversampling (SMOTE) distorts empirical probability distributions, undermining downstream probability calibration.
- Operating on true calibrated probabilities allows business teams to adjust thresholds dynamically as risk policies change, without retraining the model.

### Alternatives Considered & Rejected
- *SMOTE / Random Oversampling*: Generates unrealistic synthetic records in 457-dimensional mixed tabular space and triples training compute.
- *scale_pos_weight in LightGBM*: Artificially scales positive class gradients, shifting output scores away from true empirical probabilities.

### Possible Improvements
- Incorporate merchant-specific fee structures and dynamic chargeback penalties into the cost optimization matrix.

---

## Decision 7: Isotonic Probability Calibration

### Context & Problem
Gradient boosted decision trees minimize ranking loss, which often yields uncalibrated probabilities that cluster near decision boundaries.

### Decision Made
Fit Scikit-learn's `IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")` on validation prediction probabilities.

### Why This Was Chosen
- **Non-Parametric**: Fits empirical probability distortions without assuming a rigid sigmoidal shape.
- **Rank Preserving**: The monotonic step function preserves ROC-AUC while reducing Expected Calibration Error (ECE) from 0.0039 to 0.0018.

### Alternatives Considered & Rejected
- *Platt Scaling (Logistic)*: Assumes a symmetric sigmoidal distortion, which does not match tree boosting outputs on skewed tabular data.
- *Uncalibrated Raw Scores*: Scores cannot be interpreted as true empirical probabilities in dollar loss calculations.

### Possible Improvements
- Re-fit the calibrator periodically on rolling windows to account for temporal drift in predicted distributions.

---

## Decision 8: Native C++ TreeSHAP for Real-Time Explainability

### Context & Problem
Adverse action compliance and fraud analyst workflows require per-prediction explanations for flagged transactions.

### Decision Made
Use LightGBM's native C++ TreeSHAP implementation via `booster.predict(..., pred_contrib=True)` in `src/fraud/explain/reasons.py`.

### Why This Was Chosen
- Computes exact feature-level log-odds contributions during tree traversal in ~22 ms compute (~33.41 ms p50 HTTP latency at c=1 on 4 workers), meeting real-time API latency budgets.
- Eliminates the need to install the heavy Python `shap` package in production containers.

### Alternatives Considered & Rejected
- *KernelSHAP / LIME*: Model-agnostic perturbation methods take 1 to 5 seconds per transaction, making them unusable in synchronous payment flows.
- *Global Feature Importance Alone*: Provides overall model insight but cannot explain individual scoring decisions to fraud analysts.

### Possible Improvements
- Implement batch caching for baseline values and pre-filter low-importance features to further optimize latency.
