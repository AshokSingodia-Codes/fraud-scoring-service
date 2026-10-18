# 06 - Deep Dive: Advanced Technical Concepts

This document provides in-depth technical explanations of the core machine learning and systems concepts implemented in this codebase, structured progressively from beginner intuition to mathematical formulations and code examples.

---

## 1. Temporal Data Splitting & Leakage Prevention

### Beginner Intuition
In standard introductory machine learning tasks, datasets are often shuffled randomly before being split into train and test sets (e.g. using `train_test_split(shuffle=True)`). 

However, in financial fraud detection, **time order matters**. Shuffling transactions across time means the model trains on transactions from May to predict transactions from March. In reality, fraudsters continuously adapt their techniques—modifying bot behavior, testing stolen cards in batches, or shifting transaction amounts. Shuffling lets the model "peek into the future," creating an unrealistic evaluation setup that will fail once deployed to production.

### Intermediate Mechanics
The IEEE-CIS dataset includes a `TransactionDT` column representing elapsed time in seconds from an arbitrary start point.
- Day 1 begins at `TransactionDT = 86400`.
- Day 182 ends at `TransactionDT = 15811131`.

To evaluate real-world performance without lookahead bias:
1. The dataset is sorted chronologically by `TransactionDT`.
2. The split boundaries are strictly sequential:
   - **Train Set (70%)**: Day 1 through Day 120 (413,378 rows).
   - **Validation Set (15%)**: Day 120 through Day 152 (88,581 rows).
   - **Test Set (15%)**: Day 152 through Day 182 (88,581 rows).

```text
Time Horizon (Day 1 ────────────────────────────── Day 182)
[====== TRAIN (Days 1–120) ======][=== VAL (120–152) ===][=== TEST (152–182) ===]
       Learn patterns               Tune parameters        Evaluate once
```

### Advanced Formulation & Project Implementation
In `src/fraud/data/split.py`:
```python
if df["TransactionDT"].is_monotonic_increasing:
    df_sorted = df
else:
    df_sorted = df.sort_values("TransactionDT", ignore_index=True)

n = len(df_sorted)
train_end = int(n * train_ratio)
val_end = int(n * (train_ratio + val_ratio))

train_df = df_sorted.iloc[:train_end]
val_df = df_sorted.iloc[train_end:val_end]
test_df = df_sorted.iloc[val_end:]
```

The test suite in `tests/test_no_leakage.py` strictly verifies:
$$\max(\text{Train}_{DT}) < \min(\text{Val}_{DT}) \quad \text{and} \quad \max(\text{Val}_{DT}) < \min(\text{Test}_{DT})$$
$$\text{Train}_{IDs} \cap \text{Val}_{IDs} = \emptyset \quad \text{and} \quad \text{Val}_{IDs} \cap \text{Test}_{IDs} = \emptyset$$

---

## 2. State-Preserving Feature Pipelines (Preventing Training-Serving Skew)

### Beginner Intuition
When a model is trained on a laptop or server, features are computed across hundreds of thousands of historical rows at once. But when deployed as an API, transactions arrive **one at a time**. 

If feature calculations depend on batch statistics that aren't stored—or if missing values are filled using numbers calculated on the test set—the live API won't match the training environment. This discrepancy is called **training-serving skew**.

### Intermediate Mechanics
To prevent skew:
1. Any transformation that learns state (such as categorical frequencies, mean values, or valid category levels) must be fit **exclusively on the training partition**.
2. That learned state must be serialized (saved to disk) and loaded directly by the live inference engine.
3. During online inference, the single incoming transaction must pass through the identical transformation logic, looking up pre-computed values from the stored state.

### Advanced Formulation & Project Implementation
The `FeatureBuilder` in `src/fraud/features/builder.py` encapsulates this pattern:

```python
class FeatureBuilder:
    def fit(self, train_df: pd.DataFrame) -> "FeatureBuilder":
        # 1. Learn frequency encodings strictly on train_df
        for col in FREQ_COLS:
            self.freq_maps[col] = train_df[col].value_counts(dropna=True).to_dict()

        # 2. Capture categorical category sets strictly from train_df
        sample_transformed = self._transform_raw(train_df)
        for col in cat_cols:
            cats = sorted([str(x) for x in sample_transformed[col].dropna().unique()])
            self.cat_categories[col] = cats

        # 3. Freeze schema ordering (457 features)
        self.final_feature_names = sample_final.columns.tolist()
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_raw(df)
        
        # Fast align: inject any missing columns as NaN
        missing_cols = [c for c in self.final_feature_names if c not in transformed.columns]
        if missing_cols:
            missing_df = pd.DataFrame(np.nan, index=transformed.index, columns=missing_cols)
            transformed = pd.concat([transformed, missing_df], axis=1)

        # Enforce exact categorical levels learned during training
        transformed = self._apply_categoricals(transformed)
        return transformed[self.final_feature_names]
```

At inference time, `FraudPredictor` loads the fitted `models/feature_builder.joblib`. If a transaction contains an unseen category (e.g. a brand-new card issuer not present during training), the lookup falls back to `0.0` rather than raising a KeyError.

---

## 3. Cost-Sensitive Decision Thresholds & Business Loss Matrices

### Beginner Intuition
By default, most binary classification systems use a probability threshold of $0.5$ (if score $\ge 0.5$, flag as fraud; otherwise approve). 

In fraud detection, a $0.5$ threshold is rarely practical:
- A missed $1,500 fraud transaction (**False Negative**) results in a direct financial loss of $1,500 plus chargeback fees.
- Flagging a legitimate transaction for review (**False Positive**) costs approximately $10.00 for an analyst to inspect it.

Because the penalties are **asymmetric**, the decision threshold must be tuned to minimize total business loss rather than maximizing generic accuracy.

### Intermediate Mechanics
Rather than relying on a single threshold, this project implements a **dual-threshold three-tier decision policy**:

```text
0.0000 ───────────── [ t_review = 0.0595 ] ───────────── [ t_block = 0.7500 ] ───────────── 1.0000
       APPROVE                           MANUAL REVIEW                           BLOCK
  (Frictionless User)                 (Analyst Work Queue)               (Automated Decline)
```

1. **`t_review` (Cost-Optimal Boundary)**: Transactions scoring above this threshold are flagged for manual review to catch fraud before settlement.
2. **`t_block` (High-Precision Boundary)**: Transactions scoring above this threshold are blocked immediately, configured to achieve $\ge 90\%$ precision to avoid declining good customers without review.

### Advanced Mathematical Formulation
In `src/fraud/models/threshold.py`, $t_{\text{review}}$ is found via grid search on the validation set, minimizing the total loss function:

$$\mathcal{L}(t) = \sum_{i \in \text{FN}(t)} \text{TransactionAmt}_i + \sum_{j \in \text{FP}(t)} C_{\text{review}}$$

Where:
- $\text{FN}(t) = \{i \mid y_i = 1 \land p_i < t\}$
- $\text{FP}(t) = \{j \mid j = 0 \land p_j \ge t\}$
- $C_{\text{review}} = \$10.00$

```python
grid_thresholds = np.linspace(0.01, 0.99, 100)
best_cost = float("inf")
best_t_review = 0.5

for t in grid_thresholds:
    flagged = val_probs >= t
    fn = (~flagged) & (y_val == 1)
    fp = flagged & (y_val == 0)

    fn_cost = np.sum(val_amounts[fn])
    fp_cost = np.sum(fp) * fp_review_cost
    total_cost = fn_cost + fp_cost

    if total_cost < best_cost:
        best_cost = total_cost
        best_t_review = float(t)
```

Applying this to our validation set yields an optimal $t_{\text{review}} = 0.0595$.

---

## 4. Probability Calibration (Isotonic Regression, Brier Score, and ECE)

### Beginner Intuition
When a model outputs a score of `0.08`, what does that number actually mean? In an uncalibrated model, it is simply an arbitrary ranking score. 

In a **calibrated** model, that number represents a true probability: out of 100 transactions given a score of `0.08`, approximately 8 are actually fraudulent. This calibration is critical when calculating dollar losses or integrating with downstream risk systems.

### Intermediate Mechanics
Gradient boosted decision trees often produce distorted, uncalibrated probabilities because their objective functions (such as binary log-loss) focus on ranking instances correctly rather than matching true empirical frequencies.

To correct for this, an **Isotonic Regression Calibrator** is fit on the held-out validation predictions:
- It fits a non-parametric, monotonically increasing step function.
- It maps raw booster probabilities to empirical risk probabilities.
- Because the mapping is monotonic, it preserves the model's ROC-AUC ranking while improving probability alignment.

### Advanced Metrics: Brier Score and Expected Calibration Error (ECE)

#### 1. Brier Score
Measures the mean squared error between predicted probabilities $p_i$ and true binary outcomes $y_i \in \{0, 1\}$:
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^{N} (p_i - y_i)^2$$
A lower Brier score indicates better calibrated, more accurate probabilities.

#### 2. Expected Calibration Error (ECE)
Evaluates calibration by partitioning predictions into $M$ equal-width probability bins $B_1, B_2, \dots, B_M$ (typically $M=10$):
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
Where $\text{acc}(B_m)$ is the fraction of positive labels in bin $m$, and $\text{conf}(B_m)$ is the mean predicted probability in bin $m$.

In `src/fraud/evaluation/metrics.py`:
```python
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            avg_prob = np.mean(y_prob[in_bin])
            accuracy = np.mean(y_true[in_bin])
            ece += np.abs(avg_prob - accuracy) * prop_in_bin
    return float(ece)
```

On our held-out test set, Isotonic Calibration improved Expected Calibration Error from **0.0039 down to 0.0018**.

---

## 5. Real-Time Explainability with TreeSHAP

### Beginner Intuition
If a bank declines a customer's purchase or holds a transaction for review, the customer or fraud investigator needs to know **why**. Showing a raw risk score of `0.85` does not explain what triggered the flag. 

SHAP (SHapley Additive exPlanations) explains the prediction by breaking down how much each individual feature pushed the score higher or lower relative to the baseline average.

### Intermediate Mechanics
Model-agnostic explainability tools (like KernelSHAP or LIME) work by repeatedly perturbing inputs and running inference hundreds of times. This approach takes several seconds per prediction, making it unusable in a live payment flow.

In contrast, **TreeSHAP** leverages the internal tree structure of decision tree ensembles (such as LightGBM):
- It computes exact Shapley values directly during tree traversal in $O(\text{depth}^2 \times \text{trees})$ time.
- In LightGBM, this is implemented in native C++ and accessed via `booster.predict(X, pred_contrib=True)`.
- It executes in **under 2 milliseconds**, making real-time explanations practical within the API response.

### Advanced Mathematical Formulation & Additive Identity
For an ensemble of $T$ trees, TreeSHAP decomposes the model's raw output margin (log-odds) into additive feature contributions:

$$f(x) = \phi_0 + \sum_{j=1}^{M} \phi_j(x)$$

Where:
- $\phi_0$ is the base expected value (bias) across the training dataset.
- $\phi_j(x)$ is the contribution of feature $j$ to the margin score of instance $x$.

In `src/fraud/explain/reasons.py`:
```python
def explain_prediction(booster: lgb.Booster, feature_row: pd.DataFrame, top_k: int = 5):
    # Returns an array of size: [num_features + 1]
    contribs = booster.predict(feature_row, pred_contrib=True)[0]
    feat_contribs = contribs[:-1]
    bias = float(contribs[-1])
    
    # Sort features by absolute impact
    abs_indices = np.argsort(-np.abs(feat_contribs))[:top_k]
    reasons = []
    for idx in abs_indices:
        fname = feature_names[idx]
        val = feature_row[fname].iloc[0]
        contrib = float(feat_contribs[idx])
        reasons.append({
            "feature": fname,
            "label": FEATURE_LABEL_MAP.get(fname, fname),
            "value": str(val),
            "contribution": round(contrib, 4),
            "direction": "increases_risk" if contrib > 0 else "lowers_risk",
        })
    return {"base_value": bias, "reasons": reasons}
```

The test in `tests/test_explain_drift.py` asserts that the sum of contributions plus bias matches the raw logit output within a tolerance of `1e-3`.

---

## 6. Concept Drift & Population Stability Index (PSI)

### Beginner Intuition
Machine learning models degrade over time as real-world behavior changes. In fraud detection, changes in consumer purchasing habits or new attack patterns will alter the distribution of incoming feature values. 

The **Population Stability Index (PSI)** is a statistical metric used to measure how much a production data distribution has shifted away from the reference training distribution.

### Intermediate Mechanics
To compute PSI for a feature or model score:
1. The reference distribution (e.g. validation set predictions) is divided into 10 quantile bins ($10\%$ of observations per bin).
2. The current production distribution is mapped into those same bin boundaries.
3. The proportions of observations in each bin are compared.

```text
PSI Benchmark Scale:
  PSI < 0.10: Stable (No significant shift; model is reliable)
  0.10 ≤ PSI ≤ 0.20: Moderate Drift (Monitor closely; feature distributions shifting)
  PSI > 0.20: Significant Drift (Action required; retrain or recalibrate model)
```

### Advanced Formulation & Project Implementation
The mathematical formula for Population Stability Index is:

$$\text{PSI} = \sum_{b=1}^{B} \left( \text{Actual}_b - \text{Expected}_b \right) \times \ln\left( \frac{\text{Actual}_b}{\text{Expected}_b} \right)$$

Where $\text{Expected}_b$ is the proportion of reference records in bin $b$, and $\text{Actual}_b$ is the proportion of production records in bin $b$.

In `src/fraud/evaluation/drift.py`:
```python
def compute_psi(expected, actual, bins=10):
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected_series, quantiles))
    
    e_counts, _ = np.histogram(expected_series, edges)
    a_counts, _ = np.histogram(actual_series, edges)

    e_pct = np.clip(e_counts / len(expected_series), 1e-6, None)
    a_pct = np.clip(a_counts / len(actual_series), 1e-6, None)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
```

In `scripts/drift_check.py`, a background daemon periodically queries recent prediction scores from the database, computes the live score PSI, and updates a Prometheus gauge (`SCORE_PSI_GAUGE`). On our held-out test set, the prediction score PSI is **0.0018**, confirming stable model output distributions over time.
