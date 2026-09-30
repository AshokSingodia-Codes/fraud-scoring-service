# Exploratory Data Analysis (EDA) Summary

## Dataset Overview
- **Total Transactions:** 590,540
- **Total Features:** 435
- **Overall Fraud Rate:** 3.50% (20,663 positive instances)
- **Time Span:** Day 1 to Day 182 (~6 months)

## Key Findings & Modeling Decisions

1. **Severe Class Imbalance (3.50% Fraud):**
   - *Decision:* Metric choices must prioritize PR-AUC (Average Precision) and recall at low alert rates over accuracy or standard ROC-AUC alone.

2. **Temporal Structure & Non-Stationarity:**
   - *Finding:* Fraud rate varies over time (spiking in certain weeks and low-volume nighttime hours).
   - *Decision:* Standard random K-Fold cross-validation will leak future information. We must use a strict **time-based split** (first 70% train, next 15% validation, final 15% test).

3. **High Missingness in Specific Groups:**
   - *Finding:* Identity features (`id_01`-`id_38`) and `V*` / `M*` blocks have high missing rates (up to 75-90%+ missingness).
   - *Decision:* LightGBM handles native missing values efficiently. Columns with **> 90% missingness** (12 columns, e.g. dist2, D7, id_07, id_08, id_18...) will be evaluated for dropping to prevent noise.

4. **Transaction Amount Log-Skewness & Fraud Patterns:**
   - *Finding:* Fraudulent transactions tend to have slightly higher median log amounts and specific transaction cents patterns (`TransactionAmt % 1`).
   - *Decision:* Engineer `log1p(TransactionAmt)`, cents fraction, and round-amount indicators.

5. **High-Cardinality Categoricals:**
   - *Finding:* Features such as `card1` (13,553 categories), `card2` (500 categories), `addr1` (332 categories), and `P_emaildomain` (59 categories) have high cardinality.
   - *Decision:* Use frequency encoding computed **strictly on TRAIN data** to prevent data leakage. Unseen serving values will map to 0.

6. **Email Domain Risk Disparity:**
   - *Finding:* Certain email providers (`P_emaildomain` vs `R_emaildomain`) display significantly elevated fraud risk compared to standard providers.
   - *Decision:* Engineer domain provider/suffix extraction and domain-match boolean features (`P_emaildomain == R_emaildomain`).

7. **Feature Concept Drift:**
   - *Finding:* Count features (`C1`-`C14`) and accumulators (`D1`-`D15`) demonstrate mean value shifts between Month 1 and Month 6.
   - *Decision:* Incorporate temporal stability checks (Population Stability Index / PSI) during robustness evaluation.

8. **Device & Identity Signal Strength:**
   - *Finding:* Transactions with identity data (`DeviceType`, `DeviceInfo`, `id_*`) have a higher fraud rate (~6-8%) than non-identity transactions.
   - *Decision:* Create a binary `has_identity` flag and missingness counter for identity fields.
