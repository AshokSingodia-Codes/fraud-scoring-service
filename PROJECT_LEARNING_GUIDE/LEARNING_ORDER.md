# Learning Order Roadmap: Step 1 → Complete Project Understanding

This roadmap provides a simple, numbered study plan to guide you through the codebase and documentation until you have a complete, build-it-from-scratch understanding of the project.

---

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        LEARNING ROADMAP OVERVIEW                       │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 1: Foundations & Architecture       (Steps 1 – 3)               │
│  PHASE 2: Data Engineering & Preprocessing (Steps 4 – 6)               │
│  PHASE 3: Modeling, Calibration & Policy   (Steps 7 – 9)               │
│  PHASE 4: Real-Time Serving & Systems      (Steps 10 – 12)             │
│  PHASE 5: Verification & Interview Prep    (Steps 13 – 14)             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundations & Architecture

### Step 1: Understand the Problem & System Goals
- **Read**: [00_PROJECT_OVERVIEW.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/00_PROJECT_OVERVIEW.md)
- **Goal**: Understand why real-time fraud scoring requires sub-300 ms latencies, how class imbalance affects performance metrics, and how the end-to-end system fits together.
- **Milestone Check**: Explain the difference between an `APPROVE`, `MANUAL REVIEW`, and `BLOCK` decision.

### Step 2: Survey the Technology Stack
- **Read**: [01_TECHNOLOGY_STACK.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/01_TECHNOLOGY_STACK.md) and [05_TECHNOLOGY_COMPARISON.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/05_TECHNOLOGY_COMPARISON.md)
- **Goal**: Understand why LightGBM, FastAPI, Isotonic Regression, and SQLite/PostgreSQL were chosen over their alternatives.
- **Milestone Check**: Explain why LightGBM was selected over deep learning for this dataset.

### Step 3: Explore the Repository Structure
- **Read**: [02_FOLDER_STRUCTURE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/02_FOLDER_STRUCTURE.md)
- **Inspect**: `configs/config.yaml` and `src/fraud/config.py`
- **Goal**: Familiarize yourself with the folder layout and configuration loader.

---

## Phase 2: Data Engineering & Preprocessing

### Step 4: Data Ingestion & Memory Optimization
- **Read**: [09_DATA_PIPELINE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/09_DATA_PIPELINE.md) (Sections 1–3)
- **Code**: `src/fraud/data/download.py` and `src/fraud/data/ingest.py`
- **Goal**: Understand how transactions and identity records are merged, and how NumPy downcasting reduces memory usage from 2.5 GB to 924 MB.
- **Milestone Check**: Explain how `downcast_df()` works for integers, floats, and strings.

### Step 5: Chronological Splitting & Leakage Prevention
- **Read**: [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Section 1) and [DATA_DECISIONS.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/DATA_DECISIONS.md) (Decision 2)
- **Code**: `src/fraud/data/split.py` and `tests/test_no_leakage.py`
- **Goal**: Learn why time-based splitting is essential in fraud detection and verify the test assertions that prevent data leakage.
- **Milestone Check**: State the exact date ranges for the Train, Validation, and Test sets.

### Step 6: Feature Engineering with FeatureBuilder
- **Read**: [DATA_DICTIONARY.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/DATA_DICTIONARY.md) and [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Section 2)
- **Code**: `src/fraud/features/builder.py` and `scripts/build_features.py`
- **Goal**: Master the `fit()` vs `transform()` pattern in `FeatureBuilder` and understand how frequency encodings are learned.
- **Milestone Check**: Trace what happens when a brand-new card issuer category is encountered at serving time.

---

## Phase 3: Modeling, Calibration & Policy

### Step 7: Baselines & Evaluation Metrics
- **Read**: [03_STEP_BY_STEP_LEARNING.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/03_STEP_BY_STEP_LEARNING.md) (Step 4)
- **Code**: `src/fraud/models/baseline.py` and `src/fraud/evaluation/metrics.py`
- **Goal**: Understand the baseline models (Dummy, Logistic Regression, Default LightGBM) and metrics (PR-AUC, ROC-AUC, Recall @ Top 1%).
- **Milestone Check**: Explain why PR-AUC is more informative than ROC-AUC when fraud prevalence is 3.5%.

### Step 8: LightGBM Training & Optuna Tuning
- **Read**: [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Section 5)
- **Code**: `src/fraud/models/train_lgbm.py` and `scripts/train.py`
- **Goal**: Study how LightGBM handles native missing values, how Optuna optimizes hyperparameters, and how early stopping prevents overfitting.
- **Milestone Check**: Explain the role of `feature_fraction`, `num_leaves`, and early stopping.

### Step 9: Probability Calibration & Business Cost Policy
- **Read**: [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Sections 3 & 4)
- **Code**: `src/fraud/models/calibrate.py`, `src/fraud/models/threshold.py`, and `scripts/evaluate_pipeline.py`
- **Goal**: Understand why Isotonic Regression improves calibration error and how $t_{\text{review}}$ is derived by minimizing the financial loss matrix.
- **Milestone Check**: Recite the cost matrix formula and explain how $t_{\text{review}} = 0.0595$ was found.

---

## Phase 4: Real-Time Serving & Systems

### Step 10: Real-Time TreeSHAP Explainability
- **Read**: [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Section 5)
- **Code**: `src/fraud/explain/reasons.py`
- **Goal**: Understand how LightGBM's native C++ `pred_contrib=True` computes exact feature-level log-odds contributions in ~22 ms compute (~33.41 ms p50 HTTP latency on 4 workers).
- **Milestone Check**: Verify the additive property: $\sum \text{contributions} + \text{bias} = \text{margin score}$.

### Step 11: The Prediction Service & FastAPI API
- **Read**: [04_CODE_FLOW.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/04_CODE_FLOW.md)
- **Code**: `src/fraud/serving/predictor.py`, `src/fraud/serving/app.py`, `src/fraud/serving/schemas.py`, `auth.py`, `ratelimit.py`, and `db.py`
- **Goal**: Follow an incoming HTTP request through authentication, rate limiting, Pydantic validation, feature transformation, inference, and audit logging.
- **Milestone Check**: Explain how FastAPI's lifespan handler keeps model artifacts loaded in memory.

### Step 12: Observability, Drift Monitoring & Frontends
- **Read**: [06_DEEP_DIVE.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/06_DEEP_DIVE.md) (Section 6)
- **Code**: `src/fraud/serving/metrics.py`, `scripts/drift_check.py`, `monitoring/prometheus.yml`, and `frontend/src/App.jsx`
- **Goal**: Learn how Prometheus monitors request latencies, how Population Stability Index (PSI) detects distribution shift, and how the React dashboard visualizes SHAP contributions.
- **Milestone Check**: Interpret a PSI score of 0.15 vs 0.25.

---

## Phase 5: Verification & Interview Prep

### Step 13: Run the Automated Test Suite
- **Run**:
  ```bash
  pytest -v tests/
  ```
- **Inspect**: `tests/test_api.py`, `tests/test_no_leakage.py`, and `tests/test_predictor.py`
- **Goal**: Verify that all unit, integration, and leakage prevention tests pass cleanly.

### Step 14: Review Interview Questions & Key Metrics
- **Read**: [07_INTERVIEW_PREPARATION.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/07_INTERVIEW_PREPARATION.md) and [08_PROJECT_GLOSSARY.md](file:///e:/7thsem%20learning/projects/froud%20detection/PROJECT_LEARNING_GUIDE/08_PROJECT_GLOSSARY.md)
- **Goal**: Rehearse core interview questions and memorize key benchmark metrics (0.9103 ROC-AUC, 0.5289 PR-AUC calibrated, 0.0018 ECE, 5.23 ms p50 latency without SHAP / 36.43 ms with SHAP on 4 workers).
- **Final Milestone**: Confidently explain the full system architecture and key trade-offs in an engineering interview.
