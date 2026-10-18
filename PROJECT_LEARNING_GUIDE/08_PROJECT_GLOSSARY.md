# 08 - Project Technical Glossary

This glossary defines technical terms, metrics, abbreviations, and engineering patterns used throughout this project in clear, practical language.

---

## 1. Machine Learning & Statistical Terms

### GBDT (Gradient Boosted Decision Trees)
An ensemble learning technique that builds decision trees sequentially. Each new tree fits to the negative gradient (pseudo-residuals) of the loss function, correcting errors made by previous trees. LightGBM, XGBoost, and CatBoost are GBDT implementations.

### LightGBM
An open-source, high-performance GBDT framework developed by Microsoft. It uses histogram-based feature binning (grouping continuous values into discrete bins) and Exclusive Feature Bundling (EFB) to achieve fast training, low memory usage, and ~0.5 ms booster inference latencies (4.82 ms p50 HTTP latency on 4 workers).

### PR-AUC (Precision-Recall Area Under Curve / Average Precision)
The area under the curve plotting Precision ($TP / (TP + FP)$) against Recall ($TP / (TP + FN)$) across all possible classification thresholds. Unlike ROC-AUC, PR-AUC is sensitive to false alarms in heavily imbalanced datasets (e.g. 3.5% fraud prevalence).

### ROC-AUC (Receiver Operating Characteristic Area Under Curve)
The area under the curve plotting True Positive Rate against False Positive Rate. An uninformative model scores 0.50; a perfect model scores 1.00. In imbalanced datasets, high True Negative counts can sometimes yield high ROC-AUC scores even when false positive rates are significant.

### Brier Score
A strictly proper scoring rule measuring the accuracy of probabilistic predictions. Computed as the mean squared difference between predicted probability $p$ and actual binary outcome $y \in \{0, 1\}$:
$$\text{Brier} = \frac{1}{N} \sum (p_i - y_i)^2$$
Lower values indicate more accurate, better-calibrated probabilities.

### ECE (Expected Calibration Error)
A metric measuring probability calibration. Predictions are partitioned into equal-width probability bins (e.g. 10 bins: 0–0.1, 0.1–0.2, etc.). ECE computes the weighted average absolute difference between the mean predicted confidence and the empirical accuracy in each bin:
$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$

### Probability Calibration
The process of adjusting raw model output scores so that they reflect true empirical probabilities. If a well-calibrated model assigns an 8% fraud probability to a cohort of transactions, approximately 8 out of every 100 will be fraudulent.

### Isotonic Regression
A non-parametric calibration method that fits a piecewise constant, monotonically non-decreasing function to map raw model margins to calibrated probabilities. Because it is monotonic, it preserves the rank-order of predictions and does not degrade ROC-AUC.

### TreeSHAP (Shapley Additive exPlanations)
An algorithm that computes exact Shapley values for tree ensembles in polynomial time $O(TLD^2)$. It decomposes a model's raw output margin into additive per-feature contributions ($\sum \phi_i(x) + \phi_0 = f(x)$), showing which features increased or decreased fraud risk.

### Recall @ Top K% Alerts
An operational metric measuring the percentage of all fraudulent transactions caught when flagging the top $K\%$ (e.g. top 1% or top 5%) highest-scoring transactions. This reflects performance under real-world analyst review bandwidth constraints.

### PSI (Population Stability Index)
A statistical metric measuring how much a production data distribution has shifted relative to a baseline reference distribution:
$$\text{PSI} = \sum (\text{Actual}_b - \text{Expected}_b) \times \ln\left(\frac{\text{Actual}_b}{\text{Expected}_b}\right)$$
Values below 0.10 indicate stability; values above 0.20 indicate significant drift requiring model retraining.

### Concept Drift & Covariate Shift
- **Covariate Shift**: Changes in the input feature distribution $P(X)$ over time while the relationship to the target $P(Y \mid X)$ remains unchanged.
- **Concept Drift**: Changes in the statistical relationship between features and the target variable $P(Y \mid X)$ over time (e.g. fraudsters switching to new techniques).

---

## 2. Feature Engineering & Preprocessing Terms

### Downcasting (Memory Optimization)
Inspecting numeric columns in a DataFrame and converting them to the smallest lossless NumPy data type (e.g. converting `int64` taking 8 bytes down to `uint8` taking 1 byte for values between 0 and 255). In this project, downcasting reduced in-memory data usage from 2.5 GB to 924 MB.

### Parquet Format
An open-source, columnar storage file format developed for the Apache Hadoop ecosystem. It provides efficient data compression (Snappy) and encoding schemes, loading significantly faster and using 10x less disk space than raw CSVs.

### Frequency Encoding (Count Encoding)
Replacing categorical levels with their frequency of occurrence in the training partition. Used for high-cardinality features like `card1` (13,553 unique categories), where One-Hot Encoding would produce an unmanageably wide matrix.

### Interaction Features
Features created by combining two or more attributes to capture joint signals (e.g. `card1_addr1 = card1 + "_" + addr1`). This captures card-and-billing-zip combinations that fraudsters frequently alter.

### Training-Serving Skew
A discrepancy between how features are computed during offline training versus online serving. Mitigated in this project by encapsulating transformation logic and learned frequencies in a serialized `FeatureBuilder` instance.

### Data Leakage
When information from outside the training partition (such as future time periods or target labels) inadvertently enters the training pipeline. Mitigated here by strict chronological splitting on `TransactionDT`.

---

## 3. Systems, Architecture & Backend Terms

### FastAPI
A modern, high-performance Python web framework for building APIs, based on Starlette and Pydantic. It provides native async execution and auto-generates interactive OpenAPI Swagger documentation.

### Pydantic (v2)
A data validation and parsing library written in Rust for Python. It enforces type hints at runtime, validates request bodies, and serializes API response models.

### Lifespan Handler (`@asynccontextmanager`)
A FastAPI context manager that runs startup and shutdown logic. In this project, it loads all machine learning model artifacts into memory once at application launch, keeping individual scoring requests fast and stateless.

### WAL Mode (Write-Ahead Logging)
A concurrency mode in SQLite (`PRAGMA journal_mode=WAL`). Rather than locking the entire database during writes, changes are appended to a separate WAL file, allowing concurrent readers to read without being blocked by writers.

### SQLAlchemy (v2.0)
A Python SQL toolkit and Object Relational Mapper (ORM). Provides a consistent database abstraction, allowing the application to use local SQLite during testing and switch to PostgreSQL in production via environment variables.

### Token Bucket Rate Limiting
A rate-limiting algorithm where a client is assigned a bucket of tokens that refill over time. Each incoming request consumes a token. If the bucket is empty, requests are rejected with HTTP 429 (Too Many Requests).

### Reverse Proxy / CORS (Cross-Origin Resource Sharing)
A security mechanism implemented via HTTP headers that specifies which domains are allowed to access resources on the server. Managed here via FastAPI's `CORSMiddleware`.

---

## 4. Observability & Operations Terms

### Prometheus
An open-source systems monitoring and alerting toolkit. It operates via a pull model, periodically scraping metrics from application HTTP endpoints (`/metrics`).

### Counter, Histogram & Gauge (Prometheus Types)
- **Counter**: A cumulative metric that only increases (e.g. `fraud_requests_total`, `fraud_decisions_total`).
- **Histogram**: Samples observations across configurable buckets to measure frequency distributions and percentiles (e.g. `fraud_api_request_latency_seconds`).
- **Gauge**: A metric that can increase or decrease (e.g. `fraud_score_psi` tracking real-time distribution drift).

### Grafana
An open-source analytics and visualization web application. It connects to Prometheus to render real-time dashboards displaying request volumes, latencies, and drift metrics.

### Locust
An open-source, scriptable load testing tool. Used in this project (`scripts/locustfile.py`) to simulate concurrent user traffic and measure throughput and latency percentiles.

### Multi-Stage Docker Build
A Dockerfile optimization pattern where dependencies and wheels are compiled in an initial builder container, and only the final compiled artifacts are copied into a lean runner image. This reduces image size and strips unnecessary development build tools.
