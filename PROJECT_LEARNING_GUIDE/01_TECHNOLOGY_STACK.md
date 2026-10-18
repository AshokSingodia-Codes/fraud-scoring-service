# 01 - Technology Stack: Selection, Alternatives, and Tradeoffs

This document details every technology, framework, library, storage engine, tool, and cloud service used across the **Real-Time Fraud Scoring Service**. For each component, we analyze why it was chosen for this specific project, evaluate realistic alternatives, and document concrete engineering tradeoffs.

---

## 1. Machine Learning & Data Processing Stack

### 1.1 Model Engine: LightGBM (v4.0+)
- **Role in Project**: Primary gradient boosted decision tree classifier (`src/fraud/models/train_lgbm.py`, saved to `models/model.txt`). Used for binary classification on tabular transaction data.
- **Why Selected**:
  - **Native Handling of Missing Values**: IEEE-CIS transactions contain 434 columns with up to 80–90% missingness (e.g. `dist2`, identity attributes). LightGBM natively routes missing values to optimal split directions without requiring expensive synthetic imputation.
  - **Native Categorical Support**: LightGBM handles high-cardinality categorical features via optimal Fisher partitioning over categorical integer indices.
  - **Ultra-Fast Real-Time Inference**: LightGBM provides a lean C++ runtime with fast prediction loops (`booster.predict()`), scoring single transactions in under 5 milliseconds.
  - **Built-in TreeSHAP (`pred_contrib=True`)**: Computes exact per-prediction feature contributions inside the compiled tree traversal without needing the slow Python `shap` package.
- **Alternatives Considered**:
  - *XGBoost*: Excellent performance, but historically slower on large tabular datasets with high cardinality, higher memory overhead during histogram construction, and heavier prediction runtime dependencies.
  - *CatBoost*: Outstanding out-of-the-box categorical handling (target encoding), but inference latency on single rows is higher than LightGBM, and memory usage during training on 590k rows is significantly larger.
  - *Deep Learning (TabNet / PyTorch MLPs)*: Deep neural nets on tabular data typically underperform tree ensembles on tabular tabular benchmarks with extreme missingness, require extensive normalization/imputation, and demand GPUs for training and inference.
  - *Random Forest (scikit-learn)*: Cannot handle native NaNs without imputation, lacks gradient boosting accuracy, produces huge model artifacts on disk, and evaluates slowly at inference time.
- **Advantages & Disadvantages**:
  - *Pros*: Extremely fast training and inference, low memory footprint via histogram binning, native NaN support, built-in TreeSHAP.
  - *Cons*: Prone to overfitting on small datasets if `num_leaves` is too high; requires careful hyperparameter tuning (`feature_fraction`, `min_child_samples`).
- **When to Use Alternatives Instead**:
  - Use **CatBoost** if the dataset consists primarily of complex categorical interactions without dense numerical structures.
  - Use **XGBoost** if your deployment ecosystem relies heavily on specialized GPU-accelerated inference engines like NVIDIA Triton.
  - Use **PyTorch/TabNet** if transactions include unstructured modalities (free-form fraud notes, audio recordings, or image attachments).

---

### 1.2 Probability Calibration: Scikit-Learn Isotonic Regression
- **Role in Project**: Post-hoc probability calibrator (`src/fraud/models/calibrate.py`, saved to `models/calibrator.joblib`).
- **Why Selected**:
  - Tree-based models minimize loss on ranking (optimizing AUC) and frequently output uncalibrated, distorted probabilities clustered around decision boundaries or biased toward class priors.
  - Isotonic Regression is a non-parametric monotonic piecewise constant fit that preserves model ranking (does not degrade ROC-AUC) while aligning predicted probabilities with empirical frequencies (slashing Expected Calibration Error from 0.0039 to 0.0018).
- **Alternatives Considered**:
  - *Platt Scaling (Sigmoid / Logistic Regression)*: Parametric method that fits a logistic function to model margins. Assumes the distortion is strictly sigmoidal. LightGBM probability distortions on skewed tabular data are often non-sigmoid.
  - *Beta Calibration*: Parametric method extending Platt scaling using Beta distributions; adds extra hyperparameter complexity without significant gain over Isotonic when validation sample size is large (> 80,000 samples).
- **Advantages & Disadvantages**:
  - *Pros*: Non-parametric, completely flexible monotonic shape, mathematically guarantees preservation of rank order.
  - *Cons*: Can overfit if fit on tiny datasets (requires $\ge 1,000$ calibration samples; here we fit on 88,581 validation rows, where it is robust).
- **When to Use Alternatives Instead**:
  - Use **Platt Scaling** if the calibration dataset is small (< 1,000 samples) to prevent stepwise overfitting.

---

### 1.3 Hyperparameter Optimization: Optuna (v3.2+)
- **Role in Project**: Automated tuning of LightGBM hyperparameters (`src/fraud/models/train_lgbm.py`).
- **Why Selected**:
  - Uses the **Tree-structured Parzen Estimator (TPE)**, an efficient Bayesian optimization algorithm that explores non-linear hyperparameter interactions far faster than random or grid search.
  - Native Python syntax with clean objective functions.
- **Alternatives Considered**:
  - *GridSearchCV / RandomizedSearchCV (scikit-learn)*: Inefficient brute-force search; ignores trial histories; wastes compute on unpromising parameter regions.
  - *Ray Tune*: Powerful distributed hyperparameter framework, but introduces excessive architectural complexity and dependencies for a single-node training job.
- **When to Use Alternatives Instead**:
  - Use **Ray Tune** if training on a distributed cluster across hundreds of GPU nodes.

---

### 1.4 Experiment Tracking: MLflow (v2.5+)
- **Role in Project**: Logs model parameters, training metrics across folds, and model artifacts (`mlflow.db` and `mlruns/`).
- **Why Selected**:
  - Open-source, local-first, lightweight tracking that works with SQLite (`mlflow.db`) without requiring an active external cloud account.
  - Native integration with LightGBM and scikit-learn.
- **Alternatives Considered**:
  - *Weights & Biases (WandB)*: Excellent UI and cloud collaboration, but requires an external SaaS account and internet connectivity.
  - *Neptune.ai / Comet.ml*: SaaS tools with vendor lock-in and account setup overhead.
  - *DVC (Data Version Control)*: Excellent for large binary artifact versioning on S3/GCS, but heavier CLI tooling for simple experiment metric logging.
- **When to Use Alternatives Instead**:
  - Use **Weights & Biases** or **Comet** if working in a geographically distributed team requiring shared cloud dashboards and real-time alerts.

---

### 1.5 Data Manipulation & Storage: Pandas & PyArrow (Parquet)
- **Role in Project**:
  - `pandas`: Data wrangling, categorical type conversions, custom frequency encoding.
  - `pyarrow`: High-performance columnar serialization for intermediate datasets (`data/interim/train_merged.parquet`).
- **Why Selected**:
  - Raw CSVs (`train_transaction.csv` + `train_identity.csv`) take ~800 MB on disk and over 2.5 GB of RAM when loaded as raw strings.
  - PyArrow Parquet provides columnar compression (Snappy) reducing disk footprint to ~90 MB, and typing downcasting in `src/fraud/data/ingest.py` drops in-memory RAM usage from 2.5 GB to 924 MB.
- **Alternatives Considered**:
  - *Polars*: Extremely fast Rust-based DataFrame library with multithreading. Excellent, but `FeatureBuilder` integrates with scikit-learn/joblib pipelines where Pandas is the standard interoperability layer.
  - *DuckDB*: Excellent for in-process SQL querying on Parquet, but not required since full feature transformations are Python-native.
- **When to Use Alternatives Instead**:
  - Use **Polars** or **DuckDB** if raw dataset size exceeds available system RAM (> 16–32 GB) to leverage lazy out-of-core streaming execution.

---

## 2. Backend & Serving Stack

### 2.1 Web Framework: FastAPI (v0.100+) & Uvicorn
- **Role in Project**: Core REST scoring service (`src/fraud/serving/app.py`). Serves `/health`, `/ready`, `/v1/score`, `/v1/score/batch`, `/v1/model/info`, and `/metrics`.
- **Why Selected**:
  - **High Asynchronous Concurrency**: Built on Starlette and `asyncio`, offering near-Go/Node throughput while retaining native Python ML interoperability.
  - **Native Pydantic v2 Integration**: Automatic request deserialization, payload type validation, and OpenAPI Swagger documentation generation (`/docs`).
  - **Lifespan Context Management**: Uses `@asynccontextmanager` to load the heavy `FraudPredictor` artifacts once into memory at application boot, avoiding per-request disk I/O.
- **Alternatives Considered**:
  - *Flask*: Synchronous WSGI framework; lacks built-in async handling, automatic schema validation, and auto-generated Swagger UI; slower under high concurrency.
  - *Django / Django REST Framework (DRF)*: Full-stack monolith with ORM and admin dashboard; heavy overhead for a lightweight real-time microservice.
  - *Triton Inference Server / TorchServe*: Specialized C++ model servers. Outstanding for GPU deep learning models, but over-engineered for a LightGBM GBDT model with custom feature engineering logic.
  - *BentoML / Ray Serve*: Good for complex multi-model pipelines, but adds framework complexity and container bloat.
- **When to Use Alternatives Instead**:
  - Use **Triton Inference Server** if serving multi-model ensembles on GPUs at 50,000+ requests per second.
  - Use **Flask** only if maintaining legacy Python 2/3 codebases.

---

### 2.2 Validation & Serialization: Pydantic (v2.0+)
- **Role in Project**: Schema enforcement for incoming transaction payloads and API response serialization (`src/fraud/serving/schemas.py`).
- **Why Selected**:
  - Pydantic v2 is written in Rust, offering a 5x–20x parsing speedup over Pydantic v1.
  - Validates business constraints (e.g. `TransactionAmt > 0`, non-empty transaction IDs) with informative 422 error payloads.
  - `ConfigDict(extra="allow")` allows accepting additional arbitrary transaction fields without rejecting client requests, ensuring backwards compatibility.
- **Alternatives Considered**:
  - *Marshmallow*: Pure Python schema library; significantly slower.
  - *FastAPI Native Dict*: Zero schema enforcement; risks runtime TypeErrors inside model inference.

---

### 2.3 Database & ORM: SQLAlchemy (v2.0+) + SQLite / PostgreSQL
- **Role in Project**: Logs every scoring decision, fraud probability, risk band, and latency to an audit table (`src/fraud/serving/db.py`).
- **Why Selected**:
  - **Database Agnostic**: Supports both local SQLite (`predictions.db`) for rapid offline testing and production PostgreSQL (or serverless Neon PostgreSQL via connection pooler `DATABASE_URL_POOLED`) without code modifications.
  - **Write-Ahead Logging (WAL)**: Configured in `db.py` (`PRAGMA journal_mode=WAL`) to allow concurrent reads and writes without database lock contention.
  - **Thread-safe**: Protected by a threading lock `_db_lock` and isolated `SessionLocal` sessions to prevent SQLite concurrency deadlocks.
- **Alternatives Considered**:
  - *Raw SQL (psycopg3 / sqlite3)*: Faster, but lacks clean migration handling and model definition abstractions.
  - *MongoDB / Document DB*: Flexible for unstructured JSON logs, but lacks relational integrity and query joins with transaction ledger databases.
- **When to Use Alternatives Instead**:
  - Use **Kafka / RabbitMQ + ClickHouse** in ultra-high-throughput enterprise banking (> 10,000 tx/sec) where synchronous or direct DB inserts become a bottleneck.

---

### 2.4 In-Memory Caching & Rate Limiting: Redis (Docker) & In-Memory Fallback
- **Role in Project**:
  - Rate-limiting protection against denial-of-service and brute-force key scraping (`src/fraud/serving/ratelimit.py`).
  - Redis image defined in `docker-compose.yml`.
  - Fallback in-memory sliding window queue implemented directly in Python when Redis is absent.
- **Why Selected**:
  - Prevents API key exhaustion and safeguards CPU compute from load spikes.
  - Graceful fallback ensures the API works in isolated standalone environments (like Render Free Tier or local tests) without crashing if Redis is offline.

---

## 3. Observability & Monitoring Stack

### 3.1 Metrics Collection: Prometheus Client (Python) & Prometheus Server
- **Role in Project**: Collects and exposes operational metrics via `/metrics` (`src/fraud/serving/metrics.py`, `monitoring/prometheus.yml`).
- **Metrics Tracked**:
  - `fraud_requests_total` (Counter with labels `endpoint`, `status`).
  - `fraud_api_request_latency_seconds` (Histogram across latency buckets).
  - `fraud_prediction_score` (Histogram tracking probability distribution across bins `[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]`).
  - `fraud_decisions_total` (Counter labeled by `APPROVE`, `REVIEW`, `BLOCK`).
  - `fraud_score_psi` (Gauge tracking Population Stability Index drift in `scripts/drift_check.py`).
- **Why Selected**:
  - Industry standard pull-based monitoring system with zero impact on request critical path.

---

### 3.2 Metrics Visualization: Grafana
- **Role in Project**: Provisioned via Docker Compose (`monitoring/grafana/dashboard.json`).
- **Why Selected**:
  - Visualizes real-time request rates, p95/p99 latency spikes, decision distributions, and model score drift on a unified operations dashboard.

---

## 4. Frontend & Presentation Stack

### 4.1 Production UI: React 19 + Vite + Tailwind CSS
- **Role in Project**: Modern, responsive production web dashboard (`frontend/`).
- **Libraries Used**:
  - `Vite 8`: Lightning-fast build tool and local development server with Hot Module Replacement (HMR).
  - `Tailwind CSS 4`: Utility-first CSS framework enabling dark-mode styling and modern cards.
  - `Recharts 3`: Composable SVG charting library for interactive horizontal SHAP contribution bar charts and probability progress gauges.
  - `Lucide React`: Clean icon library.
  - `Axios`: HTTP client communicating with FastAPI endpoints.
- **Why Selected**:
  - Provides a frictionless, state-of-the-art interface for bank fraud operations teams.
  - Decoupled from Python runtime; can be deployed independently to edge CDNs (Vercel, Cloudflare Pages).

---

### 4.2 Prototyping UI: Streamlit (v1.24+)
- **Role in Project**: Single-file Python analytical web dashboard (`app_ui/streamlit_app.py`).
- **Why Selected**:
  - Allows data scientists to test model features, visualize Matplotlib SHAP graphs, and experiment with presets without writing HTML/JS/CSS.

---

## 5. DevOps, Infrastructure & CI/CD Stack

### 5.1 Containerization: Docker & Multi-Stage Builds
- **Role in Project**: `Dockerfile` and `docker-compose.yml`.
- **Why Selected**:
  - Multi-stage build (`python:3.11-slim` builder stage $\rightarrow$ runner stage) keeps final image size minimal and strips development compilers (`gcc`).
  - Non-root container execution (`appuser:1000`) for production container security.
  - Bundles dependencies (`libgomp1` required by LightGBM C++ OpenMP).

---

### 5.2 CI/CD: GitHub Actions
- **Role in Project**: `.github/workflows/ci.yml`.
- **Why Selected**:
  - Automatically runs `ruff check .` for linting and `pytest` for unit/integration/leakage tests on every pull request and push to `main`.

---

### 5.3 Cloud Deployments
- **Render (`render.yaml`)**: Hosts backend Docker container web service with native `/health` probes.
- **Vercel (`vercel.json`)**: Hosts the static Vite React production build with edge global caching.
- **Neon PostgreSQL (`neon.ts`, `scratch/test_neon_conn.py`)**: Serverless Postgres branch with connection pooling for cloud persistence.

---

## 6. Comprehensive Technology Matrix Summary

| Layer | Technology | Primary Function | Primary Reason for Choice | Primary Alternative |
|---|---|---|---|---|
| **ML Engine** | LightGBM | Gradient Boosted Trees | Fast inference, native NaN & categorical handling | XGBoost / CatBoost |
| **Calibration** | Isotonic Regression | Probability alignment | Non-parametric, preserves ROC-AUC rank | Platt Scaling |
| **Explainability** | TreeSHAP (native) | Real-time reason codes | $O(\text{depth} \times \text{trees})$ inside LightGBM | KernelSHAP / LIME |
| **Data Format** | PyArrow Parquet | Columnar interim data | 10x smaller file size, fast typing | Feather / CSV |
| **Backend API** | FastAPI + Uvicorn | Scoring microservice | Async, Pydantic validation, OpenAPI docs | Flask / Django |
| **Audit Database** | SQLAlchemy + SQLite/PG | Prediction logging | Multi-engine support, WAL mode | MongoDB / ClickHouse |
| **Metrics** | Prometheus | Telemetry & PSI drift | Pull-based, zero serving latency impact | Datadog / StatsD |
| **Dashboard** | Grafana | System & ML dashboard | Pre-configured metrics visualization | Kibana |
| **Frontend UI** | React + Vite + Tailwind | Fraud analyst portal | Interactive Recharts SHAP graphs, fast UX | Next.js / Angular |
| **CI/CD** | GitHub Actions | Automated tests & lints | Zero setup overhead, native GitHub integration | GitLab CI / Jenkins |
