# 05 - Technology Comparison & Design Rationale

This document presents structured comparisons between each major technology used in this project and its industry alternatives, detailing why the chosen tool was selected for this specific real-time fraud scoring workload.

---

## 1. Machine Learning Engines: LightGBM vs XGBoost vs CatBoost vs Deep Learning

| Evaluation Criterion | LightGBM (Selected) | XGBoost | CatBoost | Deep Learning (TabNet / MLP) |
|---|---|---|---|---|
| **Training Speed (590k rows)** | **Fastest** (~3–4 mins on CPU via histogram binning) | Moderate (slower histogram building on large feature counts) | Moderate to Slow (heavy symmetric tree building) | Very Slow (multiple epochs, GPU required) |
| **Inference Latency (single row)** | **Ultra-Fast** (~0.5 ms booster predict / 4.82 ms HTTP p50 on 4 workers) | Fast (~5–10 ms) | Moderate (~10–20 ms) | Slow (~25–60 ms PyTorch runtime overhead) |
| **Handling of Missing Values** | **Native** (Learns optimal split routing for NaNs) | **Native** (Routes NaNs to default branch) | **Native** (Treats NaNs as min/max value) | **Fails without Imputation** (Requires SimpleImputer or KNN) |
| **Categorical Feature Support** | **Native** (Optimal Fisher partition on integer indices) | Experimental / One-Hot Encoding preferred | **Best-in-class** (Online target statistics encoding) | Requires Entity Embeddings / One-Hot |
| **Memory Consumption** | **Very Low** (Histogram binning discretizes floats to uint8) | Moderate | High (stores complex categorical combinations) | High (GPU VRAM / tensor buffers) |
| **Built-in Explainability** | **Native TreeSHAP** (`pred_contrib=True` in C++) | Native TreeSHAP | Native TreeSHAP | Slow (Integrated Gradients / KernelSHAP) |
| **Hardware Requirements** | Lightweight CPU (Runs easily on standard instances) | CPU / GPU | CPU / GPU | Heavy GPU required |

### Why This Project Selected LightGBM
1. **Handling Extreme Missingness**: The IEEE-CIS dataset contains 434 columns where several blocks (`dist2`, `D7`, `id_01`–`id_38`) are 75% to 90%+ missing. LightGBM handles these missing values natively during split finding, avoiding the computational cost and bias of synthetic imputation pipelines.
2. **Strict Latency Budget**: Payment gateways require response times under 300 ms. LightGBM's compiled tree evaluation adds minimal overhead (~0.5 ms booster predict), leaving ample headroom for network transit and feature transformation.
3. **Built-in TreeSHAP**: LightGBM computes exact feature-level log-odds contributions during tree traversal without requiring the heavy Python `shap` package.

---

## 2. Web Serving Frameworks: FastAPI vs Flask vs Django vs Triton

| Evaluation Criterion | FastAPI (Selected) | Flask | Django / DRF | Triton Inference Server |
|---|---|---|---|---|
| **Concurrency Model** | **Asynchronous (`asyncio` / Starlette)** | Synchronous WSGI (requires Gevent/Gunicorn workers) | Synchronous WSGI / Heavy ASGI | C++ Multi-Model Server |
| **Request Throughput** | **High** (~4,000–8,000 req/sec on light endpoints) | Moderate (~1,000–2,500 req/sec) | Low to Moderate (~800–1,500 req/sec) | **Extreme** (10,000+ req/sec on GPU/C++) |
| **Data Validation** | **Built-in via Rust Pydantic v2** | None (Manual parsing or Marshmallow) | Heavy Django Forms / Serializers | Rigid binary tensor buffers |
| **API Documentation** | **Automatic Interactive OpenAPI/Swagger (`/docs`)** | None (Requires Flasgger extension) | Third-party (drf-spectacular) | None (gRPC / HTTP proto contracts) |
| **Feature Transformation Integration** | **Seamless Python Integration** | Seamless Python Integration | Python, but heavy framework overhead | Poor (Requires custom C++/Python backends) |
| **Learning Curve** | Low / Intuitive | Very Low | High (Large monolithic framework) | High (Requires C++, Docker, model repo layout) |

### Why This Project Selected FastAPI
1. **Combined Async Serving and Schema Validation**: FastAPI pairs asynchronous I/O with Pydantic v2 validation. Incoming JSON requests are validated and typed before execution, returning clear 422 errors for malformed inputs (e.g. negative amounts).
2. **Automated Documentation**: FastAPI automatically generates interactive Swagger documentation at `/docs`, simplifying testing for payment integration engineers.
3. **Predictor Lifespan Management**: FastAPI's `@asynccontextmanager` loads all machine learning artifacts into memory once at application startup, keeping serving-time request processing stateless and fast.

---

## 3. Probability Calibration: Isotonic Regression vs Platt Scaling (Sigmoid) vs Beta Calibration

| Evaluation Criterion | Isotonic Regression (Selected) | Platt Scaling (Logistic / Sigmoid) | Beta Calibration |
|---|---|---|---|
| **Mathematical Formulation** | Non-parametric, piecewise monotonic constant fit | Parametric logistic fit: $P(y=1\|f) = \frac{1}{1 + e^{Af + B}}$ | Parametric fit using Beta distribution: $1 / (1 + 1/(e^c \cdot (p^a / (1-p)^b)))$ |
| **Shape Flexibility** | **Completely flexible** (fits any monotonic curve) | Rigid sigmoidal shape | Flexible inverted-S / asymmetric shapes |
| **Preservation of ROC-AUC Rank** | **Strictly Preserved** (Monotonic step function) | Strictly Preserved | Strictly Preserved |
| **Sample Size Requirement** | Requires $\ge 1,000$ calibration samples | Works well even on small samples ($< 500$) | Works on moderate samples ($> 1,000$) |
| **Risk of Overfitting** | Moderate if validation set is too small | Low | Low to Moderate |
| **Performance in This Project** | **Brier Score: 0.0221, ECE: 0.0018** | Brier Score: 0.0248, ECE: 0.0062 | Brier Score: 0.0230, ECE: 0.0035 |

### Why This Project Selected Isotonic Regression
1. **Large Validation Split**: With 88,581 samples in the validation set, there is sufficient data to fit a non-parametric isotonic step function without overfitting.
2. **Non-Sigmoidal Calibration Curve**: Gradient boosted decision tree margins on heavily imbalanced tabular data often display asymmetric distribution distortions that do not follow a standard logistic sigmoid. Isotonic Regression fits this empirical curve directly, reducing Expected Calibration Error (ECE) to 0.0018.

---

## 4. Databases & Storage: PostgreSQL vs SQLite vs MySQL vs MongoDB

| Evaluation Criterion | PostgreSQL / Neon (Selected for Prod) | SQLite with WAL (Selected for Local) | MySQL | MongoDB |
|---|---|---|---|---|
| **Storage Architecture** | Client-Server Relational DB | In-process Serverless File DB | Client-Server Relational DB | Document-Oriented NoSQL |
| **Concurrent Write Performance** | **High** (MVCC with row-level locks) | Moderate (WAL mode allows concurrent readers, serialized writers) | High (InnoDB engine MVCC) | High (WiredTiger document-level locking) |
| **Setup & Maintenance Overhead** | Moderate (Requires DB instance / connection pooler) | **Zero** (Zero-configuration local file `predictions.db`) | Moderate | Moderate (Requires replica sets for resilience) |
| **ACID Compliance** | **Strict** | **Strict** | Strict (InnoDB) | Tunable / Document-level |
| **Cloud Serverless Support** | **Excellent** (Neon branch-per-environment, auto-suspend) | Poor in serverless containers (stateless disk wipes) | Moderate (PlanetScale) | Moderate (MongoDB Atlas) |
| **Structured Querying for Audit Logs** | **Native SQL** (Time-range queries, indexing, joins) | **Native SQL** | Native SQL | Aggregation pipelines (JSON syntax) |

### Why This Project Supports Both SQLite (WAL) and PostgreSQL
1. **Local Developer Experience**: By default, `src/fraud/serving/db.py` uses SQLite with Write-Ahead Logging (`PRAGMA journal_mode=WAL`). Developers and CI runners can run tests and the API immediately without configuring Docker containers or remote databases.
2. **Production Migration via SQLAlchemy**: The application uses SQLAlchemy 2.0 ORM models (`PredictionLog`). Switching to a production PostgreSQL database (such as serverless Neon via `DATABASE_URL_POOLED`) requires only updating an environment variable, with zero changes to application code.

---

## 5. Rate Limiting & Caching: In-Memory Token Bucket vs Redis vs Cloudflare

| Evaluation Criterion | In-Memory Sliding Window (Selected Fallback) | Redis (Selected for Distributed Prod) | Cloudflare / API Gateway Rate Limiting |
|---|---|---|---|
| **Architecture** | Python process memory (`defaultdict(list)`) | Distributed In-Memory Key-Value Store | Edge Network / Reverse Proxy |
| **Latency Overhead** | **Sub-microsecond (< 0.05 ms)** | Low (1–3 ms network round-trip) | Edge-evaluated (Zero origin overhead) |
| **Multi-Worker Synchronization** | Isolated per Uvicorn worker process | **Globally synchronized across all workers/containers** | Globally synchronized across edge network |
| **Operational Dependency** | **Zero external dependencies** | Requires running Redis container / cluster | Requires SaaS subscription & DNS proxying |
| **Failure Behavior** | Fails closed or open within Python process | Requires fallback handling if Redis disconnects | Edge-managed |

### Why This Project Uses a Hybrid Approach
1. **Containerized Production**: `docker-compose.yml` provisions a dedicated `redis:7-alpine` container for distributed rate limiting across horizontal API replicas.
2. **Resilient Local Execution**: `src/fraud/serving/ratelimit.py` implements an in-memory sliding window timestamp cleaner as a built-in fallback. If Redis is unavailable or when running on lightweight cloud free tiers (e.g. Render free instance), the API continues operating without throwing connection errors.

---

## 6. Frontend Frameworks: React 19 + Vite vs Streamlit vs Next.js vs Vue

| Evaluation Criterion | React 19 + Vite (Selected for Production UI) | Streamlit (Selected for Analyst Portal) | Next.js (SSR / Full-Stack) | Vue 3 / Nuxt |
|---|---|---|---|---|
| **Target User Experience** | **Modern, responsive production web dashboard** | Rapid Python prototype / analyst exploration | Full-stack production web application | Modern responsive web dashboard |
| **UI Customization & Theming** | **Complete control** (Tailwind CSS 4, custom dark mode, glassmorphism) | Restricted to Streamlit component grid | Complete control | Complete control |
| **Charting & Interactivity** | **High** (Recharts SVG tooltips, dynamic bar coloring) | Moderate (Static Matplotlib / Altair re-renders) | High | High |
| **Hosting & Deployment** | **Static build** deployed globally to edge CDNs (Vercel) | Requires long-running Python server instance | Requires Node.js server or Vercel SSR compute | Static build on edge CDN |
| **Development Speed for Python Devs** | Moderate (Requires JS/JSX knowledge) | **Very Fast** (Pure Python scripting) | Moderate | Moderate |

### Why This Project Provides Both Frontends
1. **React 19 + Vite (`frontend/`)**: Provides a responsive, branded interface for production deployments, featuring interactive Recharts SHAP graphs, preset loaders, and batch scoring tables. It compiles down to static HTML/CSS/JS deployable on Vercel with minimal operational overhead.
2. **Streamlit (`app_ui/streamlit_app.py`)**: Included for data scientists who prefer working entirely in Python to inspect model behavior and test scenarios without frontend tooling.
