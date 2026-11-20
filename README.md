# Real-Time Fraud Detection & Decision Engine

[![CI/CD Pipeline](https://github.com/AshokSingodia-Codes/fraud-scoring-service/actions/workflows/ci.yml/badge.svg)](https://github.com/AshokSingodia-Codes/fraud-scoring-service/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![React 19](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev/)
[![Tailwind CSS v4](https://img.shields.io/badge/Tailwind-CSS%20v4-38bdf8.svg)](https://tailwindcss.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-82%20Passed%20(100%25)-success.svg)](tests/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

An enterprise-grade, real-time fraud scoring platform and decision engine built on the Kaggle IEEE-CIS Fraud Detection dataset (590K+ transactions). It evaluates credit card payments in milliseconds, outputting calibrated fraud probabilities, asymmetric cost-optimal business decisions (`APPROVE`, `MANUAL REVIEW`, `BLOCK`), per-prediction TreeSHAP adverse action reason codes, dynamic card velocity features, and real-time streaming feature drift monitoring.

---

## 🔗 Live Deployments & Application Links

| Component | Platform | Live URL & Endpoints | Status |
| :--- | :--- | :--- | :--- |
| **Frontend Web Application** | Vercel | [https://fraud-scoring-service.vercel.app](https://fraud-scoring-service.vercel.app) | [![Vercel](https://img.shields.io/badge/Vercel-Live%20Production-success?logo=vercel)](https://fraud-scoring-service.vercel.app) |
| **Backend Scoring API** | Render | [https://fraud-scoring-api.onrender.com](https://fraud-scoring-api.onrender.com) | [![Render](https://img.shields.io/badge/Render-Live%20API-success?logo=render)](https://fraud-scoring-api.onrender.com) |
| **Interactive API Documentation** | Swagger UI | [https://fraud-scoring-api.onrender.com/docs](https://fraud-scoring-api.onrender.com/docs) | [![Docs](https://img.shields.io/badge/Swagger-OpenAPI%203.0-blue?logo=swagger)](https://fraud-scoring-api.onrender.com/docs) |
| **Service Health Check** | Render | [https://fraud-scoring-api.onrender.com/health](https://fraud-scoring-api.onrender.com/health) | `{"status": "ok"}` |
| **Prometheus Telemetry** | Render | [https://fraud-scoring-api.onrender.com/metrics](https://fraud-scoring-api.onrender.com/metrics) | Real-time PSI & Latency |
| **Project Evaluation & Resume Guide** | Documentation | [reports/EVALUATION_AND_RESUME_GUIDE.md](reports/EVALUATION_AND_RESUME_GUIDE.md) | Verified Benchmarks & Highlights |

---

## 🌟 System Architecture

```mermaid
flowchart TD
    subgraph ClientLayer [1. Client & Presentation Layer - Vercel]
        UI["React 19 Single Page App<br/>(Tailwind CSS v4 / Recharts)"]
        Gateway["Payment Gateway / E-Commerce Backend"]
    end

    subgraph APILayer [2. Serving & Ingestion Layer - Render Web Service]
        RateLimiter["Sliding Window Rate Limiter<br/>(In-Memory / Redis ZSET)"]
        Val["Pydantic v2 Request Validation<br/>(Strict Type Coercion & Bounds)"]
        Predictor["FraudPredictor Core Coordinator"]
        BG["FastAPI BackgroundTasks<br/>(Zero Latency Async Offload)"]
    end

    subgraph MLEngine [3. Real-Time ML Inference Pipeline]
        VStore["Online Velocity Store<br/>(10m, 30m, 1h, 24h Card Bursts)"]
        FB["FeatureBuilder<br/>(456 features / Pre-allocated NumPy)"]
        LGBM["LightGBM Champion Classifier<br/>(500 trees / num_leaves=101)"]
        Cal["Probability Calibrator<br/>(Isotonic Regression / ECE: 0.0045)"]
        Policy["Cost-Optimal Policy Engine<br/>(t_review: 0.0496, t_block: 0.7534)"]
        SHAP["Native C-TreeSHAP Engine<br/>(Top Adverse Action Reason Codes)"]
    end

    subgraph ObservabilityAudit [4. Persistence & Observability Plane]
        DB[("PostgreSQL / SQLite<br/>Audit Store")]
        Prom["Prometheus Exporter<br/>(/metrics)"]
        Drift["Streaming PSI Drift Monitor<br/>(Sliding Reservoir)"]
        Shadow["Challenger Shadow Evaluator<br/>(Candidate Model Telemetry)"]
    end

    UI -->|HTTPS / JSON| RateLimiter
    Gateway -->|HTTPS / JSON| RateLimiter
    RateLimiter --> Val --> Predictor
    Predictor --> VStore
    Predictor --> FB --> LGBM --> Cal --> Policy
    Predictor -.->|If explain=True| SHAP
    Predictor --> BG
    BG --> DB
    BG --> Prom
    BG --> Drift
    BG --> Shadow
```

### Architecture Component Breakdown

```
[ Client / Browser ] (React 19 on Vercel)
         │
         │  HTTPS / JSON  (Open Access, No Auth Barrier)
         ▼
[ FastAPI Gateway on Render ]
    ├── In-memory / Redis Sliding-Window Rate Limiter
    ├── Pydantic v2 Schema Validation
    └── FraudPredictor Engine (< 15 ms latency SLA)
         ├── Online Velocity Store (10m, 30m, 1h, 24h burst tracking)
         ├── FeatureBuilder (456 numerical & interaction features)
         ├── LightGBM Champion Model (C++ inference)
         ├── Isotonic Probability Calibration (ECE: 0.0045)
         ├── Dual Decision Policy (APPROVE / MANUAL REVIEW / BLOCK)
         └── Native TreeSHAP Reason Code Generator
         │
         └── BackgroundTasks (Async non-blocking offload):
              ├── Audit Logger (PostgreSQL / SQLite)
              ├── Streaming PSI Drift Monitor
              ├── Prometheus Metrics Collector
              └── Shadow Model Evaluation
```

1. **Client & Presentation Layer (React 19 on Vercel):**
   - Modern, lightweight single-page application built with React 19, Vite, and Tailwind CSS.
   - Provides intuitive real-time transaction checking, instant decision badges, TreeSHAP contribution visualizer, audit log search by transaction ID, and PDF/CSV report exports.
   - Zero authentication barrier: users can immediately verify transactions without login, registration, or API key configuration.

2. **Ingestion & Serving Layer (FastAPI on Render):**
   - High-throughput asynchronous REST API built with FastAPI and Uvicorn.
   - Sub-5ms fast-path decisioning with single-threaded BLAS/OMP thread safety.
   - Cross-origin resource sharing (CORS) configured for smooth Vercel-to-Render communication.

3. **Feature Engineering & Velocity Engine:**
   - 456 features extracted using pre-allocated NumPy buffers for zero runtime allocation overhead.
   - Real-time online card velocity store measuring burst transactions across 10-minute, 30-minute, 1-hour, and 24-hour windows.

4. **Machine Learning Inference Pipeline:**
   - **Champion Classifier:** LightGBM gradient boosted decision tree (500 estimators, 101 leaves, learning rate 0.05).
   - **Probability Calibration:** Isotonic regression mapper reducing Expected Calibration Error from 0.0123 to 0.0045.
   - **Cost-Optimal Policy:** Dual operational thresholds ($t_{\text{review}} = 0.0496$, $t_{\text{block}} = 0.7534$) minimizing expected financial loss ($C_{\text{FP}} = \$10.00$, $C_{\text{FN}} = \text{TransactionAmt}$).
   - **Explainability:** Native TreeSHAP algorithm calculating exact Shapley values and top negative contribution codes.

5. **Persistence & Observability Plane:**
   - **Audit Store:** Relational persistence with SQLAlchemy supporting Neon PostgreSQL or local SQLite fallback.
   - **Telemetry:** Prometheus metrics exporter exposing request latency histograms, decision counters, and PSI drift gauges.
   - **Continuous Shadow Validation:** Asynchronous execution of candidate models to compare score divergence before production promotion.

---

## 📊 Measured Model Performance & Business Impact

All metrics are **empirically measured** on a strict chronological held-out test partition (88,581 transactions, 3,083 fraudulent, 3.48% prevalence):

### 1. Classification & Discrimination Performance
| Metric | Final Calibrated LightGBM | Logistic Regression Baseline | Dummy Baseline | Relative Gain |
| :--- | :--- | :--- | :--- | :--- |
| **ROC-AUC** | **0.9046** (95% CI: [0.8987, 0.9110]) | 0.8362 | 0.5000 | **+8.2% vs Logistic** |
| **PR-AUC (Average Precision)** | **0.5289** (95% CI: [0.5236, 0.5582]) | 0.3806 | 0.0343 | **15.4x vs Baseline (+39% vs Logistic)** |
| **Recall @ Precision = 50%** | **53.07%** | 33.83% | 0.03% | Intercepts majority of fraud at low false alert rate |
| **Recall @ Precision = 80%** | **34.19%** | 16.96% | 0.00% | High-confidence automated declines |
| **Recall @ Top 1% Alerts** | **25.56%** | 20.61% | 1.12% | 1 in 4 frauds flagged in the top 1% traffic |
| **Recall @ Top 5% Alerts** | **58.55%** | 46.15% | 6.15% | Captures nearly 60% of all frauds in top 5% |
| **Brier Score** | **0.0222** | 0.1255 | 0.0332 | Well-calibrated probabilities |
| **Expected Calibration Error (ECE)** | **0.0045** | 0.2691 | 0.0008 | **63.4% reduction in calibration error** |

### 2. Asymmetric Cost Matrix & Decision Policy
Thresholds were optimized by minimizing expected financial loss ($C_{\text{FP}} = \$10.00$ human review overhead, $C_{\text{FN}} = \text{Transaction Amount}$ chargeback loss):

$$\text{Expected Loss} = \sum_{\text{False Negatives}} \text{TransactionAmt} + \sum_{\text{False Positives}} \$10.00$$

| Decision Tier | Threshold | Action & Characteristics | Business Outcome |
| :--- | :--- | :--- | :--- |
| **APPROVE** | Score $< 0.0496$ | Immediate automated pass; zero friction | Captures 89.16% of legitimate traffic with no disruption |
| **MANUAL REVIEW** | $0.0496 \le \text{Score} < 0.7534$ | Enriched analyst queue (15.2% - 19.4% fraud rate, **5.6x base traffic rate**) | Intercepts **53.35%** of all fraud dollars ($250,555.41) |
| **AUTOMATED BLOCK** | Score $\ge 0.7534$ | High-precision automated decline (Target precision $\ge 90\%$) | Declines **19.82%** of fraud dollars ($93,066.30) with <1.3% traffic block |
| **COMBINED IMPACT** | — | Intercepts **72.49% of fraud transactions** and **73.17% of fraud dollars** ($343,621) | **$269,971.69 saved (57.49% net loss reduction)** |

### 3. Real-Time Serving Benchmarks
- **Fast-Path Latency (without SHAP)**: Server-side **p50: 1.40 ms**, **p95: 1.96 ms**; Client round-trip **p50: 3.59 ms**, **p95: 4.47 ms**.
- **Explainability Path (with TreeSHAP)**: Server-side **p50: 22.44 ms**, **p95: 40.39 ms**; Client round-trip **p50: 24.15 ms**, **p95: 43.05 ms**.
- **Reliability**: **0.0% error rate** under sustained load testing across 1, 2, and 4 worker configurations.

---

## 🚀 End-to-End Cloud Deployment Guide

Follow this guide to deploy the backend on **Render** and the frontend on **Vercel** with full end-to-end communication.

```mermaid
flowchart LR
    Vercel["Frontend (Vercel)<br/>https://fraud-app.vercel.app"] -->|HTTPS / JSON Requests| Render["Backend API (Render)<br/>https://fraud-api.onrender.com"]
    Render -->|Read/Write| Postgres[("PostgreSQL / Neon")]
    Render -->|Atomic ZSET| Redis[("Redis / Upstash")]
```

### Step 1: Deploy Backend to Render

1. Log in to [Render](https://dashboard.render.com/) and click **New + > Web Service**.
2. Connect your GitHub repository: `https://github.com/AshokSingodia-Codes/fraud-scoring-service`.
3. Configure the service settings:
   - **Name:** `fraud-scoring-api`
   - **Environment:** `Python`
   - **Region:** Choose the region closest to your users (e.g. `Oregon (US West)` or `Frankfurt (EU)`).
   - **Branch:** `main`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn src.fraud.serving.app:app --host 0.0.0.0 --port $PORT --workers 2`
   - **Health Check Path:** `/health`
4. Set Environment Variables in Render:
   - `APP_ENV`: `prod`
   - `ANYIO_MAX_THREADS`: `40`
   - `DATABASE_URL`: Your PostgreSQL connection string (e.g. from Render Postgres or [Neon](https://neon.tech/): `postgresql+psycopg://user:password@host/dbname`). If omitted, the service automatically uses local SQLite storage.
   - `REDIS_URL`: *(Optional)* Redis URL (e.g. from [Upstash](https://upstash.com/)). If omitted, in-memory rate limiting and velocity storage activate automatically.
5. Click **Create Web Service**.
6. Once deployed, note your backend URL: e.g. `https://fraud-scoring-api.onrender.com`. Test it by opening:
   ```bash
   curl https://fraud-scoring-api.onrender.com/health
   # Returns: {"status": "ok"}
   ```

### Step 2: Deploy Frontend to Vercel

1. Log in to [Vercel](https://vercel.com/) and click **Add New... > Project**.
2. Import the same GitHub repository: `AshokSingodia-Codes/fraud-scoring-service`.
3. Configure the Vercel project:
   - **Framework Preset:** `Vite`
   - **Root Directory:** If deploying from the root of the repo, leave root directory as `./` (the root `vercel.json` and `package.json` will build and publish automatically). Alternatively, set Root Directory to `frontend`.
   - **Build Command:** `npm run build`
   - **Output Directory:** `dist`
4. Add Environment Variables in Vercel:
   - `VITE_API_URL`: Your live Render backend URL from Step 1 (e.g. `https://fraud-scoring-api.onrender.com`).
5. Click **Deploy**.
6. Vercel will build the React application and deploy it to a URL like `https://fraud-scoring-service.vercel.app`.

### Step 3: Verify End-to-End Integration

1. Open your Vercel deployment in your browser.
2. The navbar status indicator in the top right will display a green indicator dot showing the backend is online.
3. If you ever update the backend URL or want to test against a different server, click the **Settings (gear icon)** in the navbar:
   - Enter your Render API URL.
   - Click **Test Connection** to verify live liveness probing.
   - Click **Save Settings** (persisted in browser `localStorage`).
4. Navigate to **Single Check**, select a preset transaction (or enter custom transaction details), and click **Check Transaction**:
   - The UI scores the transaction against your live Render backend, returning the decision badge, probability score, velocity counters, and interactive TreeSHAP feature contribution cards.
5. Search the transaction under **Audit Log** (`Find by ID`) or browse **Saved** to inspect historical transaction records.

---

## 📡 API Reference & Endpoints

All endpoints are open access for friction-free transaction inspection (no API keys, registration, or login required).

### 1. Health & Probes
- `GET /health` / `GET /healthz`: Liveness probe (`{"status": "ok"}`).
- `GET /ready`: Readiness probe verifying champion model and feature builder state (`{"status": "ready", "model_version": "1.0.0"}`).
- `GET /metrics`: Prometheus formatted metrics export.

### 2. Model Telemetry & Drift Monitoring
- `GET /v1/model/info`: Returns feature count, version, training date, and calibrated thresholds.
- `GET /v1/model/drift`: Real-time streaming Population Stability Index (PSI) and health status (`HEALTHY`, `MODERATE_DRIFT`, `CRITICAL_DRIFT`).
- `GET /v1/model/shadow`: Status and specification of Champion vs Challenger shadow model.

### 3. Scoring Endpoints
- `POST /v1/score?explain=true`: Real-time transaction scoring with SHAP feature explanations.
  ```bash
  curl -X POST "https://fraud-scoring-api.onrender.com/v1/score?explain=true" \
    -H "Content-Type: application/json" \
    -d '{
      "transaction_id": "tx_prod_001",
      "TransactionDT": 86400,
      "TransactionAmt": 350.00,
      "ProductCD": "W",
      "card1": 1000,
      "card2": 555.0,
      "addr1": 150.0,
      "P_emaildomain": "gmail.com"
    }'
  ```
- `POST /v1/score/batch?explain=false`: High-throughput batch scoring (up to 1,000 transactions per payload).

### 4. Audit Log CRUD Endpoints
- `GET /v1/predictions/{transaction_id}`: Retrieve latest audit log entry by ID.
- `GET /v1/predictions?limit=50&offset=0&decision=review`: Paginated list of historical audit logs with filtering.
- `POST /v1/predictions`: Manually insert an audit record.
- `PATCH /v1/predictions/{transaction_id}`: Update review decisions and analyst notes.
- `DELETE /v1/predictions/{transaction_id}`: Purge audit records by ID.

---

## 🛠️ Local Development Quickstart

```bash
# 1. Clone repository
git clone https://github.com/AshokSingodia-Codes/fraud-scoring-service.git
cd fraud-scoring-service

# 2. Setup Python environment
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt -r requirements-dev.txt

# 3. Run all unit and integration tests
.venv/Scripts/pytest tests/

# 4. Start backend server (defaults to localhost:8000 with local SQLite audit DB)
.venv/Scripts/uvicorn src.fraud.serving.app:app --host 127.0.0.1 --port 8000

# 5. Start React frontend
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 3000
```
Open **`http://localhost:3000`** in your browser.

---

## 💼 Resume / CV Project Presentation

### Machine Learning / MLOps Engineer Bullet Points

> **Real-Time Fraud Detection & Decision Engine** | *Python, FastAPI, LightGBM, Scikit-learn, Redis, PostgreSQL, Prometheus, Docker, React 19*
> - Architected and deployed an enterprise-grade real-time fraud scoring microservice processing payment transactions with **sub-5ms fast-path p95 latency** (<45ms with full SHAP explanations) and **0.0% error rate** under load testing.
> - Engineered 456 numerical, interaction, frequency, and sliding-window online card velocity features trained on 590K+ IEEE-CIS transactions using a strict **leakage-free 70/15/15 chronological split**.
> - Optimized gradient boosted trees (LightGBM) achieving **0.9046 ROC-AUC** and **0.5289 PR-AUC** on held-out test data (a 15.4x improvement over baseline), calibrating probabilities via Isotonic Regression to reduce Expected Calibration Error (ECE) by **63.4%** (to 0.0045).
> - Designed an asymmetric cost-optimization decision matrix establishing dual thresholds (`t_review` = 0.0496, `t_block` = 0.7534) that intercepted **73.17% of fraudulent dollars** ($343,621) and achieved a **57.49% direct chargeback loss reduction** ($269,972 net savings).
> - Built native TreeSHAP reason code extraction, an online sliding-window velocity store, streaming PSI data drift detection, champion/challenger shadow routing, and a full RESTful audit logging system with Prometheus metrics and a React 19 analytics dashboard.

### 1-Line Resume Hook:
> *Built an end-to-end real-time fraud scoring service (LightGBM, Isotonic calibration, cost-optimized decisioning) serving predictions at sub-5ms p95 latency, intercepting 73.17% of fraudulent transaction dollars and reducing financial chargeback loss by 57.49%.*

---

## 📜 License
This project is licensed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
