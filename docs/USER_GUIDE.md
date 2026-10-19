# Production User Guide & Operational Walkthrough

A practical, step-by-step operational guide for developers, ML engineers, and risk analysts using the **Real-Time Fraud Scoring & Streaming ML Platform**.

---

## 1. Quickstart: Launching the Stack

You can launch the complete system locally with either Docker Compose or direct processes.

### Option A: Complete Docker Compose Mesh (Recommended)
Spins up the FastAPI scoring engine, Postgres database, Redis velocity store, Prometheus scraper, and auto-provisioned Grafana dashboards:

```bash
docker compose up --build -d
```

| Service | Access URL | Default Credentials / Notes |
|---|---|---|
| **FastAPI Backend** | `http://localhost:8000` | Docs at `/docs`, Health at `/health` |
| **Grafana ML Dashboards** | `http://localhost:3000` | User: `admin` / Password: `admin` |
| **Prometheus Scraper** | `http://localhost:9090` | Scrapes `/metrics` every 5 seconds |
| **Redis Store** | `localhost:6379` | Sliding window rate limits & velocity cache |
| **PostgreSQL Database** | `localhost:5432` | User: `postgres`, Pass: `postgres`, DB: `fraud_db` |

---

### Option B: Local Developer Execution

#### 1. Backend Server
```bash
# In project root with .venv activated
export APP_ENV=dev
uvicorn src.fraud.serving.app:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Frontend React Dashboard
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 3000
```
Open **`http://localhost:3000`** in your browser.

---

## 2. Frontend Features Walkthrough

The web dashboard provides an interactive, production-grade interface designed for risk operations and ML evaluation:

### 1. Header Navigation & Health Bar
- **System Health Pill**: Top right displays real-time health (`Live` / `Degraded` / `Offline`) checking the backend `/ready` probe.
- **Settings Modal**: Allows configuring custom Backend API URLs and API Keys (defaults to `demo-api-key-12345`).

### 2. Single Transaction Check (`CheckOnePage`)
- **Quick-Fill Buttons**: Load one-click realistic test cases:
  - *Standard Safe Purchase*: Typical card transaction, small amount, trusted email domain.
  - *Suspicious Fraud Pattern*: High amount, burner email, unusual device profile.
- **Real-Time Risk Scoring**:
  - Displays calibrated **Fraud Probability** (e.g. `0.0758` = 7.58%).
  - Shows cost-optimal **Decision Tier**: `APPROVE` (Green), `MANUAL REVIEW` (Amber), or `BLOCK` (Red).
  - Displays **Risk Band**: `LOW`, `MEDIUM`, or `HIGH`.
- **Top SHAP Contributing Factors**:
  - Plain-English breakdown of why the model flagged or cleared the transaction (e.g. `Card1 Frequency`, `Amount Cents Part`, `Email Domain Match Flag`).
  - Color-coded: Red badges indicate factors raising risk; green badges indicate trust signals.
- **Live Velocity Store Panel**:
  - Displays sliding-window card velocity metrics computed via Redis:
    - **10 Minutes**: Count & Total Amount ($)
    - **1 Hour**: Count & Total Amount ($)
    - **24 Hours**: Count & Total Amount ($)
- **One-Click Actions**:
  - **Save to Vault**: Stores audit snapshot in local storage with custom analyst notes.
  - **Download PDF**: Exports a branded, printable one-page risk report for regulatory audit trails.

### 3. Bulk Batch Scoring (`CheckBulkPage`)
- Paste CSV or JSON arrays containing up to 1,000 transactions at once.
- Evaluates transactions in high-throughput batch mode.
- Summary analytics bar displays:
  - Approval Rate (%)
  - Manual Review Queue Count
  - Auto-Blocked Percentage
  - Mean Processing Latency

### 4. Audit Log Vault (`SavedReportsPage` & `FindByIdPage`)
- Search any transaction ID to query historical audit logs saved in PostgreSQL/SQLite.
- View archived probabilities, decisions, and exact timestamp recorded at time of decision.

### 5. Model Specifications & Live Drift (`ModelInfoPage`)
- Inspect model architecture (LightGBM GBDT), active feature count (40+ features), and threshold policies ($10 review cost vs preventing chargebacks).
- **Live Population Stability Index (PSI) Panel**:
  - Real-time distribution shift monitoring against 590K baseline training records.
  - Color-coded status indicator:
    - **`HEALTHY`** (PSI < 0.10)
    - **`MODERATE_DRIFT`** (0.10 ≤ PSI < 0.25)
    - **`CRITICAL_DRIFT`** (PSI ≥ 0.25)
  - Live gauges for `TransactionAmt`, `card1`, `addr1`, and `card2`.

---

## 3. Backend API Reference & Curl Walkthrough

### Authenticating
All protected endpoints require the `X-API-Key` header:
```bash
-H "X-API-Key: demo-api-key-12345"
```

### 1. Live Scoring with Dynamic Velocity
```bash
curl -X POST "http://localhost:8000/v1/score?explain=true" \
  -H "X-API-Key: demo-api-key-12345" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "tx_prod_9001",
    "TransactionDT": 86400,
    "TransactionAmt": 125.00,
    "ProductCD": "W",
    "card1": 1000,
    "card2": 555.0,
    "P_emaildomain": "gmail.com"
  }'
```

**Example JSON Response:**
```json
{
  "transaction_id": "tx_prod_9001",
  "fraud_probability": 0.0758,
  "decision": "review",
  "risk_band": "MEDIUM",
  "thresholds_used": { "t_review": 0.15, "t_block": 0.85 },
  "latency_ms": 1.45,
  "model_version": "0.1.0",
  "reasons": [
    {
      "feature": "TransactionAmt",
      "contribution": 0.21,
      "direction": "increases_risk",
      "label": "Transaction Amount"
    }
  ],
  "velocity_features": {
    "card1_tx_cnt_10m": 0.0,
    "card1_amt_sum_10m": 0.0,
    "card1_tx_cnt_1h": 0.0,
    "card1_amt_sum_1h": 0.0,
    "card1_tx_cnt_24h": 0.0,
    "card1_amt_sum_24h": 0.0
  }
}
```

### 2. Inspecting Model Drift Status
```bash
curl -X GET "http://localhost:8000/v1/model/drift" \
  -H "X-API-Key: demo-api-key-12345"
```

**Example JSON Response:**
```json
{
  "window_size": 1000,
  "current_samples": 42,
  "mean_fraud_probability": 0.076,
  "status": "HEALTHY",
  "feature_psi": {
    "TransactionAmt": 0.012,
    "card1": 0.008,
    "addr1": 0.005
  },
  "critical_features": []
}
```

### 3. Scraping Prometheus Observability Metrics
```bash
curl -X GET "http://localhost:8000/metrics"
```
Exports `fraud_requests_total`, `fraud_api_request_latency_seconds`, `fraud_prediction_score`, `fraud_feature_psi`, and `fraud_model_health_status`.

---

## 4. Troubleshooting & FAQ

### Issue: "Invalid API key" (403 Forbidden)
- **Cause**: The `X-API-Key` sent does not match `ALLOWED_API_KEYS`.
- **Fix**: Open the **Settings** modal in the frontend (gear icon in navbar) and ensure the API Key is set to `demo-api-key-12345`. When testing via terminal, ensure `APP_ENV=dev` is exported.

### Issue: "Cannot reach fraud service" / CORS Error
- **Cause**: Backend server is not running on port 8000 or the URL in frontend settings is incorrect.
- **Fix**: Check `curl http://localhost:8000/health`. Ensure the API Base URL in Settings is `http://localhost:8000`.

### Issue: Redis Unavailable in Local Dev
- **Cause**: Local environment doesn't have a running Redis instance.
- **Graceful Behavior**: The system automatically switches to the in-memory fallback stores (`InMemoryVelocityStore` and `InMemorySlidingWindow`) without dropping requests or causing errors.
