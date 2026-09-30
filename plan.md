# PLAN: Real-Time Fraud Scoring Service (IEEE-CIS Fraud Detection)

An end-to-end build plan: data, features, models, evaluation, API, monitoring, CI/CD and deployment.
It is written so a human, or an AI coding agent (for example Antigravity), can execute it stage by stage.

---

## 0. How to use this file with an AI IDE (read first)

### 0.1 Master prompt (paste this to the agent, then attach this file)

```
You are building the project described in plan.md. Work strictly stage by stage.
Rules:
1. Complete one stage at a time. At the end of each stage, run the stage's
   acceptance checks and tests, then stop and print a short status report.
   Do not start the next stage until the checks pass.
2. Never invent or hard-code metrics. Every number in README, reports and
   model card must come from JSON files in reports/ produced by code.
3. Never commit the raw dataset, API keys, tokens or .env files.
4. Use a time-based split only. Never shuffle rows across time.
5. Every fitted object (encoders, medians, calibrators) is fit on TRAIN ONLY.
6. Training and serving must call the same FeatureBuilder code.
7. If something in the plan is impossible or ambiguous, ask me instead of guessing.
8. Keep functions small, typed and tested. Use ruff for lint and pytest for tests.
```

### 0.2 Per-stage prompt template

```
Execute Stage N of plan.md ("<stage title>"). Create exactly the files listed
in that stage, implement the tasks in order, run the acceptance checks, and
report: files changed, commands run, test results, and the measured metrics.
Stop after the report.
```

### 0.3 Things only the human can do
- Create a Kaggle account, accept the IEEE-CIS competition rules, and create an API token (Stage 1).
- Create accounts for the deployment services (Stage 13).
- Review every metric before it goes on the resume.

---

## 1. Project overview

**Goal:** a production-style fraud scoring service. It takes a card transaction, returns a calibrated fraud probability, an approve / review / block decision, and the top reasons behind the score.

**Dataset:** Kaggle IEEE-CIS Fraud Detection (https://www.kaggle.com/c/ieee-fraud-detection).
- `train_transaction.csv` (about 590k rows, about 3.5% fraud) and `train_identity.csv` (about 144k rows, so identity data exists for only some transactions). Verify exact counts in Stage 2.
- `TransactionDT` is a time offset in seconds (not a real date). The training data spans roughly six months.
- Kaggle test labels are hidden. **You will split the labelled training data by time** into your own train, validation and test sets.

**Why this is a strong resume project:** imbalanced data, leakage-safe time splits, calibration, cost-based thresholds, explainability, drift checks, and a tested, containerized, monitored API.

**Success criteria (definition of done at the end):**
- [ ] PR-AUC and ROC-AUC reported on a time-held-out test set, with baseline comparison
- [ ] Calibrated probabilities (Brier score and reliability plot reported)
- [ ] Cost-optimal threshold with a documented cost assumption
- [ ] Per-prediction explanations returned by the API
- [ ] API load-tested with throughput and p95 latency reported
- [ ] Docker Compose stack (API, Postgres, Redis, Prometheus, Grafana, MLflow)
- [ ] CI passing on GitHub Actions
- [ ] Public live demo URL, model card, and evaluation report

---

## 2. Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.11 |
| Data | pandas, pyarrow (Parquet), numpy |
| Modeling | scikit-learn, LightGBM, Optuna (tuning) |
| Explainability | LightGBM native SHAP (`pred_contrib=True`), shap for plots |
| Experiment tracking | MLflow |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Storage | Postgres (prediction log, API keys), Redis (cache, rate limit) |
| Monitoring | prometheus-client, Prometheus, Grafana |
| Testing | pytest, httpx, Locust (load test) |
| Quality | ruff, pre-commit |
| Packaging | Docker, Docker Compose |
| CI/CD | GitHub Actions |
| Demo UI | Streamlit |
| Deployment | Render (API), Neon (Postgres), Upstash (Redis), Streamlit Community Cloud or Hugging Face Spaces (UI) |

Pin exact versions after the first successful install (`pip freeze > requirements.lock`).

---

## 3. Repository structure

```
fraud-scoring-service/
├── README.md
├── plan.md
├── Makefile
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore                  # data/, mlruns/, *.parquet, .env, *.csv
├── Dockerfile
├── docker-compose.yml
├── render.yaml
├── .github/workflows/ci.yml
├── configs/config.yaml
├── data/                       # gitignored
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   └── 01_eda.ipynb
├── src/fraud/
│   ├── __init__.py
│   ├── config.py
│   ├── data/        download.py  ingest.py  split.py
│   ├── features/    builder.py
│   ├── models/      baseline.py  train_lgbm.py  tune.py  calibrate.py  threshold.py
│   ├── evaluation/  metrics.py  temporal.py  drift.py
│   ├── explain/     reasons.py
│   └── serving/     app.py  schemas.py  predictor.py  auth.py  ratelimit.py
│                    db.py  cache.py  metrics.py
├── scripts/
│   ├── download_data.sh
│   ├── train.py
│   ├── evaluate.py
│   ├── make_report.py
│   └── locustfile.py
├── models/                     # small artifacts (model.txt, joblibs, json)
├── reports/                    # metrics_*.json, figures/, model_card.md, evaluation_report.md
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── app_ui/streamlit_app.py
└── tests/
    ├── conftest.py
    ├── fixtures/               # tiny synthetic data only
    ├── test_split.py
    ├── test_no_leakage.py
    ├── test_features.py
    ├── test_metrics.py
    ├── test_predictor.py
    └── test_api.py
```

**Important:** the competition rules do not allow redistributing the dataset. Never commit raw or processed data. Provide a download script instead. Test fixtures must be synthetic.

---

## 4. Timeline

| Path | Stages | Approx. days |
|---|---|---|
| **MVP path** | 0, 1, 3, 4, 6, 7, 10, 12, 13, 14 | 6-7 |
| **Full path** | 0 to 14 | 12-14 |

Do the MVP path first so you have something deployable, then add the remaining stages.

---

# STAGES

Each stage lists: goal, tasks, files, acceptance checks (the gate), and what results to expect.

---

## Stage 0: Repo, environment and tooling (0.5 day)

**Goal:** a clean, reproducible skeleton.

**Tasks**
1. Create the repo `fraud-scoring-service` and initialize git.
2. Create the folder structure from Section 3 with empty `__init__.py` files.
3. Create a virtual environment (`python -m venv .venv`) and install dependencies:
   `pandas pyarrow numpy scikit-learn lightgbm optuna mlflow fastapi uvicorn[standard] pydantic pydantic-settings sqlalchemy psycopg[binary] redis prometheus-client httpx streamlit matplotlib seaborn pyyaml joblib`
   Dev: `pytest pytest-cov ruff pre-commit locust kaggle`
4. Write `requirements.txt` and `requirements-dev.txt`.
5. Write `.gitignore` (data, models cache, `.env`, `mlruns/`, `.venv/`, `*.parquet`, `*.csv`).
6. Write `configs/config.yaml` with: paths, random seed (42), split fractions (0.70 / 0.15 / 0.15), cost assumptions (FP review cost, FN cost = transaction amount), thresholds placeholders, model version string.
7. Write `src/fraud/config.py` that loads the YAML and environment variables.
8. Write a `Makefile` with targets: `setup`, `data`, `train`, `eval`, `test`, `lint`, `serve`, `docker-up`, `load-test`.
9. Add ruff config and a pre-commit hook.
10. Add a placeholder test and make sure `pytest` runs.

**Acceptance checks**
- [ ] `make lint` and `make test` pass on the empty skeleton
- [ ] `git status` shows no data or secret files

---

## Stage 1: Data download and ingestion (0.5 day)

**Goal:** raw CSVs converted into compact, typed Parquet files.

**Human step:** accept the competition rules at the Kaggle page, create an API token, and place `kaggle.json` in `~/.kaggle/` (permissions 600).

**Tasks**
1. `scripts/download_data.sh`: `kaggle competitions download -c ieee-fraud-detection -p data/raw` then unzip.
2. `src/fraud/data/ingest.py`:
   - Read `train_transaction.csv` and `train_identity.csv`.
   - Left-join on `TransactionID`.
   - Downcast numeric columns (`float64` to `float32`, integers to the smallest safe type).
   - Convert string columns to pandas `category`.
   - Save `data/interim/train_merged.parquet`.
   - Print shape, memory usage, fraud rate, and column groups.
3. Save a `reports/data_profile.json` with row count, column count, fraud rate, and the share of rows with identity data.
4. Add `make data`.

**Acceptance checks**
- [ ] Parquet loads in a few seconds
- [ ] Row count matches the raw transaction file
- [ ] Memory usage is well below the raw CSV footprint

**Expected results:** about 590k rows and 430+ columns after merging; fraud rate about 3.5%; identity fields missing for most rows. Record the actual values in `data_profile.json`.

---

## Stage 2: Exploratory data analysis (0.5-1 day)

**Goal:** understand the data before modeling and document what matters.

**Tasks** (in `notebooks/01_eda.ipynb`, export key figures to `reports/figures/`)
1. Class balance and fraud rate over time (weekly).
2. `TransactionDT` distribution; convert to day index (`DT // 86400`) and hour of day.
3. Missingness heatmap by column group: `C*`, `D*`, `M*`, `V*`, `id_*`, `card*`, `addr*`.
4. `TransactionAmt` distribution (log scale) split by fraud class.
5. Fraud rate by `ProductCD`, `card4`, `card6`, `DeviceType`, email domain.
6. Cardinality of categorical columns; list high-cardinality ones (`card1`, `card2`, `addr1`, `P_emaildomain`, `DeviceInfo`).
7. Distribution shift check: compare first-month vs last-month means of a few features.
8. Write `reports/eda_summary.md` with 8-10 findings and the modeling decisions they imply.

**Acceptance checks**
- [ ] EDA summary lists which columns will be dropped (for example those over 90% missing and non-informative), which will be encoded, and why

---

## Stage 3: Time-based split and validation framework (0.5 day)

**Goal:** a leakage-safe evaluation setup used by every later stage.

**Tasks**
1. `src/fraud/data/split.py`: sort by `TransactionDT`; first 70% = train, next 15% = validation, final 15% = test. Save the boundaries in `reports/split_info.json`.
2. Add an expanding-window CV helper (for example 4 folds by time) for hyperparameter tuning.
3. `tests/test_split.py` and `tests/test_no_leakage.py`:
   - max train time < min validation time < min test time
   - no duplicated `TransactionID` across splits
   - fraud rate per split is printed and stored

Snippet:
```python
df = df.sort_values("TransactionDT").reset_index(drop=True)
n = len(df)
train = df.iloc[: int(0.70 * n)]
val   = df.iloc[int(0.70 * n): int(0.85 * n)]
test  = df.iloc[int(0.85 * n):]
```

**Acceptance checks**
- [ ] Leakage tests pass
- [ ] Split info JSON saved

**Rule from here on:** the test set is used once, at the end of Stage 6 and again in Stage 9 for the final report. All tuning uses validation or time-CV only.

---

## Stage 4: Baseline models (0.5 day)

**Goal:** honest baselines to compare against.

**Tasks**
1. Dummy baselines: predict prior probability (PR-AUC equals the fraud rate).
2. Logistic regression baseline: median-impute numerics, one-hot or frequency-encode a few low-cardinality categoricals, standard scale, `class_weight="balanced"`.
3. LightGBM with default parameters on raw features (categoricals as `category`).
4. `src/fraud/evaluation/metrics.py`, implementing:
   - PR-AUC (average precision), ROC-AUC
   - Recall at fixed precision (0.5, 0.8)
   - Recall when flagging the top 1% and 5% of transactions
   - Brier score and expected calibration error (ECE)
   - Expected cost at a given threshold
5. Save validation metrics to `reports/metrics_baselines.json`.
6. Log all runs in MLflow.

**Acceptance checks**
- [ ] Unit tests for metrics on toy arrays pass
- [ ] Baseline table exists

**Expected results (approximate, validation set, time split; treat as sanity ranges, not promises)**

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Dummy | 0.50 | about the fraud rate (0.03-0.04) |
| Logistic regression | 0.80-0.87 | 0.20-0.40 |
| LightGBM default | 0.88-0.93 | 0.45-0.65 |

**Red flag:** ROC-AUC above about 0.98 on a time split means leakage. Stop and investigate.

---

## Stage 5: Feature engineering (1-1.5 days)

**Goal:** a single `FeatureBuilder` that is fit on train only and reused at serving time.

**Design rule:** only features computable from the single incoming transaction plus static tables learned from train. No features that need other live transactions or history, or you cannot serve them in real time without a feature store.

**Tasks** (`src/fraud/features/builder.py`, class `FeatureBuilder` with `fit(train_df)` and `transform(df)`)
1. **Time features:** day index, hour of day, day of week from `TransactionDT`.
2. **Amount features:** log amount, cents part (`amt % 1`), amount is round-number flag.
3. **Email features:** split `P_emaildomain` and `R_emaildomain` into provider and suffix; flag when purchaser and recipient domains match.
4. **Frequency encoding** (counts learned on train only) for `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `P_emaildomain`, `DeviceInfo`, `id_30`, `id_31`, `id_33`; unseen values map to 0.
5. **Categorical handling:** fixed category lists learned on train; unseen values become NaN.
6. **Column selection:** drop columns flagged in the EDA (very high missingness, constants). Keep `V*` columns initially, then test dropping highly correlated ones (correlation above 0.95) and compare validation PR-AUC.
7. **Missingness flags:** count of missing values in `id_*` block, and an `has_identity` flag.
8. **Card-and-address combination key** as a frequency-encoded feature (for example `card1 + addr1` as a string). Compute counts on train only. Compare validation PR-AUC with and without it.
9. Serialize with joblib to `models/feature_builder.joblib`; also write `models/feature_spec.json` (final feature names, dtypes, category lists).
10. `tests/test_features.py`: same input gives same output; unseen category is handled; missing columns become NaN; output columns and order match the spec.

**Ablation table** (`reports/metrics_feature_ablation.json`): validation PR-AUC for raw features, plus each feature group added in turn.

**Acceptance checks**
- [ ] `fit` uses train only (asserted in a test)
- [ ] Transform of a single row takes a few milliseconds
- [ ] Ablation shows which groups help

**Expected results:** modest but real gains over the raw-feature LightGBM (for example +0.01 to +0.04 PR-AUC). If a feature gives a huge jump, check for leakage.

---

## Stage 6: LightGBM training, tuning and tracking (1 day)

**Goal:** the best model under time-based validation.

**Tasks**
1. `train_lgbm.py`: train on train, early-stop on validation, `objective="binary"`, metric `average_precision`. Try `scale_pos_weight` and compare with no weighting (weighting can hurt calibration; you will calibrate in Stage 7).
2. `tune.py`: Optuna, 30-60 trials, objective = mean PR-AUC across the expanding-window time CV folds. Search: `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l1`, `lambda_l2`, `max_bin`.
3. Optional comparison: XGBoost or CatBoost with the same features, reported in the same table.
4. Refit the best config on train (keep validation for calibration and thresholds; do not train on validation).
5. Save `models/model.txt` (LightGBM native format) and log everything in MLflow (params, metrics, feature importance, model file).
6. **Evaluate on the test set once** and store `reports/metrics_test_stage6.json`.

**Acceptance checks**
- [ ] MLflow shows all runs with params and metrics
- [ ] Test metrics saved once and not used to tune anything

**Expected results (approximate, time-held-out test)**

| Metric | Expected range |
|---|---|
| ROC-AUC | 0.90-0.95 |
| PR-AUC | 0.55-0.75 |
| Recall when flagging top 1% of transactions | 0.35-0.55 |

For reference, top Kaggle solutions reached roughly 0.96 ROC-AUC on the hidden test set using heavy group-based features that are not real-time servable. Your numbers will likely be a little lower under a strict time split, and that is fine. Report them honestly.

---

## Stage 7: Calibration and decision thresholds (0.5 day)

**Goal:** probabilities that mean something, and thresholds tied to business cost.

**Tasks**
1. `calibrate.py`: fit isotonic regression (and compare Platt scaling) on the validation set predictions. Save `models/calibrator.joblib`.
2. Report before and after: Brier score, ECE, reliability diagram (`reports/figures/calibration.png`). Evaluate calibration on the test set.
3. `threshold.py`: compute two thresholds on validation data.
   - **Review threshold `t_review`:** minimizes expected cost, where cost = FN loss (transaction amount) + FP cost (fixed manual-review cost from config, an assumption you must state).
   - **Block threshold `t_block`:** the lowest threshold with precision at or above 0.90 on validation.
4. Save `models/thresholds.json`: `{"t_review": ..., "t_block": ..., "cost_assumptions": {...}}`.
5. Report on test: precision, recall, alert rate and total cost at each threshold, compared with "approve everything".
6. Decision policy: score below `t_review` = **approve**; between = **review**; at or above `t_block` = **block**.

**Acceptance checks**
- [ ] Calibration improves Brier score or ECE on validation and test
- [ ] Thresholds chosen on validation only; test used for reporting

**Expected results:** ECE drops noticeably after calibration (often by more than half). At `t_block`, test precision should be close to 0.9 (some drift from validation is normal). The cost-optimal policy should beat "approve everything" in total cost; if it does not, revisit the cost assumptions.

---

## Stage 8: Explainability (0.5 day)

**Goal:** per-prediction reasons that are cheap enough for real-time serving.

**Tasks**
1. Use LightGBM's native contributions: `booster.predict(X, pred_contrib=True)` returns per-feature contributions and a bias term (no `shap` package needed at serving time).
2. `explain/reasons.py`: return the top 5 features by absolute contribution, with feature name, value and direction (raises or lowers risk). Map cryptic feature names to readable labels using a small dictionary (`V*` columns stay as-is and are marked "anonymized").
3. Offline: global SHAP summary and bar plots on a sample of 10-20k validation rows (`reports/figures/shap_summary.png`).
4. Sanity check: the sum of contributions plus bias matches the raw model score (test this).

**Acceptance checks**
- [ ] Contribution-sum test passes
- [ ] Explanation for one row takes only a few milliseconds

---

## Stage 9: Robustness, drift and final evaluation (0.5-1 day)

**Goal:** show the model behaves over time and document weaknesses.

**Tasks**
1. **Temporal degradation:** compute PR-AUC per month of the test period (or per week) and plot (`reports/figures/temporal_pr_auc.png`).
2. **Drift:** implement PSI (Population Stability Index) in `evaluation/drift.py`. Compute PSI between train and test for the top 20 features and for the model score. Flag PSI above 0.2.
3. **Segment analysis:** PR-AUC and recall by `ProductCD`, by card type and by amount bucket. Note weak segments.
4. **Ablation summary:** baseline vs. engineered vs. tuned vs. calibrated, in one table.
5. **Error analysis:** 20 highest-scoring false positives and 20 missed frauds; write 5 observations.
6. `scripts/evaluate.py` produces `reports/evaluation_report.md` from the JSON files (no hand-typed numbers).

Snippet (PSI):
```python
def psi(expected, actual, bins=10):
    edges = np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    e = np.histogram(expected.dropna(), edges)[0] / max(expected.notna().sum(), 1)
    a = np.histogram(actual.dropna(), edges)[0] / max(actual.notna().sum(), 1)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))
```

**Acceptance checks**
- [ ] Evaluation report generated by code
- [ ] Known weaknesses listed honestly (they go into the model card)

**Expected results:** some performance decline in later months is normal. Score PSI should be small to moderate; a few features may exceed 0.2, which shows why drift monitoring matters.

---

## Stage 10: Inference API (1.5 days)

**Goal:** a fast, safe, tested scoring service.

**Tasks**
1. `serving/predictor.py`: loads `feature_builder.joblib`, `model.txt`, `calibrator.joblib` and `thresholds.json` once at startup. Method `score(dict) -> result` that builds features, predicts, calibrates, applies the decision policy and computes top reasons.
2. `serving/schemas.py` (Pydantic v2):
   - `Transaction`: `transaction_id` (str, required), `TransactionDT` (int), `TransactionAmt` (float, over 0), and optional raw fields (`ProductCD`, `card1`...`card6`, `addr1`, `addr2`, `P_emaildomain`, `R_emaildomain`, `C*`, `D*`, `M*`, `DeviceType`, `DeviceInfo`, `id_*`, `V*`). Missing fields become NaN. Reject unknown types cleanly.
   - `ScoreResponse`: `transaction_id`, `fraud_probability`, `decision`, `risk_band`, `reasons[]`, `model_version`, `thresholds`, `latency_ms`.
3. `serving/app.py` endpoints:
   - `GET /health` (liveness) and `GET /ready` (model loaded, DB and cache reachable)
   - `GET /metrics` (Prometheus)
   - `GET /v1/model/info` (version, training date, thresholds, feature count, headline metrics from the JSON reports)
   - `POST /v1/score` single transaction
   - `POST /v1/score/batch` up to 1,000 transactions
   - `GET /v1/predictions/{transaction_id}` (from Postgres)
4. **Auth:** `X-API-Key` header; hashed keys stored in Postgres (or an env list for the free deployment).
5. **Rate limiting:** token bucket in Redis, in-memory fallback when Redis is not configured.
6. **Idempotency:** optional `Idempotency-Key` header; a repeated key returns the stored response.
7. **Caching:** cache by transaction id and payload hash in Redis with a TTL.
8. **Persistence:** log each prediction (id, timestamp, score, decision, model version, latency, hashed API key, a subset of input fields) to Postgres via SQLAlchemy. Never log full payloads with personal data.
9. **Graceful degradation:** if Redis or Postgres is unavailable, scoring still works and the failure is logged (needed for the free deployment).
10. **Prometheus metrics:** request count by endpoint and status, latency histogram, score histogram, decision counts, cache hit ratio, model version info gauge.
11. **Structured JSON logging** with a request id middleware.
12. **Error handling:** consistent JSON errors, timeouts, and input size limits.
13. Configuration through environment variables via `pydantic-settings`.

**Tests**
- `test_predictor.py`: loads a tiny synthetic model fixture; missing-field payload works; unseen category works.
- `test_api.py` (httpx TestClient): auth required, invalid payload gives 422, batch limit enforced, idempotent repeat, `/ready` behaviour, rate limit returns 429.
- A **training-serving parity test**: features built by the batch pipeline and by the single-row path are identical for the same rows.

**Acceptance checks**
- [ ] All API tests pass
- [ ] Manual `curl` of `/v1/score` returns a sensible JSON response

**Targets (design goals, measure them in Stage 11):** p95 latency of a single score under 50 ms without explanations on a laptop-class CPU, and a few hundred requests per second per worker on the batch path.

---

## Stage 11: Testing, load testing and monitoring (1 day)

**Goal:** prove reliability with numbers.

**Tasks**
1. Raise unit test coverage on `src/fraud` to at least 80% (`pytest --cov`).
2. `scripts/locustfile.py`: 90% `/v1/score` traffic, 10% `/v1/score/batch`; realistic payloads sampled from a synthetic generator (not raw data).
3. Run three load profiles against the Docker Compose stack: 10, 50 and 100 concurrent users, 2 minutes each. Record requests per second, median and p95/p99 latency, error rate. Save to `reports/load_test.json` and a summary table.
4. Compare latency with explanations on vs. off (a query flag `?explain=true`, default on for single scores, off for batch).
5. `monitoring/prometheus.yml` scrapes `/metrics`. Build a Grafana dashboard JSON with panels: request rate, p95 latency, error rate, score distribution, decision mix, cache hit ratio.
6. **Drift monitor job** (`scripts/drift_check.py`): compares the recent score distribution in Postgres with the training score distribution using PSI and exposes the result as a Prometheus gauge.
7. Save dashboard screenshots to `reports/figures/`.

**Acceptance checks**
- [ ] Load test JSON exists and error rate is under 1%
- [ ] Grafana dashboard shows live data during the load test

**Expected results (depends on your machine; report what you measure):** single-score p95 in the tens of milliseconds without explanations; explanations add a few extra milliseconds; throughput scales roughly with the number of worker processes until CPU saturates.

---

## Stage 12: Docker, Compose and CI/CD (0.5-1 day)

**Goal:** one-command local stack and automated checks.

**Tasks**
1. `Dockerfile`: multi-stage; slim Python base; install only runtime dependencies; copy `src/` and `models/`; non-root user; `HEALTHCHECK`; run with `uvicorn --workers 2`.
2. `docker-compose.yml` services: `api`, `postgres`, `redis`, `prometheus`, `grafana`, `mlflow` (optional), `ui`. Use named volumes and a `.env` file.
3. Database init: SQLAlchemy `create_all` at startup or a simple migration script.
4. `.github/workflows/ci.yml`: on push and pull request: install, `ruff check`, `pytest --cov` (uses only synthetic fixtures), build the Docker image, run a container smoke test hitting `/health`.
5. Optional CD job: on tags, build and push the image to GitHub Container Registry.
6. Add status badges to the README.

**Acceptance checks**
- [ ] `docker compose up` starts the full stack and `/ready` returns OK
- [ ] CI is green on GitHub

---

## Stage 13: Deployment and live demo (1 day)

**Goal:** a public URL an interviewer can open.

**Suggested free-tier setup**
- **API:** Render web service (Docker) using `render.yaml`.
- **Postgres:** Neon free tier (`DATABASE_URL` env var).
- **Redis:** Upstash free tier (`REDIS_URL`), or leave unset to use the in-memory fallback.
- **UI:** Streamlit app on Streamlit Community Cloud or a Hugging Face Space.

**Tasks**
1. Bundle `models/` in the image (a LightGBM model is a few MB). Do not ship data.
2. `render.yaml`: service definition, environment variables, health check path `/health`.
3. Set secrets in the hosting dashboard (never in the repo): API keys, `DATABASE_URL`, `REDIS_URL`.
4. Deploy, then verify `/ready`, `/docs` and a scored request with `curl`.
5. Build `app_ui/streamlit_app.py`: a form with the main fields (amount, product, card type, email domain, device type) plus a JSON editor; shows probability, decision, risk band and a bar chart of the top reasons; includes a few synthetic sample transactions. Add a clear "demo only" notice.
6. Note the free-tier behaviour in the README (cold starts after inactivity, ephemeral disks). Prediction logging goes to Neon so it survives restarts.
7. Prometheus and Grafana run locally in Compose; the deployed API exposes `/metrics`. Use dashboard screenshots in the README.

**Acceptance checks**
- [ ] Public API `/docs` opens
- [ ] Demo UI scores a sample transaction end to end
- [ ] README lists the live links

---

## Stage 14: Documentation, model card and resume (0.5 day)

**Tasks**
1. `scripts/make_report.py` fills README tables from `reports/*.json` (never typed by hand).
2. **README sections:** summary and live links; architecture diagram (Mermaid); dataset and split description; results tables (baselines, ablation, final, calibration, thresholds and cost); load test table; how to run locally; API examples with `curl`; limitations.
3. `reports/model_card.md`: intended use, training data, split, metrics, calibration, thresholds and cost assumptions, explanations, known weaknesses (weak segments, drift), ethical notes (no protected attributes; anonymized features), and out-of-scope uses.
4. Pin the repo on GitHub, add topics and a description.
5. Resume bullet (fill with your measured values):
   *Built a real-time fraud scoring service on the IEEE-CIS dataset (LightGBM, calibrated probabilities, cost-optimized thresholds, per-prediction explanations) reaching PR-AUC X on a time-held-out test set; served via FastAPI with Postgres, Redis and Prometheus at Y req/s and p95 latency of Z ms; containerized with CI/CD and deployed publicly.*

**Acceptance checks**
- [ ] Every number in the README traces to a JSON file in `reports/`
- [ ] A new person can run the project from the README on a clean machine

---

# 5. Expected results summary (before deployment)

These are approximate sanity ranges for a strict time-based split, not guarantees. Always report what you measure.

| Stage | Result | Expected range | Red flag |
|---|---|---|---|
| 1 | Fraud rate | about 3.5% | Very different from about 3.5% (bad join or read) |
| 3 | Leakage tests | All pass | Any failure |
| 4 | Logistic regression ROC-AUC / PR-AUC | 0.80-0.87 / 0.20-0.40 | Above 0.95 |
| 4 | LightGBM default ROC-AUC / PR-AUC | 0.88-0.93 / 0.45-0.65 | Above 0.98 |
| 5 | Feature gain in PR-AUC | +0.01 to +0.04 | Jump above +0.15 (likely leakage) |
| 6 | Tuned LightGBM, test ROC-AUC / PR-AUC | 0.90-0.95 / 0.55-0.75 | Test much better than validation |
| 7 | ECE after calibration | Clearly lower than before | Worse than before |
| 7 | Precision at `t_block` (test) | Near 0.90 | Below 0.7 |
| 7 | Cost vs. approve-all | Lower cost | Higher cost |
| 8 | Contribution-sum check | Matches model score | Mismatch |
| 9 | Monthly PR-AUC | Mild decline over time | Sudden collapse |
| 10 | API tests | All pass | Any failure |
| 11 | Load test error rate | Under 1% | Over 1% |
| 11 | Single-score p95 (no explanations) | Tens of ms | Hundreds of ms |
| 12 | Compose and CI | Green | Red |

---

# 6. Common pitfalls

1. **Random splits.** They inflate scores a lot on this data. Use time splits only.
2. **Fitting encoders on all data.** Fit everything on train only.
3. **Using test data for tuning.** Tune on validation or time-CV; touch test once for reporting.
4. **Features you cannot compute in real time** (for example counts over other transactions). Avoid them, or say clearly that they are offline-only.
5. **Accuracy as a metric.** Use PR-AUC, recall at fixed precision, and cost.
6. **`scale_pos_weight` without calibration.** It skews probabilities; calibrate in Stage 7.
7. **Redistributing the dataset.** Not allowed; keep it out of git and images.
8. **Hand-typed metrics.** Always generate from JSON.
9. **Silent training-serving skew.** Keep the parity test.
10. **Logging personal data.** Log only what you need and hash identifiers.

---

# 7. Final definition of done

- [ ] All stages' acceptance checks passed
- [ ] `reports/` contains: `data_profile.json`, `split_info.json`, `metrics_baselines.json`, `metrics_feature_ablation.json`, `metrics_test_stage6.json`, `load_test.json`, `evaluation_report.md`, `model_card.md`, and figures
- [ ] Live API and demo UI links work
- [ ] README has architecture diagram, results tables, run instructions and limitations
- [ ] CI badge green, repo pinned, resume bullet filled with real numbers
- [ ] You can explain in an interview: why a time split, why PR-AUC, how calibration and cost thresholds work, how the explanations are computed, and what the load test showed
