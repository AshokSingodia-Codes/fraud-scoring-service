# Benchmark & Ablation Suite

This directory contains the single unified benchmarking and ablation suite for the Real-Time Fraud Scoring Service (`run_benchmark.py`).

## Prerequisites
- Server running on `http://127.0.0.1:8000` with `RATE_LIMIT_REQUESTS=1000000` and `ANYIO_MAX_THREADS=200`.
- All background tasks stopped prior to measurement.

## Recommended Server Startup
```bash
# In PowerShell:
$env:RATE_LIMIT_REQUESTS="1000000"
$env:ANYIO_MAX_THREADS="200"
$env:OPENBLAS_NUM_THREADS="1"
$env:APP_ENV="dev"
$env:ALLOWED_API_KEYS="demo-api-key-12345"
uvicorn src.fraud.serving.app:app --host 127.0.0.1 --port 8000 --workers 4
```

## Running Benchmarks

### 1. Measure Client Ceiling (/health)
Measures the client load generator's throughput and CPU ceiling at concurrency 1, 5, 25, 50 across 4 client worker processes:
```bash
python scripts/benchmarks/run_benchmark.py --mode health --url http://127.0.0.1:8000
```

### 2. Full Sustained Benchmark
Executes 3 sustained runs per profile across concurrency 1, 5, 10, 25, 50, and 100 (stress point) for both `explain=false` and `explain=true`, logging per-process CPU and flagging `client_limited`:
```bash
python scripts/benchmarks/run_benchmark.py --mode full --url http://127.0.0.1:8000 --out reports/load_test_workers_4.json
```

### 3. Concurrency Ablation Study
Executes 5 repeats each of Variants A-E in randomized order at c=50, explain=false, running Variant A at start and end for drift detection:
```bash
python scripts/benchmarks/run_benchmark.py --mode ablation --out reports/ablation_results.json
```
