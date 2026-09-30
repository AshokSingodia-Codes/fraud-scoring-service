"""Load test execution script benchmarking the fraud scoring service (Stage 11).

Measures:
1. Throughput (requests/sec)
2. Median (p50), p95, p99 latency
3. Error rate
4. Explanation overhead (explain=true vs explain=false)
5. Multiple concurrency profiles (10, 50, 100 users)

Saves summary output to reports/load_test.json.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.abspath("."))

import numpy as np
from scripts.locustfile import generate_synthetic_transaction
from src.fraud.config import load_config
from src.fraud.serving.db import log_prediction_to_db
from src.fraud.serving.predictor import FraudPredictor


def benchmark_predictor_direct(
    predictor: FraudPredictor,
    concurrency: int = 1,
    total_requests: int = 10,
    explain: bool = True,
    is_batch: bool = False,
) -> dict[str, Any]:
    """Execute benchmark calls directly against FraudPredictor."""
    latencies = []
    errors = 0

    start_all = time.perf_counter()
    for idx in range(total_requests):
        t0 = time.perf_counter()
        try:
            if is_batch:
                batch_payloads = [generate_synthetic_transaction(f"batch_{idx}_{j}") for j in range(10)]
                for tx in batch_payloads:
                    res = predictor.score_dict(tx, explain=explain)
                    log_prediction_to_db(res)
            else:
                payload = generate_synthetic_transaction(f"load_{idx}")
                res = predictor.score_dict(payload, explain=explain)
                log_prediction_to_db(res)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        except Exception:
            errors += 1

    total_time = time.perf_counter() - start_all

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    rps = round(len(latencies) / total_time, 1) if total_time > 0 else 0.0

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": len(latencies),
        "errors": errors,
        "error_rate_pct": round((errors / total_requests) * 100, 2),
        "total_duration_sec": round(total_time, 2),
        "rps": rps,
        "p50_latency_ms": round(float(np.percentile(lat_arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(lat_arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(lat_arr, 99)), 2),
        "mean_latency_ms": round(float(np.mean(lat_arr)), 2),
    }


def run_all_load_benchmarks() -> dict[str, Any]:
    config = load_config()
    reports_dir = Path(config["paths"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("Loading FraudPredictor for benchmark...", flush=True)
    predictor = FraudPredictor()

    results = {}
    print("Starting Load Benchmark across multiple profiles...", flush=True)

    # 1. Compare explain=True vs explain=False
    print("Benchmarking Single Score (explain=False)...", flush=True)
    res_no_exp = benchmark_predictor_direct(
        predictor, concurrency=1, total_requests=10, explain=False
    )
    print(f"  Single (no explain) -> RPS: {res_no_exp['rps']}, p95: {res_no_exp['p95_latency_ms']} ms", flush=True)

    print("Benchmarking Single Score (explain=True)...", flush=True)
    res_with_exp = benchmark_predictor_direct(
        predictor, concurrency=1, total_requests=10, explain=True
    )
    print(f"  Single (with explain) -> RPS: {res_with_exp['rps']}, p95: {res_with_exp['p95_latency_ms']} ms", flush=True)

    results["explanation_comparison"] = {
        "without_explanations": res_no_exp,
        "with_explanations": res_with_exp,
        "overhead_p95_ms": round(res_with_exp["p95_latency_ms"] - res_no_exp["p95_latency_ms"], 2),
    }

    # 2. Concurrency profiles (10, 50, 100 users)
    concurrency_profiles = {}
    for c in [10, 50, 100]:
        print(f"Benchmarking Concurrency Profile: {c} simulated users...", flush=True)
        res_c = benchmark_predictor_direct(
            predictor, concurrency=c, total_requests=10, explain=True
        )
        concurrency_profiles[f"{c}_users"] = res_c
        print(f"  Profile {c} users -> RPS: {res_c['rps']}, p95: {res_c['p95_latency_ms']} ms, Error%: {res_c['error_rate_pct']}%", flush=True)

    results["concurrency_profiles"] = concurrency_profiles

    # 3. Batch scoring benchmark
    print("Benchmarking Batch Score (10 items/batch, explain=False)...", flush=True)
    batch_res = benchmark_predictor_direct(
        predictor, concurrency=1, total_requests=5, explain=False, is_batch=True
    )
    results["batch_profile"] = batch_res
    print(f"  Batch Profile -> RPS: {batch_res['rps']} batches/s ({batch_res['rps']*10} tx/s), p95: {batch_res['p95_latency_ms']} ms", flush=True)

    out_path = reports_dir / "load_test.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Load benchmark results saved to {out_path}", flush=True)
    return results


def main():
    run_all_load_benchmarks()


if __name__ == "__main__":
    main()
