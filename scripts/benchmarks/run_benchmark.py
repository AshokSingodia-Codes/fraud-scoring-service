"""Unified benchmarking and ablation suite for Real-Time Fraud Scoring Service.

Features:
- Multi-process client load generation (default 4 client worker processes).
- Measures /health client ceiling at c=1, 5, 25, 50.
- Per-process client CPU monitoring; flags "client_limited": true if any client process > 70% CPU.
- Strict HTTP 200 accounting for RPS and latency percentiles. Loud failure on HTTP 429.
- Status code breakdown and error classification (timeout, connect, 429, 5xx).
- Randomized ablation study (Variants A-E, 5 repeats each, A at start and A at end for drift detection).
- Optional psutil dependency (graceful degradation if missing).
"""

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fraud.utils.synthetic import generate_synthetic_transaction  # noqa: E402

# Optional psutil import
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark")


def client_worker_process(
    worker_id: int,
    base_url: str,
    endpoint: str,
    method: str,
    concurrency: int,
    duration_sec: float,
    max_requests: int,
    explain: bool,
    api_key: str,
    result_queue: mp.Queue,
):
    """Sub-process running async requests using its own event loop and httpx client."""
    async def _run():
        headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        limits = httpx.Limits(
            max_keepalive_connections=max(50, concurrency * 2),
            max_connections=max(100, concurrency * 4),
            keepalive_expiry=30.0,
        )
        timeout = httpx.Timeout(connect=15.0, read=30.0, write=15.0, pool=15.0)

        latencies_200 = []
        status_codes = {}
        error_types = {"timeout": 0, "connect": 0, "429": 0, "5xx": 0, "other": 0}
        total_reqs = 0

        # CPU monitoring for this worker process
        proc = psutil.Process() if HAS_PSUTIL else None
        if proc:
            proc.cpu_percent()  # initialize baseline

        actual_duration = 0.0
        try:
            async with httpx.AsyncClient(base_url=base_url, headers=headers, limits=limits, timeout=timeout) as client:
                semaphore = asyncio.Semaphore(concurrency)
                stop_event = asyncio.Event()

                async def _worker():
                    nonlocal total_reqs
                    while not stop_event.is_set():
                        async with semaphore:
                            if stop_event.is_set():
                                break
                            if max_requests > 0 and total_reqs >= max_requests:
                                stop_event.set()
                                break

                            total_reqs += 1
                            t0 = time.perf_counter()
                            try:
                                if method == "GET":
                                    res = await client.get(endpoint)
                                else:
                                    tx = generate_synthetic_transaction()
                                    res = await client.post(f"{endpoint}?explain={str(explain).lower()}", json=tx)

                                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                                code = str(res.status_code)
                                status_codes[code] = status_codes.get(code, 0) + 1

                                if res.status_code == 200:
                                    latencies_200.append(elapsed_ms)
                                elif res.status_code == 429:
                                    error_types["429"] += 1
                                    logger.error("FATAL: HTTP 429 Rate Limit hit in worker %d!", worker_id)
                                elif res.status_code >= 500:
                                    error_types["5xx"] += 1
                                else:
                                    error_types["other"] += 1
                            except httpx.TimeoutException:
                                error_types["timeout"] += 1
                            except httpx.ConnectError:
                                error_types["connect"] += 1
                            except Exception as e:
                                error_types["other"] += 1
                                logger.debug("Worker %d exception: %s", worker_id, e)

                workers = [asyncio.create_task(_worker()) for _ in range(concurrency)]

                start_wall = time.perf_counter()
                while time.perf_counter() - start_wall < duration_sec and not stop_event.is_set():
                    await asyncio.sleep(0.1)

                stop_event.set()
                for w in workers:
                    w.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
                actual_duration = time.perf_counter() - start_wall
        except Exception as e:
            logger.debug("Worker %d error: %s", worker_id, e)
        finally:
            worker_cpu = proc.cpu_percent() if proc else 0.0
            result_queue.put({
                "worker_id": worker_id,
                "duration": actual_duration if actual_duration > 0 else duration_sec,
                "total_requests": total_reqs,
                "latencies_200": latencies_200,
                "status_codes": status_codes,
                "error_types": error_types,
                "cpu_percent": worker_cpu,
            })

    asyncio.run(_run())


def run_distributed_load(
    base_url: str,
    endpoint: str = "/v1/score",
    method: str = "POST",
    concurrency: int = 1,
    num_client_procs: int = 4,
    duration_sec: float = 15.0,
    max_requests: int = 1000,
    explain: bool = False,
    api_key: str = "demo-api-key-12345",
) -> dict[str, Any]:
    """Dispatch load across num_client_procs processes and aggregate results."""
    # Ensure at least 1 process, up to concurrency
    procs_count = min(num_client_procs, max(1, concurrency))
    per_proc_concurrency = max(1, concurrency // procs_count)
    per_proc_requests = max(1, max_requests // procs_count) if max_requests > 0 else 0

    result_queue = mp.Queue()
    processes = []
    for wid in range(procs_count):
        p = mp.Process(
            target=client_worker_process,
            args=(
                wid,
                base_url,
                endpoint,
                method,
                per_proc_concurrency,
                duration_sec,
                per_proc_requests,
                explain,
                api_key,
                result_queue,
            ),
        )
        processes.append(p)
        p.start()

    results = []
    for _ in range(procs_count):
        try:
            results.append(result_queue.get(timeout=duration_sec + 15.0))
        except Exception as e:
            logger.warning("Queue get timeout after %s s: %s", duration_sec + 15.0, e)

    for p in processes:
        p.join(timeout=5.0)
        if p.is_alive():
            p.terminate()

    # Aggregate
    all_latencies = []
    total_reqs = 0
    merged_codes = {}
    merged_errors = {"timeout": 0, "connect": 0, "429": 0, "5xx": 0, "other": 0}
    client_cpus = []
    max_dur = max(r["duration"] for r in results)

    for r in results:
        all_latencies.extend(r["latencies_200"])
        total_reqs += r["total_requests"]
        client_cpus.append(r["cpu_percent"])
        for k, v in r["status_codes"].items():
            merged_codes[k] = merged_codes.get(k, 0) + v
        for k, v in r["error_types"].items():
            merged_errors[k] += v

    if merged_errors["429"] > 0:
        raise RuntimeError(f"Run failed loudly: encountered {merged_errors['429']} HTTP 429 rate limit responses!")

    success_count = len(all_latencies)
    error_count = sum(merged_errors.values())
    rps = round(success_count / max_dur, 1) if max_dur > 0 else 0.0

    if all_latencies:
        p50 = round(float(np.percentile(all_latencies, 50)), 2)
        p95 = round(float(np.percentile(all_latencies, 95)), 2)
        p99 = round(float(np.percentile(all_latencies, 99)), 2) if len(all_latencies) >= 1000 else None
        mean = round(float(np.mean(all_latencies)), 2)
    else:
        p50, p95, p99, mean = None, None, None, None

    max_client_cpu = max(client_cpus) if client_cpus else 0.0
    client_limited = max_client_cpu > 70.0

    return {
        "concurrency": concurrency,
        "client_processes": procs_count,
        "explain": explain,
        "duration_sec": round(max_dur, 2),
        "total_requests": total_reqs,
        "successful_requests": success_count,
        "errors": error_count,
        "rps": rps,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean,
        "client_limited": client_limited,
        "per_process_client_cpu": client_cpus,
        "max_client_cpu_pct": max_client_cpu,
        "status_codes": merged_codes,
        "error_types": merged_errors,
    }


def benchmark_health_ceiling(base_url: str) -> dict[str, Any]:
    """Measure the client's own ceiling against /health at concurrency 1, 5, 25, 50."""
    logger.info("=== Measuring Client Ceiling against /health ===")
    concurrencies = [1, 5, 25, 50]
    ceiling_results = {}
    for c in concurrencies:
        res = run_distributed_load(
            base_url=base_url,
            endpoint="/health",
            method="GET",
            concurrency=c,
            num_client_procs=4,
            duration_sec=10.0,
            max_requests=5000,
            explain=False,
        )
        ceiling_results[f"c{c}"] = res
        logger.info(
            "/health c=%d: RPS=%.1f, p50=%.2fms, p95=%.2fms, max_client_cpu=%.1f%%, client_limited=%s",
            c, res["rps"], res["p50_ms"] or 0.0, res["p95_ms"] or 0.0, res["max_client_cpu_pct"], res["client_limited"]
        )
    return ceiling_results


def run_full_benchmark(
    base_url: str,
    output_path: str = "reports/load_test_workers_4.json",
    concurrencies: list[int] | None = None,
) -> dict[str, Any]:
    """Execute sustained load test across selected concurrencies (default [1, 5, 10])."""
    if concurrencies is None:
        concurrencies = [1, 5, 10]
    profiles = {}

    for c in concurrencies:
        for explain in [False, True]:
            key = f"c{c}_explain_{str(explain).lower()}"
            is_stress = (c == 100)
            runs = []
            logger.info("Running profile %s (3 repeats)...", key)
            for r in range(3):
                run_res = run_distributed_load(
                    base_url=base_url,
                    endpoint="/v1/score",
                    method="POST",
                    concurrency=c,
                    num_client_procs=4,
                    duration_sec=15.0,
                    max_requests=1000,
                    explain=explain,
                )
                runs.append(run_res)
                time.sleep(1.0)

            med_rps = float(np.median([r["rps"] for r in runs]))
            med_p50 = float(np.median([r["p50_ms"] for r in runs if r["p50_ms"] is not None]))
            med_p95 = float(np.median([r["p95_ms"] for r in runs if r["p95_ms"] is not None]))
            p99_vals = [r["p99_ms"] for r in runs if r.get("p99_ms") is not None]
            med_p99 = float(np.median(p99_vals)) if p99_vals else None
            tot_errors = sum(r["errors"] for r in runs)
            max_client_cpu = max(r["max_client_cpu_pct"] for r in runs)
            client_limited = any(r["client_limited"] for r in runs)

            # Aggregate status codes and error types
            agg_codes = {}
            agg_errors = {"timeout": 0, "connect": 0, "429": 0, "5xx": 0, "other": 0}
            for r in runs:
                for code, count in r["status_codes"].items():
                    agg_codes[code] = agg_codes.get(code, 0) + count
                for etype, count in r["error_types"].items():
                    agg_errors[etype] += count

            profiles[key] = {
                "concurrency": c,
                "explain": explain,
                "is_stress_point": is_stress,
                "repeats": 3,
                "median_p50_ms": round(med_p50, 2),
                "median_p95_ms": round(med_p95, 2),
                "median_p99_ms": round(med_p99, 2) if med_p99 else None,
                "median_rps": round(med_rps, 1),
                "total_errors": tot_errors,
                "client_limited": client_limited,
                "max_client_cpu_pct": round(max_client_cpu, 1),
                "status_codes": agg_codes,
                "error_types": agg_errors,
                "runs": runs,
            }
            logger.info(
                "Profile %s: Median RPS=%.1f, p50=%.2fms, p95=%.2fms, Errors=%d, client_limited=%s",
                key, med_rps, med_p50, med_p95, tot_errors, client_limited
            )

    import hashlib
    import subprocess

    def file_sha256(filepath: str) -> str | None:
        p = Path(filepath)
        if not p.exists():
            return None
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        git_sha = "unknown"

    py_procs = []
    if HAS_PSUTIL:
        for p in psutil.process_iter(["pid", "name"]):
            try:
                if "python" in p.info["name"].lower():
                    py_procs.append({"pid": p.info["pid"], "name": p.info["name"]})
            except Exception:
                pass

    report_data = {
        "metadata": {
            "workers": 4,
            "client_processes": 4,
            "os_cpu_count": os.cpu_count(),
            "python_version": sys.version,
            "git_sha": git_sha,
            "models_model_txt_sha256": file_sha256("models/model.txt"),
            "models_calibrator_joblib_sha256": file_sha256("models/calibrator.joblib"),
            "python_processes_at_start": py_procs,
            "duration_sec_per_profile": 15.0,
            "max_requests_per_profile": 1000,
            "repeats": 3,
            "server_url": base_url,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "profiles": profiles,
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report_data, f, indent=2)

    logger.info("Saved full benchmark results to %s", output_path)
    return report_data


def wait_for_server(url: str = "http://127.0.0.1:8000/health", timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(url)
                if res.status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def kill_all_server_processes():
    """Ensure no leftover python/uvicorn server processes exist on port 8000."""
    try:
        import subprocess
        cmd = "Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"
        subprocess.run(["powershell", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

    import socket
    for _ in range(20):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 8000))
            s.close()
            break
        except OSError:
            time.sleep(0.5)


def start_server_process(workers: int = 4, env_vars: dict | None = None):
    import subprocess
    kill_all_server_processes()
    env = os.environ.copy()
    env["APP_ENV"] = "dev"
    env["ALLOWED_API_KEYS"] = "demo-api-key-12345"
    env["RATE_LIMIT_REQUESTS"] = "1000000"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    if env_vars:
        env.update(env_vars)

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.fraud.serving.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--workers",
        str(workers),
    ]
    logger.info("Starting server: workers=%d, env=%s", workers, env_vars or {})
    log_file = open("reports/server_current.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    if not wait_for_server(timeout=40.0):
        proc.kill()
        log_file.close()
        raise RuntimeError("Server failed to start within timeout.")
    return proc


def run_ablation_variant(variant_name: str, workers: int, env_vars: dict, repeats: int = 5) -> dict:
    import subprocess
    logger.info("==================================================")
    logger.info("Running Variant: %s (%d repeats)", variant_name, repeats)
    logger.info("==================================================")

    server_proc = start_server_process(workers=workers, env_vars=env_vars)
    rps_list, p50_list, p95_list, errors_list = [], [], [], []

    try:
        # Warmup
        run_distributed_load(
            base_url="http://127.0.0.1:8000",
            concurrency=5,
            num_client_procs=2,
            duration_sec=3.0,
            max_requests=50,
            explain=False,
        )

        for rep in range(repeats):
            logger.info("Variant '%s' repeat %d/%d...", variant_name, rep + 1, repeats)
            res = run_distributed_load(
                base_url="http://127.0.0.1:8000",
                concurrency=50,
                num_client_procs=4,
                duration_sec=8.0,
                max_requests=500,
                explain=False,
            )
            rps_list.append(res["rps"])
            if res["p50_ms"]:
                p50_list.append(res["p50_ms"])
            if res["p95_ms"]:
                p95_list.append(res["p95_ms"])
            errors_list.append(res["errors"])
            time.sleep(1.0)
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        kill_all_server_processes()
        time.sleep(1.0)

    return {
        "variant": variant_name,
        "workers": workers,
        "env": env_vars,
        "repeats": repeats,
        "rps_mean": round(float(np.mean(rps_list)), 1),
        "rps_std": round(float(np.std(rps_list)), 1),
        "rps_min": round(float(np.min(rps_list)), 1),
        "rps_max": round(float(np.max(rps_list)), 1),
        "p50_mean": round(float(np.mean(p50_list)), 2) if p50_list else None,
        "p95_mean": round(float(np.mean(p95_list)), 2) if p95_list else None,
        "total_errors": sum(errors_list),
        "raw_rps": rps_list,
    }


def execute_ablation_study(output_path: str = "reports/ablation_results.json"):
    random.seed(42)
    variants = [
        ("B. 1 worker (ANYIO_MAX_THREADS=40)", 1, {"ANYIO_MAX_THREADS": "40"}),
        ("C. 2 workers (ANYIO_MAX_THREADS=40)", 2, {"ANYIO_MAX_THREADS": "40"}),
        ("D. AnyIO thread limit 200 (4 workers)", 4, {"ANYIO_MAX_THREADS": "200"}),
        ("E. AnyIO thread limit 80 (4 workers)", 4, {"ANYIO_MAX_THREADS": "80"}),
    ]
    random.shuffle(variants)
    results = []

    # 1. Baseline A at start
    logger.info("--- Step 1: Baseline A at start ---")
    res_a_start = run_ablation_variant(
        "A. Baseline at start (4 workers, ANYIO_MAX_THREADS=40)",
        workers=4,
        env_vars={"ANYIO_MAX_THREADS": "40"},
        repeats=5,
    )
    results.append(res_a_start)

    # 2. Variants B-E
    for name, workers, env_vars in variants:
        res = run_ablation_variant(name, workers=workers, env_vars=env_vars, repeats=5)
        results.append(res)

    # 3. Baseline A at end
    logger.info("--- Step 3: Baseline A at end (drift check) ---")
    res_a_end = run_ablation_variant(
        "A. Baseline at end (4 workers, ANYIO_MAX_THREADS=40)",
        workers=4,
        env_vars={"ANYIO_MAX_THREADS": "40"},
        repeats=5,
    )
    results.append(res_a_end)

    drift_diff = abs(res_a_start["rps_mean"] - res_a_end["rps_mean"])
    pooled_spread = max(res_a_start["rps_std"], res_a_end["rps_std"])
    drift_status = "Significant drift" if drift_diff > pooled_spread * 1.5 else "Stable (within noise spread)"
    # Minimum Detectable Effect (MDE): for n=5 per group, alpha=0.05, 80% power: MDE ~ 2.01 * pooled_std
    mde_rps = round(2.01 * res_a_start["rps_std"], 1)

    logger.info("Baseline Drift: start=%.1f RPS, end=%.1f RPS (diff=%.1f, spread=%.1f, status=%s)",
                res_a_start["rps_mean"], res_a_end["rps_mean"], drift_diff, pooled_spread, drift_status)
    logger.info("Baseline std=%.1f RPS -> Minimum Detectable Effect (MDE)=%.1f RPS (alpha=0.05, power=0.80, n=5)",
                res_a_start["rps_std"], mde_rps)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "drift_analysis": {
                "a_start_rps_mean": res_a_start["rps_mean"],
                "a_end_rps_mean": res_a_end["rps_mean"],
                "drift_diff": round(drift_diff, 1),
                "status": drift_status,
                "baseline_std": res_a_start["rps_std"],
                "minimum_detectable_effect_rps": mde_rps,
            },
            "variants": results,
        }, f, indent=2)

    logger.info("Saved ablation results to %s", output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmarking & Ablation Suite")
    parser.add_argument("--mode", choices=["health", "full", "quick", "ablation"], default="health")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.mode == "health":
        benchmark_health_ceiling(args.url)
    elif args.mode == "full":
        run_full_benchmark(args.url, args.out or "reports/load_test_workers_4.json")
    elif args.mode == "ablation":
        execute_ablation_study(args.out or "reports/ablation_results.json")



if __name__ == "__main__":
    main()


