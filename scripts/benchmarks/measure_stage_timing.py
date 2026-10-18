"""Script to measure per-stage timings against real running Uvicorn server (4 workers) on 200 validation rows."""
import argparse
import json
import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("measure_stages")

STAGE_MAP = {
    "req_val": "request_validation",
    "auth": "auth",
    "rate": "ratelimit",
    "feat": "feature_build",
    "pred": "predict",
    "cal": "calibrate",
    "shap": "shap",
    "reasons": "reasons_build",
    "audit": "audit_write",
    "ser": "serialization",
    "total_handler": "total_handler",
}

def parse_server_timing(header_str: str) -> dict[str, float]:
    out = {}
    if not header_str:
        return out
    for part in header_str.split(","):
        part = part.strip()
        m = re.match(r"([a-zA-Z_]+);dur=([0-9\.]+)", part)
        if m:
            k = m.group(1)
            v = float(m.group(2))
            out[STAGE_MAP.get(k, k)] = v
    return out

def get_200_validation_rows(parquet_path: str = "data/interim/train_merged.parquet") -> list[dict]:
    df = pd.read_parquet(parquet_path)
    val_start = int(len(df) * 0.70)
    val_end = int(len(df) * 0.85)
    val_slice = df.iloc[val_start:val_end]
    sampled = val_slice.sample(n=200, random_state=42)
    # Convert numpy types and NaNs cleanly
    records = []
    for row in sampled.to_dict(orient="records"):
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = float(v)
            else:
                clean_row[k] = str(v)
        clean_row["transaction_id"] = str(clean_row.get("TransactionID", "tx_val_sample"))
        records.append(clean_row)
    return records

def run_measurement(base_url: str = "http://127.0.0.1:8000", out_path: str = "reports/stage_timing.json"):
    logger.info("Extracting 200 real validation-split full payloads...")
    records = get_200_validation_rows()
    headers = {"X-API-Key": "demo-api-key-12345", "Content-Type": "application/json"}

    client = httpx.Client(timeout=30.0)

    # Health check
    logger.info("Verifying server is ready at %s...", base_url)
    res_ready = client.get(f"{base_url}/health")
    if res_ready.status_code != 200:
        raise RuntimeError(f"Server at {base_url} not healthy: {res_ready.status_code}")

    # Warmup
    logger.info("Warming up server with 10 requests...")
    for i in range(10):
        client.post(f"{base_url}/v1/score?explain=false", headers=headers, json=records[i])

    modes = [("explain_false", False), ("explain_true", True)]
    results = {}

    for mode_name, explain in modes:
        logger.info("Measuring %s across 200 validation rows at concurrency 1...", mode_name)
        stage_accum = {stage: [] for stage in STAGE_MAP.values()}
        client_latencies = []
        network_overheads = []  # client_observed - total_handler
        unaccounted_handler = []  # total_handler - sum(stages)

        sub_stages = [
            "request_validation", "auth", "ratelimit", "feature_build",
            "predict", "calibrate", "shap", "reasons_build", "audit_write", "serialization"
        ]

        for i, payload in enumerate(records):
            t0 = time.perf_counter()
            resp = client.post(
                f"{base_url}/v1/score?explain={str(explain).lower()}",
                headers=headers,
                json=payload,
            )
            client_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code != 200:
                logger.error("Request %d failed with status %d: %s", i, resp.status_code, resp.text)
                continue

            client_latencies.append(client_ms)
            server_timing = parse_server_timing(resp.headers.get("Server-Timing", ""))
            for st, dur in server_timing.items():
                if st in stage_accum:
                    stage_accum[st].append(dur)

            handler_ms = server_timing.get("total_handler", 0.0)
            network_overheads.append(max(0.0, client_ms - handler_ms))
            stages_sum = sum(server_timing.get(s, 0.0) for s in sub_stages)
            unaccounted_handler.append(max(0.0, handler_ms - stages_sum))

        mode_stats = {}
        for st in sub_stages + ["total_handler"]:
            arr = stage_accum[st]
            if arr:
                mode_stats[st] = {
                    "p50_ms": round(float(np.percentile(arr, 50)), 2),
                    "p95_ms": round(float(np.percentile(arr, 95)), 2),
                    "mean_ms": round(float(np.mean(arr)), 2),
                }
            else:
                mode_stats[st] = {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0}

        mode_stats["client_observed"] = {
            "p50_ms": round(float(np.percentile(client_latencies, 50)), 2),
            "p95_ms": round(float(np.percentile(client_latencies, 95)), 2),
            "mean_ms": round(float(np.mean(client_latencies)), 2),
        }
        mode_stats["client_minus_total_handler"] = {
            "p50_ms": round(float(np.percentile(network_overheads, 50)), 2),
            "p95_ms": round(float(np.percentile(network_overheads, 95)), 2),
            "mean_ms": round(float(np.mean(network_overheads)), 2),
        }
        mode_stats["total_handler_minus_sum_stages"] = {
            "p50_ms": round(float(np.percentile(unaccounted_handler, 50)), 2),
            "p95_ms": round(float(np.percentile(unaccounted_handler, 95)), 2),
            "mean_ms": round(float(np.mean(unaccounted_handler)), 2),
        }

        results[mode_name] = mode_stats

    final_report = {
        "metadata": {
            "target": "real uvicorn server (single laptop, loopback, 4 workers)",
            "server_url": base_url,
            "concurrency": 1,
            "validation_rows": 200,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "explain_false": results["explain_false"],
        "explain_true": results["explain_true"],
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(final_report, f, indent=2)

    logger.info("Successfully written reports/stage_timing.json")
    print(json.dumps(final_report, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--out", default="reports/stage_timing.json")
    args = parser.parse_args()
    run_measurement(args.url, args.out)
