"""Script to print median/min/max per profile and client_limited flags across the 5 load test runs."""
import glob
import json

import numpy as np

files = sorted(glob.glob("reports/load_test_20261019_074005_run*.json"))
print(f"Reading {len(files)} files: {files}\n")

runs_data = []
for f in files:
    with open(f) as fp:
        runs_data.append(json.load(fp))

profiles = [
    "c1_explain_false",
    "c1_explain_true",
    "c5_explain_false",
    "c5_explain_true",
    "c10_explain_false",
    "c10_explain_true",
]

print("=" * 80)
print("PER-RUN CLIENT_LIMITED FLAGS & METRICS")
print("=" * 80)
for idx, (f, data) in enumerate(zip(files, runs_data)):
    print(f"Run {idx+1} ({f}):")
    for p in profiles:
        prof = data["profiles"][p]
        rps = prof["median_rps"]
        p50 = prof["median_p50_ms"]
        cl = prof["client_limited"]
        print(f"  {p:<18}: RPS={rps:6.1f}, p50={p50:6.2f}ms, client_limited={cl}")
    print()

print("=" * 80)
print("AGGREGATE PROFILE SUMMARY (Median, Min, Max & Screening Rule)")
print("=" * 80)
header = f"{'Profile':<18} | {'Median RPS':>10} | {'Min RPS':>8} | {'Max RPS':>8} | {'Spread %':>10} | {'Client Limited':>15} | {'Report Status':<20}"
print(header)
print("-" * len(header))

for p in profiles:
    rps_vals = [data["profiles"][p]["median_rps"] for data in runs_data]
    cl_vals = [data["profiles"][p]["client_limited"] for data in runs_data]
    med_rps = float(np.median(rps_vals))
    min_rps = float(np.min(rps_vals))
    max_rps = float(np.max(rps_vals))
    spread_pct = ((max_rps - min_rps) / med_rps) * 100.0 if med_rps > 0 else 0.0
    any_cl = any(cl_vals)

    status = "REPORT"
    if spread_pct > 15.0 and any_cl:
        status = "EXCLUDE (spread > 15%, client_limited)"
    elif spread_pct > 15.0:
        status = "EXCLUDE (spread > 15%)"
    elif any_cl:
        status = "EXCLUDE (client_limited)"

    print(
        f"{p:<18} | {med_rps:10.1f} | {min_rps:8.1f} | {max_rps:8.1f} | {spread_pct:9.1f}% | {str(any_cl):>15} | {status:<20}"
    )

print("=" * 80)
