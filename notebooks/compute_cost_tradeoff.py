# Phase 3, cost/accuracy tradeoff -- the actual deliverable the research
# question asks for (README: "characterizing the tradeoff between
# accuracy, latency, and compute cost"). Escalation % is a proxy for cost,
# not cost itself; this converts the honest, held-out-evaluated threshold
# sweep into real wall-clock latency and a GPU-hardware cost proxy.
#
# Reads results/phase3_sweep_heldout.csv plus per-problem mean latency from
# results/pilot_small.json / results/pilot_large.json (mean_latency_per_problem_seconds,
# already logged there -- NOT recomputed here). GPU count per model isn't
# in those JSON files; it's the actual tensor_parallel_size each model runs
# with (pilot.py, DECISIONS.md 2026-08-09): small model fits on one T4,
# large model needs both.
#
# No GPU needed. Run after router_sweep_heldout.py:
#   !python notebooks/compute_cost_tradeoff.py
from __future__ import annotations

import csv
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SWEEP_CSV = RESULTS_DIR / "phase3_sweep_heldout.csv"
SMALL_PATH = RESULTS_DIR / "pilot_small.json"
LARGE_PATH = RESULTS_DIR / "pilot_large.json"
OUT_CSV = RESULTS_DIR / "phase3_cost_tradeoff.csv"

SMALL_GPU_COUNT = 1  # tensor_parallel_size=1
LARGE_GPU_COUNT = 2  # tensor_parallel_size=2 -- a second, distinct cost axis from latency

NEW_FIELDS = [
    "blended_latency_seconds",
    "latency_savings_pct_vs_large_only",
    "blended_gpu_seconds",
    "gpu_seconds_savings_pct_vs_large_only",
]


def load_latencies() -> tuple[float, float]:
    small = json.loads(SMALL_PATH.read_text())
    large = json.loads(LARGE_PATH.read_text())
    return (
        small["mean_latency_per_problem_seconds"],
        large["mean_latency_per_problem_seconds"],
    )


def main() -> None:
    missing = [p for p in (SWEEP_CSV, SMALL_PATH, LARGE_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        return

    small_latency, large_latency = load_latencies()
    small_gpu_s = small_latency * SMALL_GPU_COUNT
    large_gpu_s = large_latency * LARGE_GPU_COUNT

    print(
        f"small model: {small_latency:.4f}s/problem on {SMALL_GPU_COUNT} GPU "
        f"({small_gpu_s:.4f} GPU-seconds/problem)"
    )
    print(
        f"large model: {large_latency:.4f}s/problem on {LARGE_GPU_COUNT} GPUs, "
        f"tensor_parallel_size=2 ({large_gpu_s:.4f} GPU-seconds/problem)"
    )
    print(
        "Note: latency and GPU-seconds are reported separately by design -- "
        "the large model is only ~1.8x slower in wall-clock latency, but "
        "consumes 2 GPUs concurrently while running, so its hardware-cost "
        "footprint is a different multiple than its latency footprint."
    )

    with SWEEP_CSV.open() as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        pct = float(r["pct_escalated"]) / 100
        blended_latency = (1 - pct) * small_latency + pct * large_latency
        blended_gpu_seconds = (1 - pct) * small_gpu_s + pct * large_gpu_s
        out_rows.append(
            {
                **r,
                "blended_latency_seconds": round(blended_latency, 5),
                "latency_savings_pct_vs_large_only": round(
                    100 * (large_latency - blended_latency) / large_latency, 2
                ),
                "blended_gpu_seconds": round(blended_gpu_seconds, 5),
                "gpu_seconds_savings_pct_vs_large_only": round(
                    100 * (large_gpu_s - blended_gpu_seconds) / large_gpu_s, 2
                ),
            }
        )

    fieldnames = list(rows[0].keys()) + NEW_FIELDS
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {OUT_CSV} ({len(out_rows)} rows)")


if __name__ == "__main__":
    main()
