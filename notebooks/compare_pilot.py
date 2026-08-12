# Phase 2 pilot comparison: reads results/pilot_small.json and
# results/pilot_large.json (written by pilot.py) and prints a side-by-side
# summary table plus the oracle ceiling routing is trying to approach.
#
# Run after both pilot.py invocations have completed:
#   !python notebooks/compare_pilot.py
from __future__ import annotations

import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FILES = {
    "small": RESULTS_DIR / "pilot_small.json",
    "large": RESULTS_DIR / "pilot_large.json",
}


def load_results() -> dict:
    missing = [key for key, path in FILES.items() if not path.exists()]
    if missing:
        for key in missing:
            print(f"Missing: {FILES[key]} (run pilot.py --model {key} first)")
        sys.exit(1)
    return {key: json.loads(path.read_text()) for key, path in FILES.items()}


def compute_oracle(data: dict) -> tuple[int, int]:
    """Accuracy if we always picked whichever model got a given problem
    right -- the ceiling a routing system is trying to approach."""
    small_by_id = {r["unique_id"]: r["correct"] for r in data["small"]["results"]}
    large_by_id = {r["unique_id"]: r["correct"] for r in data["large"]["results"]}
    common_ids = set(small_by_id) & set(large_by_id)
    if len(common_ids) != len(small_by_id) or len(common_ids) != len(large_by_id):
        print(
            f"Note: small has {len(small_by_id)} problems, large has "
            f"{len(large_by_id)}; oracle computed over the "
            f"{len(common_ids)} problems common to both.\n"
        )
    n_correct = sum(1 for uid in common_ids if small_by_id[uid] or large_by_id[uid])
    return n_correct, len(common_ids)


def print_table(data: dict, oracle_correct: int, oracle_n: int) -> None:
    headers = [
        "model",
        "accuracy",
        "correct/total",
        "load_s",
        "gen_s",
        "mean_latency_s",
    ]
    rows = []
    for key in ("small", "large"):
        d = data[key]
        acc = d["n_correct"] / d["n"] if d["n"] else 0.0
        rows.append(
            (
                d["model"],
                f"{acc:.1%}",
                f"{d['n_correct']}/{d['n']}",
                f"{d['load_time_seconds']:.1f}",
                f"{d['generation_time_seconds']:.1f}",
                f"{d['mean_latency_per_problem_seconds']:.2f}",
            )
        )
    oracle_acc = oracle_correct / oracle_n if oracle_n else 0.0
    rows.append(
        (
            "oracle (best of both)",
            f"{oracle_acc:.1%}",
            f"{oracle_correct}/{oracle_n}",
            "-",
            "-",
            "-",
        )
    )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def fmt_row(cells: tuple) -> str:
        return "  ".join(cell.ljust(width) for cell, width in zip(cells, widths))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


if __name__ == "__main__":
    data = load_results()
    oracle_correct, oracle_n = compute_oracle(data)
    print_table(data, oracle_correct, oracle_n)
