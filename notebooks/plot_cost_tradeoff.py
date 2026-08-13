# Plots the actual cost/accuracy tradeoff (not escalation % as a proxy):
# x-axis is latency savings vs. always using the large model, y-axis is
# held-out accuracy, one point per threshold from results/phase3_cost_tradeoff.csv,
# with small-only and large-only marked as the two endpoints of the curve.
#
# No GPU needed. Run after compute_cost_tradeoff.py:
#   !python notebooks/plot_cost_tradeoff.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COST_CSV = RESULTS_DIR / "phase3_cost_tradeoff.csv"
SPLIT_PATH = RESULTS_DIR / "train_test_split.json"
SMALL_PATH = RESULTS_DIR / "pilot_small.json"
LARGE_PATH = RESULTS_DIR / "pilot_large.json"
OUT_PNG = RESULTS_DIR / "phase3_cost_tradeoff_plot.png"


def load_rows() -> list[dict]:
    with COST_CSV.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pct_escalated"] = float(r["pct_escalated"])
        r["accuracy"] = float(r["accuracy"])
        r["latency_savings_pct_vs_large_only"] = float(r["latency_savings_pct_vs_large_only"])
    return rows


def held_out_endpoints() -> dict:
    """small-only and large-only accuracy on the held-out 150, plus the
    small-only endpoint's real x-position. The small model is only ~1.83x
    faster, not infinitely faster, so 0% escalation does NOT mean 100%
    latency savings -- it means 100*(large_latency-small_latency)/large_latency,
    the actual ceiling for this model pair. Getting this wrong would put
    the small-only endpoint way off to the right of where the real curve's
    data points end up."""
    split = json.loads(SPLIT_PATH.read_text())
    held_out_ids = set(split["held_out"])
    small_correct = {r["unique_id"]: r["correct"] for r in json.loads(SMALL_PATH.read_text())["results"]}
    large_data = json.loads(LARGE_PATH.read_text())
    small_data = json.loads(SMALL_PATH.read_text())
    large_correct = {r["unique_id"]: r["correct"] for r in large_data["results"]}
    ids = [u for u in held_out_ids if u in small_correct and u in large_correct]
    n = len(ids)

    small_latency = small_data["mean_latency_per_problem_seconds"]
    large_latency = large_data["mean_latency_per_problem_seconds"]
    max_savings_pct = 100 * (large_latency - small_latency) / large_latency

    return {
        "small_only_accuracy": sum(small_correct[u] for u in ids) / n,
        "small_only_savings_pct": max_savings_pct,
        "large_only_accuracy": sum(large_correct[u] for u in ids) / n,
        "oracle_accuracy": sum(1 for u in ids if small_correct[u] or large_correct[u]) / n,
    }


def main() -> None:
    missing = [p for p in (COST_CSV, SPLIT_PATH, SMALL_PATH, LARGE_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        return

    rows = load_rows()
    endpoints = held_out_endpoints()

    fig, ax = plt.subplots(figsize=(8, 6))

    for signal, style, color in (
        ("mean_logprob", "o-", "tab:blue"),
        ("min_logprob", "s-", "tab:orange"),
    ):
        signal_rows = sorted(
            (r for r in rows if r["signal"] == signal),
            key=lambda r: r["latency_savings_pct_vs_large_only"],
        )
        if signal_rows:
            ax.plot(
                [r["latency_savings_pct_vs_large_only"] for r in signal_rows],
                [100 * r["accuracy"] for r in signal_rows],
                style,
                color=color,
                label=signal,
                markersize=4,
            )

    fv_rows = [r for r in rows if r["signal"] == "format_validity"]
    if fv_rows:
        r = fv_rows[0]
        ax.plot(
            r["latency_savings_pct_vs_large_only"], 100 * r["accuracy"],
            "D", color="black", label="format_validity (single point)", markersize=8,
        )

    # The two fixed endpoints of the tradeoff curve: always-large (0%
    # savings) and always-small (100% savings).
    ax.plot(0, 100 * endpoints["large_only_accuracy"], "*", color="firebrick", markersize=18, zorder=5)
    ax.annotate(
        f"large-only\n({endpoints['large_only_accuracy']:.1%}, 0% savings)",
        xy=(0, 100 * endpoints["large_only_accuracy"]), xytext=(8, -18),
        textcoords="offset points", fontsize=8, color="firebrick",
    )
    small_x = endpoints["small_only_savings_pct"]
    ax.plot(small_x, 100 * endpoints["small_only_accuracy"], "*", color="gray", markersize=18, zorder=5)
    ax.annotate(
        f"small-only\n({endpoints['small_only_accuracy']:.1%}, {small_x:.1f}% savings)",
        xy=(small_x, 100 * endpoints["small_only_accuracy"]), xytext=(8, -18),
        textcoords="offset points", fontsize=8, color="gray",
    )

    ax.axhline(
        100 * endpoints["oracle_accuracy"], linestyle="--", linewidth=1, color="seagreen", alpha=0.7
    )
    ax.annotate(
        f"oracle ceiling ({endpoints['oracle_accuracy']:.1%})",
        xy=(2, 100 * endpoints["oracle_accuracy"]), xytext=(0, 3),
        textcoords="offset points", fontsize=8, color="seagreen",
    )

    ax.set_xlabel("Latency savings vs. always using the large model (%)")
    ax.set_ylabel("Accuracy on held-out set (%)")
    ax.set_title("Where a cascade router can land on the accuracy/cost curve\n(held-out n=150, honest fit-on-train/eval-on-held-out thresholds)")
    max_x = max([r["latency_savings_pct_vs_large_only"] for r in rows] + [small_x])
    ax.set_xlim(-3, max_x + 5)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
