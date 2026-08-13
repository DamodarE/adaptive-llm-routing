# Phase 3: plot the accuracy-vs-escalation tradeoff from results/phase3_sweep.csv.
#
# Reads the CSV written by router_sweep.py and plots accuracy against
# %-escalated-to-large-model for each confidence signal, with small-only /
# large-only / oracle reference lines pulled from the pilot results if
# present. Kept as a separate, committed script (not an ad hoc notebook
# cell) so the plot is actually reproducible from what's checked in.
#
# Run after router_sweep.py:
#   !python notebooks/plot_router_sweep.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SWEEP_CSV = RESULTS_DIR / "phase3_sweep.csv"
OUT_PNG = RESULTS_DIR / "router_sweep_plot.png"


def load_sweep_rows() -> list[dict]:
    with SWEEP_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            row["pct_escalated"] = float(row["pct_escalated"])
            row["accuracy"] = float(row["accuracy"])
            rows.append(row)
        return rows


def load_baselines() -> dict:
    """small-only / large-only / oracle reference values, if the pilot
    result files are present; omitted from the plot otherwise."""
    baselines = {}
    small_path = RESULTS_DIR / "pilot_small.json"
    large_path = RESULTS_DIR / "pilot_large.json"
    if small_path.exists():
        d = json.loads(small_path.read_text())
        baselines["small-only"] = d["n_correct"] / d["n"]
    if large_path.exists():
        d = json.loads(large_path.read_text())
        baselines["large-only"] = d["n_correct"] / d["n"]
    if small_path.exists() and large_path.exists():
        small = {r["unique_id"]: r["correct"] for r in json.loads(small_path.read_text())["results"]}
        large = {r["unique_id"]: r["correct"] for r in json.loads(large_path.read_text())["results"]}
        common = set(small) & set(large)
        baselines["oracle"] = sum(1 for uid in common if small[uid] or large[uid]) / len(common)
    return baselines


def main() -> None:
    if not SWEEP_CSV.exists():
        print(f"Missing: {SWEEP_CSV} (run router_sweep.py first)")
        return

    rows = load_sweep_rows()
    baselines = load_baselines()

    fig, ax = plt.subplots(figsize=(8, 6))

    for signal, style in (("mean_logprob", "o-"), ("min_logprob", "s-")):
        signal_rows = sorted(
            (r for r in rows if r["signal"] == signal), key=lambda r: r["pct_escalated"]
        )
        if signal_rows:
            ax.plot(
                [r["pct_escalated"] for r in signal_rows],
                [100 * r["accuracy"] for r in signal_rows],
                style,
                label=signal,
                markersize=4,
            )

    fv_rows = [r for r in rows if r["signal"] == "format_validity"]
    if fv_rows:
        r = fv_rows[0]
        ax.plot(
            r["pct_escalated"], 100 * r["accuracy"], "D", color="black",
            label="format_validity (single point)", markersize=8,
        )

    for name, acc, color in (
        ("small-only", baselines.get("small-only"), "gray"),
        ("large-only", baselines.get("large-only"), "firebrick"),
        ("oracle", baselines.get("oracle"), "seagreen"),
    ):
        if acc is not None:
            ax.axhline(100 * acc, linestyle="--", linewidth=1, color=color, alpha=0.7)
            ax.annotate(
                f"{name} ({100 * acc:.1f}%)",
                xy=(0, 100 * acc),
                xytext=(5, 3),
                textcoords="offset points",
                ha="left",
                fontsize=8,
                color=color,
            )

    ax.set_xlabel("% of problems escalated to large model")
    ax.set_ylabel("Overall accuracy (%)")
    ax.set_title("Cascade router: accuracy vs. escalation rate (MATH-500, n=500)")
    ax.set_xlim(-2, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
