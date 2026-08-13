# Headline cost/accuracy chart: one clear story ("same accuracy, less
# compute"), not a full signal comparison -- that's what
# plot_cost_tradeoff.py is for. x-axis is GPU-seconds savings (the
# stronger, more legible cost number: ~43% vs ~27% for latency, since the
# large model's tensor_parallel_size=2 footprint is the bigger cost lever
# than its wall-clock speed). Only the mean_logprob curve is plotted -- the
# real, working signal -- with the large-only baseline and the
# matching-accuracy point called out directly on the chart.
#
# No GPU needed. Run after compute_cost_tradeoff.py:
#   !python notebooks/plot_cost_accuracy_headline.py
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
COST_CSV = RESULTS_DIR / "phase3_cost_tradeoff.csv"
OUT_PNG = RESULTS_DIR / "cost_accuracy_headline_plot.png"

# Reference palette (see dataviz skill): categorical slot 1 (blue) for the
# one real series, status "good" green for the savings callout/region,
# chart chrome/ink roles for everything else -- never the series color for
# text.
BLUE = "#2a78d6"
GOOD_GREEN = "#0ca30c"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

LARGE_ONLY_ACCURACY = 0.8066666666666666  # held-out (n=150) large-only baseline


def load_mean_logprob_rows() -> list[dict]:
    with COST_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["signal"] == "mean_logprob"]
    for r in rows:
        r["accuracy"] = float(r["accuracy"])
        r["gpu_seconds_savings_pct_vs_large_only"] = float(r["gpu_seconds_savings_pct_vs_large_only"])
    return sorted(rows, key=lambda r: r["gpu_seconds_savings_pct_vs_large_only"])


def find_headline_point(rows: list[dict]) -> dict:
    """The point with the most GPU-seconds savings that still matches or
    exceeds the large-only baseline accuracy -- the actual headline."""
    candidates = [r for r in rows if r["accuracy"] >= LARGE_ONLY_ACCURACY]
    return max(candidates, key=lambda r: r["gpu_seconds_savings_pct_vs_large_only"])


def main() -> None:
    if not COST_CSV.exists():
        print(f"Missing: {COST_CSV}")
        return

    rows = load_mean_logprob_rows()
    headline = find_headline_point(rows)
    headline_x = headline["gpu_seconds_savings_pct_vs_large_only"]
    headline_y = 100 * headline["accuracy"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # "You get this much for free" region: from 0 savings to the headline
    # point, accuracy still at/above the large-only baseline.
    ax.axvspan(0, headline_x, color=GOOD_GREEN, alpha=0.08, zorder=0)

    # The one real series.
    ax.plot(
        [r["gpu_seconds_savings_pct_vs_large_only"] for r in rows],
        [100 * r["accuracy"] for r in rows],
        "-",
        color=BLUE,
        linewidth=2.5,
        zorder=3,
    )
    ax.plot(
        [r["gpu_seconds_savings_pct_vs_large_only"] for r in rows],
        [100 * r["accuracy"] for r in rows],
        "o",
        color=BLUE,
        markersize=5,
        zorder=3,
    )

    # Large-only baseline: reference line + explicit point + callout.
    ax.axhline(100 * LARGE_ONLY_ACCURACY, color=TEXT_MUTED, linewidth=1, linestyle="--", zorder=1)
    ax.plot(0, 100 * LARGE_ONLY_ACCURACY, "D", color=TEXT_SECONDARY, markersize=10, zorder=4)
    ax.annotate(
        "Large model only\n(no routing)",
        xy=(0, 100 * LARGE_ONLY_ACCURACY),
        xytext=(10, 22),
        textcoords="offset points",
        fontsize=10,
        color=TEXT_SECONDARY,
        arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, linewidth=1),
    )

    # The headline point itself.
    ax.plot(headline_x, headline_y, "o", color=GOOD_GREEN, markersize=12, zorder=5)
    ax.annotate(
        f"Same accuracy,\n{headline_x:.0f}% less GPU cost",
        xy=(headline_x, headline_y),
        xytext=(headline_x - 27, headline_y - 8.5),
        fontsize=13,
        fontweight="bold",
        color=GOOD_GREEN,
        arrowprops=dict(arrowstyle="->", color=GOOD_GREEN, linewidth=1.8),
        zorder=6,
    )

    ax.set_xlabel("GPU-seconds saved vs. always using the large model (%)", fontsize=11, color=TEXT_SECONDARY)
    ax.set_ylabel("Accuracy on held-out problems (%)", fontsize=11, color=TEXT_SECONDARY)

    ax.set_ylim(70, 85)
    max_x = max(r["gpu_seconds_savings_pct_vs_large_only"] for r in rows)
    ax.set_xlim(-3, max_x + 5)

    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.grid(axis="x", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)

    # Title stacked above the subtitle via suptitle/figure-text (not
    # ax.set_title twice), with explicit y positions and a reserved top
    # margin -- avoids the two overlapping.
    fig.suptitle(
        "Same accuracy, less compute",
        fontsize=20, fontweight="bold", color=TEXT_PRIMARY, y=0.99,
    )
    fig.text(
        0.5, 0.925,
        "Routing easy problems to a small model frees up real GPU capacity, with no accuracy loss",
        ha="center", fontsize=11, color=TEXT_SECONDARY,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=140, facecolor=SURFACE)
    print(f"Wrote {OUT_PNG}")
    print(f"Headline point: {headline_x:.1f}% GPU-seconds savings at {headline_y:.1f}% accuracy")


if __name__ == "__main__":
    main()
