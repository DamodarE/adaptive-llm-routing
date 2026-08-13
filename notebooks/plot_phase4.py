# Phase 4: plot heuristic (mean_logprob, min_logprob) vs. learned router
# curves together, all evaluated on the same held-out 150 -- an honest,
# apples-to-apples comparison (as opposed to plot_router_sweep.py, which
# plots the full-500 fit-and-eval-on-same-data numbers).
#
# Run after train_learned_router.py (needs results/phase3_sweep_heldout.csv
# and results/phase4_learned_router.csv):
#   !python notebooks/plot_phase4.py
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
HEURISTIC_CSV = RESULTS_DIR / "phase3_sweep_heldout.csv"
LEARNED_CSV = RESULTS_DIR / "phase4_learned_router.csv"
SPLIT_PATH = RESULTS_DIR / "train_test_split.json"
SMALL_PATH = RESULTS_DIR / "small_with_logprobs.json"
LARGE_PATH = RESULTS_DIR / "pilot_large.json"
OUT_PNG = RESULTS_DIR / "phase4_comparison_plot.png"


def load_csv_rows(path: Path) -> list[dict]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["pct_escalated"] = float(r["pct_escalated"])
        r["accuracy"] = float(r["accuracy"])
    return rows


def held_out_baselines() -> dict:
    """small-only / large-only / oracle accuracy on the held-out 150 only
    -- the correct reference lines for a held-out-only plot, not the
    full-500 baselines plot_router_sweep.py uses."""
    split = json.loads(SPLIT_PATH.read_text())
    held_out_ids = set(split["held_out"])
    small = {r["unique_id"]: r["correct"] for r in json.loads(SMALL_PATH.read_text())["results"]}
    large = {r["unique_id"]: r["correct"] for r in json.loads(LARGE_PATH.read_text())["results"]}
    ids = [u for u in held_out_ids if u in small and u in large]
    n = len(ids)
    return {
        "small-only": sum(small[u] for u in ids) / n,
        "large-only": sum(large[u] for u in ids) / n,
        "oracle": sum(1 for u in ids if small[u] or large[u]) / n,
    }


def main() -> None:
    missing = [p for p in (HEURISTIC_CSV, LEARNED_CSV) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        return

    heuristic_rows = load_csv_rows(HEURISTIC_CSV)
    learned_rows = load_csv_rows(LEARNED_CSV)
    baselines = held_out_baselines() if SPLIT_PATH.exists() and SMALL_PATH.exists() else {}

    fig, ax = plt.subplots(figsize=(8, 6))

    for signal, style, color in (
        ("mean_logprob", "o-", "tab:blue"),
        ("min_logprob", "s-", "tab:orange"),
    ):
        rows = sorted(
            (r for r in heuristic_rows if r["signal"] == signal), key=lambda r: r["pct_escalated"]
        )
        if rows:
            ax.plot(
                [r["pct_escalated"] for r in rows],
                [100 * r["accuracy"] for r in rows],
                style,
                color=color,
                label=f"{signal} (heuristic)",
                markersize=4,
            )

    learned_sorted = sorted(learned_rows, key=lambda r: r["pct_escalated"])
    if learned_sorted:
        ax.plot(
            [r["pct_escalated"] for r in learned_sorted],
            [100 * r["accuracy"] for r in learned_sorted],
            "^-",
            color="tab:green",
            label="learned router (logistic regression)",
            markersize=5,
            linewidth=2,
        )

    for name, color in (("small-only", "gray"), ("large-only", "firebrick"), ("oracle", "seagreen")):
        if name in baselines:
            acc = baselines[name]
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
    ax.set_ylabel("Accuracy on held-out set (%)")
    ax.set_title("Phase 3 heuristic vs. Phase 4 learned router (held-out n=150)")
    ax.set_xlim(-2, 100)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
