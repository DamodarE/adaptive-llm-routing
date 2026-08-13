# Phase 3, held-out evaluation: the honest version of router_sweep.py's
# numbers. Thresholds are selected using ONLY the train split
# (results/train_test_split.json), then evaluated on the held-out split --
# not fit and evaluated on the same 500 problems, which is what
# router_sweep.py / phase3_sweep.csv do.
#
# Reuses router_sweep.py's load_records/percentile/evaluate_threshold
# directly (imported, not re-typed) so the two stay consistent. Does not
# modify or overwrite router_sweep.py or phase3_sweep.csv.
#
# No GPU needed. Run after make_split.py and router_sweep.py:
#   !python notebooks/router_sweep_heldout.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_sweep as rs  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SPLIT_PATH = RESULTS_DIR / "train_test_split.json"
ORIGINAL_CSV = RESULTS_DIR / "phase3_sweep.csv"
OUT_CSV = RESULTS_DIR / "phase3_sweep_heldout.csv"


def load_split() -> tuple[set[str], set[str]]:
    split = json.loads(SPLIT_PATH.read_text())
    return set(split["train"]), set(split["held_out"])


def sweep_heldout(
    records: list[dict], train_ids: set[str], held_out_ids: set[str]
) -> tuple[list[dict], int, int]:
    train_records = [r for r in records if r["unique_id"] in train_ids]
    held_out_records = [r for r in records if r["unique_id"] in held_out_ids]

    rows = []
    for signal in ("mean_logprob", "min_logprob"):
        # Thresholds are percentiles of the TRAIN split's distribution only.
        train_values = [r[signal] for r in train_records if r[signal] is not None]
        thresholds = sorted({rs.percentile(train_values, p) for p in rs.PERCENTILES})
        for t in thresholds:
            # Evaluated on the HELD-OUT split only -- unseen during
            # threshold selection.
            rows.append(rs.evaluate_threshold(held_out_records, signal, t))

    # format_validity's threshold (1) isn't fit from data, so there's
    # nothing to leak here -- evaluated on held-out anyway for a fair,
    # apples-to-apples comparison against the other two signals.
    rows.append(rs.evaluate_threshold(held_out_records, "format_validity", 1))
    return rows, len(train_records), len(held_out_records)


def load_original_by_signal() -> dict[str, list[dict]]:
    if not ORIGINAL_CSV.exists():
        return {}
    with ORIGINAL_CSV.open() as f:
        rows = list(csv.DictReader(f))
    by_signal: dict[str, list[dict]] = {}
    for r in rows:
        r["pct_escalated"] = float(r["pct_escalated"])
        r["accuracy"] = float(r["accuracy"])
        by_signal.setdefault(r["signal"], []).append(r)
    return by_signal


def nearest(rows: list[dict], target_pct: float) -> dict:
    return min(rows, key=lambda r: abs(r["pct_escalated"] - target_pct))


def split_baselines(records: list[dict]) -> dict[str, float]:
    """small-only / large-only / oracle accuracy *within this specific
    split* -- needed because raw accuracy isn't comparable across splits
    of different size/composition on its own (see print_comparison)."""
    n = len(records)
    return {
        "small_only": sum(r["small_correct"] for r in records) / n,
        "large_only": sum(r["large_correct"] for r in records) / n,
        "oracle": sum(r["small_correct"] or r["large_correct"] for r in records) / n,
    }


def print_comparison(
    heldout_rows: list[dict],
    original_by_signal: dict[str, list[dict]],
    full_baselines: dict[str, float],
    heldout_baselines: dict[str, float],
) -> None:
    heldout_by_signal: dict[str, list[dict]] = {}
    for r in heldout_rows:
        heldout_by_signal.setdefault(r["signal"], []).append(r)

    print(
        "\n--- Split baselines (these differ just from which problems "
        "landed in which split, before any thresholding) ---"
    )
    for name, baselines in (("full-500", full_baselines), ("held-out (150)", heldout_baselines)):
        print(
            f"  {name:16s} small-only={baselines['small_only']:.1%}  "
            f"large-only={baselines['large_only']:.1%}  oracle={baselines['oracle']:.1%}"
        )

    print(
        "\n--- Full-500 (fit + eval on same 500) vs. held-out "
        "(fit on 350 train, eval on unseen 150) ---\n"
        "raw accuracy is not directly comparable across splits of different "
        "composition -- 'lift' below is accuracy minus that split's OWN "
        "large-only baseline, which is the apples-to-apples number."
    )
    header = (
        f"{'signal':<18}{'target':<8}{'full500 acc':<13}{'full500 lift':<14}"
        f"{'heldout acc':<13}{'heldout lift':<14}"
    )
    print(header)
    print("-" * len(header))
    for signal in ("mean_logprob", "min_logprob", "format_validity"):
        ho_rows = heldout_by_signal.get(signal, [])
        orig_rows = original_by_signal.get(signal, [])
        if not ho_rows or not orig_rows:
            continue
        targets = (20, 50) if len(ho_rows) > 1 else (None,)
        for target in targets:
            ho_r = ho_rows[0] if target is None else nearest(ho_rows, target)
            orig_r = orig_rows[0] if target is None else nearest(orig_rows, target)
            orig_lift = orig_r["accuracy"] - full_baselines["large_only"]
            ho_lift = ho_r["accuracy"] - heldout_baselines["large_only"]
            label = f"~{target}%" if target is not None else "n/a"
            print(
                f"{signal:<18}{label:<8}{orig_r['accuracy']:<13.1%}{orig_lift:<+14.1%}"
                f"{ho_r['accuracy']:<13.1%}{ho_lift:<+14.1%}"
            )


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "signal",
                "threshold",
                "pct_escalated",
                "accuracy",
                "false_escalation_rate",
                "dangerous_non_escalation_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    if not SPLIT_PATH.exists():
        print(f"Missing: {SPLIT_PATH} (run notebooks/make_split.py first)")
        sys.exit(1)

    train_ids, held_out_ids = load_split()
    records = rs.load_records()
    rows, n_train, n_held_out = sweep_heldout(records, train_ids, held_out_ids)
    print(f"train records matched: {n_train}  held_out records matched: {n_held_out}")

    write_csv(rows)

    held_out_records = [r for r in records if r["unique_id"] in held_out_ids]
    print_comparison(
        rows,
        load_original_by_signal(),
        full_baselines=split_baselines(records),
        heldout_baselines=split_baselines(held_out_records),
    )
