# Phase 3: heuristic router threshold sweep.
#
# Reads results/small_with_logprobs.json (small model, 500 problems, with
# per-problem mean_logprob/min_logprob from rerun_small_with_logprobs.py)
# and results/pilot_large.json (large model baseline, untouched). For each
# candidate confidence signal, sweeps thresholds and computes what a
# heuristic cascade router would achieve: keep a problem on the small model
# if its signal >= threshold, else escalate to the large model.
#
# Run after both input files exist:
#   !python notebooks/router_sweep.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SMALL_PATH = RESULTS_DIR / "small_with_logprobs.json"
LARGE_PATH = RESULTS_DIR / "pilot_large.json"
OUT_CSV = RESULTS_DIR / "phase3_sweep.csv"

PERCENTILES = list(range(5, 100, 5))  # 5, 10, ..., 95 -- 19 thresholds (~20)


def load_records() -> list[dict]:
    missing = [p for p in (SMALL_PATH, LARGE_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"Missing: {p}")
        sys.exit(1)

    small = json.loads(SMALL_PATH.read_text())
    large = json.loads(LARGE_PATH.read_text())

    small_by_id = {r["unique_id"]: r for r in small["results"]}
    large_by_id = {r["unique_id"]: r for r in large["results"]}
    common_ids = sorted(set(small_by_id) & set(large_by_id))
    if len(common_ids) != len(small_by_id) or len(common_ids) != len(large_by_id):
        print(
            f"Note: small has {len(small_by_id)} problems, large has "
            f"{len(large_by_id)}; sweep computed over the "
            f"{len(common_ids)} problems common to both.\n"
        )

    records = []
    for uid in common_ids:
        s, l = small_by_id[uid], large_by_id[uid]
        predicted = s["predicted"]
        records.append(
            {
                "unique_id": uid,
                "small_correct": s["correct"],
                "large_correct": l["correct"],
                "mean_logprob": s.get("mean_logprob"),
                "min_logprob": s.get("min_logprob"),
                "format_validity": 1 if predicted not in (None, "") else 0,
            }
        )
    return records


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (numpy's default method), no numpy
    dependency required."""
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (pct / 100) * (len(s) - 1)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def evaluate_threshold(records: list[dict], signal: str, threshold: float) -> dict:
    n = len(records)
    n_escalated = 0
    n_correct = 0

    # false_escalation_rate and dangerous_non_escalation_rate are rates
    # *within the subset the error type can apply to* (like a false-positive
    # rate), not raw fractions of all 500 problems:
    #   false_escalation_rate = of problems the small model already got
    #     right, what fraction got escalated anyway (wasted compute).
    #   dangerous_non_escalation_rate = of problems where escalating would
    #     have flipped wrong -> right, what fraction were NOT escalated
    #     (missed accuracy -- the more costly error type per PROJECT_PLAN.md
    #     §5).
    small_correct_n = 0
    false_escalations = 0
    at_risk_n = 0
    dangerous_non_escalations = 0

    for r in records:
        value = r[signal]
        # A missing signal (e.g. zero-token generation) can't be compared
        # to a threshold -- treat it as minimum confidence, i.e. always
        # escalate, rather than crashing the sweep.
        escalate = value is None or value < threshold
        n_escalated += escalate
        n_correct += r["large_correct"] if escalate else r["small_correct"]

        if r["small_correct"]:
            small_correct_n += 1
            if escalate:
                false_escalations += 1

        if (not r["small_correct"]) and r["large_correct"]:
            at_risk_n += 1
            if not escalate:
                dangerous_non_escalations += 1

    return {
        "signal": signal,
        "threshold": threshold,
        "pct_escalated": 100 * n_escalated / n,
        "accuracy": n_correct / n,
        "false_escalation_rate": (
            false_escalations / small_correct_n if small_correct_n else 0.0
        ),
        "dangerous_non_escalation_rate": (
            dangerous_non_escalations / at_risk_n if at_risk_n else 0.0
        ),
    }


def sweep(records: list[dict]) -> list[dict]:
    rows = []
    for signal in ("mean_logprob", "min_logprob"):
        values = [r[signal] for r in records if r[signal] is not None]
        thresholds = sorted({percentile(values, p) for p in PERCENTILES})
        for t in thresholds:
            rows.append(evaluate_threshold(records, signal, t))

    # format_validity is binary -- one operating point, not a percentile
    # sweep: escalate iff predicted answer failed to parse.
    rows.append(evaluate_threshold(records, "format_validity", 1))
    return rows


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


def print_summary(rows: list[dict]) -> None:
    print("\n--- Best accuracy near 20% and 50% escalation ---")
    header = f"{'signal':<18}{'target %':<10}{'actual %':<10}{'accuracy':<10}"
    print(header)
    print("-" * len(header))
    for signal in ("mean_logprob", "min_logprob", "format_validity"):
        signal_rows = [r for r in rows if r["signal"] == signal]
        if not signal_rows:
            continue
        if len(signal_rows) == 1:
            r = signal_rows[0]
            print(
                f"{signal:<18}{'n/a':<10}{r['pct_escalated']:<10.1f}"
                f"{r['accuracy']:<10.1%}"
            )
            continue
        for target in (20, 50):
            nearest = min(signal_rows, key=lambda r: abs(r["pct_escalated"] - target))
            print(
                f"{signal:<18}{target:<10}{nearest['pct_escalated']:<10.1f}"
                f"{nearest['accuracy']:<10.1%}"
            )


if __name__ == "__main__":
    records = load_records()
    rows = sweep(records)
    write_csv(rows)
    print_summary(rows)
