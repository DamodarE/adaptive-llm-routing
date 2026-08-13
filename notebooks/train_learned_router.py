# Phase 4: learned router. A logistic regression predicting P(small model
# is wrong on this problem), trained on the 350-problem train split only,
# then used as a confidence signal and threshold-swept the same way
# router_sweep_heldout.py does for the heuristic signals -- fit on train,
# evaluated on the 150-problem held-out split.
#
# Features: mean_logprob, min_logprob, format_validity, response_length
# (word count of the generation -- cheap proxy, no extra generation needed).
# Label: small model wrong (1) / correct (0), from small_with_logprobs.json
# alone. pilot_large.json is only needed downstream, for the same
# escalate-vs-not evaluation logic router_sweep.py already uses (was
# escalating to the large model actually the right call for this problem).
#
# No GPU needed. Run after router_sweep_heldout.py (needs
# results/train_test_split.json, results/small_with_logprobs.json,
# results/pilot_large.json):
#   !pip install -q scikit-learn
#   !python notebooks/train_learned_router.py
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_sweep as rs  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SPLIT_PATH = RESULTS_DIR / "train_test_split.json"
SMALL_PATH = RESULTS_DIR / "small_with_logprobs.json"
LARGE_PATH = RESULTS_DIR / "pilot_large.json"
HEURISTIC_HELDOUT_CSV = RESULTS_DIR / "phase3_sweep_heldout.csv"
OUT_CSV = RESULTS_DIR / "phase4_learned_router.csv"
OUT_COEFFICIENTS = RESULTS_DIR / "phase4_router_coefficients.json"

FEATURE_NAMES = ["mean_logprob", "min_logprob", "format_validity", "response_length"]


def load_joined_records() -> dict[str, dict]:
    """One record per problem: features, small/large correctness, keyed by
    unique_id. response_length is a word count of the small model's own
    generation -- already in small_with_logprobs.json, no rerun needed."""
    small = json.loads(SMALL_PATH.read_text())["results"]
    large = {r["unique_id"]: r["correct"] for r in json.loads(LARGE_PATH.read_text())["results"]}

    records = {}
    for r in small:
        if r["unique_id"] not in large:
            continue
        if r["mean_logprob"] is None or r["min_logprob"] is None:
            # Zero-token generation -- no signal to build features from.
            # Doesn't occur in the current data (checked), but don't train
            # on a fabricated value if it ever does.
            continue
        records[r["unique_id"]] = {
            "unique_id": r["unique_id"],
            "mean_logprob": r["mean_logprob"],
            "min_logprob": r["min_logprob"],
            "format_validity": 1 if r["predicted"] not in (None, "") else 0,
            "response_length": len(r["generation"].split()),
            "small_correct": r["correct"],
            "large_correct": large[r["unique_id"]],
        }
    return records


def to_xy(records: list[dict]) -> tuple[list[list[float]], list[int]]:
    X = [[r[f] for f in FEATURE_NAMES] for r in records]
    y = [0 if r["small_correct"] else 1 for r in records]  # 1 = small model wrong
    return X, y


def fit_model(train_records: list[dict]) -> Pipeline:
    X_train, y_train = to_xy(train_records)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("logreg", LogisticRegression()),
        ]
    )
    model.fit(X_train, y_train)
    return model


def save_coefficients(model: Pipeline) -> None:
    logreg = model.named_steps["logreg"]
    # Coefficients are on the *standardized* feature scale (StandardScaler
    # ran first), so magnitudes are comparable across features with very
    # different raw units (logprobs ~ -3..0, response_length ~ 0..1000+) --
    # a fair "which feature mattered most" comparison, not just an artifact
    # of units.
    coefs = dict(zip(FEATURE_NAMES, logreg.coef_[0].tolist()))
    ranked = sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)

    out = {
        "label": "P(small model wrong) -- positive coefficient = pushes toward WRONG",
        "intercept": logreg.intercept_[0],
        "coefficients_standardized": coefs,
        "ranked_by_magnitude": [name for name, _ in ranked],
    }
    OUT_COEFFICIENTS.write_text(json.dumps(out, indent=2))

    print("\n--- Learned router coefficients (standardized scale) ---")
    print("Positive = pushes toward predicting the small model is WRONG.\n")
    for name, coef in ranked:
        direction = "-> more likely WRONG" if coef > 0 else "-> more likely CORRECT"
        print(f"  {name:18s} {coef:+.3f}  {direction}")
    print(f"  {'intercept':18s} {logreg.intercept_[0]:+.3f}")
    print(f"Wrote {OUT_COEFFICIENTS}")


def sanity_check_direction(model: Pipeline, records: list[dict], label: str) -> None:
    """Confidence should be higher, on average, for problems the small
    model actually got right -- if this comes out backwards, the
    correct/wrong <-> predict_proba column mapping has a sign bug, and
    everything downstream would be silently inverted."""
    X, _ = to_xy(records)
    wrong_col = list(model.classes_).index(1)
    p_wrong = model.predict_proba(X)[:, wrong_col]
    confidence = 1 - p_wrong

    conf_when_correct = [c for c, r in zip(confidence, records) if r["small_correct"]]
    conf_when_wrong = [c for c, r in zip(confidence, records) if not r["small_correct"]]
    mean_correct = sum(conf_when_correct) / len(conf_when_correct)
    mean_wrong = sum(conf_when_wrong) / len(conf_when_wrong)
    print(
        f"[{label}] mean confidence when small model correct: {mean_correct:.3f}  "
        f"when wrong: {mean_wrong:.3f}"
    )
    assert mean_correct > mean_wrong, (
        f"Sanity check FAILED on {label}: confidence should be higher when the "
        "small model is actually correct. Signal is inverted -- do not trust "
        "downstream numbers."
    )


def score_all(model: Pipeline, records: list[dict]) -> list[dict]:
    X, _ = to_xy(records)
    wrong_col = list(model.classes_).index(1)
    p_wrong = model.predict_proba(X)[:, wrong_col]
    scored = []
    for r, pw in zip(records, p_wrong):
        scored.append({**r, "learned_score": 1 - pw})  # confidence = P(correct)
    return scored


def sweep_learned(train_scored: list[dict], held_out_scored: list[dict]) -> list[dict]:
    values = [r["learned_score"] for r in train_scored]
    thresholds = sorted({rs.percentile(values, p) for p in rs.PERCENTILES})
    rows = []
    for t in thresholds:
        row = rs.evaluate_threshold(held_out_scored, "learned_score", t)
        row["signal"] = "learned_router"
        rows.append(row)
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


def nearest(rows: list[dict], target_pct: float) -> dict:
    return min(rows, key=lambda r: abs(float(r["pct_escalated"]) - target_pct))


def print_comparison(learned_rows: list[dict], held_out_records: list[dict]) -> None:
    n = len(held_out_records)
    oracle_acc = sum(r["small_correct"] or r["large_correct"] for r in held_out_records) / n
    large_only_acc = sum(r["large_correct"] for r in held_out_records) / n

    heuristic_rows = []
    if HEURISTIC_HELDOUT_CSV.exists():
        with HEURISTIC_HELDOUT_CSV.open() as f:
            heuristic_rows = [r for r in csv.DictReader(f) if r["signal"] == "mean_logprob"]

    print("\n--- Held-out (n=150) comparison at matched escalation rates ---")
    print(f"large-only baseline: {large_only_acc:.1%}   oracle ceiling: {oracle_acc:.1%}\n")
    header = f"{'target %':<10}{'heuristic (mean_logprob)':<28}{'learned router':<18}"
    print(header)
    print("-" * len(header))
    for target in (20, 30, 50):
        h = nearest(heuristic_rows, target) if heuristic_rows else None
        l = nearest(learned_rows, target)
        h_str = f"{float(h['accuracy']):.1%} (@{float(h['pct_escalated']):.1f}%)" if h else "n/a"
        l_str = f"{l['accuracy']:.1%} (@{l['pct_escalated']:.1f}%)"
        print(f"{target:<10}{h_str:<28}{l_str:<18}")

    if heuristic_rows:
        h20, l20 = nearest(heuristic_rows, 20), nearest(learned_rows, 20)
        delta = l20["accuracy"] - float(h20["accuracy"])
        verdict = "BEATS" if delta > 0.01 else ("LOSES TO" if delta < -0.01 else "TIES WITH")
        print(
            f"\nAt ~20% escalation, the learned router {verdict} the mean_logprob "
            f"heuristic by {delta:+.1%}."
        )


if __name__ == "__main__":
    for path in (SPLIT_PATH, SMALL_PATH, LARGE_PATH):
        if not path.exists():
            print(f"Missing: {path}")
            sys.exit(1)

    split = json.loads(SPLIT_PATH.read_text())
    train_ids, held_out_ids = set(split["train"]), set(split["held_out"])

    all_records = load_joined_records()
    train_records = [all_records[u] for u in train_ids if u in all_records]
    held_out_records = [all_records[u] for u in held_out_ids if u in all_records]
    print(f"train records: {len(train_records)}  held_out records: {len(held_out_records)}")

    model = fit_model(train_records)
    save_coefficients(model)

    sanity_check_direction(model, train_records, "train")
    sanity_check_direction(model, held_out_records, "held_out")

    train_scored = score_all(model, train_records)
    held_out_scored = score_all(model, held_out_records)

    rows = sweep_learned(train_scored, held_out_scored)
    write_csv(rows)
    print_comparison(rows, held_out_records)
