# Reproducible train/held-out split of the 500 MATH-500 problems, stratified
# by subject (recovered from unique_id, e.g. "test/precalculus/807.json")
# so both splits have a similar subject mix. Written once and reused by
# both Phase 3's held-out re-evaluation and Phase 4's learned router, so
# train/test discipline stays consistent across both.
#
# No GPU needed -- reads results/pilot_small.json (any pilot result file
# works; only unique_id is used) and writes results/train_test_split.json.
#
# Run:
#   !python notebooks/make_split.py
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from random import Random

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SOURCE_PATH = RESULTS_DIR / "pilot_small.json"
OUT_PATH = RESULTS_DIR / "train_test_split.json"

SEED = 42
TRAIN_FRACTION = 0.7


def subject_of(unique_id: str) -> str:
    m = re.match(r"test/([^/]+)/", unique_id)
    return m.group(1) if m else "unknown"


def stratified_split(
    unique_ids: list[str], train_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    by_subject: dict[str, list[str]] = defaultdict(list)
    # Sort first so the per-subject shuffle below is deterministic
    # regardless of the input list's original order.
    for uid in sorted(unique_ids):
        by_subject[subject_of(uid)].append(uid)

    rng = Random(seed)
    raw_counts = {}
    for subject, ids in by_subject.items():
        rng.shuffle(ids)
        raw = len(ids) * train_fraction
        raw_counts[subject] = (ids, int(raw))

    # Largest-remainder apportionment: per-subject train counts are
    # rounded down individually, then the shortfall against the overall
    # rounded target is handed out, one problem each, to the subjects with
    # the largest fractional remainder -- so per-subject counts stay exact
    # integers and still sum to exactly round(n * train_fraction) overall,
    # not just approximately.
    target_train_total = round(len(unique_ids) * train_fraction)
    base_total = sum(n for _, n in raw_counts.values())
    n_to_bump = target_train_total - base_total
    remainders = sorted(
        ((len(ids) * train_fraction - n, subject) for subject, (ids, n) in raw_counts.items()),
        reverse=True,
    )
    bump_subjects = {subject for _, subject in remainders[:n_to_bump]}

    train, held_out = [], []
    for subject, (ids, n_train) in raw_counts.items():
        if subject in bump_subjects:
            n_train += 1
        train.extend(ids[:n_train])
        held_out.extend(ids[n_train:])

    train.sort()
    held_out.sort()
    return train, held_out


def main() -> None:
    if not SOURCE_PATH.exists():
        print(f"Missing: {SOURCE_PATH} (run pilot.py --model small first)")
        return

    data = json.loads(SOURCE_PATH.read_text())
    unique_ids = [r["unique_id"] for r in data["results"]]

    train, held_out = stratified_split(unique_ids, TRAIN_FRACTION, SEED)

    counts = defaultdict(lambda: {"train": 0, "held_out": 0})
    for uid in train:
        counts[subject_of(uid)]["train"] += 1
    for uid in held_out:
        counts[subject_of(uid)]["held_out"] += 1

    out = {
        "seed": SEED,
        "train_fraction": TRAIN_FRACTION,
        "n_train": len(train),
        "n_held_out": len(held_out),
        "train": train,
        "held_out": held_out,
        "subject_counts": dict(sorted(counts.items())),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))

    print(f"train: {len(train)}  held_out: {len(held_out)}")
    for subject, c in sorted(counts.items()):
        print(f"  {subject:25s} train={c['train']:3d}  held_out={c['held_out']:3d}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
