# Decisions Log

Running record of choices made, why, and what alternatives were rejected.
Update this as you go — it's the "why" behind PROJECT_PLAN.md's "what."

Format per entry: date, decision, reasoning, alternatives considered.

---

## 2026-08-07 — Model pair: Qwen2.5-Math-1.5B/7B-Instruct
**Decision:** Use same-family math-specialized models (1.5B small, 7B large).
**Why:** Isolates model size as the variable — avoids confounds from
different tokenizers/training recipes.
**Fallback if needed:** Qwen2.5-1.5B/7B-Instruct (non-math) if the CoT
accuracy gap on MATH-500 turns out too small (unknown until Phase 2).

## 2026-08-07 — Dataset: MATH-500 (not GSM8K, not full MATH)
**Decision:** MATH-500 as primary benchmark.
**Why:** GSM8K — both models near-ceiling, no routing signal. Full MATH
(~5000 problems) — same accuracy-gap profile as MATH-500 but 10x the
inference cost for iteration, without much added robustness.
**Alternatives considered:** GSM8K (rejected — ceiling effect), full MATH
test set (rejected — cost/benefit).

## 2026-08-07 — Grading: math-verify (symbolic equivalence)
**Decision:** Use `math-verify` package, not naive string match.
**Why:** MATH answers are LaTeX expressions; string matching would
undercount correct answers with different formatting.
**Risk flagged:** needs validation against a hand-checked sample before
trusting it for the full baseline run.

## 2026-08-07 — Compute: Kaggle first, PACE Phoenix in parallel
**Decision:** Start immediately on Kaggle (free, 30 GPU-hrs/week), PACE
application running in parallel for larger-scale needs.
**Why:** No approval wait blocks Phase 0-2 work; PACE is for headroom on
Phase 5+ stretch goals (quantization ablations).

## 2026-08-07 — Scope: Phases 0-4 are "done," 5-7 are stretch
**Decision:** Hard bar for done = research design → baselines → heuristic
router → learned router, with real accuracy-vs-compute table.
**Why:** Resume timeline matters more than exhaustiveness. Quantization/full
ablations/engineering polish add scope risk without changing the core
contribution.

---

<!-- Add new entries above this line, newest at top or bottom — pick one and
stay consistent. -->
