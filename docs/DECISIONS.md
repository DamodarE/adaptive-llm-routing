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

## 2026-08-09 — Library: transformers (not vllm) for Phase 1
**Decision:** Use `transformers` for model loading/inference in Phase 1.
**Why:** Simpler setup and debugging while getting the environment working.
**Revisit:** Reconsider `vllm` at Phase 2 for throughput once running the
full MATH-500 benchmark makes inference speed matter.

## 2026-08-09 — Compute: Kaggle T4x2 (not single T4)
**Decision:** Use Kaggle's T4x2 accelerator setting, not a single T4.
**Why:** The 7B model is ~15GB in FP16, which barely fits a single 16GB T4
with no headroom left for inference (KV cache, activations).

## 2026-08-09 — Phase 1 sanity check — passed.
Both Qwen2.5-Math models (1.5B, 7B) loaded successfully via `transformers`
with `device_map="auto"` on Kaggle T4x2, split automatically across GPUs by
Accelerate. Both correctly solved the test problem (2x+5=17 -> x=6). Peak
memory: 7B model used 13.03GB/13.10GB on GPU 0, 4.06GB/4.10GB on GPU 1 —
comfortable headroom, no OOM risk observed. Confirms environment is ready
for Phase 2.

## 2026-08-10 — Grading: math-verify with explicit LaTeX-environment wrapping
**Decision:** Wrap both gold and predicted answers in `$...$` before calling
`math_verify.parse`/`verify`, instead of passing raw strings.
**Why:** math-verify has native support for tuples, intervals, and finite
sets (see its `grader.py` and README's set-theory example) — no custom
multi-value comparator needed. But its LaTeX extraction only activates
inside a LaTeX environment delimiter; the README states "the latex must be
placed in latex environment to be parsable." Unwrapped strings risk falling
through to plain-expression extraction and mis-grading multi-value answers.
**Verification status:** Confirmed by reading math-verify's GitHub README
and source (native tuple/set/interval comparison in `grader.py`, LaTeX
delimiter requirement stated in the README). Not yet confirmed by running
it — this environment has Python 3.9 and math-verify requires >=3.10.
Needs a smoke test against real MATH-500 tuple/interval answers (e.g. on
Kaggle/PACE) before trusting it for reported results.

---

<!-- Add new entries above this line, newest at top or bottom — pick one and
stay consistent. -->
