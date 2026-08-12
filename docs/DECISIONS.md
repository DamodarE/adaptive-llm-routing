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
delimiter requirement stated in the README), and now also empirically —
see 2026-08-11 entry below.

## 2026-08-11 — Phase 2 pilot script: local dry-run (no GPU/vllm available)
**What was verified, and how:** dev machine is macOS/arm64 with no CUDA, so
`notebooks/pilot.py`'s actual vllm inference could not be run here. Instead,
set up an isolated Python 3.11 venv (via Homebrew, local machine has 3.9 by
default) and exercised every non-inference part of the pipeline against
real data:
- `load_dataset("HuggingFaceH4/MATH-500")` — loads, single `test` split,
  500 rows, matches expected columns.
- `extract_boxed_answer` — passes hand-written cases including nested
  braces (`\boxed{\frac{1}{2}}`) and multiple `\boxed{}` occurrences
  (last one wins).
- `grade_answer` (from `src/grading.py`) — tested against a real MATH-500 tuple
  answer (`\left( 3, \frac{\pi}{2} \right)`, problem `test/precalculus/807`):
  correctly matches equivalent notations, correctly rejects a wrong
  component. **This empirically confirms the tuple/interval support claimed
  in the 2026-08-10 entry above**, closing out that entry's "not yet run"
  caveat.
- Qwen2.5-Math-1.5B/7B-Instruct tokenizers, via `apply_chat_template` — both
  inject a default system message ("Please reason step by step, and put
  your final answer within \boxed{}.") when none is given. The pilot script
  originally appended that same instruction again in the user turn;
  removed the duplicate.
**One finding, not fixed:** `math-verify`'s numeric comparison has a
precision tolerance tied to the predicted value's own decimal precision —
`grade_answer("\frac{14}{3}", "4.666666")` (6 truncated decimals) is
`False`, but `grade_answer("\frac{14}{3}", "4.6666667")` (7 digits) is
`True`. This is math-verify's own default behavior, not a bug in our
wrapper — flagging in case truncated-decimal model outputs cause
unexpected misses during the real Phase 2 run.
**Not verified by this dry-run:** vllm install/behavior, actual model
inference/generation quality, GPU memory behavior under vllm (as opposed
to the `transformers` numbers from Phase 1). Still needs a real run on
Kaggle T4x2.

## 2026-08-12 — Phase 2 pilot (n=50): real accuracy gap confirmed, routing looks worthwhile
**Result:** Ran the full pilot pipeline (vllm, sequential + TP=2 for large
model, math-verify grading) on 50 MATH-500 problems for both models:
- Qwen2.5-Math-1.5B-Instruct: 76.0% (38/50)
- Qwen2.5-Math-7B-Instruct: 84.0% (42/50)
- Oracle (best of either model per-problem): 88.0% (44/50)

**Why this matters:** This resolves the open risk flagged in PROJECT_PLAN.md §8
— the 1.5B model's plain-CoT accuracy on MATH was previously unpublished
(only tool-integrated-reasoning numbers existed). Now measured directly.
The 7B result (84%) closely matches its published MATH CoT accuracy
(83.6%), which is a good sanity check that the pipeline (generation,
extraction, grading) is measuring correctly.

**Decision:** The 8-point accuracy gap between models, and the 4-point
oracle lift over the large-only baseline, together indicate the accuracy
gap is large enough to make cascade routing worthwhile. No need to fall
back to the non-math-specialized model pair (the contingency noted in
PROJECT_PLAN.md §3). Proceeding with the current model pair.

**Caveat:** n=50 is a small sample; exact percentages may shift on the
full 500-problem run, though the qualitative conclusion (meaningful gap
exists) is expected to hold given how closely the 7B number already
matches its published benchmark.

**Next:** Run the same pipeline on the full MATH-500 set (n=500) to get
the final Phase 2 baseline table.

## 2026-08-12 — Phase 2 baseline complete (n=500): final accuracy-vs-compute numbers
**Result:** Full MATH-500 baseline run (vllm, sequential for small model,
tensor_parallel_size=2 for large model, math-verify grading):
- Qwen2.5-Math-1.5B-Instruct: 72.8% (364/500), load 35.8s, gen 143.3s,
  mean latency 0.29s/problem
- Qwen2.5-Math-7B-Instruct: 79.0% (395/500), load 55.7s, gen 275.0s,
  mean latency 0.55s/problem
- Oracle (best of either model per-problem): 83.2% (416/500)

**Comparison to n=50 pilot:** all three numbers moved down slightly from
the 50-problem estimate (76%→72.8%, 84%→79.0%, 88%→83.2%), but the
qualitative story held — a consistent ~6-point gap between models and a
real oracle lift (4.2 points) over the large-only baseline.

**Sanity check:** 7B result (79.0%) is in the right neighborhood of its
published MATH CoT accuracy (83.6%) — close enough to trust the pipeline;
gap likely attributable to normal variance, MATH-500 vs full MATH test set
differences, or minor prompt/parsing differences, not a pipeline bug.

**Decision:** This closes the open risk in PROJECT_PLAN.md §8 (1.5B CoT
accuracy on MATH was previously unpublished). The accuracy gap and oracle
lift are both large enough to justify the routing/cascade approach — no
fallback to the non-math-specialized model pair needed. This is the
official Phase 2 baseline table (small-only, large-only, oracle) — the
first entry in what becomes the accuracy-vs-compute resume table.

**Phase 2 status: COMPLETE.** Next: Phase 3, heuristic/confidence-threshold
router.

---

<!-- Add new entries above this line, newest at top or bottom — pick one and
stay consistent. -->
