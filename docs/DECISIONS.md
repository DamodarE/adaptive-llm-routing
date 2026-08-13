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
- Qwen2.5-Math-1.5B-Instruct: 73.4% (367/500), load 35.8s, gen 143.3s,
  mean latency 0.29s/problem
- Qwen2.5-Math-7B-Instruct: 79.0% (395/500), load 55.7s, gen 275.0s,
  mean latency 0.55s/problem
- Oracle (best of either model per-problem): 83.4% (417/500)

**Correction (2026-08-13):** small-model and oracle counts above were
originally logged as 72.8% (364/500) and 83.2% (416/500). Corrected against
the actual committed `results/pilot_small.json`/`results/pilot_large.json`
(367/500 and 417/500 respectively) during a repo-wide review — likely
either a transcription slip when these were first reported, or vllm's
greedy decoding not being perfectly run-to-run reproducible (batch
composition can affect floating-point accumulation order). The 7B number
(395/500) was correct as originally logged. Numbers below and in
`README.md`/`PROJECT_PLAN.md` reflect the corrected values.

**Comparison to n=50 pilot:** all three numbers moved down slightly from
the 50-problem estimate (76%→73.4%, 84%→79.0%, 88%→83.4%), but the
qualitative story held — a consistent ~5.6-point gap between models and a
real oracle lift (4.4 points) over the large-only baseline.

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

## 2026-08-13 — Held-out train/test split added; Phase 3 re-evaluated honestly
**Decision:** Split the 500 MATH-500 problems into a 350-problem train set
and a 150-problem held-out set (`notebooks/make_split.py`, seed=42,
stratified by subject via largest-remainder apportionment so each split's
subject mix matches the full set), saved to
`results/train_test_split.json` so it's reproducible and reused by both
Phase 3's re-evaluation and Phase 4's learned router.
**Why:** the last review (2026-08-12/13) flagged that every number in
`phase3_sweep.csv` was fit (percentile thresholds) and evaluated on the
same 500 problems — no held-out validation anywhere in the pipeline. That
needed fixing before Phase 4, where training a classifier with zero
held-out evaluation would be a much more serious version of the same
problem.
**What changed when re-evaluated honestly:** `notebooks/router_sweep_heldout.py`
selects thresholds from the train split only, then evaluates on the
held-out 150 (`results/phase3_sweep_heldout.csv`, does not overwrite the
original). Raw held-out accuracy came out *higher* than the full-500
numbers at every comparable escalation rate (e.g. mean_logprob at ~20%
escalation: 76.4% full-500 vs. 80.0% held-out) — but that is **not**
evidence the thresholds generalize better. The held-out 150 problems
happen to have an easier baseline purely from which problems landed in
that split (large-only accuracy on held-out alone is 80.7% vs. 79.0% on
the full 500, oracle 85.3% vs. 83.4% — before any thresholding at all).
Normalizing for this (reporting lift over each split's own large-only
baseline instead of raw accuracy) is the apples-to-apples comparison:
mean_logprob's lift at ~20% escalation is -2.6% on full-500 vs. -0.7% on
held-out; at ~50%, -0.6% vs. +0.7%. min_logprob is similar. **Conclusion:
no evidence of meaningful threshold overfitting** — held-out lift is
comparable to, if anything marginally better than, the original
fit-and-eval-on-same-data numbers. This isn't a strong guarantee either
way, though: n=150 is small enough that these ~1-2 point deltas are within
plausible sampling noise, not a precise measurement.
**Next:** Phase 4's learned router will train on the 350-problem train
split and report final numbers on the same 150-problem held-out split, so
train/test discipline is consistent with what's here.

## 2026-08-14 — Phase 4 learned router: doesn't beat the heuristic, reporting honestly
**What was built:** a logistic regression (`sklearn`, `StandardScaler` +
default L2) predicting P(small model wrong), trained on the 350-problem
train split only (`notebooks/train_learned_router.py`). Features:
`mean_logprob`, `min_logprob`, `format_validity`, and `response_length`
(word count of the small model's generation — a free, cheap feature already
available in `small_with_logprobs.json`). Its output probability was
threshold-swept the same way as the heuristics (percentiles of the TRAIN
distribution, evaluated on the 150-problem held-out split), written to
`results/phase4_learned_router.csv`. A sanity check (mean predicted
confidence must be higher on problems the small model actually got right)
passed on both splits before trusting any downstream number — train: 0.851
vs. 0.406; held-out: 0.851 vs. 0.481.

**Result — reported honestly per PROJECT_PLAN.md's framing commitment: the
learned router does not beat the simple mean_logprob heuristic on held-out
data.** At matched escalation rates (all held-out, n=150, large-only
baseline 80.7%, oracle 85.3%):

| ~target escalation | heuristic (mean_logprob) | learned router |
|---|---|---|
| 20% | 80.0% (@20.7%) | 78.0% (@18.7%) |
| 30% | 79.3% (@28.7%) | 78.0% (@30.7%) |
| 50% | 81.3% (@49.3%) | 79.3% (@48.0%) |

The learned router loses by roughly 1.3–2.0 points at every matched rate
checked, and the full curve (`results/phase4_comparison_plot.png`) shows
mean_logprob at or above the learned router across nearly the whole
escalation range. Plausible explanation, not confirmed: n=350 training
examples for a 4-feature model is enough to learn *something* but not much
more than what the single best heuristic feature already captures, and the
logistic regression doesn't obviously buy anything a simple threshold on
`mean_logprob` didn't already have.

**Coefficients (standardized scale, so magnitudes are comparable across
features with very different raw units) — the quotable finding:**

| feature | coefficient | direction |
|---|---|---|
| `format_validity` | -1.360 | strongest predictor — a parseable `\boxed{}` answer means much more likely correct |
| `response_length` | +1.080 | longer generations are more likely wrong |
| `mean_logprob` | -0.899 | higher (more confident) average token logprob means more likely correct, as expected |
| `min_logprob` | +0.146 | **counterintuitive sign** — see below |

`min_logprob`'s positive coefficient looks backwards (a more-confident
worst-token should mean *less* likely wrong) until you check it in
isolation: `min_logprob`'s own univariate logistic coefficient is -0.467
(correctly signed) and its univariate correlation with wrongness is -0.218,
both in the expected direction. It flips sign only once `mean_logprob` is
also in the model — the two are correlated at r=0.525, `mean_logprob` has
the stronger standalone relationship with correctness, and it absorbs most
of their shared signal, leaving `min_logprob` a slightly reversed
"leftover" coefficient. This is a real, verified multicollinearity effect
(checked directly, not assumed), not a bug — but it's a good example of why
a coefficient's sign in a multivariate model can't be read the same way as
a univariate correlation.

**Decision:** ship the heuristic (mean_logprob threshold) as the
recommended router for now, not the learned model — it's simpler, cheaper,
and currently more accurate on held-out data. Keep the learned-router code
and this result in the project as evidence of following through on the
plan's "honest analysis if it doesn't win" commitment, not as the
recommended approach. Revisit if `level`/`subject` features get added
(currently not captured) or if n grows beyond MATH-500's 500 problems —
a 4-feature model trained on 350 examples may simply not have enough
signal or data to beat a well-chosen single heuristic yet.

---

<!-- Add new entries above this line, newest at top or bottom — pick one and
stay consistent. -->
