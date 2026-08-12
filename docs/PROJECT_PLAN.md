# Project Plan: Adaptive LLM Routing for Efficient Mathematical Reasoning

## 1. Research Question

Can we retain most of a large model's mathematical reasoning accuracy while
substantially cutting inference cost, by sending easy problems to a small
model and only escalating hard problems to a large model?

The objective is not maximum accuracy. It's characterizing the tradeoff
between accuracy, latency, and compute cost — and showing where on that
curve a routing/cascade system can land.

## 2. Relationship to Prior Work

This project extends, rather than repeats, [friend]'s knowledge-distillation
project. His work asks: *can a small model learn to imitate a large one
during training?* This project assumes two already-trained models exist and
asks a different, inference-time question: *given a query, which model
should answer it?* Distillation and routing are complementary — a system
could use both — but the contribution here is entirely in the
model-selection/inference layer, not training.

**Terminology note:** architecturally, what this project builds is what the
literature calls an *LLM cascade* — query the cheap model first, escalate
based on a confidence signal — rather than a *router* in the strict sense
(deciding which model to use before either one runs). FrugalGPT (Chen,
Zaharia, and Zou, 2023) is the foundational paper for this cascade pattern,
and reports matching a strong model's performance at up to 98% lower cost by
exploiting exactly this kind of query-difficulty skew. "Routing" is used
colloquially throughout this project to match the common usage, but the
precise term is worth knowing.

### Related work (starting points, not exhaustive)

- **FrugalGPT** (Chen, Zaharia, Zou, 2023) — foundational LLM cascade work;
  sequentially queries cheaper models first, escalating based on a learned
  reliability score.
- **Hybrid LLM** (Ding et al., 2024) — routes queries to a small or large
  model based on predicted query difficulty.
- **RouteLLM** (Ong et al., 2025) — learns routers from preference data to
  choose between weaker/stronger models.
- Entropy-trajectory work (2026) showing that whether a model's token-level
  entropy decreases monotonically during chain-of-thought predicts answer
  reliability — a directly usable signal for the Phase 3 heuristic router,
  evaluated on MATH-500 specifically.

## 3. Models

| | Small | Large |
|---|---|---|
| Model | Qwen2.5-Math-1.5B-Instruct | Qwen2.5-Math-7B-Instruct |
| Parameters | 1.5B | 7B |
| Memory (FP16) | ~3GB | ~15GB |
| Mode | CoT only | CoT only |

Same model family (both math-specialized Qwen2.5-Math instruct models) —
isolates model *size* as the variable being studied, avoiding confounds from
different tokenizers or training recipes between the small and large model.

CoT-only (no tool-integrated/code-execution mode) by design: simpler
pipeline, no code sandbox to secure, and matches the confidence signals
planned for Phase 3 (token probability, entropy, answer formatting).

**Known unknown:** the 7B model's CoT scores are published (95.2% GSM8K,
83.6% MATH). The 1.5B model's plain-CoT score is not — its only published
number (~80% MATH) uses tool-integrated reasoning, which we won't be using.
**Phase 2's full MATH-500 run (n=500) produced the real number:** 1.5B CoT
accuracy is 72.8% (364/500), vs. 7B's 79.0% (395/500) — a ~6-point gap,
confirmed large enough to proceed with this model pair. If the CoT accuracy
gap between 1.5B and 7B on MATH-500 had turned out too small to give
routing anything meaningful to do, the fallback would have been the
non-math-specialized pair (Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct),
which trades a lower accuracy ceiling for a likely larger gap — not needed.

## 4. Dataset

**Primary: MATH-500** — a 500-problem curated test subset of the Hendrycks
MATH benchmark.

Why MATH-500 over the alternatives considered:
- **vs. GSM8K:** both models are expected to be near-ceiling on GSM8K,
  giving routing little real work to do. MATH shows a meaningfully larger
  gap between the two model sizes.
- **vs. full MATH test set (~5,000 problems):** same accuracy-gap profile as
  MATH-500, but ~10x the inference calls per pass — expensive across
  baselines, threshold sweeps, and router training/eval iterations, without
  adding much beyond statistical robustness.

MATH-500 also ships with per-problem difficulty labels (1–5), usable later
as a router feature or a stratification axis in results.

**Grading:** exact match plus symbolic/LaTeX equivalence checking via the
`math-verify` package (not naive string matching — MATH answers are
LaTeX-formatted expressions, not bare integers like GSM8K).

**Optional/secondary (not required for Phase 0–4):** GSM8K, as a
grading-simplicity sanity check on the eval harness.

## 5. Metrics

**Model quality**
- Exact match / symbolic equivalence accuracy

**Efficiency**
- Mean, p50, p95 latency
- Throughput
- GPU memory
- % of queries routed to the large model
- Relative compute vs. large-only baseline

**Routing quality**
- Accuracy-vs-compute curve (the resume centerpiece table)
- Routing precision/recall
- False escalation rate — easy problem sent to the large model (wasted
  compute)
- Dangerous non-escalation rate — hard problem kept on the small model
  (accuracy loss). Treated as the more costly error type, not symmetric with
  false escalation.

## 6. Compute Plan

- **Immediate start:** Kaggle Notebooks (free, 30 GPU-hrs/week, T4×2 or
  P100) — used for environment setup, sanity checks, and early Phase 1–2
  work. No application/approval needed.
- **Parallel track:** PACESHIP application submitted for PACE Phoenix
  access (~2 week review) — larger, no-weekly-reset allocation for full-scale
  runs and any Phase 5+ stretch work (quantization ablations in particular).
- **Estimated need for Phases 0–4:** ~20–50 GPU-hours total. This is
  inference-only work (no model training) over a 500-problem set — well
  within either free tier alone.

## 7. Scope

**Phases 0–4 are the bar for "done."** Research design → baselines →
heuristic router → learned router, with a real accuracy-vs-compute table as
the output.

**Phases 5–7 (quantization, full ablation suite, engineering polish) are
stretch goals**, attempted only if time and compute headroom remain after
Phase 4 is solid.

| Phase | Deliverable |
|---|---|
| 0 | This document |
| 1 | Reproducible environment (Kaggle now, PACE once access clears) |
| 2 | Baselines: small-only, large-only, oracle — first benchmark table — COMPLETE |
| 3 | Heuristic/confidence-threshold router — first accuracy-vs-compute curve |
| 4 | Learned router (lightweight classifier), compared against the heuristic baseline — including honest analysis if it doesn't win |
| 5–7 | Stretch: quantization ablations, full experiment suite, repo polish |

## 8. Open Risks

- **1.5B CoT accuracy unknown until Phase 2 — RESOLVED.** See
  docs/DECISIONS.md, 2026-08-12 entry ("Phase 2 baseline complete") for
  full numbers.
- **PACE access timing** — mitigated by starting on Kaggle immediately in
  parallel; nothing in Phases 0–2 blocks on PACE clearing.
- **MATH-500 grading correctness** — symbolic equivalence checking is
  trickier than exact match; `math-verify` needs validation against a
  hand-checked sample before trusting it for the full baseline run.
