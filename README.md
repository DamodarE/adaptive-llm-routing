# Adaptive LLM Routing for Efficient Mathematical Reasoning

Cascade system that routes math problems to a small model (Qwen2.5-Math-1.5B)
or escalates to a large model (Qwen2.5-Math-7B) based on a confidence
signal, trading a small accuracy loss for large compute savings.

- Full design: [`PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)
- Why choices were made: [`DECISIONS.md`](docs/DECISIONS.md)

## Structure
```
notebooks/   Kaggle/exploratory notebooks
src/         Reusable eval harness, routing logic
data/        MATH-500, cached outputs
results/     Benchmark tables, plots
docs/        Write-ups, resume-ready summaries
```

## Status
Phase 1 (environment setup) — in progress.
