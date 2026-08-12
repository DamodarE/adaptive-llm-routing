# Phase 2 pilot: CoT generation + grading on a 50-problem MATH-500 subset,
# using vllm for inference.
#
# Must be run on Kaggle with Settings -> Accelerator -> GPU T4 x2.
#
# vllm's LLM class initializes its own CUDA/distributed context on the
# process's visible devices; that context can't be remapped mid-process, and
# running two LLM() instances concurrently in one process is not a
# supported/reliable vllm usage pattern. So each model gets its own process
# invocation, run sequentially (not concurrently) -- not two engines
# coexisting in one process.
#
# The two models don't have the same GPU footprint. The 1.5B model fits
# comfortably on one T4, so it's pinned to a single GPU via
# CUDA_VISIBLE_DEVICES. The 7B model does not -- Phase 1's own measurements
# (docs/DECISIONS.md, 2026-08-09) show it split ~13GB/~4GB across both GPUs
# under transformers' device_map="auto", so under vllm it needs
# tensor_parallel_size=2 and both GPUs visible; a single-GPU invocation for
# it will fail.
#
# Kaggle cells:
#   !pip install -q vllm datasets
#   !pip install -q "math-verify[antlr4_13_2]"
#   !CUDA_VISIBLE_DEVICES=0 python notebooks/pilot.py --model small --gpu 0
#   !CUDA_VISIBLE_DEVICES=0,1 python notebooks/pilot.py --model large --gpu 0,1
#
# Each invocation writes results/pilot_{model}.json independently; run both
# before comparing them.
from __future__ import annotations

import sys

if sys.version_info < (3, 10):
    print(
        f"math-verify requires Python >=3.10, found "
        f"{sys.version_info.major}.{sys.version_info.minor} — see "
        f"docs/DECISIONS.md 2026-08-10 entry"
    )
    sys.exit(1)

import argparse
import json
import os
import time
from pathlib import Path

MODELS = {
    "small": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "large": "Qwen/Qwen2.5-Math-7B-Instruct",
}

DATASET_ID = "HuggingFaceH4/MATH-500"

def extract_boxed_answer(text: str) -> str | None:
    """Extract the contents of the last \\boxed{...} in text, handling
    nested braces (e.g. \\boxed{\\frac{1}{2}})."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    idx += len("\\boxed")
    while idx < len(text) and text[idx] != "{":
        idx += 1
    if idx >= len(text):
        return None
    depth = 0
    start = idx
    for i in range(idx, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i]
    return None


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def print_hand_check_examples(results: list[dict], k: int = 5) -> None:
    """Print a spread of examples for eyeballing -- every len(results)/k-th
    one, not just the first k, so the same problems aren't reviewed on
    every run."""
    n = len(results)
    k = min(k, n)
    if k == 0:
        return
    indices = sorted({int(i * n / k) for i in range(k)})

    print(f"\n--- {len(indices)} hand-check examples ---")
    for idx in indices:
        r = results[idx]
        status = "CORRECT" if r["correct"] else "INCORRECT"
        print(f"\n[{status}] {r['unique_id']}")
        print(f"  problem:    {_truncate(r['problem'], 150)}")
        print(f"  gold:       {r['gold']}")
        print(f"  predicted:  {r['predicted']}")
        print(f"  generation: {_truncate(r['generation'], 300)}")
    print()


def load_pilot_problems(n: int):
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID)
    split = "test" if "test" in ds else list(ds.keys())[0]
    return ds[split].select(range(n))


def run_model(model_key: str, n: int, out_path: Path):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = MODELS[model_key]
    problems = load_pilot_problems(n)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = []
    for problem in problems["problem"]:
        # Both Qwen2.5-Math-Instruct tokenizers inject a default system
        # message ("Please reason step by step, and put your final answer
        # within \boxed{}.") when none is given -- confirmed by rendering
        # apply_chat_template with no system message, for both 1.5B and 7B.
        # No need to add it again in the user turn.
        messages = [{"role": "user", "content": problem}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    load_start = time.perf_counter()
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,
        dtype="float16",
        tensor_parallel_size=2 if model_key == "large" else 1,
    )
    load_time_seconds = time.perf_counter() - load_start

    sampling_params = SamplingParams(temperature=0, max_tokens=1024)
    generation_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generation_time_seconds = time.perf_counter() - generation_start
    # vllm batches all prompts into a single generate() call, so this is a
    # batch average (total generation time / n problems), not a true
    # per-request latency -- vllm doesn't expose that by default. Good
    # enough for pilot-stage comparison; revisit if real per-request
    # p50/p95 is needed later.
    mean_latency_per_problem_seconds = generation_time_seconds / len(prompts)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from grading import grade_answer

    results = []
    n_correct = 0
    for problem, gold, output in zip(problems, problems["answer"], outputs):
        generation = output.outputs[0].text
        predicted = extract_boxed_answer(generation)
        correct = predicted is not None and grade_answer(gold, predicted)
        n_correct += correct
        results.append(
            {
                "unique_id": problem["unique_id"],
                "problem": problem["problem"],
                "gold": gold,
                "generation": generation,
                "predicted": predicted,
                "correct": correct,
            }
        )

    print_hand_check_examples(results)

    out_data = {
        "model": model_name,
        "n": len(results),
        "n_correct": n_correct,
        "load_time_seconds": load_time_seconds,
        "generation_time_seconds": generation_time_seconds,
        "mean_latency_per_problem_seconds": mean_latency_per_problem_seconds,
        "results": results,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"{model_name}: {n_correct}/{len(results)} correct")
    print(
        f"Load time: {load_time_seconds:.1f}s | "
        f"Generation time: {generation_time_seconds:.1f}s | "
        f"Mean latency/problem: {mean_latency_per_problem_seconds:.2f}s"
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), required=True)
    parser.add_argument("--gpu", required=True, help="informational only; "
                         "must match whatever CUDA_VISIBLE_DEVICES was set to "
                         "before invoking this script (e.g. '0' for --model "
                         "small, '0,1' for --model large)")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    out_path = args.out or Path(f"results/pilot_{args.model}.json")
    run_model(args.model, args.n, out_path)
