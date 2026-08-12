# Phase 3 prep: re-run the small model (Qwen2.5-Math-1.5B-Instruct) on the
# full 500-problem MATH-500 set with per-token logprobs enabled, to build a
# confidence signal for the Phase 3 heuristic router.
#
# Same model load, sampling params, dataset, and single-GPU usage as
# pilot.py's small-model path -- imported directly from pilot.py rather than
# re-typed, so the two stay identical. The only addition is
# SamplingParams(logprobs=1), which is enough to recover the logprob of the
# actually-generated token at each position: decoding is greedy
# (temperature=0), so the sampled token is always the top-1 token and will
# always be present when logprobs=1 is requested.
#
# Does not touch the large model or results/pilot_large.json.
#
# Kaggle cell (single GPU, same as pilot.py's small-model invocation):
#   !CUDA_VISIBLE_DEVICES=0 python notebooks/rerun_small_with_logprobs.py
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pilot  # noqa: E402  (runs pilot.py's Python-version guard on import too)

MODEL_KEY = "small"
OUT_PATH = Path(__file__).resolve().parent.parent / "results" / "small_with_logprobs.json"


def _truncate(text: str, max_len: int) -> str:
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def token_logprobs(output) -> list[float]:
    """Per-generated-token logprob of the actually-sampled token, read out
    of vllm's CompletionOutput.logprobs (list[dict[token_id, Logprob]])."""
    completion = output.outputs[0]
    logprobs_list = completion.logprobs
    if not logprobs_list:
        return []
    return [
        position_logprobs[token_id].logprob
        for token_id, position_logprobs in zip(completion.token_ids, logprobs_list)
    ]


def print_hand_check_examples(results: list[dict], k: int = 5) -> None:
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
        print(f"  problem:      {_truncate(r['problem'], 150)}")
        print(f"  gold:         {r['gold']}")
        print(f"  predicted:    {r['predicted']}")
        print(f"  generation:   {_truncate(r['generation'], 300)}")
        print(f"  mean_logprob: {r['mean_logprob']}")
        print(f"  min_logprob:  {r['min_logprob']}")
    print()


def main() -> None:
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    model_name = pilot.MODELS[MODEL_KEY]
    problems = pilot.load_pilot_problems(500)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": problem}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for problem in problems["problem"]
    ]

    load_start = time.perf_counter()
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.85,
        dtype="float16",
        tensor_parallel_size=1,
    )
    load_time_seconds = time.perf_counter() - load_start

    sampling_params = SamplingParams(temperature=0, max_tokens=1024, logprobs=1)
    generation_start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    generation_time_seconds = time.perf_counter() - generation_start
    mean_latency_per_problem_seconds = generation_time_seconds / len(prompts)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from grading import grade_answer

    results = []
    n_correct = 0
    for problem, gold, output in zip(problems, problems["answer"], outputs):
        generation = output.outputs[0].text
        predicted = pilot.extract_boxed_answer(generation)
        correct = predicted is not None and grade_answer(gold, predicted)
        n_correct += correct

        lps = token_logprobs(output)
        # None (not 0.0) when generation produced zero tokens, so downstream
        # code can tell "no signal" apart from "very confident."
        mean_logprob = sum(lps) / len(lps) if lps else None
        min_logprob = min(lps) if lps else None

        results.append(
            {
                "unique_id": problem["unique_id"],
                "problem": problem["problem"],
                "gold": gold,
                "generation": generation,
                "predicted": predicted,
                "correct": correct,
                "mean_logprob": mean_logprob,
                "min_logprob": min_logprob,
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
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out_data, indent=2))
    print(f"{model_name}: {n_correct}/{len(results)} correct")
    print(
        f"Load time: {load_time_seconds:.1f}s | "
        f"Generation time: {generation_time_seconds:.1f}s | "
        f"Mean latency/problem: {mean_latency_per_problem_seconds:.2f}s"
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    main()
