# Phase 1: Environment setup and sanity check.
#
# Must be run on Kaggle with Settings -> Accelerator -> GPU T4 x2.
#
# Loads Qwen2.5-Math-1.5B-Instruct and Qwen2.5-Math-7B-Instruct one at a
# time, runs a single test math prompt through each, and reports GPU memory
# usage, to confirm the environment can load and run both models before
# Phase 2 (full benchmark evaluation) begins.

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMALL_MODEL = "Qwen/Qwen2.5-Math-1.5B-Instruct"
LARGE_MODEL = "Qwen/Qwen2.5-Math-7B-Instruct"

TEST_PROMPT = (
    "Solve for x: 2x + 5 = 17. Show your reasoning step by step, "
    "then give the final answer."
)


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Loaded {model_name}")
    print(f"Device map: {model.hf_device_map}")
    return model, tokenizer


def run_prompt(model, tokenizer, prompt, max_new_tokens=512):
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def check_gpu_memory():
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU {i}: allocated={allocated:.2f} GB, reserved={reserved:.2f} GB")


if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU count: {torch.cuda.device_count()}")

    # Small model
    small_model, small_tokenizer = load_model(SMALL_MODEL)
    response = run_prompt(small_model, small_tokenizer, TEST_PROMPT)
    print(f"\n--- {SMALL_MODEL} response ---\n{response}\n")
    check_gpu_memory()

    del small_model, small_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Large model
    large_model, large_tokenizer = load_model(LARGE_MODEL)
    response = run_prompt(large_model, large_tokenizer, TEST_PROMPT)
    print(f"\n--- {LARGE_MODEL} response ---\n{response}\n")
    check_gpu_memory()

    del large_model, large_tokenizer
    gc.collect()
    torch.cuda.empty_cache()
