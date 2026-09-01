#!/usr/bin/env python3
"""Run the common single-GPU BF16 smoke test for one local model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


DEFAULT_PROMPT = """Given the following table and hypothesis,
determine their logical relationship.

Labels:
Entailment
Contradiction
Neutral

Table:
Olympic: 1912
Mixed gender: Yes

Hypothesis:
The sport became an Olympic sport in 1912.

Return only the final answer in the form:

FINAL: Entailment"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable reasoning templates on models that support this option",
    )
    return parser.parse_args()


def render_prompt(tokenizer: AutoTokenizer, enable_thinking: bool) -> str:
    messages = [{"role": "user", "content": DEFAULT_PROMPT}]
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    return DEFAULT_PROMPT


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")

    visible_gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible_gpus:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to exactly one GPU before running")
    if "," in visible_gpus:
        raise SystemExit("This project requires exactly one visible GPU per model")

    settings = {
        "model": str(model_path),
        "cuda_visible_devices": visible_gpus,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": args.enable_thinking,
    }
    print("=== settings ===")
    print(json.dumps(settings, indent=2))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt = render_prompt(tokenizer, args.enable_thinking)

    print("\n=== rendered prompt ===")
    print(prompt)
    print("\n=== loading model ===", flush=True)

    llm = LLM(
        model=str(model_path),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    print("MODEL_LOAD_OK")

    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0, max_tokens=args.max_tokens),
    )
    raw_output = outputs[0].outputs[0].text
    print("\n=== raw output ===")
    print(raw_output)


if __name__ == "__main__":
    main()
