#!/usr/bin/env python3
"""Run one model/condition over InfoTabS splits with resume support."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from experiment_config import (
    DATA_ROOT,
    MODEL_SPECS,
    PREDICTIONS_ROOT,
    PROJECT_ROOT,
    PROMPT_VERSION,
    SPLITS,
    build_user_prompt,
    model_path,
    resolve_model_key,
)
from parse_prediction import INVALID, VALID_LABELS, parse_prediction
from reasoning import render_chat, supports_thinking_toggle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model alias or repository ID")
    parser.add_argument("--condition", choices=("default", "thinking", "no_thinking"))
    parser.add_argument("--split", action="append", choices=SPLITS, help="Repeat for selected splits; default: all")
    parser.add_argument("--input", type=Path, help="Custom JSONL input; only valid with one --split")
    parser.add_argument("--output", type=Path, help="Custom output JSONL; only valid with one --split")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, help="Default is 1024 for reasoning and 256 otherwise")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, help="Limit each split for smoke tests")
    parser.add_argument(
        "--guided-labels",
        action="store_true",
        help="Constrain generated output to one of the three labels (retry use)",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            for field in ("sample_id", "split", "table", "hypothesis", "gold"):
                if field not in record:
                    raise ValueError(f"{path}:{line_number}: missing {field}")
            records.append(record)
    if len({record["sample_id"] for record in records}) != len(records):
        raise ValueError(f"{path}: duplicate sample_id")
    return records


def key_for(record: dict) -> str:
    return ":".join((record.get("split", ""), record.get("sample_id", "")))


def read_existing(path: Path, invalid_path: Path) -> dict[str, dict]:
    """Keep valid rows for resume and preserve invalid attempts separately."""
    valid_rows: dict[str, dict] = {}
    invalid_rows: list[dict] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("parsed_prediction") in VALID_LABELS:
                    valid_rows[key_for(row)] = row
                else:
                    invalid_rows.append(row)
    if invalid_path.is_file():
        with invalid_path.open("r", encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    try:
                        invalid_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in valid_rows.values():
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    if invalid_rows:
        with invalid_path.open("w", encoding="utf-8", newline="\n") as destination:
            for row in invalid_rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
    return valid_rows


def append_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        destination.flush()
        os.fsync(destination.fileno())


def main() -> None:
    args = parse_args()
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or "," in os.environ["CUDA_VISIBLE_DEVICES"]:
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to exactly one GPU")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")

    model_key = resolve_model_key(args.model)
    spec = MODEL_SPECS[model_key]
    if spec.get("backend") == "ollama":
        raise SystemExit(
            "This model is configured for Ollama; use scripts/run_ollama.py "
            "instead of the vLLM runner"
        )
    selected_splits = tuple(args.split or SPLITS)
    if args.input and len(selected_splits) != 1:
        raise SystemExit("--input is only valid with one --split")
    if args.output and len(selected_splits) != 1:
        raise SystemExit("--output is only valid with one --split")

    model_dir = model_path(model_key)
    if not model_dir.is_dir():
        raise SystemExit(f"model directory does not exist: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    probe = build_user_prompt("Olympic: 1912", "The sport became an Olympic sport in 1912.")
    reasoning_info = supports_thinking_toggle(tokenizer, probe)
    supports_toggle = bool(reasoning_info["supports_toggle"])
    condition = args.condition
    if condition is None:
        if supports_toggle:
            raise SystemExit("This model supports thinking control; pass --condition thinking or --condition no_thinking")
        condition = "default"
    if condition in {"thinking", "no_thinking"} and not supports_toggle:
        raise SystemExit(f"{model_key} has no reliable thinking toggle; use --condition default")
    if condition == "default" and supports_toggle:
        raise SystemExit(f"{model_key} requires an explicit thinking condition")

    reasoning_condition = condition == "thinking" or (model_key == "deepseek" and condition == "default")
    max_tokens = args.max_tokens if args.max_tokens is not None else (1024 if reasoning_condition else 256)
    if max_tokens < 1:
        raise SystemExit("--max-tokens must be positive")

    settings = {
        "model_key": model_key,
        "model": spec["repo_id"],
        "model_path": str(model_dir),
        "condition": condition,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "temperature": 0,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "batch_size": args.batch_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "reasoning_toggle": reasoning_info,
        "prompt_version": PROMPT_VERSION,
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)

    print("Loading vLLM model...", flush=True)
    llm = LLM(
        model=str(model_dir),
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    print("MODEL_LOAD_OK", flush=True)
    sampling_kwargs = {"temperature": 0, "max_tokens": max_tokens}
    if args.guided_labels:
        sampling_kwargs["guided_decoding"] = GuidedDecodingParams(
            choice=["Entailment", "Contradiction", "Neutral"]
        )
    sampling = SamplingParams(**sampling_kwargs)
    run_started = datetime.now(timezone.utc).isoformat()

    for split in selected_splits:
        input_path = args.input.expanduser().resolve() if args.input else DATA_ROOT / f"{split}.jsonl"
        output_path = args.output.expanduser().resolve() if args.output else PREDICTIONS_ROOT / f"{spec['output_prefix']}_{condition}" / f"{split}.jsonl"
        invalid_path = output_path.with_suffix(".invalid.jsonl")
        metadata_path = output_path.with_suffix(".metadata.json")
        records = load_records(input_path)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        completed = read_existing(output_path, invalid_path)
        pending = [record for record in records if key_for(record) not in completed]
        valid_count = len(completed)
        invalid_count = 0
        print(f"{split}: total={len(records)} resumed={valid_count} pending={len(pending)}", flush=True)

        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [render_chat(tokenizer, build_user_prompt(item["table"], item["hypothesis"]), condition) for item in batch]
            outputs = llm.generate(prompts, sampling)
            valid_rows = []
            invalid_rows = []
            for record, output in zip(batch, outputs):
                raw_output = output.outputs[0].text
                parsed = parse_prediction(raw_output)
                row = {
                    "sample_id": record["sample_id"],
                    "split": split,
                    "gold": record["gold"],
                    "model": spec["repo_id"],
                    "model_key": model_key,
                    "condition": condition,
                    "raw_output": raw_output,
                    "parsed_prediction": parsed,
                }
                if parsed in VALID_LABELS:
                    valid_rows.append(row)
                else:
                    invalid_rows.append(row)
            append_rows(output_path, valid_rows)
            append_rows(invalid_path, invalid_rows)
            valid_count += len(valid_rows)
            invalid_count += len(invalid_rows)
            print(f"{split}: processed={start + len(batch)}/{len(pending)} valid={len(valid_rows)} invalid={len(invalid_rows)}", flush=True)

        metadata = {
            **settings,
            "split": split,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "invalid_path": str(invalid_path),
            "run_started_utc": run_started,
            "run_finished_utc": datetime.now(timezone.utc).isoformat(),
            "requested_count": len(records),
            "resumed_count": len(completed),
            "valid_count_this_run": valid_count,
            "invalid_count_this_run": invalid_count,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
