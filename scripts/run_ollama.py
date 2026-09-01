#!/usr/bin/env python3
"""Run the Llama 3.1 Ollama model over InfoTabS JSONL with resume support."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from experiment_config import (
    DATA_ROOT,
    MODEL_SPECS,
    PREDICTIONS_ROOT,
    PROMPT_VERSION,
    SPLITS,
    build_user_prompt,
    resolve_model_key,
)
from parse_prediction import INVALID, VALID_LABELS, parse_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama", help="Ollama-backed model alias")
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--ollama-model", help="Override the Ollama model tag")
    parser.add_argument("--split", action="append", choices=SPLITS)
    parser.add_argument("--input", type=Path, help="Custom JSONL input; only valid with one --split")
    parser.add_argument("--output", type=Path, help="Custom output JSONL; only valid with one --split")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int)
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
    valid_rows: dict[str, dict] = {}
    invalid_rows: list[dict] = []
    for candidate in (path, invalid_path):
        if not candidate.is_file():
            continue
        with candidate.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if candidate == path and row.get("parsed_prediction") in VALID_LABELS:
                    valid_rows[key_for(row)] = row
                else:
                    invalid_rows.append(row)
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


def normalized_host(value: str) -> str:
    value = value.rstrip("/")
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value


def post_generate(host: str, model: str, prompt: str, max_tokens: int, timeout: float) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens},
    }
    request = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail[:500]}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc
    if body.get("error"):
        raise RuntimeError(f"Ollama error: {body['error']}")
    message = body.get("message") or {}
    result = message.get("content")
    if not isinstance(result, str):
        raise RuntimeError(f"Ollama response has no text: {body!r}")
    return result


def generate_with_retry(args: argparse.Namespace, model: str, prompt: str) -> str:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            return post_generate(args.host, model, prompt, args.max_tokens, args.request_timeout)
        except RuntimeError as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            delay = args.retry_delay * (2**attempt)
            print(f"request failed ({exc}); retrying in {delay:g}s", file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError(str(last_error))


def main() -> None:
    args = parse_args()
    if args.max_tokens < 1 or args.max_retries < 0 or args.request_timeout <= 0:
        raise SystemExit("max-tokens must be positive; retries must be non-negative; timeout must be positive")
    model_key = resolve_model_key(args.model)
    spec = MODEL_SPECS[model_key]
    if spec.get("backend") != "ollama":
        raise SystemExit(f"{model_key} is not configured for Ollama")
    selected_splits = tuple(args.split or SPLITS)
    if args.input and len(selected_splits) != 1:
        raise SystemExit("--input is only valid with one --split")
    if args.output and len(selected_splits) != 1:
        raise SystemExit("--output is only valid with one --split")
    ollama_model = args.ollama_model or spec["ollama_model"]
    args.host = normalized_host(args.host)
    settings = {
        "model_key": model_key,
        "model": spec["repo_id"],
        "backend": "ollama",
        "ollama_host": args.host,
        "ollama_model": ollama_model,
        "condition": "default",
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "request_timeout": args.request_timeout,
        "max_retries": args.max_retries,
        "prompt_version": PROMPT_VERSION,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "managed by Ollama service"),
    }
    print(json.dumps(settings, ensure_ascii=False, indent=2), flush=True)
    run_started = datetime.now(timezone.utc).isoformat()

    for split in selected_splits:
        input_path = args.input.expanduser().resolve() if args.input else DATA_ROOT / f"{split}.jsonl"
        output_path = args.output.expanduser().resolve() if args.output else PREDICTIONS_ROOT / f"{spec['output_prefix']}_default" / f"{split}.jsonl"
        invalid_path = output_path.with_suffix(".invalid.jsonl")
        metadata_path = output_path.with_suffix(".metadata.json")
        records = load_records(input_path)
        if args.max_samples is not None:
            if args.max_samples < 1:
                raise SystemExit("--max-samples must be positive")
            records = records[: args.max_samples]
        completed = read_existing(output_path, invalid_path)
        pending = [record for record in records if key_for(record) not in completed]
        invalid_count = 0
        print(f"{split}: total={len(records)} resumed={len(completed)} pending={len(pending)}", flush=True)
        for index, record in enumerate(pending, start=1):
            prompt = build_user_prompt(record["table"], record["hypothesis"])
            try:
                raw_output = generate_with_retry(args, ollama_model, prompt)
            except RuntimeError as exc:
                raw_output = f"OLLAMA_ERROR: {exc}"
            parsed = parse_prediction(raw_output)
            row = {
                "sample_id": record["sample_id"],
                "split": split,
                "gold": record["gold"],
                "model": spec["repo_id"],
                "model_key": model_key,
                "backend": "ollama",
                "ollama_model": ollama_model,
                "condition": "default",
                "raw_output": raw_output,
                "parsed_prediction": parsed,
            }
            if parsed in VALID_LABELS:
                append_rows(output_path, [row])
            else:
                append_rows(invalid_path, [row])
                invalid_count += 1
            print(f"{split}: processed={index}/{len(pending)} parsed={parsed}", flush=True)

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
            "valid_count_this_run": len(pending) - invalid_count,
            "invalid_count_this_run": invalid_count,
        }
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
