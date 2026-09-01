#!/usr/bin/env python3
"""Build wide and long InfoTabS prediction tables after coverage validation."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiment_config import DATA_ROOT, MODEL_SPECS, PREDICTIONS_ROOT, PROJECT_ROOT, RESULTS_ROOT, SPLITS
from parse_prediction import VALID_LABELS
from validate_predictions import condition_plan, load_jsonl


BASE_COLUMNS = ["Split", "Sample", "Gold"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_ROOT)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--capabilities", type=Path, default=RESULTS_ROOT / "reasoning_capabilities.json")
    parser.add_argument("--coverage", type=Path, default=RESULTS_ROOT / "coverage_report.json")
    return parser.parse_args()


def model_columns(plan: dict[str, list[str]]) -> list[tuple[str, str, str]]:
    columns = []
    for key in ("qwen",):
        for condition in plan.get(key, []):
            columns.append((key, condition, f"Qwen_{'Thinking' if condition == 'thinking' else 'NoThinking'}"))
    columns.append(("llama", "default", "Llama"))
    for key in ("gemma", "mistral"):
        if key in plan:
            columns.append((key, "default", MODEL_SPECS[key]["display"]))
    for condition in plan.get("deepseek", []):
        name = "DeepSeek_Thinking" if condition == "thinking" else "DeepSeek_NoThinking" if condition == "no_thinking" else "DeepSeek"
        columns.append(("deepseek", condition, name))
    if "phi" in plan:
        columns.append(("phi", "default", "Phi"))
    return columns


def read_prediction_map(path: Path) -> dict[str, dict]:
    rows = load_jsonl(path)
    result = {}
    for row in rows:
        if row.get("sample_id") in result:
            raise ValueError(f"duplicate prediction sample_id in {path}: {row.get('sample_id')}")
        if row.get("parsed_prediction") not in VALID_LABELS:
            raise ValueError(f"invalid prediction in {path}: {row.get('sample_id')}")
        result[row["sample_id"]] = row
    return result


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def command_output(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr).strip()


def package_versions() -> dict[str, str]:
    names = ("torch", "vllm", "transformers", "tokenizers", "accelerate", "safetensors", "huggingface-hub")
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    predictions_root = args.predictions_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    coverage_path = args.coverage.expanduser().resolve()
    if not coverage_path.is_file():
        raise SystemExit(f"missing coverage report: {coverage_path}")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if not coverage.get("valid"):
        raise SystemExit("coverage report is invalid; resolve missing/duplicate/INVALID rows first")
    plan = condition_plan(args.capabilities.expanduser().resolve(), predictions_root)
    columns = model_columns(plan)

    dataset = {split: load_jsonl(data_root / f"{split}.jsonl") for split in SPLITS}
    prediction_maps = {}
    for split in SPLITS:
        prediction_maps[split] = {}
        for model_key, condition, column in columns:
            if model_key == "llama" and model_key not in plan:
                continue
            spec = MODEL_SPECS[model_key]
            path = predictions_root / f"{spec['output_prefix']}_{condition}" / f"{split}.jsonl"
            prediction_maps[split][(model_key, condition)] = read_prediction_map(path)

    wide_rows = []
    long_rows = []
    for split in SPLITS:
        for record in dataset[split]:
            sample_id = record["sample_id"]
            wide = {"Split": split, "Sample": sample_id, "Gold": record["gold"]}
            for model_key, condition, column in columns:
                if model_key == "llama" and model_key not in plan:
                    wide[column] = "NA"
                    long_rows.append({"Split": split, "Sample": sample_id, "Gold": record["gold"], "Model": "meta-llama/Llama-3.1-8B-Instruct", "Condition": "default", "Prediction": "NA", "RawOutputPath": ""})
                    continue
                row = prediction_maps[split][(model_key, condition)].get(sample_id)
                if row is None:
                    raise SystemExit(f"missing {split}/{sample_id}/{model_key}/{condition}")
                wide[column] = row["parsed_prediction"]
                raw_path = predictions_root / f"{MODEL_SPECS[model_key]['output_prefix']}_{condition}" / f"{split}.jsonl"
                long_rows.append({"Split": split, "Sample": sample_id, "Gold": record["gold"], "Model": MODEL_SPECS[model_key]["repo_id"], "Condition": condition, "Prediction": row["parsed_prediction"], "RawOutputPath": str(raw_path.relative_to(PROJECT_ROOT)).replace("\\", "/")})
            wide_rows.append(wide)

    wide_columns = BASE_COLUMNS + [column for _, _, column in columns]
    write_csv(results_root / "all_predictions.csv", wide_columns, wide_rows)
    for split in SPLITS:
        write_csv(results_root / f"{split}_predictions.csv", wide_columns, [row for row in wide_rows if row["Split"] == split])
    write_csv(results_root / "predictions_long.csv", ["Split", "Sample", "Gold", "Model", "Condition", "Prediction", "RawOutputPath"], long_rows)

    dataset_manifest_path = data_root / "dataset_manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8")) if dataset_manifest_path.is_file() else None
    condition_runs = []
    for metadata_path in sorted(predictions_root.glob("*/*.metadata.json")):
        try:
            condition_runs.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            raise SystemExit(f"invalid run metadata: {metadata_path}")
    capabilities_path = args.capabilities.expanduser().resolve()
    capabilities = json.loads(capabilities_path.read_text(encoding="utf-8")) if capabilities_path.is_file() else None
    run_metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "prompt_version": "infotabs-predictions-only-v1",
        "prompt_template": "Given table and hypothesis; labels Entailment/Contradiction/Neutral; return FINAL: <label>",
        "dataset_manifest": dataset_manifest,
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "packages": package_versions(),
        "torch_cuda": command_output([sys.executable, "-c", "import torch; print(torch.version.cuda)"]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "dtype": "bfloat16",
        "temperature": 0,
        "tensor_parallel_size": 1,
        "models": {key: {"repo_id": spec["repo_id"], "path": str(PROJECT_ROOT / spec["relative_path"]), "conditions": plan.get(key, [])} for key, spec in MODEL_SPECS.items()},
        "reasoning_capabilities": capabilities,
        "condition_runs": condition_runs,
        "llama": ({"repo_id": MODEL_SPECS["llama"]["repo_id"], "condition": "default", "backend": "ollama", "ollama_model": MODEL_SPECS["llama"]["ollama_model"], "prediction": "available from Ollama outputs"} if "llama" in plan else {"repo_id": MODEL_SPECS["llama"]["repo_id"], "condition": "default", "prediction": "NA", "reason": "Ollama output directory not present"}),
        "coverage_report": str(coverage_path),
        "outputs": ["all_predictions.csv", "dev_predictions.csv", "alpha1_predictions.csv", "alpha2_predictions.csv", "alpha3_predictions.csv", "train_predictions.csv", "predictions_long.csv"],
    }
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(wide_rows)} rows to {results_root / 'all_predictions.csv'}")
    print(f"Wrote {len(long_rows)} rows to {results_root / 'predictions_long.csv'}")


if __name__ == "__main__":
    main()
