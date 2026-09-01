#!/usr/bin/env python3
"""Shared configuration for the InfoTabS prediction-only experiment."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "processed"
PREDICTIONS_ROOT = PROJECT_ROOT / "predictions_raw"
RESULTS_ROOT = PROJECT_ROOT / "results"

SPLITS = ("dev", "alpha1", "alpha2", "alpha3", "train")
PROMPT_VERSION = "infotabs-predictions-only-v1"

MODEL_SPECS = {
    "llama": {
        "repo_id": "meta-llama/Llama-3.1-8B-Instruct",
        "display": "Llama",
        "relative_path": "models/llama-3.1-8b-instruct",
        "output_prefix": "llama_ollama",
        "backend": "ollama",
        "ollama_model": "llama3.1:8b",
    },
    "qwen": {
        "repo_id": "Qwen/Qwen3-8B",
        "display": "Qwen",
        "relative_path": "models/qwen3-8b",
        "output_prefix": "qwen",
    },
    "gemma": {
        "repo_id": "google/gemma-3-12b-it",
        "display": "Gemma",
        "relative_path": "models/gemma-3-12b-it",
        "output_prefix": "gemma",
    },
    "mistral": {
        "repo_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "display": "Mistral",
        "relative_path": "models/mistral-nemo-instruct-2407",
        "output_prefix": "mistral",
    },
    "deepseek": {
        "repo_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "display": "DeepSeek",
        "relative_path": "models/deepseek-r1-distill-qwen-7b",
        "output_prefix": "deepseek",
    },
    "phi": {
        "repo_id": "microsoft/Phi-4-mini-instruct",
        "display": "Phi",
        "relative_path": "models/phi-4-mini-instruct",
        "output_prefix": "phi",
    },
}


def resolve_model_key(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in MODEL_SPECS:
        return normalized
    for key, spec in MODEL_SPECS.items():
        if normalized == spec["repo_id"].lower():
            return key
    raise ValueError(
        f"Unknown model {value!r}; choose one of {', '.join(MODEL_SPECS)}"
    )


def model_path(model_key: str) -> Path:
    return PROJECT_ROOT / MODEL_SPECS[model_key]["relative_path"]


def build_user_prompt(table: str, hypothesis: str) -> str:
    return f"""Given the following table and hypothesis,
determine their logical relationship.

Labels:
Entailment
Contradiction
Neutral

Table:
{table}

Hypothesis:
{hypothesis}

Reason about the table carefully.
Return the final answer in the form:

FINAL: Entailment"""
