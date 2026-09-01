#!/usr/bin/env python3
"""Download one official model snapshot to its canonical local directory."""

from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODELS = {
    "qwen3-8b": "Qwen/Qwen3-8B",
    "llama-3.1-8b-instruct": "meta-llama/Llama-3.1-8B-Instruct",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "mistral-nemo-instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",
    "deepseek-r1-distill-qwen-7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct",
}

OLLAMA_ONLY = {"llama-3.1-8b-instruct"}

# Avoid duplicate or converted weight formats while preserving the official
# Hugging Face safetensors checkpoint and all tokenizer/configuration files.
IGNORE_PATTERNS = (
    "*.gguf",
    "*.pth",
    "*.pt",
    "original/*",
    "consolidated.safetensors",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument(
        "--models-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ignored(filename: str) -> bool:
    return any(fnmatch.fnmatch(filename, pattern) for pattern in IGNORE_PATTERNS)


def dry_run(repo_id: str) -> None:
    info = HfApi().model_info(repo_id, files_metadata=True)
    files = []
    total = 0
    for sibling in info.siblings or []:
        filename = sibling.rfilename
        if ignored(filename):
            continue
        size = sibling.size or 0
        total += size
        files.append({"path": filename, "size": size})
    print(
        json.dumps(
            {
                "repo_id": repo_id,
                "revision": info.sha,
                "file_count": len(files),
                "download_bytes": total,
                "download_gib": round(total / 1024**3, 2),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    if args.model in OLLAMA_ONLY:
        raise SystemExit(
            "Llama-3.1-8B-Instruct is configured for Ollama in this project; "
            "do not download the gated Hugging Face snapshot. See "
            "env/install_ollama_user.sh and env/pull_llama_ollama.sh."
        )
    repo_id = MODELS[args.model]
    target = args.models_root.expanduser().resolve() / args.model

    if args.dry_run:
        dry_run(repo_id)
        return

    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target,
        ignore_patterns=list(IGNORE_PATTERNS),
    )
    print(f"Downloaded {repo_id} to {target}")


if __name__ == "__main__":
    main()
