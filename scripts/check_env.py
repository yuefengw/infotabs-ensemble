#!/usr/bin/env python3
"""Report the runtime versions and GPU inventory required by the project."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


PACKAGES = (
    "torch",
    "vllm",
    "transformers",
    "tokenizers",
    "accelerate",
    "safetensors",
    "huggingface-hub",
    "sentencepiece",
    "protobuf",
    "pandas",
    "numpy",
    "tqdm",
    "PyYAML",
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"

    output = (result.stdout or result.stderr).strip()
    return output or f"no output (exit code {result.returncode})"


def main() -> None:
    report = {
        "platform": platform.platform(),
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "project_root": str(Path(__file__).resolve().parents[1]),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "packages": {name: package_version(name) for name in PACKAGES},
    }

    try:
        import torch

        report["torch_runtime"] = {
            "cuda_available": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu_count_visible": torch.cuda.device_count(),
            "gpus_visible": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": ".".join(
                        str(part) for part in torch.cuda.get_device_capability(index)
                    ),
                    "memory_gib": round(
                        torch.cuda.get_device_properties(index).total_memory / 1024**3,
                        2,
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # Keep diagnostics usable in a partial environment.
        report["torch_runtime"] = {"error": repr(exc)}

    print(json.dumps(report, indent=2, ensure_ascii=True))
    print("\n=== nvidia-smi ===")
    print(run_command(["nvidia-smi"]))
    print("\n=== nvcc --version ===")
    print(run_command(["nvcc", "--version"]))


if __name__ == "__main__":
    main()
