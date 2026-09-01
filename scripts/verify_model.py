#!/usr/bin/env python3
"""Verify that a local model directory contains a complete HF checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--hash", action="store_true", help="Hash all safetensors files")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    model_dir = args.model.expanduser().resolve()
    required = model_dir / "config.json"
    weights = sorted(model_dir.rglob("*.safetensors"))
    tokenizer_files = sorted(model_dir.glob("tokenizer*"))

    errors = []
    if not required.is_file():
        errors.append("missing config.json")
    if not weights:
        errors.append("no safetensors weights found")
    if not tokenizer_files:
        errors.append("no tokenizer files found")

    report = {
        "model": str(model_dir),
        "valid": not errors,
        "errors": errors,
        "weight_files": [
            {
                "path": str(path.relative_to(model_dir)),
                "bytes": path.stat().st_size,
                **({"sha256": sha256(path)} if args.hash else {}),
            }
            for path in weights
        ],
        "total_weight_gib": round(sum(path.stat().st_size for path in weights) / 1024**3, 2),
        "tokenizer_files": [path.name for path in tokenizer_files],
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
