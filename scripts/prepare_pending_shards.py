#!/usr/bin/env python3
"""Create disjoint JSONL inputs for samples not yet predicted successfully."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_config import DATA_ROOT, MODEL_SPECS, PREDICTIONS_ROOT
from parse_prediction import VALID_LABELS


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, action="append", default=[])
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be positive")

    records = load_jsonl(DATA_ROOT / f"{args.split}.jsonl")
    spec = MODEL_SPECS[args.model]
    primary = PREDICTIONS_ROOT / f"{spec['output_prefix']}_{args.condition}" / f"{args.split}.jsonl"
    completed = {
        row.get("sample_id")
        for row in load_jsonl(primary)
        if row.get("parsed_prediction") in VALID_LABELS
    }
    for extra_dir in args.extra_dir:
        resolved_extra_dir = extra_dir.expanduser().resolve()
        shard_names = {
            f"out_{spec['output_prefix']}_{args.condition}_{args.split}_part*.jsonl",
            f"out_{args.model}_{args.condition}_{args.split}_part*.jsonl",
        }
        for shard_name in shard_names:
            for path in resolved_extra_dir.glob(shard_name):
                if path.name.endswith(".invalid.jsonl"):
                    continue
                completed.update(
                    row.get("sample_id")
                    for row in load_jsonl(path)
                    if row.get("parsed_prediction") in VALID_LABELS
                )
    pending = [row for row in records if row.get("sample_id") not in completed]
    args.output_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    shard_paths = []
    for index in range(args.shards):
        shard = pending[index::args.shards]
        part_index = args.start_index + index
        path = args.output_dir.expanduser().resolve() / f"{args.model}_{args.condition}_{args.split}_part{part_index:02d}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as destination:
            for row in shard:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")
        shard_paths.append({"path": str(path), "count": len(shard)})

    print(json.dumps({
        "model": args.model,
        "condition": args.condition,
        "split": args.split,
        "expected": len(records),
        "completed": len(completed),
        "pending": len(pending),
        "shards": shard_paths,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
