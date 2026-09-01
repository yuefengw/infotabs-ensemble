#!/usr/bin/env python3
"""Merge completed prediction shards into the canonical prediction JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
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


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=tuple(MODEL_SPECS))
    parser.add_argument("--condition", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    args = parser.parse_args()

    spec = MODEL_SPECS[args.model]
    output_dir = PREDICTIONS_ROOT / f"{spec['output_prefix']}_{args.condition}"
    primary = output_dir / f"{args.split}.jsonl"
    invalid_path = primary.with_suffix(".invalid.jsonl")
    metadata_path = primary.with_suffix(".metadata.json")
    dataset = load_jsonl(DATA_ROOT / f"{args.split}.jsonl")

    valid_by_id = {
        row["sample_id"]: row
        for row in load_jsonl(primary)
        if row.get("parsed_prediction") in VALID_LABELS and row.get("sample_id")
    }
    invalid_rows = load_jsonl(invalid_path)
    shard_paths = []
    for shard_dir in args.shard_dir:
        shard_paths.extend(
            path
            for path in shard_dir.expanduser().resolve().glob(
                f"out_{args.model}_{args.condition}_{args.split}_part*.jsonl"
            )
            if not path.name.endswith(".invalid.jsonl")
        )
    shard_paths = sorted(shard_paths)
    if not shard_paths:
        raise SystemExit(f"no shard outputs found in {args.shard_dir}")

    added = 0
    duplicate_ids = []
    for shard_path in shard_paths:
        for row in load_jsonl(shard_path):
            sample_id = row.get("sample_id")
            if row.get("parsed_prediction") not in VALID_LABELS or not sample_id:
                continue
            if sample_id in valid_by_id:
                duplicate_ids.append(sample_id)
                continue
            valid_by_id[sample_id] = row
            added += 1
        invalid_rows.extend(load_jsonl(shard_path.with_suffix(".invalid.jsonl")))

    expected_ids = [row["sample_id"] for row in dataset]
    missing = [sample_id for sample_id in expected_ids if sample_id not in valid_by_id]
    extra = sorted(set(valid_by_id) - set(expected_ids))
    if duplicate_ids:
        raise SystemExit(f"duplicate shard sample IDs: {sorted(set(duplicate_ids))}")
    if missing or extra:
        raise SystemExit(json.dumps({"missing": missing, "extra": extra}, ensure_ascii=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(primary, [valid_by_id[sample_id] for sample_id in expected_ids])
    if invalid_rows:
        write_jsonl(invalid_path, invalid_rows)

    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["run_finished_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["requested_count"] = len(dataset)
        metadata["valid_count_this_run"] = len(valid_by_id)
        metadata["invalid_count_this_run"] = len(invalid_rows)
        metadata["shard_merge"] = {"shard_count": len(shard_paths), "rows_added": added}
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "model": args.model,
        "condition": args.condition,
        "split": args.split,
        "expected": len(expected_ids),
        "completed": len(valid_by_id),
        "rows_added": added,
        "invalid_attempts": len(invalid_rows),
        "shards": len(shard_paths),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
