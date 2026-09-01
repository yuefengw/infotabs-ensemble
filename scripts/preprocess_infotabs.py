#!/usr/bin/env python3
"""Convert the official InfoTabS parquet splits to stable JSONL records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


SOURCES = {
    "train": "train-00000-of-00001.parquet",
    "dev": "dev-00000-of-00001.parquet",
    "alpha1": "test1-00000-of-00001.parquet",
    "alpha2": "test2-00000-of-00001.parquet",
    "alpha3": "test3-00000-of-00001.parquet",
}
REQUIRED_COLUMNS = {"table", "question", "answer"}
LABELS = {
    "entailment": "E",
    "contradiction": "C",
    "neutral": "N",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_value(value: object, field: str, row_number: int) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"row {row_number}: {field} is empty")
    value = str(value).strip()
    if not value:
        raise ValueError(f"row {row_number}: {field} is empty")
    return value


def normalize_gold(value: object, row_number: int) -> str:
    value_text = text_value(value, "answer", row_number).lower().rstrip(".!。")
    if value_text not in LABELS:
        raise ValueError(f"row {row_number}: unsupported answer label {value!r}")
    return LABELS[value_text]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "infotabs")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed")
    parser.add_argument("--source-repo", default="table-benchmark/infotabs")
    parser.add_argument("--source-revision", default="5e26f025fbbf2055f6905cf553ad9c81048891ea")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_root = args.input_root.expanduser().resolve() / "data"
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_repo": args.source_repo,
        "source_revision": args.source_revision,
        "split_mapping": SOURCES,
        "splits": {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    for split, filename in SOURCES.items():
        source_path = parquet_root / filename
        if not source_path.is_file():
            raise SystemExit(f"missing source parquet: {source_path}")
        frame = pd.read_parquet(source_path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise SystemExit(f"{source_path}: missing columns {sorted(missing)}")

        output_path = output_root / f"{split}.jsonl"
        records = []
        for index, row in frame.iterrows():
            row_number = int(index) + 1
            record = {
                "sample_id": f"{split}_{int(index):06d}",
                "split": split,
                "source_id": None if pd.isna(row.get("original_dataset_id")) else str(row.get("original_dataset_id")),
                "dataset_name": None if pd.isna(row.get("dataset_name")) else str(row.get("dataset_name")),
                "table_title": None if pd.isna(row.get("table_title")) else str(row.get("table_title")),
                "table": text_value(row["table"], "table", row_number),
                "hypothesis": text_value(row["question"], "question", row_number),
                "gold": normalize_gold(row["answer"], row_number),
            }
            records.append(record)

        sample_ids = [record["sample_id"] for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise SystemExit(f"{split}: duplicate sample IDs")
        with output_path.open("w", encoding="utf-8", newline="\n") as destination:
            for record in records:
                destination.write(json.dumps(record, ensure_ascii=False) + "\n")

        label_counts = {label: sum(record["gold"] == label for record in records) for label in ("E", "C", "N")}
        manifest["splits"][split] = {
            "source_file": str(source_path),
            "source_sha256": sha256(source_path),
            "output_file": str(output_path),
            "count": len(records),
            "label_counts": label_counts,
            "columns": list(frame.columns),
        }
        print(f"{split}: {len(records)} records, labels={label_counts}")

    manifest_path = output_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
