#!/usr/bin/env python3
"""Validate prediction coverage before building the final tables."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from experiment_config import DATA_ROOT, MODEL_SPECS, PREDICTIONS_ROOT, RESULTS_ROOT, SPLITS
from parse_prediction import VALID_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--predictions-root", type=Path, default=PREDICTIONS_ROOT)
    parser.add_argument("--capabilities", type=Path, default=RESULTS_ROOT / "reasoning_capabilities.json")
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "coverage_report.json")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def condition_plan(capabilities_path: Path, predictions_root: Path = PREDICTIONS_ROOT) -> dict[str, list[str]]:
    if capabilities_path.is_file():
        report = json.loads(capabilities_path.read_text(encoding="utf-8"))
        plan = {}
        for key, item in report.get("models", {}).items():
            if item.get("available"):
                plan[key] = list(item.get("conditions") or ["default"])
        if plan:
            llama_dir = predictions_root / f"{MODEL_SPECS['llama']['output_prefix']}_default"
            if llama_dir.is_dir() and any(llama_dir.glob("*.jsonl")):
                plan["llama"] = ["default"]
            return plan
    plan = {"qwen": ["thinking", "no_thinking"], "gemma": ["default"], "mistral": ["default"], "deepseek": ["default"], "phi": ["default"]}
    llama_dir = predictions_root / f"{MODEL_SPECS['llama']['output_prefix']}_default"
    if llama_dir.is_dir() and any(llama_dir.glob("*.jsonl")):
        plan["llama"] = ["default"]
    return plan


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    predictions_root = args.predictions_root.expanduser().resolve()
    plan = condition_plan(args.capabilities.expanduser().resolve(), predictions_root)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "valid": True,
        "conditions": plan,
        "splits": {},
    }
    expected_by_split = {}
    gold_by_split = {}
    for split in SPLITS:
        dataset_rows = load_jsonl(data_root / f"{split}.jsonl")
        ids = [row.get("sample_id") for row in dataset_rows]
        expected_by_split[split] = {sample_id for sample_id in ids if sample_id}
        gold_by_split[split] = {row.get("sample_id"): row.get("gold") for row in dataset_rows}
        if len(ids) != len(expected_by_split[split]):
            raise SystemExit(f"{split}: dataset sample IDs are missing or duplicated")

    for split in SPLITS:
        report["splits"][split] = {}
        expected = expected_by_split[split]
        for model_key, conditions in plan.items():
            if model_key not in MODEL_SPECS:
                continue
            report["splits"][split][model_key] = {}
            prefix = MODEL_SPECS[model_key]["output_prefix"]
            for condition in conditions:
                output_path = predictions_root / f"{prefix}_{condition}" / f"{split}.jsonl"
                invalid_path = output_path.with_suffix(".invalid.jsonl")
                rows = load_jsonl(output_path)
                invalid_attempts = load_jsonl(invalid_path)
                seen = {}
                duplicates = []
                invalid_final = []
                gold_mismatch = []
                for row in rows:
                    sample_id = row.get("sample_id")
                    if sample_id in seen:
                        duplicates.append(sample_id)
                    seen[sample_id] = row
                    if row.get("parsed_prediction") not in VALID_LABELS:
                        invalid_final.append(sample_id)
                    dataset_gold = gold_by_split[split].get(sample_id)
                    if dataset_gold is not None and row.get("gold") != dataset_gold:
                        gold_mismatch.append(sample_id)
                actual = set(seen)
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                entry = {
                    "output_path": str(output_path),
                    "expected": len(expected),
                    "completed": sum(row.get("parsed_prediction") in VALID_LABELS for row in seen.values()),
                    "invalid_final": len([x for x in invalid_final if x]),
                    "invalid_attempts": len(invalid_attempts),
                    "missing": missing,
                    "duplicate": sorted(set(x for x in duplicates if x)),
                    "extra": extra,
                    "gold_mismatch": sorted(set(x for x in gold_mismatch if x)),
                }
                entry["valid"] = not (entry["missing"] or entry["duplicate"] or entry["extra"] or entry["invalid_final"] or entry["gold_mismatch"])
                report["splits"][split][model_key][condition] = entry
                report["valid"] = report["valid"] and entry["valid"]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
