#!/usr/bin/env python3
"""Detect real chat-template thinking controls for the available models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from experiment_config import MODEL_SPECS, PROJECT_ROOT, RESULTS_ROOT, build_user_prompt, model_path
from reasoning import chat_template_source, supports_thinking_toggle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "reasoning_capabilities.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    probe = build_user_prompt("Olympic: 1912", "The sport became an Olympic sport in 1912.")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "probe_prompt": probe,
        "models": {},
    }
    for key, spec in MODEL_SPECS.items():
        path = model_path(key)
        if not path.is_dir():
            report["models"][key] = {"available": False, "reason": f"missing {path}"}
            continue
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        toggle = supports_thinking_toggle(tokenizer, probe)
        conditions = ["thinking", "no_thinking"] if toggle["supports_toggle"] else ["default"]
        report["models"][key] = {
            "available": True,
            "repo_id": spec["repo_id"],
            "model_path": str(path),
            "chat_template_sha256": __import__("hashlib").sha256(chat_template_source(tokenizer).encode("utf-8")).hexdigest(),
            **toggle,
            "conditions": conditions,
        }
        print(f"{key}: supports_toggle={toggle['supports_toggle']} conditions={conditions}")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
