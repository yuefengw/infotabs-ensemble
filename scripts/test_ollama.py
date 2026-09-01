#!/usr/bin/env python3
"""Check the Ollama API and run the project smoke prompt for Llama 3.1."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

from experiment_config import MODEL_SPECS, build_user_prompt
from parse_prediction import parse_prediction
from run_ollama import normalized_host, post_generate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default=MODEL_SPECS["llama"]["ollama_model"])
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    host = normalized_host(args.host)
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=10) as response:
            tags = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        raise SystemExit(f"Ollama API unavailable at {host}: {exc}") from exc
    names = [item.get("name") for item in tags.get("models", [])]
    if args.model not in names:
        raise SystemExit(f"Ollama model {args.model!r} is not installed; available: {names}")
    prompt = build_user_prompt("Olympic: 1912\nMixed gender: Yes", "The sport became an Olympic sport in 1912.")
    raw_output = post_generate(host, args.model, prompt, args.max_tokens, args.timeout)
    print(json.dumps({
        "backend": "ollama",
        "host": host,
        "ollama_model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "parsed_prediction": parse_prediction(raw_output),
        "raw_output": raw_output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
