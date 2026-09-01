#!/usr/bin/env python3
"""Conservative parser for the three InfoTabS labels."""

from __future__ import annotations

import argparse
import json
import re
import sys


VALID_LABELS = {"E", "C", "N"}
LABEL_TO_SHORT = {
    "entailment": "E",
    "contradiction": "C",
    "neutral": "N",
}
SHORT_TO_LABEL = {value: key.title() for key, value in LABEL_TO_SHORT.items()}
INVALID = "INVALID"

_LABEL = r"(Entailment|Contradiction|Neutral)"
_FINAL_RE = re.compile(r"\bFINAL\s*[:\uFF1A]\s*" + _LABEL + r"\b", re.I)
_ANSWER_RE = re.compile(r"\bAnswer\s*[:\uFF1A-]\s*" + _LABEL + r"\b", re.I)


def _short(label: str) -> str:
    return LABEL_TO_SHORT[label.strip().lower()]


def parse_prediction(raw_output: str) -> str:
    """Return E/C/N only when the output contains a reliable final label."""

    if not isinstance(raw_output, str) or not raw_output.strip():
        return INVALID

    final_matches = _FINAL_RE.findall(raw_output)
    if final_matches:
        return _short(final_matches[-1])

    answer_matches = _ANSWER_RE.findall(raw_output)
    if answer_matches:
        return _short(answer_matches[-1])

    # Accept a bare label, including a single trailing punctuation mark, but
    # never infer a label from arbitrary reasoning text.
    for line in reversed(raw_output.splitlines()):
        cleaned = re.sub(r"[\s`*_~.,;:!?\u3002\uFF01]+$", "", line.strip())
        cleaned = re.sub(r"^[\s`*_~]+", "", cleaned)
        if cleaned.lower() in LABEL_TO_SHORT:
            return _short(cleaned)

    cleaned = re.sub(r"[\s`*_~.,;:!?\u3002\uFF01]+$", "", raw_output.strip())
    if cleaned.lower() in LABEL_TO_SHORT:
        return _short(cleaned)
    return INVALID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Raw model output; otherwise read stdin")
    args = parser.parse_args()
    raw = args.text if args.text is not None else sys.stdin.read()
    print(json.dumps({"prediction": parse_prediction(raw), "raw_output": raw}))


if __name__ == "__main__":
    main()
