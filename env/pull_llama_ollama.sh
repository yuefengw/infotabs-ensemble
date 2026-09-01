#!/usr/bin/env bash
# Pull and verify the official Ollama Llama 3.1 8B tag.
set -euo pipefail

model="${1:-llama3.1:8b}"
export PATH="$HOME/.local/bin:$PATH"
command -v ollama >/dev/null || { printf 'ollama is not on PATH\n' >&2; exit 1; }
ollama pull "$model"
ollama show "$model" >/dev/null
printf 'OLLAMA_MODEL_READY %s\n' "$model"
