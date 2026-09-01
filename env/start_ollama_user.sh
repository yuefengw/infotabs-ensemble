#!/usr/bin/env bash
# Start Ollama in the current user's session when user systemd is unavailable.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"
export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

command -v ollama >/dev/null || { printf 'ollama is not on PATH\n' >&2; exit 1; }
if ollama ps >/dev/null 2>&1; then
  printf 'OLLAMA_ALREADY_RUNNING host=%s gpu=%s\n' "$OLLAMA_HOST" "$CUDA_VISIBLE_DEVICES"
  exit 0
fi

log_file="${OLLAMA_LOG_FILE:-$HOME/.ollama/ollama.log}"
mkdir -p "$(dirname "$log_file")"
nohup ollama serve >>"$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "${OLLAMA_PID_FILE:-$HOME/.ollama/ollama.pid}"
printf 'OLLAMA_STARTED pid=%s host=%s gpu=%s log=%s\n' "$pid" "$OLLAMA_HOST" "$CUDA_VISIBLE_DEVICES" "$log_file"
