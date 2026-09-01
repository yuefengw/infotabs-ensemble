#!/usr/bin/env bash
# Install the official Ollama Linux archive without requiring sudo.
set -euo pipefail

usage() {
  printf 'Usage: %s --archive /path/to/ollama-linux-amd64.tar.zst [--install-dir DIR]\n' "$0"
}

archive=""
install_dir="${OLLAMA_INSTALL_DIR:-$HOME/.local/ollama}"
gpu="${CUDA_VISIBLE_DEVICES:-5}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) archive="${2:-}"; shift 2 ;;
    --install-dir) install_dir="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done
[[ -n "$archive" ]] || { usage >&2; exit 2; }
[[ -f "$archive" ]] || { printf 'archive not found: %s\n' "$archive" >&2; exit 1; }

mkdir -p "$install_dir" "$HOME/.local/bin" "$HOME/.config/systemd/user"
case "$archive" in
  *.tar.zst) command -v zstd >/dev/null || { printf 'zstd is required for .tar.zst archives\n' >&2; exit 1; }; zstd -dc "$archive" | tar -xf - -C "$install_dir" ;;
  *.tgz|*.tar.gz) tar -xzf "$archive" -C "$install_dir" ;;
  *) printf 'unsupported archive format: %s\n' "$archive" >&2; exit 1 ;;
esac
if [[ -x "$install_dir/bin/ollama" ]]; then
  ln -sfn "$install_dir/bin/ollama" "$HOME/.local/bin/ollama"
elif [[ -x "$install_dir/ollama" ]]; then
  ln -sfn "$install_dir/ollama" "$HOME/.local/bin/ollama"
else
  printf 'archive does not contain an executable Ollama binary under bin/ollama or ollama\n' >&2
  exit 1
fi

cat > "$HOME/.config/systemd/user/ollama.service" <<EOF
[Unit]
Description=Ollama user service for InfoTabS Llama 3.1
After=default.target

[Service]
Type=simple
ExecStart=$HOME/.local/bin/ollama serve
Environment=OLLAMA_HOST=127.0.0.1:11434
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_MAX_LOADED_MODELS=1
Environment=CUDA_VISIBLE_DEVICES=$gpu
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

export PATH="$HOME/.local/bin:$PATH"
printf 'Installed: '; ollama --version
printf 'Binary: %s\n' "$HOME/.local/bin/ollama"
printf 'Service unit: %s\n' "$HOME/.config/systemd/user/ollama.service"
printf 'GPU binding: CUDA_VISIBLE_DEVICES=%s\n' "$gpu"
printf 'Start with: systemctl --user daemon-reload && systemctl --user enable --now ollama\n'
