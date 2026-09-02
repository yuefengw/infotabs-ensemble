#!/usr/bin/env bash
# Import an offline Ollama bundle into the user's standard model directory.
set -euo pipefail

bundle_dir="${1:-${OLLAMA_BUNDLE_DIR:-$HOME/linux_wheels/ollama-llama3.1-8b}}"
models_dir="${OLLAMA_MODELS:-$HOME/.ollama/models}"
model_name="${OLLAMA_MODEL_NAME:-llama3.1}"
model_tag="${OLLAMA_MODEL_TAG:-8b}"
manifest="$bundle_dir/manifest.json"

[[ -f "$manifest" ]] || { printf 'manifest not found: %s\n' "$manifest" >&2; exit 1; }
[[ -d "$bundle_dir/blobs" ]] || { printf 'blob directory not found: %s\n' "$bundle_dir/blobs" >&2; exit 1; }

python3 - "$manifest" "$bundle_dir/blobs" "$models_dir" "$model_name" "$model_tag" <<'PY'
import hashlib
import json
import pathlib
import shutil
import sys

manifest_path = pathlib.Path(sys.argv[1])
source_dir = pathlib.Path(sys.argv[2])
models_dir = pathlib.Path(sys.argv[3])
model_name = sys.argv[4]
model_tag = sys.argv[5]
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

entries = [manifest["config"], *manifest["layers"]]
destination_blobs = models_dir / "blobs"
destination_blobs.mkdir(parents=True, exist_ok=True)
for entry in entries:
    digest = entry["digest"]
    expected_hash = digest.split(":", 1)[1]
    source = source_dir / digest.replace(":", "-")
    if not source.is_file():
        raise SystemExit(f"blob not found: {source}")
    if source.stat().st_size != entry["size"] or sha256(source) != expected_hash:
        raise SystemExit(f"blob integrity check failed: {source}")
    destination = destination_blobs / source.name
    if destination.is_file() and destination.stat().st_size == entry["size"] and sha256(destination) == expected_hash:
        print(f"EXISTS {destination}")
    else:
        shutil.copy2(source, destination)
        print(f"INSTALLED {destination}")

manifest_destination = models_dir / "manifests" / "registry.ollama.ai" / "library" / model_name / model_tag
manifest_destination.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(manifest_path, manifest_destination)
print(f"MANIFEST {manifest_destination}")
PY

printf 'OLLAMA_BUNDLE_IMPORTED model=%s:%s models_dir=%s\n' "$model_name" "$model_tag" "$models_dir"
