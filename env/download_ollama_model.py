#!/usr/bin/env python3
"""Download an Ollama registry model bundle with resume and SHA-256 checks."""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import requests


REGISTRY = os.environ.get("OLLAMA_REGISTRY", "https://registry.ollama.ai")
MODEL = os.environ.get("OLLAMA_MODEL_NAME", "llama3.1")
TAG = os.environ.get("OLLAMA_MODEL_TAG", "8b")
ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "ollama-llama3.1-8b")
PROXY = os.environ.get("OLLAMA_DOWNLOAD_PROXY", "http://127.0.0.1:10808")
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None


def blob_path(digest: str) -> pathlib.Path:
    return ROOT / "blobs" / digest.replace(":", "-")


def sha256_file(path: pathlib.Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def fetch_blob(session: requests.Session, digest: str, expected_size: int) -> None:
    target = blob_path(digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    current = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={current}-"} if current else {}
    url = f"{REGISTRY}/v2/library/{MODEL}/blobs/{digest}"
    print(f"FETCH {digest} size={expected_size} resume={current}", flush=True)
    with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
        if current and response.status_code == 200:
            current = 0
            partial.unlink(missing_ok=True)
        response.raise_for_status()
        mode = "ab" if current and response.status_code == 206 else "wb"
        total = current
        with partial.open(mode) as output:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    output.write(chunk)
                    total += len(chunk)
                    if total // (256 * 1024 * 1024) != (total - len(chunk)) // (256 * 1024 * 1024):
                        print(f"PROGRESS {digest} {total}/{expected_size}", flush=True)
    if partial.stat().st_size != expected_size:
        raise RuntimeError(f"size mismatch for {digest}: {partial.stat().st_size} != {expected_size}")
    actual = sha256_file(partial)
    expected = digest.split(":", 1)[1]
    if actual != expected:
        raise RuntimeError(f"sha256 mismatch for {digest}: {actual}")
    partial.replace(target)
    print(f"READY {target} sha256:{actual}", flush=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if PROXIES:
        session.proxies.update(PROXIES)
    manifest_url = f"{REGISTRY}/v2/library/{MODEL}/manifests/{TAG}"
    response = session.get(manifest_url, timeout=(30, 60))
    response.raise_for_status()
    manifest = response.json()
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    entries = [manifest["config"], *manifest["layers"]]
    for entry in entries:
        digest = entry["digest"]
        target = blob_path(digest)
        if target.exists() and target.stat().st_size == entry["size"] and sha256_file(target) == digest.split(":", 1)[1]:
            print(f"EXISTS {target}", flush=True)
            continue
        fetch_blob(session, digest, entry["size"])
    print(f"DONE root={ROOT}", flush=True)


if __name__ == "__main__":
    main()
