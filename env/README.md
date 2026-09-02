# Offline environment artifacts

Place Linux x86_64 wheels or Conda packages for the dedicated `infotabs`
environment here when direct package installation is unavailable. Do not put
Windows wheels in this directory.

## Ollama Offline Bundle

On a connected machine, download and verify an official Ollama Registry bundle:

```bash
OLLAMA_DOWNLOAD_PROXY=http://127.0.0.1:10808 \
python env/download_ollama_model.py /path/to/ollama-llama3.1-8b
```

Upload that directory to the target server, then import it without network
access:

```bash
bash env/import_ollama_bundle.sh /path/to/ollama-llama3.1-8b
```

The importer verifies every blob's size and SHA-256 before writing the standard
`$OLLAMA_MODELS/manifests/registry.ollama.ai/library/llama3.1/8b` tag manifest.
