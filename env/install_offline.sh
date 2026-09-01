#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${HOME}/anaconda3/envs/infotabs/bin/python"
WHEEL_DIR="${PROJECT_ROOT}/env/wheels"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing environment Python: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${WHEEL_DIR}" ]]; then
    echo "Missing offline wheel directory: ${WHEEL_DIR}" >&2
    exit 1
fi

"${PYTHON}" -m pip install \
    --no-index \
    --find-links "${WHEEL_DIR}" \
    --requirement "${PROJECT_ROOT}/requirements-top-level.txt"

"${PYTHON}" -m pip check
"${PYTHON}" -m pip freeze | sort > "${PROJECT_ROOT}/requirements-lock.txt"
