#!/usr/bin/env bash
# One-shot setup for Nightjar on macOS and Linux.
set -euo pipefail
cd "$(dirname "$0")"

if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python 3.9+ is required but was not found on PATH." >&2
    exit 1
fi

exec "$PY" install.py "$@"
