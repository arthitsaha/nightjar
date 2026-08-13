#!/usr/bin/env bash
# Pull the latest Nightjar on macOS or Linux. Works with or without git.
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

exec "$PY" update.py "$@"
