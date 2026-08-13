#!/usr/bin/env bash
# Launch Nightjar on macOS or Linux. Ctrl+C here quits.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "No .venv found - run ./install.sh first." >&2
    exit 1
fi

exec .venv/bin/python nightjar.py "$@"
