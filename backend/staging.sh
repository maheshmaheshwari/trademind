#!/bin/bash
# Staging — stable server, no auto-reload, INFO logs, 2 workers.
# Use this for QA / pre-release testing on the local machine.
set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

source venv/bin/activate

# Kill anything already on the port
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

# Prevent macOS OpenMP deadlock in ML libs
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OPENBLAS_NUM_THREADS=1

export LOG_LEVEL=INFO

trap 'lsof -ti:8000 | xargs kill -9 2>/dev/null' EXIT

echo "▶  Starting TradeMind backend [STAGING] on port 8000 — 2 workers"
uvicorn api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 2 \
    --log-level info \
    --access-log
