#!/bin/bash
# Production — max workers, WARNING logs, output written to daily log file.
# Use this for live trading runs. No auto-reload, no debug noise.
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

export LOG_LEVEL=WARNING

LOG_DIR="logs/$(date +%Y-%m-%d)"
LOG_FILE="$LOG_DIR/server.log"
mkdir -p "$LOG_DIR"

trap 'lsof -ti:8000 | xargs kill -9 2>/dev/null' EXIT

echo "▶  Starting TradeMind backend [PRODUCTION] on port 8000 — 4 workers"
echo "   Logs → $LOG_FILE"

uvicorn api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level warning \
    --no-access-log \
    >> "$LOG_FILE" 2>&1
