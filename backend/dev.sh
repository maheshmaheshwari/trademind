#!/bin/bash
set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

source venv/bin/activate

# Kill any existing process on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
export OPENBLAS_NUM_THREADS=1

trap 'lsof -ti:8000 | xargs kill -9 2>/dev/null; kill 0' EXIT
watchfiles --filter python "bash -c 'OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OPENBLAS_NUM_THREADS=1 lsof -ti:8000 | xargs kill -9 2>/dev/null; sleep 0.5; OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OPENBLAS_NUM_THREADS=1 uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4'" api/ analysis/ trading/ database/ collectors/ scheduler/
