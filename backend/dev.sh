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

# Dev never runs the cron jobs — the deployed server owns the scheduler.
# Running both against the same Timescale Cloud instance duplicates collector
# and signal writes. Override with ENABLE_SCHEDULER=true bash dev.sh if you
# genuinely need to test a job locally.
export ENABLE_SCHEDULER="${ENABLE_SCHEDULER:-false}"

# Dev never applies the schema either. watchfiles restarts uvicorn on every save
# below, and api/server.py's startup_event calls init_database() with no
# APP_ENV — i.e. against PROD. That meant an unfinished edit to schema_pg.py was
# applied to production the moment it hit disk. Override with
# ENABLE_SCHEMA_INIT=true bash dev.sh if you deliberately want dev to migrate.
export ENABLE_SCHEMA_INIT="${ENABLE_SCHEMA_INIT:-false}"

trap 'lsof -ti:8000 | xargs kill -9 2>/dev/null; kill 0' EXIT
watchfiles --filter python "bash -c 'OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OPENBLAS_NUM_THREADS=1 lsof -ti:8000 | xargs kill -9 2>/dev/null; sleep 0.5; OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE OPENBLAS_NUM_THREADS=1 uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4'" api/ analysis/ trading/ database/ collectors/ scheduler/
