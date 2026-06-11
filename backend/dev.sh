#!/bin/bash
set -e

BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

source venv/bin/activate

# Kill any existing process on port 8000
lsof -ti :8000 | xargs kill -9 2>/dev/null || true

trap 'lsof -ti:8000 | xargs kill -9 2>/dev/null; kill 0' EXIT
watchfiles --filter python "bash -c 'lsof -ti:8000 | xargs kill -9 2>/dev/null; sleep 0.5; uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4'" api/ analysis/ trading/ database/ collectors/ scheduler/
