#!/bin/bash
set -e
cd /Users/maheshmaheshwari/Documents/trademind/backend
source venv/bin/activate

echo "$(date) — Waiting for price backfill (PID 41412) to finish..."
while kill -0 41412 2>/dev/null; do sleep 10; done
echo "$(date) — Price backfill done."

# ── Phase 2b: Historical indicators ──────────────────────────────────────────
echo "$(date) — Starting historical indicator backfill..."
PYTHONPATH=. python collectors/backfill_indicators_historical.py \
  > logs/backfill_indicators_3yr_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "$(date) — Indicators done."

# ── Phase 3: NSE news sentiment backfill ─────────────────────────────────────
echo "$(date) — Starting NSE announcements backfill..."
PYTHONPATH=. python collectors/nse_announcements_collector.py \
  > logs/nse_backfill_$(date +%Y%m%d_%H%M%S).log 2>&1
echo "$(date) — News sentiment done."

echo "$(date) — Full pipeline complete."
