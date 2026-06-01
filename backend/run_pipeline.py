"""
TradeMind AI — Full Data + Signal Pipeline

Runs the three steps in order:
  1. Fetch today's EOD prices from Angel One
  2. Recalculate technical indicators for all stocks
  3. Regenerate ML trade signals from final_models/

Usage:
    cd backend && source venv/bin/activate
    python run_pipeline.py
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from scheduler.jobs import collect_eod_data_job, calculate_indicators_job, generate_trade_signals_job

print('1. Fetching EOD prices from Angel One...')
collect_eod_data_job()

print('\n2. Recalculating technical indicators...')
calculate_indicators_job()

print('\n3. Regenerating ML trade signals...')
generate_trade_signals_job()

print('\n✅ Pipeline complete.')
