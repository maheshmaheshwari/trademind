import logging
logging.basicConfig(level=logging.INFO)
from scheduler.jobs import collect_eod_data_job, calculate_indicators_job, generate_trade_signals_job
from database.db import sync_trade_signals_to_turso

print('1. Collecting EOD Data...')
collect_eod_data_job()

print('2. Calculating Indicators...')
calculate_indicators_job()

print('3. Generating Signals...')
generate_trade_signals_job()

print('4. Syncing to Turso...')
count = sync_trade_signals_to_turso()
print(f'✅ Synced {count} signals to Turso!')
