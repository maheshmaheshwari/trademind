#!/bin/bash
set -e
cd /Users/maheshmaheshwari/Documents/trademind/backend
source venv/bin/activate
export PYTHONPATH=.

LOG="logs/manual_jobs_$(date +%Y%m%d_%H%M%S).log"
echo "Starting all jobs — logging to $LOG"

run() {
    echo ""
    echo "=============================================="
    echo "$(date '+%H:%M:%S') ▶ $1"
    echo "=============================================="
}

# 1. EOD Prices (Jun 4 + Jun 5 — last 5 days)
run "EOD Price Collection"
python update_stocks_angel.py >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ EOD prices done"

# 2. Index & Market Overview
run "Index & Market Overview"
python -c "
from collectors.index_collector import collect_index_daily
collect_index_daily()
print('Index data done')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ Market overview done"

# 3. Technical Indicators (latest date per stock)
run "Technical Indicators (today)"
python -c "
from analysis.signals import process_all_stocks
result = process_all_stocks()
print(f'Indicators done: {result[\"processed\"]} stocks')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ Indicators done"

# 4. FII/DII Data
run "FII/DII Data"
python -c "
from collectors.fii_collector import collect_fii_dii_data
from database.db import insert_market_overview
data = collect_fii_dii_data()
if data:
    insert_market_overview({'date': data['date'], 'fii_net': data['fii_net'], 'dii_net': data['dii_net']})
    print(f'FII/DII stored: FII={data[\"fii_net\"]}Cr DII={data[\"dii_net\"]}Cr')
else:
    print('No FII/DII data')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ FII/DII done"

# 5. NSE Delivery %
run "NSE Delivery % (backfill 5 days)"
python -c "
from collectors.delivery_collector import backfill
total = backfill(days=5)
print(f'Delivery: {total} records stored')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ Delivery % done"

# 6. RSS Market News
run "RSS Market News"
python -c "
from collectors.rss_collector import collect_all_rss
result = collect_all_rss()
print(f'RSS news: {result[\"total\"]} new articles')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ RSS news done"

# 7. yfinance Per-Stock News
run "yfinance Per-Stock News"
python -c "
from collectors.yfinance_news_collector import collect_all
result = collect_all()
print(f'yfinance news: {result[\"total\"]} new articles')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ yfinance news done"

# 8. FinBERT Score pending news
run "FinBERT News Scoring"
python -c "
from collectors.gdelt_collector import score_pending_news
count = score_pending_news(batch_limit=1000)
print(f'FinBERT scored: {count} articles')
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ FinBERT scoring done"

# 9. Refresh news_daily_sentiment continuous aggregate
run "Refresh news_daily_sentiment"
python -c "
from database.db import get_connection
conn = get_connection()
conn.autocommit = True
conn.cursor().execute(\"CALL refresh_continuous_aggregate('news_daily_sentiment', NULL, NULL)\")
print('news_daily_sentiment refreshed')
conn.autocommit = False
conn.close()
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ Sentiment aggregate refreshed"

# 10. Historical indicator backfill (all dates for new Jun 4+5 prices)
run "Historical Indicator Backfill (Jun 4+5)"
python -c "
from collectors.backfill_indicators_historical import backfill_all
backfill_all()
" >> "$LOG" 2>&1
echo "$(date '+%H:%M:%S') ✅ Historical indicators done"

echo ""
echo "=============================================="
echo "$(date '+%H:%M:%S') 🏁 ALL JOBS COMPLETE"
echo "=============================================="
echo "Now ready to start model retraining."
