#!/bin/bash
cd "$(cd "$(dirname "$0")/.." && pwd)"
source venv/bin/activate
export PYTHONPATH=.
LOG="logs/manual_jobs_$(date +%Y%m%d_%H%M%S).log"

run() { echo ""; echo "$(date '+%H:%M:%S') ▶ $1"; }

# Step 2: Index & Market Overview
run "Index & Market Overview"
python3 -c "
from collectors.index_collector import collect_index_daily
collect_index_daily()
print('Market overview done')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ Market overview done" || echo "$(date '+%H:%M:%S') ⚠️ Market overview failed (non-critical)"

# Step 3: FII/DII Data
run "FII/DII Data"
python3 -c "
from collectors.fii_collector import collect_fii_dii_data
from database.db import insert_market_overview
data = collect_fii_dii_data()
if data:
    insert_market_overview({'date': data['date'], 'fii_net': data['fii_net'], 'dii_net': data['dii_net']})
    print(f'FII={data[\"fii_net\"]}Cr DII={data[\"dii_net\"]}Cr')
else:
    print('No FII/DII data today')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ FII/DII done" || echo "$(date '+%H:%M:%S') ⚠️ FII/DII failed"

# Step 4: NSE Delivery %
run "NSE Delivery % backfill"
python3 -c "
from collectors.delivery_collector import backfill
total = backfill(days=5)
print(f'Delivery: {total} records')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ Delivery done" || echo "$(date '+%H:%M:%S') ⚠️ Delivery failed"

# Step 5: RSS Market News
run "RSS Market News"
python3 -c "
from collectors.rss_collector import collect_all_rss
r = collect_all_rss()
print(f'{r[\"total\"]} new articles')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ RSS news done" || echo "$(date '+%H:%M:%S') ⚠️ RSS failed"

# Step 6: yfinance Per-Stock News
run "yfinance Per-Stock News"
python3 -c "
from collectors.yfinance_news_collector import collect_all
r = collect_all()
print(f'{r[\"total\"]} new articles')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ yfinance news done" || echo "$(date '+%H:%M:%S') ⚠️ yfinance failed"

# Step 7: FinBERT Score pending
run "FinBERT News Scoring"
python3 -c "
from collectors.gdelt_collector import score_pending_news
count = score_pending_news(batch_limit=2000)
print(f'Scored: {count} articles')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ FinBERT done" || echo "$(date '+%H:%M:%S') ⚠️ FinBERT failed"

# Step 8: Refresh news_daily_sentiment
run "Refresh news_daily_sentiment"
python3 -c "
from database.db import get_connection
conn = get_connection()
conn.autocommit = True
conn.cursor().execute(\"CALL refresh_continuous_aggregate('news_daily_sentiment', NULL, NULL)\")
print('Refreshed')
conn.autocommit = False
conn.close()
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ Sentiment aggregate refreshed" || echo "$(date '+%H:%M:%S') ⚠️ Aggregate failed"

# Step 9: Technical Indicators (latest date)
run "Technical Indicators (today)"
python3 -c "
from analysis.signals import process_all_stocks
result = process_all_stocks()
print(f'Processed: {result[\"processed\"]} stocks')
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ Indicators done" || echo "$(date '+%H:%M:%S') ⚠️ Indicators failed"

# Step 10: Historical Indicator Backfill (new price days)
run "Historical Indicator Backfill"
python3 -c "
from collectors.backfill_indicators_historical import backfill_all
backfill_all()
" >> "$LOG" 2>&1 && echo "$(date '+%H:%M:%S') ✅ Historical indicators done" || echo "$(date '+%H:%M:%S') ⚠️ Backfill failed"

echo ""
echo "=============================================="
echo "$(date '+%H:%M:%S') 🏁 ALL JOBS COMPLETE — Ready for model training"
echo "=============================================="
