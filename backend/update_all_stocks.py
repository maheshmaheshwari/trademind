"""
Update all 499 Nifty 500 stocks to today's date.
Gets symbols from the database (not the hardcoded 50-stock list).
Uses yfinance with batch downloading for efficiency.
"""
import sys
import os
import time
import logging
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from database.db import get_connection, init_database, insert_prices, get_all_symbols

init_database()

# Get all symbols from database
all_symbols = get_all_symbols()
print(f"\n📊 Total symbols in database: {len(all_symbols)}")

# Check what needs updating
conn = get_connection()
today = datetime.now().strftime('%Y-%m-%d')
print(f"📅 Today: {today}")

# Get latest date for each symbol
rows = conn.execute(
    "SELECT symbol, MAX(date) as latest_date FROM prices WHERE interval = '1d' GROUP BY symbol"
).fetchall()
symbol_dates = {r[0]: r[1] for r in rows}
conn.close()

# Find stocks that need updating
needs_update = []
for sym in all_symbols:
    last = symbol_dates.get(sym, '2020-01-01')
    if last < today:
        needs_update.append((sym, last))

print(f"📈 Stocks needing update: {len(needs_update)} / {len(all_symbols)}")
if needs_update:
    dates = [d for _, d in needs_update]
    from collections import Counter
    date_dist = Counter(dates)
    print("   Date distribution of last update:")
    for d, c in sorted(date_dist.items()):
        print(f"     {d}: {c} stocks")

# Process in batches to avoid yfinance rate limiting
BATCH_SIZE = 20
total_inserted = 0
failed = []
success = 0

print(f"\n🚀 Starting batch update ({BATCH_SIZE} stocks per batch)...\n")

for i in range(0, len(needs_update), BATCH_SIZE):
    batch = needs_update[i:i+BATCH_SIZE]
    batch_symbols = [sym for sym, _ in batch]
    batch_num = (i // BATCH_SIZE) + 1
    total_batches = (len(needs_update) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"  Batch {batch_num}/{total_batches}: {len(batch_symbols)} stocks...")
    
    try:
        # Download all in one go
        data = yf.download(
            batch_symbols, 
            period="5d", 
            interval="1d", 
            group_by="ticker",
            progress=False,
            threads=True
        )
        
        if data.empty:
            print(f"    ⚠️ No data returned for batch {batch_num}")
            failed.extend(batch_symbols)
            continue
        
        for sym in batch_symbols:
            try:
                if len(batch_symbols) == 1:
                    df = data
                else:
                    df = data[sym] if sym in data.columns.get_level_values(0) else pd.DataFrame()
                
                if df.empty or df.dropna(how='all').empty:
                    failed.append(sym)
                    continue
                
                df = df.dropna(subset=['Close'])
                
                prices = []
                for idx, row in df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
                    
                    # Skip if we already have this date
                    last_date = symbol_dates.get(sym, '2020-01-01')
                    if date_str <= last_date:
                        continue
                    
                    prices.append({
                        'symbol': sym,
                        'date': date_str,
                        'open': float(row.get('Open', 0) or 0),
                        'high': float(row.get('High', 0) or 0),
                        'low': float(row.get('Low', 0) or 0),
                        'close': float(row.get('Close', 0) or 0),
                        'volume': int(row.get('Volume', 0) or 0),
                        'interval': '1d',
                    })
                
                if prices:
                    insert_prices(prices)
                    total_inserted += len(prices)
                    success += 1
                else:
                    success += 1  # Already up to date
                    
            except Exception as e:
                logger.error(f"    Failed {sym}: {e}")
                failed.append(sym)
        
    except Exception as e:
        logger.error(f"    Batch {batch_num} failed: {e}")
        failed.extend(batch_symbols)
    
    # Rate limit between batches
    time.sleep(1)

print(f"\n{'='*60}")
print(f"✅ Update complete!")
print(f"   New rows inserted: {total_inserted}")
print(f"   Successful: {success}")
print(f"   Failed: {len(failed)}")
if failed:
    print(f"   Failed tickers: {failed[:20]}{'...' if len(failed) > 20 else ''}")

# Verify final state
conn = get_connection()
final = conn.execute(
    "SELECT MAX(date) as latest, COUNT(DISTINCT symbol) as symbols FROM prices WHERE interval = '1d'"
).fetchone()
print(f"\n📊 Final state: {final[1]} symbols, latest date: {final[0]}")

rows3 = conn.execute(
    """SELECT MAX(date) as latest_date, COUNT(*) as num 
     FROM (SELECT symbol, MAX(date) as date FROM prices WHERE interval = '1d' GROUP BY symbol) 
     GROUP BY date ORDER BY latest_date DESC LIMIT 5"""
).fetchall()
print("   Date distribution:")
for r in rows3:
    print(f"     {r[0]}: {r[1]} stocks")
conn.close()
