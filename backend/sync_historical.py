import sys
import os
import sqlite3
import time
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from database.db import get_remote_turso_connection

def sync_historical_data():
    local_db_path = "nifty500_remote.db"
    if not os.path.exists(local_db_path):
        print(f"File not found: {local_db_path}")
        return

    print(f"1. Reading historical data from {local_db_path}...")
    local_conn = sqlite3.connect(local_db_path)
    local_cursor = local_conn.cursor()

    # Read prices
    local_cursor.execute("SELECT symbol, exchange, date, time, open, high, low, close, volume, interval FROM prices")
    price_rows = local_cursor.fetchall()

    # Read technical indicators
    local_cursor.execute('''
        SELECT symbol, date, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
               sma_20, sma_50, sma_200, ema_9, ema_21, atr_14, adx_14, stoch_k, stoch_d, obv,
               support_1, support_2, support_3, resistance_1, resistance_2, resistance_3, signal, signal_strength
        FROM technical_indicators
    ''')
    indicator_rows = local_cursor.fetchall()

    print(f"Found {len(price_rows)} prices and {len(indicator_rows)} indicators to sync.")
    local_conn.close()

    if not price_rows and not indicator_rows:
        return

    print("2. Connecting to Turso...")
    remote_conn = get_remote_turso_connection()
    remote_cursor = remote_conn.cursor()

    batch_size = 500

    # Sync Prices
    if price_rows:
        print(f"Pushing {len(price_rows)} prices to Turso in batches of {batch_size}...")
        for i in range(0, len(price_rows), batch_size):
            batch = price_rows[i:i+batch_size]
            try:
                # Using executemany for efficiency
                remote_cursor.executemany('''
                    INSERT OR IGNORE INTO prices 
                    (symbol, exchange, date, time, open, high, low, close, volume, interval)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                remote_conn.commit()
            except Exception as e:
                print(f"Error in prices batch {i}: {e}")
                # Fallback to single inserts if batch fails
                for row in batch:
                    try:
                        remote_cursor.execute('''
                            INSERT OR IGNORE INTO prices 
                            (symbol, exchange, date, time, open, high, low, close, volume, interval)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', row)
                    except:
                        pass
                remote_conn.commit()
            
            if (i+1) % 5000 == 0 or i+batch_size >= len(price_rows):
                print(f"  ...synced {min(i+batch_size, len(price_rows))}/{len(price_rows)} prices")

    # Sync Indicators
    if indicator_rows:
        print(f"Pushing {len(indicator_rows)} indicators to Turso in batches of {batch_size}...")
        for i in range(0, len(indicator_rows), batch_size):
            batch = indicator_rows[i:i+batch_size]
            try:
                remote_cursor.executemany('''
                    INSERT OR IGNORE INTO technical_indicators 
                    (symbol, date, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
                     sma_20, sma_50, sma_200, ema_9, ema_21, atr_14, adx_14, stoch_k, stoch_d, obv,
                     support_1, support_2, support_3, resistance_1, resistance_2, resistance_3, signal, signal_strength)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', batch)
                remote_conn.commit()
            except Exception as e:
                print(f"Error in indicators batch {i}: {e}")
                for row in batch:
                    try:
                        remote_cursor.execute('''
                            INSERT OR IGNORE INTO technical_indicators 
                            (symbol, date, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower,
                             sma_20, sma_50, sma_200, ema_9, ema_21, atr_14, adx_14, stoch_k, stoch_d, obv,
                             support_1, support_2, support_3, resistance_1, resistance_2, resistance_3, signal, signal_strength)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', row)
                    except:
                        pass
                remote_conn.commit()

            if (i+1) % 5000 == 0 or i+batch_size >= len(indicator_rows):
                print(f"  ...synced {min(i+batch_size, len(indicator_rows))}/{len(indicator_rows)} indicators")

    print("✅ Full historical sync complete!")

if __name__ == "__main__":
    sync_historical_data()
