"""
Calculate day-wise historical indicators for ALL dates in the database.

Instead of only the latest day, this calculates indicators for every
trading day where we have enough data (minimum 200 days lookback for SMA-200).

For each stock:
  1. Load all 5yr price data
  2. Calculate rolling indicators (RSI, MACD, BB, SMA, EMA, etc.)
  3. Store every day's indicators in the technical_indicators table
  4. Generate signals for every day
"""
import math
import os
import time
import libsql_experimental as libsql
import pandas as pd
from tqdm import tqdm
from analysis.indicators import calculate_all
from analysis.signals import generate_signal, _safe_float


def calculate_historical_indicators():
    """Calculate indicators for every date across all stocks."""
    
    conn = libsql.connect("nifty500.db")
    
    # Get all stock symbols (exclude indices)
    symbols = [r[0] for r in conn.execute(
        "SELECT DISTINCT symbol FROM prices WHERE interval='1d' ORDER BY symbol"
    ).fetchall()]
    
    print(f"📊 Calculating day-wise indicators for {len(symbols)} symbols...\n")
    
    # Clear existing indicators (we're recalculating everything)
    conn.execute("DELETE FROM technical_indicators")
    conn.execute("DELETE FROM ai_signals")
    conn.commit()
    print("🗑️  Cleared old indicators & signals\n")
    
    total_indicator_rows = 0
    total_signal_rows = 0
    
    for sym_idx, symbol in enumerate(symbols, 1):
        # Load all price data for this stock
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
            FROM prices WHERE symbol=? AND interval='1d'
            ORDER BY date ASC""",
            (symbol,)
        ).fetchall()
        
        if len(rows) < 50:
            print(f"  [{sym_idx:2d}/{len(symbols)}] {symbol:20s} — skipped (only {len(rows)} rows)")
            continue
        
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = pd.DataFrame(rows, columns=cols)
        
        # Calculate ALL indicators on the full DataFrame
        # This gives us rolling values for every row
        df = calculate_all(df)
        
        # We need at least 200 rows for SMA-200, but we can start saving
        # indicators from row 50+ (enough for most indicators)
        min_row = 50  # Start from row 50 (enough for RSI-14, MACD, BB, SMA-50)
        
        indicator_rows = []
        signal_rows = []
        
        for i in range(min_row, len(df)):
            row = df.iloc[i]
            date = row["date"]
            
            indicator_rows.append((
                symbol, date,
                _safe_float(row.get("rsi_14")),
                _safe_float(row.get("macd")),
                _safe_float(row.get("macd_signal")),
                _safe_float(row.get("macd_hist")),
                _safe_float(row.get("bb_upper")),
                _safe_float(row.get("bb_middle")),
                _safe_float(row.get("bb_lower")),
                _safe_float(row.get("sma_20")),
                _safe_float(row.get("sma_50")),
                _safe_float(row.get("sma_200")),
                _safe_float(row.get("ema_9")),
                _safe_float(row.get("ema_21")),
                _safe_float(row.get("atr_14")),
                _safe_float(row.get("adx_14")),
                _safe_float(row.get("stoch_k")),
                _safe_float(row.get("stoch_d")),
                _safe_float(row.get("obv")),
                _safe_float(row.get("support_1")),
                _safe_float(row.get("support_2")),
                _safe_float(row.get("support_3")),
                _safe_float(row.get("resistance_1")),
                _safe_float(row.get("resistance_2")),
                _safe_float(row.get("resistance_3")),
                None,  # signal — set below
                None,  # signal_strength — set below
            ))
        
        # Batch insert indicators
        for r in indicator_rows:
            conn.execute(
                """INSERT OR REPLACE INTO technical_indicators
                (symbol, date, rsi_14, macd, macd_signal, macd_hist,
                 bb_upper, bb_middle, bb_lower,
                 sma_20, sma_50, sma_200, ema_9, ema_21,
                 atr_14, adx_14, stoch_k, stoch_d, obv,
                 support_1, support_2, support_3,
                 resistance_1, resistance_2, resistance_3,
                 signal, signal_strength)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                r
            )
        conn.commit()
        total_indicator_rows += len(indicator_rows)
        
        # Generate signal only for the LATEST day and save to ai_signals
        signal, strength, reasons = generate_signal(df)
        latest = df.iloc[-1]
        close_price = float(latest.get("close", 0))
        atr = _safe_float(latest.get("atr_14"))
        
        target_price = None
        stop_loss = None
        if close_price and atr:
            if "BUY" in signal:
                target_price = round(close_price + (2 * atr), 2)
                stop_loss = round(close_price - (1.5 * atr), 2)
            elif "SELL" in signal:
                target_price = round(close_price - (2 * atr), 2)
                stop_loss = round(close_price + (1.5 * atr), 2)
        
        import json
        conn.execute(
            """INSERT INTO ai_signals
            (symbol, signal, confidence, model_version, target_price, stop_loss, reasoning, features_used)
            VALUES (?,?,?,?,?,?,?,?)""",
            (symbol, signal, strength, "v1.0.0-rules", target_price, stop_loss,
             json.dumps(reasons), json.dumps({"indicators": ["rsi_14","macd","sma_50","sma_200","bb","ema","adx","stoch"]}))
        )
        conn.commit()
        total_signal_rows += 1
        
        # Update the signal in the latest indicator row
        conn.execute(
            "UPDATE technical_indicators SET signal=?, signal_strength=? WHERE symbol=? AND date=?",
            (signal, strength, symbol, str(latest["date"]))
        )
        conn.commit()
        
        print(f"  [{sym_idx:2d}/{len(symbols)}] {symbol:20s} ✅ {len(indicator_rows):>5} indicator days  |  {signal:12s} ({strength:.0f}%)")
    
    conn.close()
    
    print(f"\n{'='*60}")
    print(f"📊 Day-Wise Indicator Calculation Complete!")
    print(f"   Total indicator rows: {total_indicator_rows:,}")
    print(f"   Total signal rows:    {total_signal_rows}")
    print(f"{'='*60}")
    
    # Sync to Turso
    from dotenv import load_dotenv
    load_dotenv()
    url = os.getenv("TURSO_DATABASE_URL", "")
    token = os.getenv("TURSO_AUTH_TOKEN", "")
    
    if url and token and "turso.io" in url:
        print("\n🔄 Syncing to Turso cloud...")
        try:
            conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
            conn.sync()
            total = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
            print(f"☁️  Turso sync complete! {total:,} indicator rows in cloud")
            conn.close()
        except Exception as e:
            print(f"⚠️  Turso sync failed: {e}")
            print("   Data is safe locally")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    calculate_historical_indicators()
