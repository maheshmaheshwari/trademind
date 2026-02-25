"""
Safest script to rebuild and sync 63k rows to Turso exactly.
This script uses a REMOTE-ONLY connection to Turso (bypassing embedded replica)
to completely eliminate the 'database disk image malformed' and hanging issues
that persist with the libsql local sync engine.
"""
import os
import pandas as pd
import yfinance as yf
import libsql_experimental as libsql
from dotenv import load_dotenv

from analysis.indicators import calculate_all
from analysis.signals import generate_signal, _safe_float
from database.models import ALL_TABLES, CREATE_INDEXES
from data.stocks_list import NIFTY_50_STOCKS, INDEX_SYMBOLS

# 1. Download
symbols_to_fetch = [s["symbol"] for s in NIFTY_50_STOCKS] + list(INDEX_SYMBOLS.values())
total_symbols = len(symbols_to_fetch)
print(f"\n📊 Downloading 5 years of history for {total_symbols} symbols in bulk...", flush=True)

df_bulk = yf.download(symbols_to_fetch, period="5y", interval="1d", group_by="ticker", auto_adjust=False, threads=True)

# 2. Remote only connection
load_dotenv(override=True)
url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")

print("\n🔄 Connecting to Turso (REMOTE ONLY)...", flush=True)
conn = libsql.connect(url, auth_token=token)

print("🗑️  Cleaning remote tables...", flush=True)
conn.execute("DROP TABLE IF EXISTS prices")
conn.execute("DROP TABLE IF EXISTS technical_indicators")
conn.execute("DROP TABLE IF EXISTS ai_signals")
conn.execute("DROP TABLE IF EXISTS market_overview")
conn.execute("DROP TABLE IF EXISTS news_sentiment")

print("📦 Creating fresh tables...", flush=True)
for sql in ALL_TABLES: conn.execute(sql)
for sql in CREATE_INDEXES: conn.execute(sql)

total_prices = 0
total_indicators = 0

print("\n⚙️  Processing and uploading to Turso (in remote batches)...", flush=True)
for idx, symbol in enumerate(symbols_to_fetch, 1):
    try:
        if symbol not in df_bulk.columns.levels[0]:
            print(f"  [{idx:2d}/{total_symbols}] {symbol:15s} ❌ missing in bulk download", flush=True)
            continue
            
        df = df_bulk[symbol].dropna(how="all").copy()
        if df.empty:
            print(f"  [{idx:2d}/{total_symbols}] {symbol:15s} ❌ no valid data", flush=True)
            continue
            
        price_rows = []
        for d, row in df.iterrows():
            date_str = d.strftime("%Y-%m-%d")
            price_rows.append((
                symbol, "NSE", date_str, None,
                round(row["Open"], 2), round(row["High"], 2),
                round(row["Low"], 2), round(row["Close"], 2),
                int(row["Volume"]), "1d"
            ))
            
        # Helper to do batch inserts manually for speed
        def manual_batch_insert(conn, table, columns, data_rows, chunk_size=500):
            if not data_rows: return
            cols_count = len(columns)
            for i in range(0, len(data_rows), chunk_size):
                chunk = data_rows[i:i + chunk_size]
                placeholders = ",".join(["(" + ",".join(["?"] * cols_count) + ")"] * len(chunk))
                flat_data = []
                for row in chunk:
                    flat_data.extend(row)
                query = f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES {placeholders}"
                conn.execute(query, tuple(flat_data))

        manual_batch_insert(
            conn, "prices",
            ["symbol", "exchange", "date", "time", "open", "high", "low", "close", "volume", "interval"],
            price_rows
        )
        
        indicator_rows = []
        sig_val = "HOLD"
        sig_str = 0.0
        
        if symbol not in INDEX_SYMBOLS.values() and len(df) >= 50:
            df_ind = df.reset_index().copy()
            df_ind = df_ind.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df_ind["date"] = df_ind["date"].dt.strftime("%Y-%m-%d")
            df_calc = calculate_all(df_ind)
            
            for i in range(50, len(df_calc)):
                row = df_calc.iloc[i]
                indicator_rows.append((
                    symbol, row["date"],
                    _safe_float(row.get("rsi_14")), _safe_float(row.get("macd")),
                    _safe_float(row.get("macd_signal")), _safe_float(row.get("macd_hist")),
                    _safe_float(row.get("bb_upper")), _safe_float(row.get("bb_middle")),
                    _safe_float(row.get("bb_lower")), _safe_float(row.get("sma_20")),
                    _safe_float(row.get("sma_50")), _safe_float(row.get("sma_200")),
                    _safe_float(row.get("ema_9")), _safe_float(row.get("ema_21")),
                    _safe_float(row.get("atr_14")), _safe_float(row.get("adx_14")),
                    _safe_float(row.get("stoch_k")), _safe_float(row.get("stoch_d")),
                    _safe_float(row.get("obv")), _safe_float(row.get("support_1")),
                    _safe_float(row.get("support_2")), _safe_float(row.get("support_3")),
                    _safe_float(row.get("resistance_1")), _safe_float(row.get("resistance_2")),
                    _safe_float(row.get("resistance_3")), None, None
                ))
                
            sig_val, sig_str, reasons = generate_signal(df_calc)
            latest = df_calc.iloc[-1]
            c_price = float(latest.get("close", 0))
            atr = _safe_float(latest.get("atr_14"))
            tp = sl = None
            if c_price and atr:
                if "BUY" in sig_val:
                    tp = round(c_price + (2 * atr), 2)
                    sl = round(c_price - (1.5 * atr), 2)
                elif "SELL" in sig_val:
                    tp = round(c_price - (2 * atr), 2)
                    sl = round(c_price + (1.5 * atr), 2)
                    
            if len(indicator_rows) > 0:
                manual_batch_insert(
                    conn, "technical_indicators",
                    ["symbol", "date", "rsi_14", "macd", "macd_signal", "macd_hist",
                     "bb_upper", "bb_middle", "bb_lower",
                     "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
                     "atr_14", "adx_14", "stoch_k", "stoch_d", "obv",
                     "support_1", "support_2", "support_3",
                     "resistance_1", "resistance_2", "resistance_3",
                     "signal", "signal_strength"],
                    indicator_rows
                )
                
                conn.execute(
                    "UPDATE technical_indicators SET signal=?, signal_strength=? WHERE symbol=? AND date=?",
                    (sig_val, sig_str, symbol, str(latest["date"]))
                )
                
            import json
            conn.execute(
                """INSERT INTO ai_signals
                (symbol, signal, confidence, model_version, target_price, stop_loss, reasoning, features_used)
                VALUES (?,?,?,?,?,?,?,?)""",
                (symbol, sig_val, sig_str, "v1.0.0-rules", tp, sl,
                 json.dumps(reasons), '{"indicators": ["rsi","macd","sma"]}')
            )
            
        conn.commit()
            
        total_prices += len(price_rows)
        total_indicators += len(indicator_rows)
        print(f"  [{idx:2d}/{total_symbols}] {symbol:15s} ✅ {len(price_rows):4d} prices | {len(indicator_rows):4d} inds | {sig_val:11s}", flush=True)
        
    except Exception as e:
        print(f"  [{idx:2d}/{total_symbols}] {symbol:15s} ❌ ERROR: {e}", flush=True)

print(f"\n✅ Build complete! {total_prices} prices, {total_indicators} indicators explicitly uploaded to Turso.", flush=True)

# Delete local embedded replica to clean slate for app usage later
try:
    for f in ["nifty500.db", "nifty500.db-wal", "nifty500.db-shm"]:
        if os.path.exists(f): os.remove(f)
except Exception:
    pass
