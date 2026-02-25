"""
Retry script: Fetch missing symbols from Angel One and write to the
LOCAL embedded SQLite replica. Optionally attempts Turso cloud sync.

Strategy:
  1. Check current local DB for already-present symbols
  2. Fetch missing ones from Angel One API
  3. Write prices, indicators, and signals into local SQLite
  4. At the very end, attempt one Turso embedded replica sync
"""
import os
import json
import time
import datetime
import sqlite3
import pandas as pd
import pyotp
import libsql_experimental as libsql
from dotenv import load_dotenv
from SmartApi import SmartConnect

from analysis.indicators import calculate_all
from analysis.signals import generate_signal, _safe_float
from database.models import ALL_TABLES, CREATE_INDEXES

load_dotenv(override=True)

# Paths
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nifty500.db")
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

def get_local_conn():
    """Get a pure local SQLite connection (no Turso sync)."""
    conn = libsql.connect(DB_PATH)
    return conn

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

# 1. Ensure local DB has tables
print("📦 Ensuring local database tables exist...", flush=True)
conn = get_local_conn()
for sql in ALL_TABLES:
    conn.execute(sql)
for sql in CREATE_INDEXES:
    conn.execute(sql)
conn.commit()

# 2. Find symbols already present locally
existing = conn.execute("SELECT DISTINCT symbol FROM prices WHERE interval = '1d'").fetchall()
existing_symbols = set(row[0] for row in existing)
conn.close()
print(f"   Found {len(existing_symbols)} symbols already in local DB.", flush=True)

# 3. Load ALL tokens and identify missing symbols
with open("data/angel_tokens.json", "r") as f:
    token_map = json.load(f)

all_symbols = [(sym, data["token"]) for sym, data in token_map.items()]
missing = [(sym, tok) for sym, tok in all_symbols if sym + ".NS" not in existing_symbols]

if not missing:
    print("✅ All symbols are already in local DB! Nothing to retry.", flush=True)
else:
    print(f"⚠️  {len(missing)} symbols missing. Fetching from Angel One...\n", flush=True)

    # 4. Authenticate with Angel One
    api_key = os.getenv("ANGEL_API_KEY")
    client_id = os.getenv("ANGEL_CLIENT_ID")
    password = os.getenv("ANGEL_PASSWORD")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET")

    print("🔑 Logging in to Angel One...", flush=True)
    smart_api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()
    login_data = smart_api.generateSession(client_id, password, totp)
    if not login_data.get("status"):
        print(f"❌ Login failed: {login_data}")
        exit(1)
    print(f"✅ Logged in as {client_id}\n", flush=True)

    # Time window
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5*365)
    end_str = end_date.strftime("%Y-%m-%d 15:30")
    start_str = start_date.strftime("%Y-%m-%d 09:15")

    total_retried = 0
    total_failed = 0
    
    # Use a single persistent local connection
    conn = get_local_conn()

    for idx, (symbol, token_id) in enumerate(missing, 1):
        symbol_ns = symbol + ".NS"
        print(f"  [{idx}/{len(missing)}] {symbol} (Token: {token_id})...", end=" ", flush=True)
        
        try:
            time.sleep(0.4)  # Angel One rate limit
            historic_param = {
                "exchange": "NSE",
                "symboltoken": token_id,
                "interval": "ONE_DAY",
                "fromdate": start_str,
                "todate": end_str
            }
            res = smart_api.getCandleData(historic_param)
            
            if not res.get("status") or not res.get("data"):
                print(f"❌ API: {res.get('message', 'No data')}")
                total_failed += 1
                continue
            
            data = res["data"]
            df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            # Prices
            price_rows = []
            for _, row in df.iterrows():
                price_rows.append((
                    symbol_ns, "NSE", row["date"], None,
                    round(row["open"], 2), round(row["high"], 2),
                    round(row["low"], 2), round(row["close"], 2),
                    int(row["volume"]), "1d"
                ))
            
            manual_batch_insert(
                conn, "prices",
                ["symbol", "exchange", "date", "time", "open", "high", "low", "close", "volume", "interval"],
                price_rows
            )
            
            # Indicators & Signals
            indicator_rows = []
            sig_val = "HOLD"
            sig_str = 0.0
            reasons = ["Insufficient data"]
            
            if len(df) >= 50:
                df_calc = calculate_all(df)
                for i in range(50, len(df_calc)):
                    row = df_calc.iloc[i]
                    indicator_rows.append((
                        symbol_ns, row["date"],
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
                
                sig_val, sig_str, reasons = generate_signal(df_calc, symbol_ns)
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
                
                if indicator_rows:
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
                        (sig_val, sig_str, symbol_ns, str(latest["date"]))
                    )
                
                conn.execute(
                    """INSERT INTO ai_signals
                    (symbol, signal, confidence, model_version, target_price, stop_loss, reasoning, features_used)
                    VALUES (?,?,?,?,?,?,?,?)""",
                    (symbol_ns, sig_val, sig_str, "v1.0.0-rules", tp, sl,
                     json.dumps(reasons), '{"indicators": ["rsi","macd","sma"]}')
                )
            
            conn.commit()
            total_retried += 1
            print(f"✅ {len(price_rows)}p | {len(indicator_rows)}i | {sig_val}", flush=True)
            
        except Exception as e:
            print(f"❌ {e}", flush=True)
            total_failed += 1

    conn.close()
    smart_api.terminateSession(client_id)

    print(f"\n{'='*60}")
    print(f"📊 Retry Results")
    print(f"{'='*60}")
    print(f"✅ Retried successfully: {total_retried}")
    print(f"❌ Failed: {total_failed}")

# 5. Print final local DB stats
print(f"\n📦 Local DB Stats:")
conn = get_local_conn()
total_symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices WHERE interval = '1d'").fetchone()[0]
total_prices = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
total_indicators = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
total_signals = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
conn.close()

print(f"   Symbols:     {total_symbols}")
print(f"   Prices:      {total_prices}")
print(f"   Indicators:  {total_indicators}")
print(f"   AI Signals:  {total_signals}")

# 6. Attempt single Turso embedded replica sync
if TURSO_DATABASE_URL and TURSO_AUTH_TOKEN:
    print(f"\n☁️  Attempting Turso embedded replica sync...", flush=True)
    try:
        sync_conn = libsql.connect(
            DB_PATH,
            sync_url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        sync_conn.sync()
        sync_conn.close()
        print("✅ Turso cloud sync completed!", flush=True)
    except Exception as e:
        print(f"⚠️  Turso sync failed: {e}", flush=True)
        print("   Data is safe locally. Cloud sync may need a plan upgrade.", flush=True)

print(f"\n✅ Done!")
