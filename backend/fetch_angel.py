"""
Fetch ALL 499 mapped Nifty 500 stocks from Angel One for 5 years,
calculate indicators, and push to Turso via pure remote HTTP.
"""
import os
import json
import time
import datetime
import pandas as pd
import pyotp
import libsql_experimental as libsql
from dotenv import load_dotenv
from SmartApi import SmartConnect

from analysis.indicators import calculate_all
from analysis.signals import generate_signal, _safe_float
from database.models import ALL_TABLES, CREATE_INDEXES
from data.stocks_list import INDEX_SYMBOLS

load_dotenv(override=True)

# 1. Connect to Turso
url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")
print(f"🔄 Connecting to Turso: {url}", flush=True)
conn = libsql.connect(url, auth_token=token)

# Start clean
print("🗑️  Cleaning remote tables...", flush=True)
conn.execute("DROP TABLE IF EXISTS prices")
conn.execute("DROP TABLE IF EXISTS technical_indicators")
conn.execute("DROP TABLE IF EXISTS ai_signals")
conn.execute("DROP TABLE IF EXISTS market_overview")
conn.execute("DROP TABLE IF EXISTS news_sentiment")

print("📦 Creating fresh tables...", flush=True)
for sql in ALL_TABLES: conn.execute(sql)
for sql in CREATE_INDEXES: conn.execute(sql)

# 2. Authenticate with Angel One
api_key = os.getenv("ANGEL_API_KEY")
client_id = os.getenv("ANGEL_CLIENT_ID")
password = os.getenv("ANGEL_PASSWORD")
totp_secret = os.getenv("ANGEL_TOTP_SECRET")

print("\n🔑 Logging in to Angel One...", flush=True)
smart_api = SmartConnect(api_key=api_key)
totp = pyotp.TOTP(totp_secret).now()
login_data = smart_api.generateSession(client_id, password, totp)
if not login_data.get("status"):
    print(f"❌ Login failed: {login_data}")
    exit(1)
print(f"✅ Logged in as {client_id}", flush=True)

# 3. Load ALL tokens
with open("data/angel_tokens.json", "r") as f:
    token_map = json.load(f)

tokens_to_fetch = []
for sym, data in token_map.items():
    tokens_to_fetch.append((sym, data["token"]))

total_symbols = len(tokens_to_fetch)
print(f"\n📊 Fetching 5 years data for {total_symbols} symbols...", flush=True)

# Helper for manual batch inserts
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

# Time window: Angel One requires exact dates
end_date = datetime.datetime.now()
start_date = end_date - datetime.timedelta(days=5*365)
end_str = end_date.strftime("%Y-%m-%d 15:30")
start_str = start_date.strftime("%Y-%m-%d 09:15")

total_prices = 0
total_indicators = 0

for idx, (symbol, token_id) in enumerate(tokens_to_fetch, 1):
    symbol_ns = symbol + ".NS"
    print(f"\n  [{idx}/{total_symbols}] {symbol} (Token: {token_id})...", end=" ", flush=True)
    
    try:
        # Rate limit: 3 requests per second
        time.sleep(0.4) 
        historic_param = {
            "exchange": "NSE",
            "symboltoken": token_id,
            "interval": "ONE_DAY",
            "fromdate": start_str,
            "todate": end_str
        }
        res = smart_api.getCandleData(historic_param)
        
        if not res.get("status") or not res.get("data"):
            print(f"❌ API Error: {res.get('message', 'No data')}")
            continue
            
        data = res["data"]
        print(f"Got {len(data)} candles.", end=" ", flush=True)
        
        # Build DataFrame for indicators
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        
        # 1. Prices
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
        
        # 2. Indicators
        indicator_rows = []
        sig_val = "HOLD"
        sig_str = 0.0
        
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
        total_prices += len(price_rows)
        total_indicators += len(indicator_rows)
        print(f"✅ Synced {len(price_rows)} prices | {len(indicator_rows)} inds | {sig_val}", flush=True)

    except Exception as e:
        print(f"❌ ERROR: {e}", flush=True)

# Fetch Indices via yfinance because Angel token issues for indices
print("\n📈 Fetching index data via yfinance...", flush=True)
for yf_sym, name in INDEX_SYMBOLS.items():
    print(f"  {name:15s} ", end="", flush=True)
    try:
        import yfinance as yf
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            print("— no data")
            continue
            
        price_rows = []
        for date_idx, row in df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            price_rows.append((
                yf_sym, "NSE", date_str, None,
                round(row["Open"], 2), round(row["High"], 2),
                round(row["Low"], 2), round(row["Close"], 2),
                int(row["Volume"]), "1d"
            ))
            
        manual_batch_insert(
            conn, "prices",
            ["symbol", "exchange", "date", "time", "open", "high", "low", "close", "volume", "interval"],
            price_rows
        )
        conn.commit()
        total_prices += len(price_rows)
        print(f"✅ {len(price_rows)} rows")
    except Exception as e:
        print(f"❌ {e}")

smart_api.terminateSession(client_id)
print(f"\n✅ Build complete! {total_prices} prices, {total_indicators} indicators explicitly uploaded to Turso.", flush=True)
conn.close()
