"""Temp script: Load remaining 24 failed stocks + indices locally."""
import time
import libsql_experimental as libsql
import yfinance as yf

# Failed stocks from first run
failed_symbols = [
    "LTIMindtree.NS", "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS",
    "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "HINDALCO.NS", "COALINDIA.NS", "SUNPHARMA.NS",
    "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "DIVISLAB.NS",
    "ULTRACEMCO.NS", "GRASIM.NS", "SHREECEM.NS", "LT.NS",
    "BHARTIARTL.NS", "TITAN.NS", "ASIANPAINT.NS", "LTIM.NS"
]

conn = libsql.connect("nifty500.db")
total_inserted = 0

print("📥 Loading remaining 24 stocks (local only, no Turso sync)...\n")

for i, symbol in enumerate(failed_symbols, 1):
    print(f"  [{i:2d}/24] {symbol:20s} ", end="", flush=True)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            print("— no data")
            continue

        rows = 0
        for date_idx, row in df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            conn.execute(
                """INSERT OR IGNORE INTO prices
                (symbol, exchange, date, time, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (symbol, "NSE", date_str, None,
                 round(row["Open"], 2), round(row["High"], 2),
                 round(row["Low"], 2), round(row["Close"], 2),
                 int(row["Volume"]), "1d")
            )
            rows += 1
        conn.commit()
        total_inserted += rows
        print(f"✅ {rows} rows")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {e}")
        time.sleep(0.5)

# Load index data
print("\n📈 Loading index data...")
indices = {"^NSEI": "NIFTY50", "^BSESN": "SENSEX", "^INDIAVIX": "INDIAVIX"}
for yf_sym, name in indices.items():
    print(f"  {name:15s} ", end="", flush=True)
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            print("— no data")
            continue
        rows = 0
        for date_idx, row in df.iterrows():
            date_str = date_idx.strftime("%Y-%m-%d")
            conn.execute(
                """INSERT OR IGNORE INTO prices
                (symbol, exchange, date, time, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (yf_sym, "NSE", date_str, None,
                 round(row["Open"], 2), round(row["High"], 2),
                 round(row["Low"], 2), round(row["Close"], 2),
                 int(row["Volume"]), "1d")
            )
            rows += 1
        conn.commit()
        total_inserted += rows
        print(f"✅ {rows} rows")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {e}")

# Final count
total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
stocks = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
conn.close()

print(f"\n{'='*50}")
print(f"📦 Local DB: {total:,} total rows across {stocks} symbols")
print(f"   New rows added: {total_inserted:,}")
print(f"{'='*50}")
