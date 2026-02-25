"""
Rebuild the entire database from scratch.
Step 1: Delete old corrupted DB
Step 2: Download all 50 stocks (5 years) — LOCAL ONLY (no Turso)
Step 3: Download indices
Step 4: Calculate indicators + signals
Step 5: Single Turso sync at the very end
"""
import os
import time
import libsql_experimental as libsql
import yfinance as yf
from dotenv import load_dotenv
load_dotenv()

# =========================
# STEP 1: Clean slate
# =========================
print("🗑️  Removing corrupted database files...")
for f in ["nifty500.db", "nifty500.db-wal", "nifty500.db-shm"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"   Deleted {f}")

# =========================
# STEP 2: Create fresh local DB + download stocks
# =========================
print("\n📦 Creating fresh database (LOCAL ONLY)...\n")
conn = libsql.connect("nifty500.db")

# Create tables
from database.models import ALL_TABLES, CREATE_INDEXES
for sql in ALL_TABLES:
    conn.execute(sql)
for sql in CREATE_INDEXES:
    conn.execute(sql)
conn.commit()
print("✅ Tables created\n")

# Stock list
stocks = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "TITAN.NS", "SUNPHARMA.NS", "ULTRACEMCO.NS", "WIPRO.NS", "ONGC.NS",
    "NTPC.NS", "TATASTEEL.NS", "POWERGRID.NS", "M&M.NS", "HCLTECH.NS",
    "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "BAJAJ-AUTO.NS",
    "DRREDDY.NS", "CIPLA.NS", "NESTLEIND.NS", "COALINDIA.NS", "TECHM.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "INDUSINDBK.NS", "JSWSTEEL.NS",
    "HINDALCO.NS", "HEROMOTOCO.NS", "DIVISLAB.NS", "APOLLOHOSP.NS",
    "GRASIM.NS", "SBILIFE.NS", "HDFCLIFE.NS", "TATACONSUM.NS",
    "SHREECEM.NS", "LTIM.NS",
]

total_rows = 0
success = 0
failed_list = []

print(f"📊 Downloading 5 years of data for {len(stocks)} stocks...\n")

for i, symbol in enumerate(stocks, 1):
    print(f"  [{i:2d}/{len(stocks)}] {symbol:20s} ", end="", flush=True)
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            print("— no data")
            failed_list.append(symbol)
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
        total_rows += rows
        success += 1
        print(f"✅ {rows} rows")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {e}")
        failed_list.append(symbol)
        time.sleep(0.5)

# =========================
# STEP 3: Download indices
# =========================
print(f"\n📈 Downloading index data...")
indices = {"^NSEI": "NIFTY50", "^BSESN": "SENSEX", "^INDIAVIX": "INDIAVIX"}
for yf_sym, name in indices.items():
    print(f"  {name:15s} ", end="", flush=True)
    try:
        ticker = yf.Ticker(yf_sym)
        df = ticker.history(period="5y", interval="1d")
        if df.empty:
            print("— no data")
            continue
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
        conn.commit()
        count = len(df)
        total_rows += count
        print(f"✅ {count} rows")
        time.sleep(0.5)
    except Exception as e:
        print(f"❌ {e}")

conn.close()

print(f"\n{'='*50}")
print(f"📦 Total: {total_rows:,} rows, {success}/{len(stocks)} stocks")
if failed_list:
    print(f"⚠️  Failed: {', '.join(failed_list)}")
print(f"{'='*50}")

# =========================
# STEP 4: Calculate indicators (still local only)
# =========================
print("\n🔬 Step 4: Calculating indicators for all stocks...")

# We need to temporarily disable Turso for this
os.environ["TURSO_DATABASE_URL"] = ""
os.environ["TURSO_AUTH_TOKEN"] = ""

# Reload db module with Turso disabled
import importlib
import database.db as db_module
importlib.reload(db_module)

from analysis.signals import process_all_stocks
import analysis.signals as sig_module
importlib.reload(sig_module)
from analysis.signals import process_all_stocks as process_all_stocks_fresh

result = process_all_stocks_fresh()

# =========================
# STEP 5: Sync to Turso
# =========================
print("\n🔄 Step 5: Syncing everything to Turso cloud...")
load_dotenv(override=True)  # Reload env vars
url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")

if url and token and "turso.io" in url:
    try:
        conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
        conn.sync()
        
        total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        ind = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
        sig = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
        syms = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
        
        print(f"\n{'='*50}")
        print(f"☁️  Turso Cloud Database Stats:")
        print(f"   Prices:     {total:>8,} rows")
        print(f"   Indicators: {ind:>8,} rows")
        print(f"   Signals:    {sig:>8,} rows")
        print(f"   Symbols:    {syms:>8}")
        print(f"{'='*50}")
        conn.close()
    except Exception as e:
        print(f"⚠️  Turso sync failed: {e}")
        print("   Data is safe in local nifty500.db")
else:
    print("   Turso not configured, skipping sync")

print("\n✅ Database rebuild complete!")
