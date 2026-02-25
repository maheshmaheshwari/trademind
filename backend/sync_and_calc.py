"""Sync local DB to Turso + calculate indicators."""
import os
import libsql_experimental as libsql
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")

# Delete the corrupted WAL files and re-sync
print("🔄 Resetting Turso replica...")
for f in ["nifty500.db-wal", "nifty500.db-shm"]:
    if os.path.exists(f):
        os.remove(f)
        print(f"   Deleted {f}")

# Connect with sync
print("🔄 Syncing to Turso cloud...")
try:
    conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
    conn.sync()
    count = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"✅ Turso sync successful! {count:,} rows in cloud")
    conn.close()
except Exception as e:
    print(f"⚠️  Turso sync failed: {e}")
    print("   Data is safe locally. Will retry sync later.")

# Calculate indicators
print("\n🔬 Calculating indicators for all stocks...")
from analysis.signals import process_all_stocks
result = process_all_stocks()
print(f"   Processed: {result['processed']} stocks")
print(f"   Failed: {result['failed']}")

# Sync indicators to Turso
print("\n🔄 Final sync...")
try:
    conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
    conn.sync()
    
    ind = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
    sig = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    stocks = conn.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
    
    print(f"\n{'='*50}")
    print(f"📦 Final Turso Cloud DB Stats:")
    print(f"   Prices:     {total:>8,} rows")
    print(f"   Indicators: {ind:>8,} rows")
    print(f"   Signals:    {sig:>8,} rows")
    print(f"   Symbols:    {stocks:>8}")
    print(f"{'='*50}")
    conn.close()
except Exception as e:
    print(f"⚠️  Final sync failed: {e}")
    # Show local stats instead
    conn = libsql.connect("nifty500.db")
    ind = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
    sig = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]
    total = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
    print(f"\n📦 Local DB Stats (Turso sync pending):")
    print(f"   Prices: {total:,}, Indicators: {ind}, Signals: {sig}")
    conn.close()

print("\n✅ Done!")
