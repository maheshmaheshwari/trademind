"""
Force-push all local data to Turso cloud.

The issue: indicators were written via local-only connection,
so they're not in the Turso WAL. This script reads all local
data and re-inserts it through a Turso-synced connection.
"""
import os
import libsql_experimental as libsql
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")

# Step 1: Read ALL data from local DB
print("📖 Reading local database...")
local = libsql.connect("nifty500.db")

prices = local.execute("SELECT * FROM prices").fetchall()
price_cols = [d[0] for d in local.execute("SELECT * FROM prices LIMIT 1").description]
print(f"   Prices: {len(prices):,} rows")

indicators = local.execute("SELECT * FROM technical_indicators").fetchall()
ind_cols = [d[0] for d in local.execute("SELECT * FROM technical_indicators LIMIT 1").description]
print(f"   Indicators: {len(indicators):,} rows")

signals = local.execute("SELECT * FROM ai_signals").fetchall()
sig_cols = [d[0] for d in local.execute("SELECT * FROM ai_signals LIMIT 1").description]
print(f"   Signals: {len(signals):,} rows")

local.close()

# Step 2: Delete old local replica + WAL files to start fresh
print("\n🗑️  Removing old replica files...")
for f in ["nifty500.db", "nifty500.db-wal", "nifty500.db-shm"]:
    if os.path.exists(f):
        os.remove(f)

# Step 3: Connect with Turso (creates fresh replica)
print("🔄 Connecting to Turso cloud...")
conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
conn.sync()

# Create tables
from database.models import ALL_TABLES, CREATE_INDEXES
for sql in ALL_TABLES:
    conn.execute(sql)
for sql in CREATE_INDEXES:
    conn.execute(sql)
conn.commit()
conn.sync()
print("✅ Tables created in Turso\n")

# Step 4: Push prices (in batches of 5000)
print(f"📤 Pushing {len(prices):,} price rows...")
batch_size = 5000
for i in range(0, len(prices), batch_size):
    batch = prices[i:i+batch_size]
    placeholders = ",".join(["?" for _ in price_cols])
    for row in batch:
        # Skip the 'id' column (auto-increment), use rest
        conn.execute(
            f"INSERT OR IGNORE INTO prices ({','.join(price_cols[1:])}) VALUES ({','.join(['?' for _ in price_cols[1:]])})",
            row[1:]
        )
    conn.commit()
    conn.sync()
    done = min(i + batch_size, len(prices))
    print(f"   {done:,}/{len(prices):,} rows synced")

# Step 5: Push indicators (in batches of 5000)
print(f"\n📤 Pushing {len(indicators):,} indicator rows...")
for i in range(0, len(indicators), batch_size):
    batch = indicators[i:i+batch_size]
    for row in batch:
        conn.execute(
            f"INSERT OR REPLACE INTO technical_indicators ({','.join(ind_cols[1:])}) VALUES ({','.join(['?' for _ in ind_cols[1:]])})",
            row[1:]
        )
    conn.commit()
    conn.sync()
    done = min(i + batch_size, len(indicators))
    print(f"   {done:,}/{len(indicators):,} rows synced")

# Step 6: Push signals
print(f"\n📤 Pushing {len(signals):,} signal rows...")
for row in signals:
    conn.execute(
        f"INSERT OR IGNORE INTO ai_signals ({','.join(sig_cols[1:])}) VALUES ({','.join(['?' for _ in sig_cols[1:]])})",
        row[1:]
    )
conn.commit()
conn.sync()
print(f"   {len(signals)} rows synced")

# Verify
print("\n📊 Verifying Turso cloud data...")
p = conn.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
i = conn.execute("SELECT COUNT(*) FROM technical_indicators").fetchone()[0]
s = conn.execute("SELECT COUNT(*) FROM ai_signals").fetchone()[0]

print(f"\n{'='*50}")
print(f"☁️  Turso Cloud Database — Final:")
print(f"   Prices:     {p:>8,} rows")
print(f"   Indicators: {i:>8,} rows")
print(f"   Signals:    {s:>8,} rows")
print(f"{'='*50}")

conn.close()
print("\n✅ All data synced to Turso! Refresh your dashboard.")
