"""
Push news_daily_sentiment table from local DB to Turso cloud.
Uses the same pattern as force_sync.py — reads local, pushes via Turso-synced connection.
"""
import os
import libsql_experimental as libsql
from dotenv import load_dotenv
load_dotenv()

url = os.getenv("TURSO_DATABASE_URL")
token = os.getenv("TURSO_AUTH_TOKEN")

if not url or not token:
    print("❌ TURSO_DATABASE_URL and TURSO_AUTH_TOKEN must be set in .env")
    exit(1)

# Step 1: Read news sentiment from local DB
print("📖 Reading local news_daily_sentiment...")
local = libsql.connect("nifty500.db")

rows = local.execute("SELECT * FROM news_daily_sentiment").fetchall()
cols = [d[0] for d in local.execute("SELECT * FROM news_daily_sentiment LIMIT 1").description]
print(f"   Total rows: {len(rows):,}")

mkt = local.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NULL").fetchone()[0]
stk = local.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]
syms = local.execute("SELECT COUNT(DISTINCT symbol) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]
print(f"   Market-wide: {mkt:,}")
print(f"   Stock-specific: {stk:,} ({syms} stocks)")
local.close()

# Step 2: Connect to Turso
print("\n🔄 Connecting to Turso cloud...")
conn = libsql.connect("nifty500.db", sync_url=url, auth_token=token)
conn.sync()

# Ensure table + indexes exist
from database.models import ALL_TABLES, CREATE_INDEXES
for sql in ALL_TABLES:
    conn.execute(sql)
for sql in CREATE_INDEXES:
    conn.execute(sql)
conn.commit()
conn.sync()
print("✅ Tables ready in Turso\n")

# Step 3: Push in batches (skip 'id' column which is auto-increment)
print(f"📤 Pushing {len(rows):,} news sentiment rows...")
batch_size = 2000
insert_cols = [c for c in cols if c != 'id']
placeholders = ','.join(['?' for _ in insert_cols])
col_names = ','.join(insert_cols)
id_idx = cols.index('id') if 'id' in cols else None

for i in range(0, len(rows), batch_size):
    batch = rows[i:i+batch_size]
    for row in batch:
        # Skip id column
        if id_idx is not None:
            values = tuple(v for j, v in enumerate(row) if j != id_idx)
        else:
            values = row
        conn.execute(
            f"INSERT OR REPLACE INTO news_daily_sentiment ({col_names}) VALUES ({placeholders})",
            values
        )
    conn.commit()
    conn.sync()
    done = min(i + batch_size, len(rows))
    print(f"   {done:,}/{len(rows):,} rows synced")

# Verify
print("\n📊 Verifying Turso cloud data...")
total = conn.execute("SELECT COUNT(*) FROM news_daily_sentiment").fetchone()[0]
mkt_c = conn.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NULL").fetchone()[0]
stk_c = conn.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]
sym_c = conn.execute("SELECT COUNT(DISTINCT symbol) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]

print(f"\n{'='*50}")
print(f"☁️  Turso Cloud — news_daily_sentiment:")
print(f"   Total:          {total:>8,} rows")
print(f"   Market-wide:    {mkt_c:>8,} rows")
print(f"   Stock-specific: {stk_c:>8,} rows ({sym_c} stocks)")
print(f"{'='*50}")

conn.close()
print("\n✅ News sentiment synced to Turso!")
