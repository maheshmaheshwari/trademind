# 04 — Data Migration & Schema Evolution

## Current State (as of 2026-03-30)

TradeMind runs on **TimescaleDB** (PostgreSQL) in Docker on port 5433.
All 948,060 rows from `nifty500.db` have been migrated.

The migration script lives at `backend/migrate_sqlite_to_pg.py`.

---

## Re-running the Migration

If you need to reload from SQLite (e.g. after a container wipe):

```bash
cd backend

# Start container first
docker start trademind-db   # or re-run docker run … see 01_database.md

# Re-initialise schema (idempotent)
python3 -c "from database.db import init_database; init_database()"

# Migrate all data
python3 migrate_sqlite_to_pg.py

# Verify
python3 migrate_sqlite_to_pg.py --dry-run
```

To migrate only one table:

```bash
python3 migrate_sqlite_to_pg.py --table prices
```

---

## Adding New Columns

```sql
-- In psql or via docker exec
ALTER TABLE prices ADD COLUMN adjusted_close DOUBLE PRECISION;
```

Then update `database/schema_pg.py` so `init_database()` stays in sync.

---

## Adding New Tables

1. Add `CREATE TABLE IF NOT EXISTS` SQL to `database/schema_pg.py` in the appropriate section.
2. If it's a time-series table, add a `create_hypertable()` call to `SQL_HYPERTABLES`.
3. Run:

```bash
python3 -c "from database.db import init_database; init_database()"
```

---

## Rebuilding from Scratch

```bash
cd backend

# 1. Wipe TimescaleDB data and re-create schema
docker exec -it trademind-db psql -U trademind -d trademind -c "
  DROP SCHEMA public CASCADE;
  CREATE SCHEMA public;
"
python3 -c "from database.db import init_database; init_database()"

# 2. Re-fetch 5 years of price data (~30 min, Angel One API)
python3 collectors/historical_bootstrap.py

# 3. Calculate technical indicators
python3 -c "
from analysis.indicators import calculate_all_indicators
calculate_all_indicators()
"

# 4. Fetch market overview (NIFTY50/500/SENSEX)
python3 collectors/index_collector.py --history

# 5. Generate trade signals
python3 generate_trades.py

# 6. (Optional, ~8 hrs) GDELT news bootstrap
python3 collectors/gdelt_collector.py --from-year 2021 --from-month 1
```

---

## Migrating to Another Machine

```bash
# Dump on source
docker exec trademind-db pg_dump -U trademind trademind | gzip > trademind.sql.gz

# Copy
scp trademind.sql.gz user@remote:~/

# On remote: start Docker, create DB, restore
docker start trademind-db
gunzip -c trademind.sql.gz | docker exec -i trademind-db psql -U trademind -d trademind
```

---

## Data Volumes (current vs. full scale)

| Table | Current | Full scale (5yr + news) | Notes |
|-------|---------|------------------------|-------|
| `prices` | 558,465 | ~650,000 | 499 stocks × 1,300 trading days |
| `technical_indicators` | 386,750 | ~625,000 | Calculated from prices |
| `news_sentiment` | 250 | ~2,500,000 | After GDELT bootstrap |
| `market_overview` | 1,300 | ~1,300 | Stable |
| `trade_signals` | 821 | ~50,000 | 1yr × 499 stocks |
| `ai_signals` | 474 | ~50,000 | One per stock per day |

---

## If You Want to Revert to SQLite

Remove `PGHOST` from `.env` — `db.py` falls back to `nifty500.db` automatically.

```bash
# In .env, comment out or remove:
# PGHOST=localhost
```

No other code changes needed.
