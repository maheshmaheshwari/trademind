# 01 — Database: TimescaleDB

## Decision: TimescaleDB (PostgreSQL)

TradeMind uses TimescaleDB running in Docker on **port 5433**.

| Concern | Decision |
|---------|----------|
| Time-series queries | Hypertables auto-partition prices / indicators / news by month |
| Compression | Chunks older than 7 days compressed automatically |
| Continuous aggregate | `news_daily_sentiment` view refreshed hourly |
| Concurrent writes | Scheduler and API write safely via psycopg2 connection pool |
| Local dev | Docker volume at `~/trademind-pgdata` — data survives container restarts |

---

## Docker Container

```bash
# Start (auto-restarts on reboot)
docker run -d --name trademind-db --restart unless-stopped \
  -e POSTGRES_PASSWORD=trademind \
  -e POSTGRES_DB=trademind \
  -e POSTGRES_USER=trademind \
  -p 5433:5432 \
  -v ~/trademind-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

# Check running
docker ps --filter name=trademind-db

# Connect via psql
docker exec -it trademind-db psql -U trademind -d trademind
```

Port **5433** is used (5432 was already in use by another local process).

---

## Connection

All database access goes through `backend/database/db.py`:

```python
from database.db import get_connection, init_database

conn = get_connection()   # returns psycopg2 connection
```

Environment variables (set in `backend/.env`):

```
PGHOST=localhost
PGPORT=5433
PGDATABASE=trademind
PGUSER=trademind
PGPASSWORD=trademind
```

When `PGHOST` is **not** set, `db.py` falls back to `nifty500.db` (SQLite) — useful for offline testing without Docker.

---

## Schema

Defined in `backend/database/schema_pg.py`. Initialised by `init_database()`.

### Regular tables (no time-series partitioning)

| Table | Rows | Notes |
|-------|------|-------|
| `users` | — | Trading accounts |
| `orders` | — | Paper + live orders |
| `positions` | — | Open positions per user |
| `portfolios` / `portfolio_sectors` / `portfolio_stocks` | — | AI-built portfolios |
| `risk_settings` | — | Per-user risk config |
| `trade_signals` | 821 | ML signals per stock per day |
| `ai_signals` | 474 | Confidence/model metadata |
| `market_overview` | 1,300 | NIFTY50/500/SENSEX daily |

### Hypertables (TimescaleDB time-series)

| Table | Rows | Partition | Chunks |
|-------|------|-----------|--------|
| `prices` | 558,465 | monthly by `date` | 64 |
| `technical_indicators` | 386,750 | monthly by `date` | 60 |
| `news_sentiment` | 250+ | monthly by `published_at` | 3 |

### Continuous Aggregate

`news_daily_sentiment` — materialised view over `news_sentiment`, refreshed every hour, covering a 3-day rolling window.

---

## Initialise / Reset

```bash
cd backend
source venv/bin/activate

# First-time setup (idempotent)
python3 -c "from database.db import init_database; init_database()"

# Row counts
python3 -c "
from database.db import get_db_stats
for table, n in get_db_stats().items():
    print(f'{table}: {n:,}')
"

# Direct SQL
docker exec -it trademind-db psql -U trademind -d trademind \
  -c "SELECT COUNT(*), MAX(date) FROM prices WHERE interval='1d';"
```

---

## Storage Estimate

| Table | Current | At full scale |
|-------|---------|---------------|
| prices | 558K rows | ~650K rows |
| technical_indicators | 387K rows | ~625K rows |
| news_sentiment | 250 rows | ~2.5M rows (after GDELT) |
| trade_signals | 821 rows | ~50K rows |
| **Total (compressed)** | **~50MB** | **~200MB** |

TimescaleDB compression reduces storage ~8–10× vs raw PostgreSQL for time-series data.

---

## Backup

```bash
# Logical dump (portable)
docker exec trademind-db pg_dump -U trademind trademind | gzip > trademind_backup_$(date +%Y%m%d).sql.gz

# Restore
gunzip -c trademind_backup_20260330.sql.gz | docker exec -i trademind-db psql -U trademind -d trademind
```
