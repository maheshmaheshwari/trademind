# TradeMind AI — Claude Code Context

## Project Overview

AI-powered trading platform for Nifty 500 stocks (Indian market).
- **Backend**: FastAPI + Python, ML models (XGBoost / LightGBM / RandomForest), TimescaleDB
- **Frontend**: React + TypeScript + Vite + MUI + TailwindCSS
- **Database**: TimescaleDB (PostgreSQL) in Docker on port 5433
- **ML**: 480 per-stock binary classification models, 6 prediction horizons (1W–6M)

---

## Repository Layout

```
trademind/
├── backend/
│   ├── api/
│   │   ├── server.py          — FastAPI app, CORS, middleware
│   │   ├── auth.py            — JWT auth
│   │   └── routes/            — prices, indicators, signals, trades, portfolio, trading
│   ├── analysis/
│   │   ├── model_training.py  — ML pipeline v4 (XGBoost, LightGBM, RF, Ensemble)
│   │   ├── indicators.py      — Technical indicator calculations (ta library)
│   │   ├── sentiment.py       — FinBERT news sentiment
│   │   └── signals.py         — Trade signal generation logic
│   ├── collectors/
│   │   ├── angel_collector.py — Angel One SmartAPI price collector
│   │   ├── gdelt_collector.py — GDELT news bootstrap (rate limit: 1 req/12s)
│   │   └── ltp_fetcher.py     — Live price fetcher
│   ├── database/
│   │   ├── db.py              — All DB access (psycopg2 for PG, sqlite3 fallback)
│   │   ├── models.py          — SQLite schema (fallback)
│   │   └── schema_pg.py       — TimescaleDB schema (hypertables, compression, cagg)
│   ├── scheduler/jobs.py      — APScheduler (EOD + hourly + weekly jobs)
│   ├── trading/               — GTT manager, price monitor, risk manager, engine
│   ├── final_models/          — 480 production .pkl models (~247MB)
│   ├── models/                — Training snapshots (v2/v3 per stock)
│   ├── data/                  — trade_signals_latest.json, angel_tokens.json
│   ├── generate_trades.py     — Regenerate all trade signals from models
│   ├── migrate_sqlite_to_pg.py — One-shot SQLite → TimescaleDB migration
│   ├── nifty500.db            — SQLite fallback (226MB, kept for reference)
│   ├── requirements.txt
│   └── .env                   — Environment variables (PG credentials, API keys)
├── frontend/
│   ├── src/
│   │   ├── pages/             — Dashboard, Market, Portfolio, Signals, Trades, Orders
│   │   ├── components/        — Layout, Navbar, Pagination
│   │   ├── AuthContext.tsx    — JWT auth context
│   │   ├── ThemeContext.tsx   — Dark/light theme
│   │   └── api.ts             — Axios API client (base URL: http://localhost:8000)
│   ├── package.json
│   └── vite.config.ts
├── docs/
│   ├── 01_database.md         — TimescaleDB architecture
│   └── 04_migration.md        — Data migration guide
├── CLAUDE.md                  — This file
└── RUNNING.md                 — How to start frontend + backend
```

---

## Database

**Engine**: TimescaleDB (PostgreSQL) in Docker — **port 5433** (not 5432).

```
PGHOST=localhost  PGPORT=5433  PGDATABASE=trademind
PGUSER=trademind  PGPASSWORD=trademind
```

All DB access goes through `backend/database/db.py`. It auto-detects PG vs SQLite:
- `PGHOST` set → psycopg2 (TimescaleDB)
- `PGHOST` not set → sqlite3 (`nifty500.db` fallback)

Key functions: `get_connection()`, `init_database()`, `get_db_stats()`, `get_trade_signals_formatted()`

**Hypertables**: `prices` (64 chunks), `technical_indicators` (60 chunks), `news_sentiment` (3 chunks)

**Continuous aggregate**: `news_daily_sentiment` — auto-refreshed hourly.

---

## ML Models

- **File**: `analysis/model_training.py` — v4
- **Models per stock**: XGBoost, XGB_HiReg, LightGBM, LGB_HiReg, RandomForest, GradBoost + Ensemble
- **Horizons**: 5d (1W), 10d (2W), 20d (1M), 40d (2M), 60d (3M), 120d (6M)
- **Target**: Raw return ≥ threshold (1.5% / 2.5% / 3.5% / 5% / 7% / 10%)
- **Features**: 96 total — returns, MA distances, Bollinger, momentum, volatility, volume, candlestick, 52-week hi/lo, price percentile, gap, calendar, alpha vs market, sentiment
- **Output**: `models/best_{symbol}_v3.pkl` — artifact with model, threshold, features, metrics

Retrain a single stock (`.env` is loaded automatically):
```bash
cd backend
source venv/bin/activate
python -c "import sys; sys.path.insert(0,'.'); from analysis.model_training import train_and_evaluate; train_and_evaluate('HDFCBANK.NS')"
```

---

## Key Environment Variables (backend/.env)

```
PGHOST=localhost
PGPORT=5433
PGDATABASE=trademind
PGUSER=trademind
PGPASSWORD=trademind

ANGEL_API_KEY=...
ANGEL_CLIENT_ID=...
ANGEL_PASSWORD=...
ANGEL_TOTP_SECRET=...

JWT_SECRET=...
PORT=8000
LOG_LEVEL=INFO
```

---

## API Base URL

Backend listens on `http://localhost:8000`. Frontend calls it via `src/api.ts`.

Key routes:
- `GET /api/signals` — latest trade signals
- `GET /api/stocks` — stock list with prices
- `GET /api/portfolio` — portfolios
- `POST /api/trades/execute` — place paper/live trade
- `GET /api/market` — market overview
- `POST /auth/login` / `POST /auth/register`

---

## Coding Conventions

- Python: all DB queries use `?` placeholders — `db.py`'s `_execute()` auto-translates to `%s` for PG
- Never use `pd.read_sql_query` with a psycopg2 connection — use `_query_to_df()` in model_training.py instead
- All collectors import `get_connection` from `database.db` — never open DB connections directly
- Frontend API calls go through `src/api.ts` — never hardcode `localhost:8000` in components

---

## Common Commands

All commands are run from `backend/`. Credentials are read from `.env` automatically.

```bash
cd /Users/mahesh/Desktop/personal/trademind/backend
source venv/bin/activate

# DB schema init (idempotent)
python -c "from database.db import init_database; init_database()"

# Row counts
python -c "from database.db import get_db_stats; [print(f'{t}: {n:,}') for t,n in get_db_stats().items()]"

# Regenerate trade signals
python generate_trades.py

# Direct DB access
docker exec -it trademind-db psql -U trademind -d trademind
```

---

## Docker (TimescaleDB)

```bash
# Start container (auto-restarts)
docker start trademind-db

# Or first-time setup
docker run -d --name trademind-db --restart unless-stopped \
  -e POSTGRES_PASSWORD=trademind -e POSTGRES_DB=trademind -e POSTGRES_USER=trademind \
  -p 5433:5432 -v ~/trademind-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16
```
