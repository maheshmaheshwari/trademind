# TradeMind AI — Claude Code Context

## Project Overview

AI-powered trading platform for Nifty 500 stocks (Indian market).
- **Backend**: FastAPI + Python, ML models (XGBoost / LightGBM / RandomForest), TimescaleDB
- **Frontend**: React + TypeScript + Vite + MUI + TailwindCSS
- **Database**: TimescaleDB (PostgreSQL), hosted on **Timescale Cloud** (managed service — not local Docker)
- **ML**: 480 per-stock binary classification models, 6 prediction horizons (1W–6M)
- **Primary data source**: Angel One SmartAPI — ALL price/OHLCV data comes from Angel One, not Yahoo Finance or any other provider

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
│   ├── scripts/                — Manual/CLI pipeline scripts (generate_trades.py, update_stocks_angel.py, retrain_*.py, run_*.sh, etc.) — imported by scheduler/jobs.py as `scripts.<name>`
│   ├── final_models/          — 480 production .pkl models (~247MB) — live, loaded by the API
│   ├── model_archives/
│   │   ├── training_snapshots/ — Per-symbol training output (v2/v3) written by model_training.py, read by scripts/retrain_failed_models.py
│   │   └── previous_models/    — Pre-retrain backups, written by scripts/retrain_walk_forward.py
│   ├── data/                  — trade_signals_latest.json, angel_tokens.json
│   ├── tests/                  — pytest suite, runs against the TEST Timescale Cloud instance only
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
│   ├── 04_migration.md        — Data migration guide
│   ├── RUNNING.md             — How to start frontend + backend
│   └── SETUP.md               — Full setup guide
├── CLAUDE.md                  — This file
└── README.md
```

---

## Database

**Engine**: TimescaleDB (PostgreSQL), hosted on **Timescale Cloud** (a managed instance — there is no local Docker container for this anymore). Connection details (host/port/credentials) live in `backend/.env` only — never hardcode them, and never commit real values outside `.env`.

```
PGHOST=<your-instance>.tsdb.cloud.timescale.com
PGPORT=<cloud-assigned-port>   # not 5433 — that was the old local-Docker port
PGDATABASE=tsdb
PGUSER=tsdbadmin
PGPASSWORD=<see backend/.env>
```

All DB access goes through `backend/database/db.py`. It auto-detects PG vs SQLite:
- `PGHOST` set → psycopg2 (TimescaleDB)
- `PGHOST` not set → sqlite3 (`nifty500.db` fallback)

Key functions: `get_connection()`, `init_database()`, `get_db_stats()`, `get_trade_signals_formatted()`

**Hypertables**: `prices` (64 chunks), `technical_indicators` (60 chunks), `news_sentiment` (3 chunks)

**Continuous aggregate**: `news_daily_sentiment` — auto-refreshed hourly.

---

## Data Sources

**All price data (OHLCV) comes exclusively from Angel One SmartAPI** — never Yahoo Finance, NSE direct, or any other provider.

- `collectors/angel_collector.py` — EOD + intraday candles via SmartAPI
- `collectors/ltp_fetcher.py` — live price (LTP) via SmartAPI
- `data/angel_tokens.json` — maps stock symbols to Angel One instrument tokens

**Angel One's corporate action behaviour (important):** Angel One sometimes retroactively adjusts historical candles after a split or bonus. This means for some stocks the price history is already adjusted by Angel One, and for others it is not. The `apply_corporate_action_adjustments()` function in `model_training.py` detects this automatically — it checks the actual price ratio on the ex-date and skips adjustment if Angel One already did it, to avoid double-adjusting.

News/sentiment sources (secondary, not price data):
- GDELT — news headlines bootstrap (`collectors/gdelt_collector.py`)
- RSS feeds — `collectors/rss_collector.py`
- NSE announcements — `collectors/nse_announcements_collector.py`
- FinBERT — sentiment scoring (`analysis/sentiment.py`)

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
PGHOST=<your-instance>.tsdb.cloud.timescale.com   # Timescale Cloud, not local Docker
PGPORT=<cloud-assigned-port>
PGDATABASE=tsdb
PGUSER=tsdbadmin
PGPASSWORD=...

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
- **Always use `release_connection(conn)` — never `conn.close()`**. `conn.close()` destroys the pool slot permanently; `release_connection` returns it to the `ThreadedConnectionPool` (maxconn=30).
- **Never use `conn.execute()` — always use `_execute(conn, sql, params)`**. psycopg2 connections have no `.execute()` method; that's SQLite syntax.
- **`insert_prices_batch` uses `DO UPDATE` for daily rows** (`time IS NULL`) so EOD data always overwrites incomplete intraday candles. Intraday rows still use `DO NOTHING`.

### Frontend optional chaining — two mandatory patterns

Apply both in every React component. No exceptions.

**Pattern 1 — safe array operations:** use `(arr ?? [])` before `.map/.filter/.sort/.reduce` — never `arr?.map()`
```tsx
// ✅  (signals ?? []).map(s => ...)
// ❌  signals?.map(s => ...)
```

**Pattern 2 — optional chaining inside callbacks:** use `i?.property` on every callback parameter, and `value?.toLocaleString(...)` on method calls — never `(value ?? 0).toLocaleString(...)`
```tsx
// ✅  .map(i => <div key={i?.symbol}>{i?.name}</div>)
// ✅  .sort((a, b) => (a?.[key] ?? 0) - (b?.[key] ?? 0))
// ✅  {value?.toLocaleString('en-IN') || 0}
// ❌  .map(i => <div key={i.symbol}>...)
// ❌  {(value ?? 0).toLocaleString('en-IN')}
```

Already applied to: DashboardPage, MarketPage, WatchlistPage, TradesPage, AutopilotPage, PortfolioPage, BacktestPage.

---

## Testing & Test Database

There are **two** Timescale Cloud instances: production (`backend/.env`) and a dedicated **test** instance (`backend/.env.test`, gitignored — template in `backend/.env.test.example`). Never test against prod directly; this is what caused a real incident (an accidental script import wrote yfinance data into prod).

**The switch**: `database/db.py` calls `load_dotenv()` for `.env` as always, then — only if `APP_ENV=test` is set — reloads `.env.test` with `override=True`, so test credentials win regardless of import order. This means `APP_ENV=test` must be set **before any `database.db` import**, including in ad-hoc shell commands:

```bash
cd backend && source venv/bin/activate
APP_ENV=test python -c "from database.db import get_connection; print(get_connection().dsn)"  # sanity-check the host before doing anything else
```

### Required workflow for `database/schema_pg.py` changes

1. Edit `schema_pg.py`.
2. Apply to the **test** instance first: `APP_ENV=test python -c "from database.db import init_database; init_database()"`.
3. Run the suite: `APP_ENV=test pytest -v` (from `backend/`).
4. Only once green, apply the same (idempotent — `CREATE TABLE/INDEX IF NOT EXISTS`) change to prod: `python -c "from database.db import init_database; init_database()"` (no `APP_ENV`).
5. Treat any write to prod as requiring explicit confirmation each time, even idempotent ones — don't assume an earlier approval covers a later, different change.

`schema_pg.py` is the single source of truth for the schema — if a table/index exists on prod but isn't in this file (this has happened: `delivery_data` and the `idx_prices_daily_unique` partial index were both created out-of-band on prod and missing here until found via test-DB testing), a fresh environment built from this file will be broken. The test DB is what catches this category of bug — a brand-new instance has nothing built out-of-band to mask gaps.

### pytest suite (`backend/tests/`)

- `conftest.py` sets `APP_ENV=test` at import time, bootstraps the schema once per session, truncates `prices`/`technical_indicators`/`trade_signals`/`news_sentiment` before every test, and provides an `api_client` fixture (`TestClient(app)`, deliberately not used as a context manager so `api/server.py`'s `startup_event` — and the real APScheduler — never runs).
- `tests/fixtures/*.json` mirror real external API response shapes (Angel One `getCandleData`/`ltpData`, yfinance `Ticker.news`) and real live API-layer responses (`/api/signals/all`, `/api/stocks`, etc.) — each fixture's `_mirrors` key states exactly which file/function's contract it represents, so it's traceable when that code changes.
- `tests/test_api_routes.py` — seeds the test DB the same way production data actually arrives (the same insert helpers/SQL the app uses), then hits the real route through `api_client` and asserts on the response. A few routes are file-backed instead of DB-backed (`/api/signals/all`, `/api/backtest/summary`) — those tests monkeypatch the route module's path constant to a fixture file so they never touch the real `backend/data/*.json` used by the live app.
- `tests/test_scheduler_jobs.py` — DB-only jobs (`calculate_indicators_job`, `cleanup_old_data_job`, `verify_data_integrity_job`) run directly against seeded test-DB data.
- `tests/test_external_api_contracts.py` — feeds the Angel One/yfinance fixtures into the actual parsing functions (`scripts/update_stocks_angel.py:fetch_candles`, `collectors/yfinance_news_collector.py:collect_stock`, etc.) with a fake API object standing in for `SmartConnect`/`yf.Ticker`, and checks the resulting DB rows — this is what would catch an Angel One/yfinance response-shape change before it breaks a live job.

Run everything: `cd backend && APP_ENV=test pytest -v`.

---

## Starting the Backend

Two modes — use **dev** during development, **prod** for stable runs.

### Development (auto-restart on file change)
```bash
cd /Users/maheshmaheshwari/Documents/trademind/backend
bash dev.sh
```
`dev.sh` uses `watchfiles` to watch `api/`, `analysis/`, `trading/`, `database/`, `collectors/`, `scheduler/`.  
Any `.py` change in those dirs kills and restarts uvicorn automatically — **no manual restart needed**.

### Production (stable, no reload)
```bash
cd /Users/maheshmaheshwari/Documents/trademind/backend
source venv/bin/activate
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Kill backend
```bash
lsof -ti :8000 | xargs kill -9
```

> `--workers 4` requires no `--reload` flag. The scheduler auto-starts in one worker only (atomic PID lock).
> Logs write to `logs/YYYY-MM-DD.log` (date-rotating, one file per day).

---

## Common Commands

All commands are run from `backend/`. Credentials are read from `.env` automatically.

```bash
cd /Users/maheshmaheshwari/Documents/trademind/backend
source venv/bin/activate

# DB schema init (idempotent)
python -c "from database.db import init_database; init_database()"

# Row counts
python -c "from database.db import get_db_stats; [print(f'{t}: {n:,}') for t,n in get_db_stats().items()]"

# Regenerate trade signals
python scripts/generate_trades.py

# Direct DB access (Timescale Cloud — credentials in backend/.env)
psql "postgres://$PGUSER:$PGPASSWORD@$PGHOST:$PGPORT/$PGDATABASE?sslmode=require"
```

---

## Database Hosting (Timescale Cloud)

The database is a managed **Timescale Cloud** instance — there is no local Docker container to start/stop. Connect directly with `psql` using the credentials in `backend/.env`:

```bash
psql "postgres://$PGUSER:$PGPASSWORD@$PGHOST:$PGPORT/$PGDATABASE?sslmode=require"
```

Provisioning, scaling, and backups are managed through the Timescale Cloud console, not via `docker run`.
