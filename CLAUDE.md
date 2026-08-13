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
│   ├── final_models/          — ~520 production .pkl models (~5.2GB) — live, loaded by the API
│   │                            NB: a CI job that syncs the full set needs ~10.5GB free
│   │                            (huggingface_hub writes a cache copy AND a target copy)
│   ├── model_archives/
│   │   ├── training_snapshots/ — Per-symbol training output (v2/v3) written by model_training.py, read by scripts/retrain_failed_models.py
│   │   └── previous_models/    — Pre-retrain backups, written by scripts/retrain_walk_forward.py
│   ├── data/                  — angel_tokens.json, retrain_results.csv (signals live in the trade_signals DB table, not JSON)
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

## Trading Calendar (NSE holidays)

`market_holidays` holds the NSE trading calendar, loaded by
`collectors/nse_holidays_collector.py` from the API behind
[nseindia.com/resources/exchange-communication-holidays](https://www.nseindia.com/resources/exchange-communication-holidays)
(`/api/holiday-master?type=trading`, segment `CM` = equity). Refreshed weekly by
the `market_holidays` scheduler job (Sun 19:30 IST).

**The feed only ever returns the current calendar year**, so rows accumulate
year by year — the upsert never deletes, and nothing else should either:
prior years are what let the price-date check tell a missed EOD collection
apart from a genuine exchange holiday.

`analysis/trading_calendar.py` is the single source of truth for "was the
market open on date X":

- `is_trading_day()` / `previous_trading_day()` / `next_trading_day()` — a
  trading day is a weekday not in `market_holidays`.
- `last_expected_trading_day()` — the newest date `prices` should already have
  a bar for; today only counts after the 15:35 IST EOD window.
- `verify_price_dates(days=N)` — expected trading days vs dates actually in
  `prices`. Reports `missing_dates`, `partial_dates` (day collected for far
  fewer symbols than usual), `unexpected_dates` (bars dated on a holiday/weekend
  → the source misdated a candle), and `stale_by_days`. Years with no holiday
  rows are reported as `uncovered_years` and deliberately **not** judged —
  without the calendar every real holiday reads as a false gap.

Consumers: `GET /api/market/holidays` and `GET /api/market/data-freshness`
(banner + drawer in `frontend/src/components/MarketBanner.tsx`),
`GET /api/market/status` (`session: "holiday"`), the weekly
`verify_data_integrity_job`, and `collect_eod_data_job`, which skips the whole
chain on a holiday.

---

## Data Sources

**All price data (OHLCV) comes exclusively from Angel One SmartAPI** — never Yahoo Finance, NSE direct, or any other provider.

- `collectors/angel_collector.py` — EOD + intraday candles via SmartAPI
- `collectors/ltp_fetcher.py` — live price (LTP) via SmartAPI
- `data/angel_tokens.json` — maps stock symbols to Angel One instrument tokens

**Angel One's corporate action behaviour (important):** Angel One sometimes retroactively adjusts historical candles after a split or bonus. This means for some stocks the price history is already adjusted by Angel One, and for others it is not. The `apply_corporate_action_adjustments()` function in `model_training.py` detects this automatically — it checks the actual price ratio on the ex-date and skips adjustment if Angel One already did it, to avoid double-adjusting.

News/sentiment sources (secondary, not price data):
- GDELT — `collectors/gdelt_collector.py`. **Non-functional in practice** —
  every request 429s and it has produced zero rows (see "Historical news
  depth" below). Its `score_pending_news()` is still very much live, though:
  that is the FinBERT scorer every other collector depends on.
- RSS feeds — `collectors/rss_collector.py`
- NSE announcements — `collectors/nse_announcements_collector.py`
- BSE announcements — `collectors/bse_announcements_collector.py`
- FinBERT — sentiment scoring (`analysis/sentiment.py`)

### Historical news depth

`news_sentiment` starts at **2023-01-01**; `prices` reaches back to 2010. Three
hard limits shape what can close that gap:

- **GDELT cannot help — it does not answer at all.** Measured 2026-08-13:
  three fetches (RELIANCE 2019-03, TCS 2019-03, INFY 2021-06) each returned
  429 on the *first* call with no prior traffic, exhausted the 60/120/180s
  retry ladder, and yielded zero articles — ~411s per request for nothing.
  `news_sentiment` has never held a single GDELT-sourced row; the media rows
  all come from RSS, Economic Times, yfinance and alphavantage. Don't plan a
  backfill around it, at any date range or symbol count.
- **Both exchange archives reach the whole range.** NSE returns 2012 Q1
  announcements and BSE serves post-2018 fine. An earlier version of this
  section claimed NSE "bottoms out around 2018" — that was wrong.
- **BSE is materially deeper before 2018**, which is why it takes the early
  window and `bse_announcements_collector.py` exists. Measured on the same
  symbol and quarter: RELIANCE 2015 Q1 → BSE 171 rows vs NSE 33; TCS 2012 Q1 →
  32 vs 19; INFY 2012 Q1 → 12 vs 12.

So the two archives tile rather than overlap (**BSE 2010-01-01 → 2017-12-31,
NSE 2018-01-01 → 2022-12-31**, existing data from 2023). The boundary year is a
coverage choice, but the *disjointness* is not optional: a company files the
same event to both exchanges under different URLs, so `uq_news_url_pubdate`
cannot dedupe across sources — overlapping windows would store it twice and
double its weight in the `news_daily_sentiment` aggregate.

`scripts/backfill_announcements.py` drives both plus the scoring pass, and
`.github/workflows/backfill-announcements.yml` runs it sharded. **Fetch shards
must pass `--fetch-only`**: `score_pending_news()` claims every globally
unscored row, so concurrent scorers redo each other's work. Score once, after.

Worth knowing before spending on depth: 450 of 537 symbols have no price bars
before 2023, so pre-2023 news only pays off for the ~7 symbols whose history
starts in 2010 — or after a deeper price backfill.

Dedupe for every news collector rides on the partial unique index
`uq_news_url_pubdate (url, published_at) WHERE url IS NOT NULL`. Without it the
`ON CONFLICT DO NOTHING` these collectors all end in is a silent no-op. It also
means **a URL must identify one filing, not one symbol-day** — the NSE backfill
previously gave every announcement the same symbol-scoped URL, so a day on
which a company filed results *and* a board-meeting notice kept only one of
them.

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

## Production config (HF Space secrets)

`backend/.env` is **never uploaded** — `deploy_space.py`'s `IGNORE_PATTERNS`
excludes `.env*`, non-negotiably. The Space reads everything from its own secret
store instead, and the authoritative list of what gets copied there is
`SECRET_KEYS` in `backend/scripts/deploy_space.py`:

```bash
cd backend && source venv/bin/activate
python scripts/deploy_space.py secrets   # upserts every SECRET_KEYS entry from .env
```

**Adding a variable to `.env` is half the job — add it to `SECRET_KEYS` too, then
re-run that command.** Editing the list alone pushes nothing. A variable missing
from the list is simply absent in production while working on every dev machine,
which is invisible until the feature fails only in prod. It has bitten three
times: `ALPHAVANTAGE_API_KEY` and `NEWSAPI_KEY` (collectors erroring ~500× a
run), and `BROKER_ENCRYPTION_KEY` (every broker credential save/read 500ing,
found 2026-08-11).

Required in production — the Space is broken or silently degraded without these:

| Variable | What breaks |
|---|---|
| `PGHOST` `PGPORT` `PGDATABASE` `PGUSER` `PGPASSWORD` | `db.py` falls back to SQLite and `nifty500.db` is not deployed — the API serves an empty database |
| `JWT_SECRET` | `api/auth.py` raises — all auth dead |
| `BROKER_ENCRYPTION_KEY` | `broker_routes.py` `_get_fernet()` raises; deliberately has **no** fallback to `JWT_SECRET` |
| `HF_TOKEN` + `MODEL_KEY` | `api/server.py` startup only calls `sync_models()` when **both** are set — otherwise the Space boots with no models |
| `CORS_ALLOWED_ORIGINS` | Falls back to localhost dev origins. Currently masked because the Vercel proxy calls the Space server-to-server (no CORS), so this only bites a browser hitting the Space directly. `*` is rejected at startup (`allow_credentials=True`) |
| `ANGEL_*` | No price collection and no live LTP — Angel One is the sole price source |

Optional: `HF_MODELS_REPO` (defaults to `{whoami}/trademind-models`),
`RESEND_*` (email), `ALPHAVANTAGE_API_KEY` / `NEWSAPI_KEY` (news),
`TEST_PG*` (pushed from `.env.test` for the `/api/health/testdb` ping).
`SPACE_ID` is platform-set — `scheduler/jobs.py` keys off it to hand the Friday
retrain to GitHub Actions.

The GitHub Actions workflows do **not** read the Space's store —
`weekly-retrain.yml` and `deploy-backend.yml` have their own repository secrets.
A rotated credential must be updated in both places.

> Fuller notes live in `docs/DEPLOYMENT.md`, but `docs/` is gitignored
> (`.gitignore:57`) — that file is local-only, so this section is the version
> that travels with the repo.

---

## API Base URL

Backend listens on `http://localhost:8000`. Frontend calls it via `src/api.ts`.

Key routes:
- `GET /api/signals` — latest trade signals
- `GET /api/stocks` — stock list with prices
- `GET /api/portfolio` — portfolios
- `POST /api/trades/execute` — place paper/live trade
- `GET /api/market` — market overview
- `GET /api/market/holidays` — NSE trading calendar (+ next holiday, today's status)
- `GET /api/market/data-freshness` — price dates verified against that calendar
- `POST /auth/login` / `POST /auth/register`

---

## Coding Conventions

- **All persisted/application data MUST live in the database (TimescaleDB), never in JSON/CSV files on disk.** This is a hard rule. Signals are in `trade_signals`, prices in `prices`, etc. — anything a route serves or the app reads back must come from a DB table. Do NOT introduce file-backed data stores (e.g. `data/*.json`, `data/*.csv`) for app data, and do NOT add API endpoints that read/write such files. Reasons: the HF Space has ephemeral disk and never receives `data/**` (see `deploy_space.py` IGNORE_PATTERNS), so any JSON/CSV is invisible in production; and there is no durability, concurrency safety, or multi-instance consistency. Known legacy violations to migrate when touched, not extend: `retrain_results.csv` (model-training stats) and any `strategy_backtest.json`. `data/angel_tokens.json` is config (instrument-token map), not app data — that one's fine.
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
- `tests/test_api_routes.py` — seeds the test DB the same way production data actually arrives (the same insert helpers/SQL the app uses), then hits the real route through `api_client` and asserts on the response. All signal routes (`/api/signals/all`, `/api/backtest/summary`, `/api/stocks`) are DB-backed — they read the `trade_signals` table, and `scripts/generate_trades.py` writes ONLY to that table (no JSON snapshots). The one remaining file-backed input is `retrain_results.csv` (model-training stats in `/api/backtest/summary`); its test monkeypatches the route module's `DATA_DIR` to a tmp dir so it never touches the real `backend/data/`.
- `tests/test_scheduler_jobs.py` — DB-only jobs (`calculate_indicators_job`, `cleanup_old_data_job`, `verify_data_integrity_job`) run directly against seeded test-DB data.
- `tests/test_external_api_contracts.py` — feeds the Angel One/yfinance fixtures into the actual parsing functions (`scripts/update_stocks_angel.py:fetch_candles`, `collectors/yfinance_news_collector.py:collect_stock`, etc.) with a fake API object standing in for `SmartConnect`/`yf.Ticker`, and checks the resulting DB rows — this is what would catch an Angel One/yfinance response-shape change before it breaks a live job.

Run everything: `cd backend && APP_ENV=test pytest -v`.

That works only because `pytest.ini` sets `pythonpath = .`. `conftest.py`
imports `database.db`, which needs `backend/` on `sys.path`; the bare `pytest`
console script does not add the cwd (`python -m pytest` does). **Remove that
line and collection dies with `ModuleNotFoundError: No module named
'database'` before a single test runs.**

Expect the full suite to take **~60 minutes** — 264 tests at roughly 4 per
minute, every one round-tripping to the Timescale Cloud test instance, with
per-test truncation and network latency dominating (CPU sits at ~2%). It is
slow, not hung: `TestSquareOff` alone takes ~3 minutes for 6 tests. Practical
consequences:

- Run it in the background and watch the log; a foreground run will hit any
  10-minute command timeout, and a long background run may be killed before it
  finishes (this has happened at 82%).
- **Never pipe the run through `tail`.** That buffers all output until exit, so
  a long run shows nothing at all, and the reported exit code is `tail`'s — a
  killed pytest still looks like `exit 0`.
- Before concluding anything is stuck, add `-o faulthandler_timeout=45` to get a
  stack dump of what is actually blocking.
- To finish an interrupted run, re-run only the files that hadn't completed
  rather than starting the whole suite over.

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
