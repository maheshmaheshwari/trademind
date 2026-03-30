# TradeMind — Implementation Overview

## Project Status (as of 2026-03-30)

| Component | Status | Notes |
|-----------|--------|-------|
| Stock OHLCV (358/499 stocks) | ✅ Live | 5 years, 2021–2026 |
| Technical Indicators | ⚠️ Stale | Last: 2026-02-23 |
| News / Sentiment | ❌ Missing | Only 1 month of data |
| Market Overview (VIX/FII) | ❌ Missing | 1 snapshot only |
| 141 stocks (no price data) | ❌ Missing | Not bootstrapped |
| ML Models (268/493 loadable) | ⚠️ Broken | sklearn version mismatch |
| Database | ⚠️ SQLite | Needs migration for scale |

---

## Implementation Docs

| # | File | What it covers |
|---|------|---------------|
| 01 | [01_database.md](01_database.md) | TimescaleDB setup, schema, Docker |
| 02 | [02_angel_one.md](02_angel_one.md) | Angel One data collection — historical + live + intraday |
| 03 | [03_news_sentiment.md](03_news_sentiment.md) | 5-year news bootstrap, GDELT, Alpha Vantage, FinBERT |
| 04 | [04_migration.md](04_migration.md) | SQLite → TimescaleDB migration |

---

## Execution Order

```
Step 1  →  01_database.md        Spin up TimescaleDB via Docker, create schema
Step 2  →  04_migration.md       Migrate existing nifty500.db → TimescaleDB
Step 3  →  02_angel_one.md       Bootstrap 5-year OHLCV for all 499 stocks via Angel One
Step 4  →  03_news_sentiment.md  Bootstrap 5-year news via GDELT + Alpha Vantage
Step 5  →  02_angel_one.md       Wire daily + intraday schedulers
```

---

## Files to Create (implementation)

```
backend/
├── collectors/
│   ├── angel_collector.py        ✅ exists
│   ├── ltp_fetcher.py            ✅ exists
│   ├── historical_bootstrap.py   ❌ to create  (Step 3)
│   ├── intraday_collector.py     ❌ to create  (Step 5)
│   ├── index_collector.py        ❌ to create  (Step 3)
│   ├── gdelt_collector.py        ❌ to create  (Step 4)
│   └── alphavantage_collector.py ❌ to create  (Step 4)
├── database/
│   ├── db.py                     ✅ exists — update connection for TimescaleDB
│   ├── models.py                 ✅ exists — SQLite schema (keep for reference)
│   └── historical_data_setup.py  ✅ exists — TimescaleDB schema + migrate()
├── update_stocks_angel.py        ✅ exists — harden (smart date detection)
└── scheduler/jobs.py             ✅ exists — add EOD + intraday jobs
```
