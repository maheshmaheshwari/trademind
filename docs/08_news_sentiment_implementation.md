# TradeMind — Data Pipeline Implementation Plan

## Status (as of Jun 4, 2026) — ALL PHASES COMPLETE ✅

| Table | Rows | Coverage |
|---|---|---|
| `prices` | 564,808 | Jan 2023 → Jun 2026, 499 stocks |
| `technical_indicators` | 208,075 | Jan 2023 → Jun 2026, every trading day, 498 stocks |
| `news_sentiment` | 199,495 | Jan 2023 → Jun 2026, 494 stocks (NSE announcements + FinBERT) |
| `news_daily_sentiment` | Auto-aggregated | TimescaleDB continuous aggregate, refreshes hourly |

---

## Phase 1 — Stock Price Backfill ✅ DONE

**Script**: `collectors/backfill_prices_angel.py`
- Angel One SmartAPI, 3 yearly chunks per stock (400-day API limit)
- Auto re-login every 150 stocks, exponential backoff on rate limits
- `--retry-failed` flag for any failed symbols
- **Result**: 564,808 rows, Jan 2023 → Jun 2026, all 499 stocks

---

## Phase 2 — Technical Indicators ✅ DONE

### Phase 2a — Latest date only
**Script**: `analysis/signals.py → process_all_stocks()`
- 499 rows (one per stock, latest date)
- Used by daily signal generation

### Phase 2b — Historical backfill (every trading day)
**Script**: `collectors/backfill_indicators_historical.py`
- Batch insert per stock (all dates in one DB round-trip)
- **Result**: 208,075 rows, every trading day, Jan 2023 → present

---

## Phase 3 — Historical News Backfill ✅ DONE

**Script**: `collectors/nse_announcements_collector.py`
- NSE corporate announcements API (free, no key, Jan 2023 → present)
- FinBERT on MPS (Apple GPU), batch size 128
- Batch DB insert per stock (1 connection vs 500 — 18 min → 17 sec per stock)
- `skip_existing=True` — safe to resume
- `--start-idx / --end-idx` for range-based runs
- **Result**: 199,495 rows, 494 symbols, Jan 2023 → Jun 2026

---

## Phase 4 — Daily News Collection ✅ DONE

### yfinance per-stock news
**Script**: `collectors/yfinance_news_collector.py`
- ~10 articles/stock × 499 stocks from Yahoo Finance
- Sources: Reuters, ET, Mint, Business Standard
- Deduplicates by URL
- FinBERT scores each headline, batch inserts per stock
- **Scheduled**: 16:45 IST daily (Mon–Fri)

### RSS market-wide news
**Script**: `collectors/rss_collector.py`
- Scrapes ET Markets, Moneycontrol, Business Standard RSS feeds
- Tags stock-specific articles by company name matching
- Unmatched → `symbol=NULL` → feeds `mkt_sentiment` ML feature
- **Scheduled**: 16:30 IST daily (Mon–Fri)

### Scheduler wiring (`scheduler/jobs.py`)
```
16:30 IST → collect_rss_news_job()      ← RSS market-wide
16:45 IST → collect_yfinance_news_job() ← yfinance per-stock
```

---

## Phase 5 — GDELT Backfill ✅ DONE (optional, background)

**Script**: `collectors/gdelt_collector.py` (pre-existing)
- Rate limited: 1 req/12s, resumes safely
- Run command:
```bash
PYTHONPATH=. python collectors/gdelt_collector.py --from-year 2023 --from-month 1 \
  > logs/gdelt_backfill_$(date +%Y%m%d).log 2>&1 &
caffeinate -dims -w $!
```

---

## Phase 6 — ML Model Retraining ✅ DONE (infrastructure)

**Script**: `retrain_walk_forward.py`
- Strict 2yr train (Jan 2023 – Dec 2024) / 1yr test (Jan 2025 – present)
- 7 models × 6 horizons per stock
- Winner by harmonic mean(accuracy, precision) + bonus if both ≥ 70%
- `--resume` flag, `--workers N` for parallel, results in `data/retrain_results.csv`

**Run**:
```bash
PYTHONPATH=. python retrain_walk_forward.py > logs/retrain_$(date +%Y%m%d_%H%M%S).log 2>&1 &
caffeinate -dims -w $!
```

---

## User-Wise Records ✅ DONE

### New DB functions (`database/db.py`)

| Function | Purpose |
|---|---|
| `get_news_for_user_watchlist(user_id)` | News feed for all stocks in user's watchlist + market-wide |
| `get_news_summary_for_user(user_id)` | Per-stock 7-day sentiment summary + portfolio sentiment |
| `get_user_signal_history(user_id)` | AI signals the user acted on (via `trade_signal_id` FK) |

### New API routes (`api/routes/news.py`)

| Endpoint | Purpose |
|---|---|
| `GET /api/news/watchlist/{userId}` | User's personalised news feed |
| `GET /api/news/watchlist/{userId}/summary` | Per-stock sentiment summary |
| `GET /api/news/stock/{symbol}` | News for a specific stock |
| `GET /api/news/market` | Market-wide news only |
| `GET /api/signals/history/{userId}` | AI signals the user has acted on |

### Frontend RTK Query hooks (`services/tradeMindApiService.ts`)

```ts
useGetUserWatchlistNewsQuery({ userId, limit })
useGetUserWatchlistSentimentQuery(userId)
useGetStockNewsQuery({ symbol, limit })
useGetMarketNewsQuery(limit)
useGetUserSignalHistoryQuery({ userId, limit })
```

---

## Data Flow Into ML Model

```
news_sentiment (raw rows — NSE + yfinance + RSS)
        │ auto-refresh every hour
        ▼
news_daily_sentiment (TimescaleDB continuous aggregate)
  symbol | date | avg_sentiment | news_count | pos_count | neg_count | max_pos | max_neg
        │
        ▼  load_data_for_symbol() — model_training.py
joined price df (prices + technical_indicators + sentiment)
        │
        ▼  engineer_features_and_target()
sentiment_1d, sentiment_3d, sentiment_7d, sentiment_14d
news_volume_spike, news_positive_ratio
sent_price_divergence, sent_extreme_pos, sent_extreme_neg
mkt_sent_3d, mkt_sent_7d
```

---

## Daily Pipeline (EOD — runs automatically via scheduler)

```
15:35 IST  EOD prices (Angel One) ──┐
                                    ├─ chained in collect_eod_data_job()
16:15 IST  Technical indicators ────┤
                                    │
17:15 IST  Trade signals ───────────┘

16:30 IST  RSS news (ET, MC, BS)
16:45 IST  yfinance news (499 stocks)
17:30 IST  Notify users of signal changes on watchlist
```

---

## Next Steps

1. **Run model retraining** — `retrain_walk_forward.py` (needs to run now that news data is complete)
2. **Phase 4 daily collectors** — yfinance + RSS will run automatically via scheduler each market close
3. **GDELT** — run in background for broader news coverage (optional)
