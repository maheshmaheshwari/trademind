# TradeMind AI — Full Review Report

**Date:** June 3, 2026  
**Fixed:** June 4, 2026  
**Branch:** `main`  
**Status:** ✅ ALL ~91 FINDINGS FIXED (including 3 previously skipped architectural items)

---

## Summary

| Priority | Findings | Status |
|---|---|---|
| Critical / High | 32 | ✅ All fixed |
| Medium | 34 | ✅ All fixed |
| Low | 25 | ✅ All fixed |
| **Total** | **~91** | **✅ Complete** |

---

## What Was Fixed

### Security (S1–S5) ✅
- **S1/S2** — JWT auth added to all trading endpoints; `req.user_id` replaced with `user["id"]` from verified token
- **S3** — Hardcoded JWT fallback secret removed; raises `RuntimeError` at startup if `JWT_SECRET` env var unset
- **S4** — Watchlist endpoints require JWT auth with IDOR check (`user["id"] != user_id → 403`)
- **S5** — GTT orders/sync endpoints require JWT auth

### Backend (B1–B11) ✅
- **B1** — psycopg2 `ThreadedConnectionPool(minconn=2, maxconn=10)` — no more new TCP connection per call
- **B2** — `insert_news` has `ON CONFLICT DO NOTHING`
- **B5** — `sys.argv` mutation removed from scheduler; `main(days=2)` parameter used directly
- **B6/B7** — `execute_signal` wrapped in `try/except/finally`; `SELECT ... FOR UPDATE` for race condition
- **B8** — `_place_angel_buy()` moved after `conn.commit()`
- **B10** — Catches `psycopg2.errors.UniqueViolation` properly
- **B11** — `final_models/` uses `os.path.abspath(__file__)` — works from any CWD

### ML Pipeline (M1–M5) ✅
- **M1** — XGBoost early stopping uses validation split from `Xtr` (no test-set leakage)
- **M2** — `.bfill()` removed from `engineer_features_and_target()`; only `.ffill()` kept
- **M4** — `sentiment.py` uses `_execute()` from `database.db` — sentiment writes now work
- **M5** — `train_and_evaluate` saves to both `models/best_v3.pkl` AND `final_models/{symbol}_final.pkl`

### Frontend Broken Features (F1–F7) ✅
- **F1** — WatchlistPage "Add" button calls `useAddToWatchlistMutation`
- **F2** — `AddPositionModal.submit()` calls `useExecuteSignalMutation`
- **F4** — SELL execution `target_price` fixed (was incorrectly set to `stop_loss`)
- **F5** — SELL button shows down-arrow SVG (was up-arrow, same as BUY)
- **F6** — `AuthContext` clears both token AND user state on token failure
- **F7** — `getStockDetail` calls correct endpoint (`/api/stocks/{symbol}`)

### Security Medium (SM1–SM5) ✅
- **SM1** — Raw exceptions hidden in responses unless `DEBUG=true`
- **SM3** — `/api/signals/refresh` requires JWT auth
- **SM4** — Minimum password raised from 4 to 8 characters
- **SM5** — `sslmode` reads from `PGSSLMODE` env var (default `"prefer"`)

### Backend Medium (BM1–BM5) ✅
- **BM1** — Price alerts have 24h cooldown (no repeat spam)
- **BM2** — `notify_signal_changes_job` DB connection leak fixed with `try/finally`
- **BM5** — Market hours check uses `datetime.now(tz=ZoneInfo("Asia/Kolkata"))` (IST)

### ML Medium (MM1–MM6) ✅
- **MM1** — `sent_extreme_neg` sign bug fixed (`> 0.8` not `< -0.8`)
- **MM3** — STRONG SELL now requires `acc >= 0.80` (same gate as STRONG BUY)
- **MM6** — FinBERT label order reads from `model.config.id2label`

### Frontend Medium (FM1–FM7) ✅
- **FM2** — Dynamic greeting (morning/afternoon/evening); real API data for delta values
- **FM3** — Date sort uses `new Date(va).getTime()` (was `va instanceof Date` — always false)
- **FM4** — `useMemo` on signals filtering in `AISignalsPage`
- **FM5** — `useDeferredValue` on watchlist search debounce

### Low Priority ✅
- Unused imports removed (`lru_cache`, `calendar` moved to top)
- STT (0.1%), SEBI (0.0001%), stamp duty (0.015%) added to fee calculation
- SMA-200 warmup: `days=365` → `days=400` in `signals.py`
- `Navbar.tsx`: `navigate('/settings/risk')` → `navigate('/settings')`
- `Layout.tsx`: account nav items now show active state
- `frontend/src/utils/format.ts` created with `inr()`, `inrCompact()`, `fmtAgo()` helpers

### Previously Skipped — Now Fixed ✅

- **B3** — Deduplicated 71,232 duplicate rows from `news_sentiment`; added partial UNIQUE index on `(url, published_at) WHERE url IS NOT NULL` (TimescaleDB requires partitioning column in unique index)
- **B4** — Added `ThreadPoolExecutor(max_workers=10)` in `api/server.py`; `run_in_thread()` helper offloads blocking psycopg2 calls from the async event loop; applied to `execute_signal` and position handlers
- **B9** — `_AngelSessionCache` singleton in `trading_engine.py`: thread-safe cached session reused across all LIVE orders, refreshes every 6 hours, auto-invalidates on 401 errors — no more fresh login per order
