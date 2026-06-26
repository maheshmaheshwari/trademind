# Full-Stack Security & Edge-Case Audit — 2026-06-24

| Field | Value |
|---|---|
| **Scope** | All backend API routes, trading engine, auth/broker integration, and frontend pages/components |
| **Method** | 5 parallel read-only agent reviews (backend auth/infra, backend trading engine, backend data routes, frontend pages, frontend core/components) |
| **Status** | Open — none of the findings below have been fixed yet |
| **Related** | [[SEC-001-mfa-scope-bypass]], [[SEC-002-otp-weak-prng]] (previously fixed, listed here for context — this audit found similar/adjacent gaps that survived that pass) |

This is a point-in-time findings report, not a fix log. File:line references are accurate as of the commit at audit time (`c37a19e`) and should be re-verified before acting on them if the code has since changed.

---

## Critical

| # | Area | File:Line | Issue |
|---|---|---|---|
| C1 | Backend / API auth | `backend/api/routes/portfolio.py` (entire file) | **No authentication on any route.** `GET/POST/PUT/DELETE /api/portfolio*` have no `Depends(get_current_user)` and `portfolios` has no `user_id` column — any unauthenticated caller can read, rebalance, or delete *any* user's portfolio by guessing an integer id (IDOR + full CRUD). |
| C2 | Backend / trading | `backend/trading/price_monitor.py` `update_position_prices` (lines 67-190) | Positions read with no row lock; two concurrent sweeps (scheduled job + synchronous `GET /portfolio` trigger) can both detect SL/Target breach and both call `square_off` on the same position — second call raises an uncaught `ValueError` that aborts the rest of the batch. |
| C3 | Backend / broker | `backend/api/routes/broker_routes.py:73-76,162` | Angel One broker password + live TOTP transit the backend in plaintext (inherent to the integration design). Logging masking works today only because field names happen to match a hardcoded substring allowlist (`server.py` `_SENSITIVE_FIELDS`) — fragile, not a structural guarantee. |

## High

| # | Area | File:Line | Issue |
|---|---|---|---|
| H1 | Backend / auth | `backend/api/routes/broker_routes.py:36` | `BROKER_ENCRYPTION_KEY` silently falls back to `JWT_SECRET` if unset, and is **not set in current `.env`**. A `JWT_SECRET` leak now also decrypts every user's stored Angel One access token. |
| H2 | Backend / auth | `backend/api/routes/auth_routes.py:334-384` | No rate limit / lockout on password-reset OTP confirm — 6-digit code (1M space) brute-forceable within its 15-minute validity window. |
| H3 | Backend / auth | `backend/api/routes/trading.py` (login route) | No rate limit / lockout on `/api/trading/login` — unlimited password-guessing. |
| H4 | Backend / auth | `broker_routes.py:56-66` | This file's own `get_current_user` is missing the `scope == "full"` check present in `auth_routes.py`/`trading.py` — an MFA-step (partial-auth) token can connect/disconnect a live Angel One broker account. |
| H5 | Backend / trading | `backend/trading/trading_engine.py` `execute_signal` (~lines 288-294, 387) | TOCTOU on new-position creation: `SELECT ... FOR UPDATE` only locks an *existing* row, so two concurrent requests for a brand-new position in the same symbol can both pass the no-existing-position check; only the DB unique constraint on the second insert catches it, after the first request's money movement already committed. |
| H6 | Backend / trading | `trading_engine.py:439-452` | LIVE BUY broker failure after DB commit is only logged — no compensating rollback of position/balance/orders. Currently dead code (LIVE mode blocked upstream) but unguarded if ever enabled. |
| H7 | Backend / trading | `backend/trading/price_monitor.py:144-174` | Manual connection lifecycle (commit/release, then re-acquire) around each `square_off` call — a mid-batch exception leaves no clear record of which positions were actually closed and the rollback acts on the wrong connection. |
| H8 | Backend / trading | `backend/trading/risk_manager.py` (`check_order`) + `trading_engine.py` `execute_signal` | Risk checks are advisory-only logic invoked solely by one HTTP route (`trading.py`); `execute_signal` itself never re-verifies daily-loss/trade-count/concentration — any other caller (future autopilot, script) bypasses all risk controls. |
| H9 | Backend / trading | `backend/api/routes/trading.py:211-243` | Check-then-act race: `check_order` and `execute_signal` run in separate transactions with no lock between them — two concurrent requests (double-click, two tabs) can both pass daily-trade-count/concentration checks against the same stale snapshot. |
| H10 | Backend / trading | `backend/trading/gtt_manager.py:303-318` | Broker-side "cancel other leg" call happens before the DB write/commit — if the DB write fails after a successful broker cancel, local state diverges from broker truth until the next sync. |
| H11 | Frontend / auth | `frontend/src/services/tradeMindInterceptor.ts:6` + `frontend/src/pages/AuthPage.tsx:166,199` | Interceptor (used by virtually every RTK Query call) reads the JWT **only from `localStorage`**, but unchecked "remember me" stores the token in `sessionStorage` only. Every request after such a login goes out unauthenticated → 401 → auto-logout. This breaks the "don't remember me" login path in production today. |
| H12 | Frontend / trades | `StockPage.tsx` (`executeBuy` 359, `executeSell` 403, `handleClosePosition` 540), `TradesPage.tsx:62` | No in-flight submit lock independent of `isLoading` — a fast double-click can fire a buy/sell/square-off mutation twice before the button disables, risking duplicate orders. |

## Medium

| # | Area | File:Line | Issue |
|---|---|---|---|
| M1 | Backend / auth | `auth_routes.py:142-224, 231-249` | `PATCH /auth/me` allows changing `email` with no re-verification; chainable with Google-auth email-based account linking → account-takeover path. |
| M2 | Backend / auth | `auth_routes.py:327` | Password-reset-request itself is unthrottled — row growth / spam vector on `password_reset_otps`. |
| M3 | Backend / server | `server.py:78-82` | CORS allow-list is dev-only (localhost); confirm production config isn't later loosened to `"*"` given `allow_credentials=True` is already set. |
| M4 | Backend / server | `broker_routes.py:167` | Raw exception text (`detail=f"...{e}"`) returned to clients on Angel One login failure — bypasses the app's global DEBUG-gated error handler. |
| M5 | Backend / trading | `trading_engine.py:349-360` | GTT placement failure for LIVE positions is only logged — BUY proceeds with no SL/Target protection and no signal of this in the API response. |
| M6 | Backend / trading | `trading_engine.py:525-631` (`square_off`) | No `try/finally` — an exception mid-function leaks the DB connection and never rolls back while holding row locks. |
| M7 | Backend / trading | `trading_engine.py:547` | `square_off` falls back to `avg_buy_price` if `current_price` is falsy, and to `0` if both are falsy — can produce a fake "100% loss" P&L with no validation. |
| M8 | Backend / trading | `price_monitor.py` `_get_db_price` (47-54) | Falls back to last DB close price with no staleness/age check if live LTP fetch fails — could be multi-day stale and still drive SL/Target decisions. |
| M9 | Backend / trading | `risk_manager.py:137-149` + `trading.py:66,217` | `max_safe_qty` is fully client-supplied; server never independently derives it from `trade_signals.recommended_volume`, so a client can omit it to disable the volume-safety check. |
| M10 | Backend / trading | `gtt_manager.py:270-337` | If `square_off` fails after an order is marked `EXECUTED`, the failure is logged but never retried — order history says EXECUTED while the position stays open and balance is never credited. |
| M11 | Backend / trading | `trading.py` `ExecuteSignalRequest` (55-67) | No `Field(gt=0)` constraints on `investment_amount`/`buy_price`/`target_price`/`stop_loss` — `buy_price=0` reaches `trading_engine.py` and raises a raw `ZeroDivisionError`. |
| M12 | Backend / autopilot | `autopilot.py` `toggle`/`_fire_pending_mandates` (152-188, 246) | No atomic claim on `authorized_trades.status`; rapid toggle on/off/on could let two background tasks both read the same `PENDING` rows before either updates status — possible duplicate execution. |
| M13 | Backend / signals | `signals.py:188-199` (`POST /signals/refresh`) | Any authenticated user (not just admin) can trigger full model-signal regeneration across ~480 models with no cooldown — repeated calls queue concurrent expensive background jobs. |
| M14 | Backend / data routes | `prices.py:55`, `indicators.py:94`, `sentiment.py:55,163,203`, `stocks.py:261-263,395,647-651`, `signals.py:56` | Recurring pattern: raw `str(e)` returned in HTTP `detail`/response body across multiple routes — internal error leakage. |
| M15 | Backend / cross-cutting | server-wide | No rate-limiting middleware anywhere (`slowapi`/`Limiter` absent) — compounds H2, H3, M2, M13, and the unauthenticated portfolio routes (C1). |
| M16 | Backend / data | `db.py` (CLAUDE.md vs actual config) | `ThreadedConnectionPool(minconn=2, maxconn=10)` in code vs. documented `maxconn=30` in CLAUDE.md — actual headroom is 3x smaller than assumed, relevant given the connection-leak findings (M6) and multi-round-trip price-monitor calls. |
| M17 | Frontend / pages | `AISignalsPage.tsx:171-176,192` | Optional-chaining convention violation — array/key access uses `s?.x` but several cell renders in the same `.map` use bare `s.sector`/`s.signal`/`s.confidence`/`s.horizon`. |
| M18 | Frontend / pages | `DashboardPage.tsx:328` | Advance/decline ratio `breadth.advances / breadth.declines` has no divide-by-zero guard (contrast with the correct `|| 1` guard in `MarketPage.tsx:82`). |
| M19 | Frontend / pages | `RiskSettingsPage.tsx:142`, `SettingsPage.tsx:834` | Numeric risk inputs (max daily loss, stop-loss %, target %, max position size) have no `min`/`max`/sanity validation before submission — negative or zero limits can be saved silently. |
| M20 | Frontend / components | `components/AddPositionModal.tsx:62` | Quantity validation (`+qty > 0`) allows fractional shares (e.g. `2.5`) with no `Number.isInteger` check, and has no upper bound or balance check (contrast with `StockDrawer.tsx`'s `insufficient` pattern). |
| M21 | Frontend / core | `services/tradeMindInterceptor.ts:16-23` | Auto-logout fires on any HTTP 401 with no single-flight guard — concurrent in-flight queries each independently dispatch `trademind:unauthorized`/`logout()` on token expiry (idempotent but redundant), and there's no refresh-token recovery path or "session expired" toast before the redirect. |
| M22 | Frontend / core | `src/api.ts:1` | Hardcoded `http://localhost:8000`, violating the project's own convention. Mostly dead code (only `getMe`/`clearToken` are imported), but in a prod build this makes the auth-bootstrap `getMe()` call on app load silently hit localhost while every other request correctly hits prod — would log every user out on load in production if this path is ever exercised. |

## Low / Informational

- `auth.py:21` — 7-day JWT with no real revocation; `DELETE /auth/sessions` is cosmetic since `decode_token` never checks `user_sessions`.
- `auth_routes.py:440-471` — TOTP re-setup doesn't require current password/TOTP as a precondition while already enabled.
- `db.py:33` — `PGSSLMODE` defaults to `"prefer"`, silently allowing an unencrypted DB connection if the server doesn't offer SSL.
- `server.py:121-157` — Sensitive-field log masking is substring-based and brittle for any newly-named credential field.
- `server.py:501-797` — `yfinance` calls for market overview/heatmap contradict the documented Angel-One-only data source policy.
- `stocks.py:332-345` — `.replace("{days}", ...)` string-interpolated SQL; currently safe (server-controlled value) but fragile if `days` is ever sourced from user input.
- `gtt_manager.py:368-374` — Rollback doesn't verify `cancel_gtt`'s return value; a failed cancel-on-rollback can leave an orphaned live GTT untracked locally.
- `price_monitor.py:19-26` — No NSE holiday calendar / explicit IST timezone in market-open check.
- `trading_engine.py` fee model — STT applied on entry but not on exit, understating exit fees in displayed P&L; P&L arithmetic uses Python `round()` on floats rather than `Decimal`.
- `orders.py` `/gtt/sync` — callable by any authenticated user, fans out one Angel One API call per pending GTT platform-wide; potential broker-session rate-limit exhaustion.
- `autopilot.py:316-336` — Ownership check on `DELETE /trades/{id}` happens after fetch (403 vs 404 distinguishes existing-but-not-owned vs nonexistent ids — minor ID enumeration).
- `PortfolioPage.tsx:29`, `RiskSettingsPage.tsx:91`, `TradesPage.tsx:49-52`, `StockPage.tsx:532,535` — `user!.id` non-null assertions evaluate before RTK Query's `skip: !user` takes effect; if a 401 sets `user` to `null` while one of these pages is mounted, the page crashes instead of gracefully skipping the query.
- `Navbar.tsx:160-170` — array guard (`?? []`) present but callback parameter `n` accessed without optional chaining (`n.id`, `n.title`, etc.).
- No `ErrorBoundary` anywhere in the frontend (`App.tsx`, `Layout.tsx`) — any render-time exception (e.g. from the heavy use of `as any` casts on API responses) blanks the entire app with no recovery UI.
- `PortfolioPage`/`WatchlistPage`/`TradesPage`/`StockPage` — no explicit `isError` UI state; a failed fetch renders an empty table indistinguishable from "no data."
- `AuthPage.tsx` writes to `localStorage`/`sessionStorage` directly in three places instead of going through a shared `setToken()` helper — risk of drift.
- `SettingsPage.tsx` Angel One credentials modal — password field lacks `autoComplete="off"`, risking browser password-manager storage of real brokerage credentials.

## Confirmed safe (no action needed)

- **No SQL injection found anywhere in scope.** Every dynamic-SQL construction point traced back to either a hardcoded whitelist (`_on_conflict_replace`, `_ALLOWED_TABLES` in `trading_engine.py`) or fully parameterized values via `_execute(conn, sql, params)`.
- **Connection-pool hygiene is correct throughout** `database/db.py` and all route files reviewed — `release_connection()` (never `conn.close()`) is used consistently, paired with `try/finally`, except for the one exception at `trading_engine.py:525-631` (M6).
- IDOR checks are correctly implemented in `watchlist.py`, `notifications.py`, `trading.py` (all user-scoped routes), and `signals.py`/`prices.py`/`news.py` (public market data, correctly unauthenticated by design).
- No `dangerouslySetInnerHTML` or raw HTML injection found anywhere in the frontend files reviewed — all dynamic content goes through JSX text interpolation (auto-escaped).
- No sensitive data found in `console.log`/`console.error` statements in any audited frontend file.

---

## Suggested fix order

1. **C1** — add `Depends(get_current_user)` + `user_id` scoping to `portfolio.py` (mirrors the existing pattern already correct in `watchlist.py`/`autopilot.py`).
2. **H11** — fix `tradeMindInterceptor.ts` to check `sessionStorage` as a fallback, matching `api.ts`'s existing dual-storage lookup.
3. **C2 / H7** — add row-level locking (`SELECT ... FOR UPDATE SKIP LOCKED` or advisory lock) in `price_monitor.py`, and wrap `square_off` callers in proper per-position exception handling so one failure doesn't abort the batch.
4. **H1 / H4** — require `BROKER_ENCRYPTION_KEY` explicitly (fail-fast like `JWT_SECRET`), and add the missing `scope == "full"` check to `broker_routes.py`'s `get_current_user`.
5. **H2 / H3 / M15** — add rate limiting (e.g. `slowapi`) to login, password-reset, and signal-refresh endpoints.
6. **H8 / H9** — move risk-check enforcement into `execute_signal` itself rather than relying solely on the calling route, and close the check-then-act race with a lock spanning both check and execution.
7. Frontend `user!.id` crashes (low effort, high visibility) — switch to `user?.id ?? 0` to match the already-correct pattern in `DashboardPage.tsx`/`AutopilotPage.tsx`.
