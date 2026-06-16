# TradeMind — Session Notes (2026-06-12)

## 1. DB Connection Stale Timeout Fix

### Problem
`GET /api/notifications` was intermittently failing with:
```
psycopg2.OperationalError: could not receive data from server: Operation timed out
SSL SYSCALL error: Operation timed out
```

Some requests failed at ~5–20ms (immediate), others at ~3,000ms. The pattern was:
- Docker's network layer silently drops idle TCP/SSL connections after a timeout period
- psycopg2's `ThreadedConnectionPool` keeps dead connections in the pool
- The next caller receives the dead connection and fails immediately
- `release_connection()` returned the dead connection back to the pool — recycling the problem

### Fix — `backend/database/db.py`

**`get_connection()`** now validates every connection before returning it:
```python
def get_connection():
    pool = _get_pool()
    for attempt in range(3):
        conn = pool.getconn()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("SELECT 1")   # lightweight liveness ping
            return conn
        except Exception:
            pool.putconn(conn, close=True)   # physically close + remove from pool
            if attempt == 2:
                raise RuntimeError("Could not obtain a healthy DB connection after 3 attempts")
```

**`release_connection()`** now discards broken connections instead of recycling them:
```python
def release_connection(conn) -> None:
    try:
        pool = _get_pool()
        if getattr(conn, "closed", 0):
            pool.putconn(conn, close=True)   # discard
        else:
            pool.putconn(conn)               # recycle
    except Exception:
        pass
```

**Cost:** ~0.5ms `SELECT 1` per request — negligible. Eliminates the entire class of stale-connection failures.

The pool already had `keepalives=1`, `keepalives_idle=60` etc. set — those help at the OS level but don't prevent Docker bridge network drops. The ping is the correct application-layer fix.

---

## 2. Live Trading Architecture

### How the current code works (LIVE mode)

`execute_signal(mode="LIVE")` in `backend/trading/trading_engine.py`:

1. Pre-trade risk checks (balance, daily loss, trade count, concentration, volume cap, market hours)
2. DB position + 3 orders inserted (ENTRY, STOP_LOSS, TARGET)
3. Real BUY LIMIT order placed on Angel One via `_place_angel_buy()` (uses cached session `_angel_cache`)
4. GTT (Good Till Triggered) rules placed on Angel One for SL and Target via `gtt_manager.place_bracket_gtts()`
5. GTT rule IDs stored in `orders` table; `sync_gtt_statuses()` runs hourly to detect triggered legs

### Known bugs that must be fixed before going live

| Severity | Bug | File / Line |
|---|---|---|
| P0 | `quantity` variable referenced before assignment in capacity check | `trading_engine.py` ~line 307 vs 311 |
| P0 | Market hours check hardcoded `"passed": True` — LIVE trades pass at any time | `risk_manager.py` line 152 |
| P1 | `gtt_manager.py` creates a fresh Angel One session (full TOTP login) on every GTT call — should share `_angel_cache` | `gtt_manager.py` `_get_angel_session()` |
| P1 | BUY order marked `EXECUTED` immediately even though Angel One LIMIT orders may not fill | `trading_engine.py` `execute_signal()` |
| P2 | LIVE mode reads `virtual_balance` for fund check instead of real Angel One margin | `trading_engine.py` line ~273 |

---

## 3. Money / Funding for Live Trades

### Core conclusion
TradeMind is **not a broker and never holds user money**. All real money lives in users' Angel One brokerage accounts. TradeMind only places orders via Angel One's SmartAPI.

### Why individual users can't have TOTP
Angel One SmartAPI (with API key + TOTP secret) is a **developer program**, not a retail feature. Regular Angel One users have a client ID + PIN but no API key or TOTP secret. Building live trading that requires each user to set up their own SmartAPI account is not viable for a consumer product.

### Correct architecture options

| Option | How | Prerequisite |
|---|---|---|
| **Angel One Partner API** | TradeMind registers as an Angel One partner. Users open/link accounts via an OAuth-style flow. TradeMind gets a master API that places orders across all user sub-accounts with one credential set — no per-user TOTP. This is how Smallcase, Sensibull, Streak work. | Angel One partner agreement |
| **Advisory only** | TradeMind generates signals, users execute manually in their broker app | SEBI Research Analyst (RA) license |
| **Paper trading** (current) | Virtual ₹10L balance, full simulation, no real money | None |

### Current `.env` assumption (wrong for production)
All LIVE trades currently use the **founder's personal Angel One credentials** from `.env`. This means all "live" orders execute from one personal brokerage account. Not viable beyond internal testing.

### Recommended product roadmap

| Phase | Deliverable | Regulatory requirement |
|---|---|---|
| Now | Paper trading + AI signals (already built) | None |
| 3–6 months | Signal recommendations with manual execution | SEBI Research Analyst registration |
| 6–12 months | Integrated execution via Angel One Partner API | Angel One partner agreement |
| 12+ months | Fully managed execution | SEBI PMS license |

### Immediate code change needed
Lock LIVE mode behind a feature gate so the platform ships cleanly as a paper + advisory product:
```python
if mode == "LIVE":
    raise HTTPException(403, "Live trading coming soon. Use PAPER mode.")
```
Remove this gate only after the Angel One partnership is in place.

---

## 4. What Was Not Changed

- No schema migrations this session
- No frontend changes
- The P0 bugs in `trading_engine.py` and `risk_manager.py` were identified but **not yet fixed** — flagged for next session
