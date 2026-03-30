# 02 — Angel One Data Collection

## What Already Exists

| File | Purpose | Status |
|------|---------|--------|
| `collectors/angel_collector.py` | `AngelCollector` class — login, LTP, historical candles | ✅ Works |
| `collectors/ltp_fetcher.py` | Lightweight live LTP batch fetch | ✅ Works |
| `update_stocks_angel.py` | EOD batch update for all 499 stocks | ✅ Works, needs hardening |
| `data/angel_tokens.json` | Token map: 499 stocks → SmartAPI token | ✅ Complete |
| `map_tokens.py` | Regenerates token map from Angel One scrip master | ✅ Works |

**Current data gap:** 141 of 499 stocks have no price data in DB.
**Price data range:** 2021-02-24 → 2026-03-04 (358 stocks only).

---

## Angel One SmartAPI — Constraints

### Authentication
- Requires TOTP — regenerates every 30 sec; must login fresh on each script run
- Session valid for 24 hours; scheduler must re-login daily before market open
- One active session per API key — concurrent logins invalidate each other

### getCandleData — per-request limits

| Interval | Max days per request |
|----------|---------------------|
| ONE_DAY | 400 days |
| ONE_HOUR | 30 days |
| THIRTY_MINUTE | 30 days |
| FIFTEEN_MINUTE | 20 days |
| FIVE_MINUTE | 5 days |
| ONE_MINUTE | 1 day |

Angel One silently truncates data beyond these limits — always chunk requests.

### Rate limit
- Safe sustained rate: **1 request per 0.4 seconds** (~2.5 req/sec)
- On `AB1010` (rate limit error): back off 5 seconds, then retry
- Index tokens (NIFTY, SENSEX, VIX) share the same rate limit pool

### Angel One Index Tokens

| Index | Token | Exchange |
|-------|-------|----------|
| NIFTY 50 | `99926000` | NSE |
| NIFTY 500 | `99926004` | NSE |
| SENSEX | `99919000` | BSE |
| INDIA VIX | `99919101` | NSE |

---

## Phase 1 — 5-Year Historical Bootstrap

**File to create:** `backend/collectors/historical_bootstrap.py`

### Goal
Fetch 5 years of daily OHLCV (2021-01-01 → today) for all 499 stocks and
store in `prices` table (`interval='1d'`). Run once; resumable with `--only-missing`.

### Chunking strategy
400-day limit → 5 chunks of 390 days (buffer for safety):

```
Chunk 1: 2021-01-01 → 2022-02-25
Chunk 2: 2022-02-26 → 2023-04-22
Chunk 3: 2023-04-23 → 2024-06-16
Chunk 4: 2024-06-17 → 2025-08-11
Chunk 5: 2025-08-12 → today
```

### Time estimate
```
499 stocks × 5 chunks × 0.4s = ~998 seconds ≈ 17 minutes
```

### Implementation outline

```python
# backend/collectors/historical_bootstrap.py

START_DATE  = "2021-01-01"
CHUNK_DAYS  = 390
RATE_SLEEP  = 0.4

def date_chunks(from_date: str, to_date: str, chunk_days: int):
    """Yield (from_str, to_str) pairs in chunk_days increments."""
    cursor = datetime.strptime(from_date, "%Y-%m-%d")
    end    = datetime.strptime(to_date,   "%Y-%m-%d")
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        yield (
            cursor.strftime("%Y-%m-%d 09:15"),
            chunk_end.strftime("%Y-%m-%d 15:30"),
        )
        cursor = chunk_end + timedelta(days=1)


def bootstrap_stock(smart_api, symbol, token, from_date, to_date):
    all_rows = []
    for chunk_from, chunk_to in date_chunks(from_date, to_date, CHUNK_DAYS):
        params = {
            "exchange":    "NSE",
            "symboltoken": token,
            "interval":    "ONE_DAY",
            "fromdate":    chunk_from,
            "todate":      chunk_to,
        }
        data = with_retry(lambda: smart_api.getCandleData(params))
        if data and data.get("data"):
            all_rows.extend(parse_candles(data["data"], symbol))
        time.sleep(RATE_SLEEP)
    insert_prices_batch(all_rows)
    return len(all_rows)


def main(only_missing=False, from_date=START_DATE):
    token_map  = load_token_map()           # 499 stocks
    smart_api  = angel_login()

    if only_missing:
        in_db = get_symbols_in_db()         # SELECT DISTINCT symbol FROM prices
        token_map = {s: v for s, v in token_map.items()
                     if f"{s}.NS" not in in_db}
        print(f"Bootstrapping {len(token_map)} missing stocks")

    for idx, (symbol, info) in enumerate(token_map.items(), 1):
        try:
            count = bootstrap_stock(smart_api, symbol, info["token"],
                                    from_date, today())
            log(f"[{idx}/{len(token_map)}] {symbol}: {count} candles")
        except SessionExpiredError:
            smart_api = angel_login()       # reconnect, retry once
            count = bootstrap_stock(smart_api, symbol, info["token"],
                                    from_date, today())
        except Exception as e:
            log_error(symbol, e)            # skip and continue


# CLI
# python historical_bootstrap.py                    → full 499 stocks, 5 years
# python historical_bootstrap.py --only-missing     → skip already-in-DB stocks
# python historical_bootstrap.py --from 2025-01-01  → partial fill
```

---

## Phase 2 — Daily EOD Update (harden existing)

**File:** `backend/update_stocks_angel.py` (already exists)

### Problems to fix

**1. Smart date detection** — fetch only missing days per symbol:
```python
# Current (bad): always fetches fixed N days regardless
rows = fetch_candles(symbol, days=args.days)

# Fixed: detect latest date per symbol, fetch only gap
latest = get_latest_date(symbol)          # MAX(date) FROM prices WHERE symbol=?
days_missing = (date.today() - latest).days + 1
if days_missing > 0:
    rows = fetch_candles(symbol, days=days_missing)
```

**2. Session refresh on mid-run expiry:**
```python
try:
    rows = fetch_candles(smart_api, symbol, ...)
except Exception as e:
    if is_session_error(e):               # checks for "token"/"session"/"invalid"
        smart_api = angel_login()
        rows = fetch_candles(smart_api, symbol, ...)   # retry once
    else:
        failed.append(symbol)
        continue
```

**3. Scheduler hook** — add to `scheduler/jobs.py`:
```python
@scheduler.scheduled_job('cron', day_of_week='mon-fri',
                          hour=15, minute=35, timezone='Asia/Kolkata')
def job_eod_update():
    """Run after NSE market close at 15:30 IST."""
    run_eod(days=2)    # today + yesterday for safety
```

---

## Phase 3 — Intraday Candles

**File to create:** `backend/collectors/intraday_collector.py`

Collects 30-min and 5-min candles during market hours for:
- Stocks with open positions
- User watchlist

Not all 499 stocks — intraday data is large and mostly unused for stocks outside positions.

### Schedule

```python
# scheduler/jobs.py additions

@scheduler.scheduled_job('cron', day_of_week='mon-fri',
                          hour='9-15', minute='*/30', timezone='Asia/Kolkata')
def job_intraday_30m():
    """Every 30 min during market hours (9:15–15:30 IST)."""
    symbols = get_open_position_symbols()   # SELECT symbol FROM positions
    collect_intraday(symbols, interval='THIRTY_MINUTE')
```

### Angel One intraday limits recap

```python
INTRADAY_LIMITS = {
    'ONE_MINUTE':      1,   # days back per request
    'FIVE_MINUTE':     5,
    'FIFTEEN_MINUTE': 20,
    'THIRTY_MINUTE':  30,
    'ONE_HOUR':       30,
}
```

### Implementation outline

```python
# backend/collectors/intraday_collector.py

MARKET_OPEN  = time(9, 15)
MARKET_CLOSE = time(15, 30)
IST          = timezone('Asia/Kolkata')

def collect_intraday(symbols: list, interval: str = 'THIRTY_MINUTE') -> int:
    if not is_market_hours():
        return 0

    smart_api = angel_login()
    today_str = date.today().strftime("%Y-%m-%d")
    total_saved = 0

    for symbol in symbols:
        token_info = _TOKEN_MAP.get(symbol.replace('.NS', ''))
        if not token_info:
            continue

        params = {
            'exchange':    'NSE',
            'symboltoken': token_info['token'],
            'interval':    interval,
            'fromdate':    f"{today_str} 09:15",
            'todate':      datetime.now(IST).strftime("%Y-%m-%d %H:%M"),
        }
        data = smart_api.getCandleData(params)
        rows = parse_candles(data['data'], symbol, interval_label(interval))
        total_saved += insert_prices_batch(rows)
        time.sleep(0.4)

    return total_saved
```

---

## Phase 4 — Index & Market Overview

**File to create:** `backend/collectors/index_collector.py`

Fetches daily OHLCV for NIFTY50, NIFTY500, SENSEX, INDIA VIX from Angel One
and populates the `market_overview` table (currently has only 1 row).

### Bootstrap (5 years of index data)
```
4 indices × 5 chunks × 0.4s = ~8 seconds
```

### Market breadth (advances/declines)
Angel One does not provide advances/declines counts.
Source: NSE India unofficial API:
```
GET https://www.nseindia.com/api/allIndices
Headers: User-Agent, Referer: https://www.nseindia.com
Parse:   entry where index == "NIFTY 500" → advances, declines, unchanged
Note:    This is a web scraper — fragile, may break on NSE changes.
```

### Implementation outline

```python
# backend/collectors/index_collector.py

INDEX_TOKENS = {
    'NIFTY50':  {'token': '99926000', 'exchange': 'NSE'},
    'NIFTY500': {'token': '99926004', 'exchange': 'NSE'},
    'SENSEX':   {'token': '99919000', 'exchange': 'BSE'},
    'INDIAVIX': {'token': '99919101', 'exchange': 'NSE'},
}

def collect_index_history(from_date='2021-01-01'):
    smart_api = angel_login()
    combined = {}                 # date → {nifty50, nifty500, sensex, vix}

    for name, info in INDEX_TOKENS.items():
        for chunk_from, chunk_to in date_chunks(from_date, today(), 390):
            data = smart_api.getCandleData({
                'exchange':    info['exchange'],
                'symboltoken': info['token'],
                'interval':    'ONE_DAY',
                'fromdate':    chunk_from,
                'todate':      chunk_to,
            })
            for candle in data['data']:
                ts, o, h, l, c, v = candle
                d = ts[:10]                 # "2024-01-15"
                combined.setdefault(d, {})[name] = c
            time.sleep(0.4)

    # Write to market_overview
    rows = []
    for date_str, vals in combined.items():
        rows.append((
            date_str,
            vals.get('NIFTY500'), None,     # nifty500_close, change_pct
            vals.get('NIFTY50'),  None,
            vals.get('SENSEX'),
            vals.get('INDIAVIX'),
            None, None, None,               # advances, declines, unchanged
            None, None, None,               # volume, fii_net, dii_net
            None, None,                     # sentiment_score, fear_greed_label
        ))
    upsert_market_overview(rows)


def collect_daily_index():
    """Called by scheduler after market close."""
    collect_index_history(from_date=yesterday())
```

---

## Error Handling

### Common Angel One errors

| Error / Message | Cause | Handling |
|-----------------|-------|---------|
| `"Invalid Token"` | Session expired | Re-login, retry once |
| `"AB1010"` | Rate limit hit | `sleep(5 × attempt)`, retry up to 3× |
| `data['data'] == []` | Holiday / stock suspended / wrong token | Skip, log warning |
| `"No data available"` | Future date or delisted stock | Skip |
| Connection timeout | Network issue | Retry 3× with exponential backoff |

### Retry wrapper

```python
import time

class SessionExpiredError(Exception): pass
class RateLimitError(Exception): pass

def with_retry(fn, max_retries=3, backoff=2.0):
    for attempt in range(max_retries):
        try:
            result = fn()
            if isinstance(result, dict):
                msg = result.get('message', '').lower()
                if 'ab1010' in msg or 'rate' in msg:
                    raise RateLimitError(msg)
                if 'invalid token' in msg or 'session' in msg:
                    raise SessionExpiredError(msg)
            return result
        except RateLimitError:
            time.sleep(5 * (attempt + 1))
        except SessionExpiredError:
            raise                   # let caller handle re-login
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(backoff * (attempt + 1))
    return None
```

---

## Token Map Maintenance

`data/angel_tokens.json` must be refreshed after each NSE index rebalancing
(quarterly: March, June, September, December).

```bash
# Run after NSE announces Nifty 500 constituent changes
cd backend && source venv/bin/activate
python map_tokens.py
# Downloads OpenAPIScripMaster.json (~50MB) from Angel One
# Filters NSE equity instruments, maps to our symbol format
# Overwrites data/angel_tokens.json
```

---

## Environment Variables

```env
ANGEL_API_KEY=your_smartapi_key          # smartapi.angelone.in dashboard
ANGEL_SECRET_KEY=your_secret_key
ANGEL_CLIENT_ID=M123456                  # your Angel One client ID
ANGEL_PASSWORD=your_angel_one_password
ANGEL_TOTP_SECRET=BASE32TOTPSECRET       # 32-char key from Google Auth setup
```

### How to get ANGEL_TOTP_SECRET
1. Angel One mobile app → Profile → SmartAPI → Enable TOTP
2. When QR code appears, tap "Can't scan? Enter manually"
3. Copy the 32-character base32 string → that is `ANGEL_TOTP_SECRET`

---

## Summary: Files to Create

| File | Phase | Est. effort |
|------|-------|-------------|
| `collectors/historical_bootstrap.py` | Phase 1 | Medium |
| `collectors/intraday_collector.py` | Phase 3 | Small |
| `collectors/index_collector.py` | Phase 4 | Small |
| Harden `update_stocks_angel.py` | Phase 2 | Small |
| Add jobs to `scheduler/jobs.py` | Phase 2+3+4 | Small |
