"""
Update all Nifty 500 stocks via Angel One SmartAPI.

Reads token mapping from data/angel_tokens.json (499 stocks),
logs in to Angel One, fetches recent EOD candles for each stock,
and upserts into the local database.

Usage:
    cd backend && source venv/bin/activate
    python update_stocks_angel.py            # default: 5 days (per-symbol gap detection)
    python update_stocks_angel.py --days 30  # 30 days history fallback for new stocks
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import date as date_type
from datetime import datetime, timedelta
from typing import Dict, List

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

from database.db import (
    get_connection,
    release_connection,
    get_latest_date,
    init_database,
    insert_prices_batch,
    _execute,
)

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ==========================================
# Config
# ==========================================
TOKENS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "angel_tokens.json")
# Angel One nominally allows ~3 req/sec, but throttles well below that under a
# sustained 500-symbol run. 0.35s sat exactly on the nominal ceiling with no
# headroom, and on 2026-08-06 the EOD collection was refused from the third
# symbol onward — 60 "Access denied because of exceeding access rate" rejections
# between 15:35:05 and 15:43:09, losing 60 of 500 stocks (coverage 440/500, 88%)
# with no restart or competing run to blame. 0.6s halves the request rate and
# costs about 2 extra minutes over the full universe.
RATE_LIMIT_SECS = float(os.environ.get("ANGEL_RATE_LIMIT_SECS", "0.6"))

# A throttled symbol used to be dropped for the day: the loop logged the failure
# and moved on, so one bad window permanently cost 12% of the universe and, with
# the EOD coverage gate in place, blocked the whole signal refresh. Rate-limit
# rejections are transient, so retry them once at the end with a much wider gap.
RETRY_RATE_LIMIT_SECS = float(os.environ.get("ANGEL_RETRY_RATE_LIMIT_SECS", "2.0"))
_RATE_LIMIT_MARKERS = ("exceeding access rate", "access denied", "too many requests")


def _is_rate_limited(err: str) -> bool:
    """True when a fetch failed because Angel One throttled us, not because the
    symbol is bad. Only these are worth retrying — a delisted token will fail
    identically on the second pass and just burn quota."""
    e = (err or "").lower()
    return any(m in e for m in _RATE_LIMIT_MARKERS)


def load_token_map() -> Dict:
    """Load the full Nifty 500 token map, fetching it from the store if absent.

    data/** is excluded from the Space deploy (deploy_space.py IGNORE_PATTERNS)
    and the container's disk is ephemeral, so angel_tokens.json only exists
    there once the startup model-store sync has run. A retry that fires before
    that sync finishes used to die on

        EOD price collection failed:
        [Errno 2] No such file or directory: '/app/data/angel_tokens.json'

    losing that day's prices entirely (scheduler_log, eod_data, 2026-07-25).
    The store is the authoritative source for this file, so fetch it rather
    than fail — `only="data"` pulls a few KB, not the ~5GB model set.
    """
    if not os.path.exists(TOKENS_FILE):
        print(f"⚠️  {TOKENS_FILE} missing — fetching data/ from the model store...")
        try:
            try:
                from scripts.model_store import sync_models      # scheduler imports us as scripts.*
            except ImportError:
                from model_store import sync_models              # run directly: scripts/ is sys.path[0]
            sync_models(only="data")
        except Exception as exc:
            print(f"❌ Could not fetch the token map from the store: {exc}")
        if not os.path.exists(TOKENS_FILE):
            raise FileNotFoundError(
                f"{TOKENS_FILE} not found and could not be fetched from the model "
                f"store. Check HF_TOKEN/MODEL_KEY, or run "
                f"`python scripts/model_store.py sync --data-only`."
            )
        print("✅ Token map recovered from the store")

    with open(TOKENS_FILE) as f:
        return json.load(f)


def angel_login() -> SmartConnect:
    """Login to Angel One SmartAPI and return the SmartConnect client."""
    api_key = os.getenv("ANGEL_API_KEY", "")
    client_id = os.getenv("ANGEL_CLIENT_ID", "")
    mpin = os.getenv("ANGEL_MPIN", "") or os.getenv("ANGEL_PASSWORD", "")
    totp_secret = os.getenv("ANGEL_TOTP_SECRET", "")

    if not all([api_key, client_id, mpin, totp_secret]):
        print("❌ Angel One credentials missing in .env")
        sys.exit(1)

    smart_api = SmartConnect(api_key=api_key)
    totp = pyotp.TOTP(totp_secret).now()

    data = smart_api.generateSession(client_id, mpin, totp)
    if not data.get("status"):
        print(f"❌ Angel One login failed: {data.get('message')}")
        sys.exit(1)

    print(f"✅ Angel One connected — Client: {client_id}")
    return smart_api


def fetch_candles(
    smart_api: SmartConnect,
    symbol: str,
    token: str,
    exchange: str,
    days: int = 5,
    from_date: str = None,
    to_date: str = None,
) -> List[tuple]:
    """
    Fetch daily candles from Angel One for a single stock.

    By default the window is the last `days` days ending now. Pass explicit
    `from_date`/`to_date` ("YYYY-MM-DD HH:MM") to fetch an arbitrary historical
    window instead — scripts/backfill_prices.py uses this to target specific
    trading days rather than "the last N days from now".

    Returns list of DB-ready tuples:
        (symbol_ns, exchange, date, time, open, high, low, close, volume, interval)
    """
    if from_date is None:
        # 00:00 rather than 09:15 — a daily candle is stamped at midnight, so a
        # 09:15 from-time drops the oldest day of the window (verified against
        # the live API, 2026-08-20). Harmless here because the rolling window
        # overlaps day to day, but it silently truncated every historical
        # backfill chunk; see build_chunks in collectors/backfill_prices_angel.py.
        from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00")
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d 15:30")

    params = {
        "exchange": exchange,
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_date,
        "todate": to_date,
    }

    data = smart_api.getCandleData(params)

    if not data.get("status") or not data.get("data"):
        return []

    symbol_ns = f"{symbol}.NS"
    rows = []
    for candle in data["data"]:
        # Angel One format: [timestamp, open, high, low, close, volume]
        ts, o, h, l, c, v = candle
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
        date_str = dt.strftime("%Y-%m-%d")

        rows.append((
            symbol_ns, exchange, date_str, None,
            round(float(o), 2), round(float(h), 2),
            round(float(l), 2), round(float(c), 2),
            int(v), "1d",
        ))

    return rows


def main(days: int = None):
    parser = argparse.ArgumentParser(description="Update stock data via Angel One")
    parser.add_argument("--days", type=int, default=5, help="Days of history to fetch for new stocks (default: 5)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Only fetch these symbols (without .NS)")
    # parse_args([]) ignores sys.argv so uvicorn's --host/--port/--workers
    # don't cause SystemExit(2) when called from the scheduler
    args = parser.parse_args([])
    if days is not None:
        args.days = days

    init_database()

    # Load tokens
    token_map = load_token_map()
    if args.symbols:
        token_map = {k: v for k, v in token_map.items() if k in args.symbols}
    total = len(token_map)
    print(f"\n📊 Nifty 500 stocks in token map: {total}")
    print(f"📅 Fetching candles (smart gap detection, fallback: {args.days} days)...\n")

    # Login
    smart_api = angel_login()

    success = 0
    failed = 0
    total_rows = 0
    failed_symbols = []
    rate_limited = []   # [(symbol, info)] — retried after the main pass

    for idx, (symbol, info) in enumerate(token_map.items(), 1):
        pct = (idx / total) * 100

        # --- Change 1: Smart date detection ---
        ns_symbol = f"{symbol}.NS"
        latest_str = get_latest_date(ns_symbol)
        if latest_str:
            latest = datetime.strptime(latest_str, "%Y-%m-%d").date()
            days_missing = (date_type.today() - latest).days + 1
            if days_missing <= 0:
                success += 1
                continue  # already up to date
        else:
            days_missing = args.days  # fallback for new stocks

        # --- Change 2: Fetch with session reconnect on token/session errors ---
        try:
            rows = fetch_candles(smart_api, symbol=symbol, token=info["token"], exchange="NSE", days=days_missing)
        except Exception as e:
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["token", "session", "invalid", "unauthorized"]):
                logger.warning(f"Session expired, reconnecting...")
                try:
                    smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
                except Exception:
                    pass
                smart_api = angel_login()
                try:
                    rows = fetch_candles(smart_api, symbol=symbol, token=info["token"], exchange="NSE", days=days_missing)
                except Exception as retry_e:
                    failed += 1
                    failed_symbols.append(symbol)
                    logger.error(f"[{idx}/{total}] {symbol} failed after reconnect: {retry_e}")
                    time.sleep(RATE_LIMIT_SECS)
                    continue
            else:
                failed += 1
                failed_symbols.append(symbol)
                # Keep throttled symbols separately — they are retryable, and a
                # dropped symbol costs a full day of prices for that stock.
                if _is_rate_limited(str(e)):
                    # Carry this symbol's own gap. It used to be re-read from
                    # `days_missing` at retry time, which by then held whatever
                    # the *last* symbol in the main loop needed — so a retried
                    # symbol fetched an arbitrary window.
                    rate_limited.append((symbol, info, days_missing))
                logger.warning(f"[{idx}/{total}] {symbol} FAILED: {e}")
                time.sleep(RATE_LIMIT_SECS)
                continue

        if rows:
            inserted = insert_prices_batch(rows, sync=False)
            total_rows += inserted
            success += 1
            if inserted > 0:
                logger.info(f"[{idx}/{total}] {symbol:15s} +{inserted} rows")
        else:
            # No data returned (could be a holiday or very recent listing)
            success += 1

        # Progress every 50 stocks
        if idx % 50 == 0:
            print(f"  ⏳ Progress: {idx}/{total} ({pct:.0f}%) — {total_rows} new rows so far")

        time.sleep(RATE_LIMIT_SECS)

    # ── Retry pass: symbols Angel One throttled ──────────────────────────────
    #
    # Only rate-limited failures, and only once. A throttled symbol is a
    # transient loss; leaving it dropped costs that stock a full day of prices,
    # and enough of them push EOD coverage under the gate and block the entire
    # signal refresh (2026-08-06: 60 throttled, coverage 440/500, signals held
    # back). Retried at RETRY_RATE_LIMIT_SECS — much slower than the main pass,
    # since being throttled is precisely the evidence we were going too fast.
    if rate_limited:
        wait = 30
        print(f"\n   🔁 {len(rate_limited)} symbol(s) were rate-limited — "
              f"pausing {wait}s, then retrying at {RETRY_RATE_LIMIT_SECS}s/request "
              f"(~{len(rate_limited) * RETRY_RATE_LIMIT_SECS / 60:.1f} min)")
        time.sleep(wait)

        recovered = 0
        for r_idx, (symbol, info, sym_days) in enumerate(rate_limited, 1):
            try:
                rows = fetch_candles(smart_api, symbol=symbol, token=info["token"],
                                     exchange="NSE", days=sym_days)
                if rows:
                    inserted = insert_prices_batch(rows, sync=False)
                    total_rows += inserted
                if symbol in failed_symbols:
                    failed_symbols.remove(symbol)
                failed -= 1
                success += 1
                recovered += 1
                logger.info(f"   [retry {r_idx}/{len(rate_limited)}] {symbol:15s} recovered")
            except Exception as e:
                logger.warning(f"   [retry {r_idx}/{len(rate_limited)}] {symbol} still failing: {e}")
            time.sleep(RETRY_RATE_LIMIT_SECS)

        print(f"   🔁 Retry recovered {recovered}/{len(rate_limited)} symbol(s)")

    # Logout
    try:
        smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
    except Exception:
        pass

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Angel One EOD update complete!")
    print(f"   Stocks processed: {success + failed}/{total}")
    print(f"   Successful:       {success}")
    print(f"   Failed:           {failed}")
    print(f"   New rows:         {total_rows}")

    if failed_symbols:
        print(f"\n   ⚠️  Failed symbols ({len(failed_symbols)}):")
        for s in failed_symbols[:20]:
            print(f"      - {s}")
        if len(failed_symbols) > 20:
            print(f"      ... and {len(failed_symbols) - 20} more")

    # Verify final state.
    #
    # Scoped to the last 7 days on purpose. This was
    #   SELECT MAX(date), COUNT(DISTINCT symbol) FROM prices WHERE interval='1d'
    # which is a full scan of a 202-chunk compressed hypertable — ~460k rows
    # decompressed to print one cosmetic line — and it OOM'd the free-tier
    # instance outright:
    #   EOD price collection failed: out of memory
    #   DETAIL: Failed while creating memory context "ExprContext"
    # (scheduler_log, eod_data, 2026-08-03), aborting the whole EOD chain after
    # the prices had already been written.
    #
    # A date-bounded predicate touches one or two chunks instead of all 202, and
    # "how many symbols reported in the last week" is the more useful number
    # here anyway — the all-time distinct count includes de-indexed names that
    # stopped reporting months ago.
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT MAX(date), COUNT(DISTINCT symbol)
              FROM prices
             WHERE interval = '1d' AND date >= CURRENT_DATE - INTERVAL '7 days'
        """)
        final = cur.fetchone()
        print(f"\n📊 DB state: {final[1]} symbols reporting in the last 7d, "
              f"latest date: {final[0]}")
    except Exception as exc:
        # Never let a summary line abort the run — the prices are already in.
        print(f"\n📊 DB state: summary query failed ({exc}) — collection itself succeeded")
    finally:
        release_connection(conn)


if __name__ == "__main__":
    main()
