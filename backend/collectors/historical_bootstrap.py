"""
Nifty 500 AI — Historical Data Bootstrap

Fetches 5 years of daily OHLCV candles for all 499 Nifty 500 stocks from
Angel One SmartAPI and stores them in the local database.

Angel One ONE_DAY candles are limited to 400 days per request.  We chunk the
5-year window into 390-day slices to stay safely under that limit.

Usage:
    cd backend
    python collectors/historical_bootstrap.py                 # all stocks, from 2021-01-01
    python collectors/historical_bootstrap.py --only-missing  # skip symbols already in DB
    python collectors/historical_bootstrap.py --from-date 2022-06-01
    python collectors/historical_bootstrap.py --symbol TCS
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Iterator, List, Optional, Tuple

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import (
    get_latest_date,
    init_database,
    insert_prices_batch,
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
RATE_LIMIT_SECS = 0.4        # 0.4 s between requests (Angel One limit ~3 req/s)
CHUNK_DAYS = 390             # Max days per Angel One ONE_DAY request (limit is 400)
DEFAULT_FROM_DATE = "2021-01-01"
PROGRESS_EVERY = 25          # Log progress every N stocks


# ==========================================
# Angel One login
# ==========================================

def _angel_login() -> SmartConnect:
    """Login to Angel One SmartAPI and return the SmartConnect client."""
    smart_api = SmartConnect(api_key=os.getenv("ANGEL_API_KEY", ""))
    totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET", "")).now()
    data = smart_api.generateSession(
        os.getenv("ANGEL_CLIENT_ID", ""),
        os.getenv("ANGEL_PASSWORD", ""),
        totp,
    )
    if not data.get("status"):
        raise RuntimeError(f"Login failed: {data.get('message')}")
    logger.info("Angel One login successful")
    return smart_api


# ==========================================
# Candle parsing
# ==========================================

def _parse_candle(candle, symbol_ns: str, interval_label: str = "1d") -> tuple:
    """
    Parse a single Angel One candle into a DB-ready tuple.

    Angel One candle format: [timestamp_str, open, high, low, close, volume]
    Timestamp example: "2024-01-15T09:15:00+0530"

    Returns:
        (symbol, exchange, date, time, open, high, low, close, volume, interval)
    """
    ts, o, h, l, c, v = candle
    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
    return (
        symbol_ns,
        "NSE",
        dt.strftime("%Y-%m-%d"),
        None,
        round(float(o), 2),
        round(float(h), 2),
        round(float(l), 2),
        round(float(c), 2),
        int(v),
        interval_label,
    )


# ==========================================
# Date chunking
# ==========================================

def date_chunks(from_date_str: str, to_date_str: str) -> Iterator[Tuple[str, str]]:
    """
    Yield (from_str, to_str) pairs in CHUNK_DAYS-day increments.

    Dates are formatted as "YYYY-MM-DD 09:15" (from) and "YYYY-MM-DD 15:30" (to)
    to match the Angel One getCandleData API expectation.

    Args:
        from_date_str: Start date as "YYYY-MM-DD"
        to_date_str:   End date as "YYYY-MM-DD"
    """
    current = datetime.strptime(from_date_str, "%Y-%m-%d")
    end = datetime.strptime(to_date_str, "%Y-%m-%d")

    while current <= end:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end)
        yield (
            current.strftime("%Y-%m-%d 09:15"),
            chunk_end.strftime("%Y-%m-%d 15:30"),
        )
        current = chunk_end + timedelta(days=1)


# ==========================================
# Fetch helpers
# ==========================================

def _fetch_chunk(
    smart_api: SmartConnect,
    token: str,
    from_str: str,
    to_str: str,
) -> Optional[list]:
    """
    Fetch ONE_DAY candles for a single date chunk.

    Returns the raw candle list from Angel One, or None on error.
    """
    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": from_str,
        "todate": to_str,
    }
    data = smart_api.getCandleData(params)
    if not data.get("status") or not data.get("data"):
        return None
    return data["data"]


def _fetch_with_retry(
    smart_api: SmartConnect,
    token: str,
    from_str: str,
    to_str: str,
    max_retries: int = 3,
) -> Optional[list]:
    """
    Fetch candles with retry on rate-limit errors (Angel One error code AB1010).

    Backs off 2 s on the first retry, 5 s on subsequent retries.
    Re-raises on non-rate-limit exceptions after exhausting retries.
    """
    backoff = [2, 5, 10]
    for attempt in range(max_retries):
        try:
            return _fetch_chunk(smart_api, token, from_str, to_str)
        except Exception as exc:
            err_str = str(exc)
            is_rate_limit = "AB1010" in err_str or "rate limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = backoff[attempt]
                logger.warning(
                    "Rate limit hit (AB1010) — waiting %ds before retry %d/%d",
                    wait, attempt + 1, max_retries - 1,
                )
                time.sleep(wait)
            else:
                raise
    return None


# ==========================================
# Session reconnect
# ==========================================

def _reconnect() -> SmartConnect:
    """Re-authenticate and return a fresh SmartConnect client."""
    logger.info("Re-establishing Angel One session...")
    try:
        return _angel_login()
    except Exception as exc:
        logger.error("Session reconnect failed: %s", exc)
        raise


# ==========================================
# Per-symbol bootstrap
# ==========================================

def bootstrap_symbol(
    smart_api: SmartConnect,
    symbol: str,
    token: str,
    from_date: str,
    to_date: str,
) -> Tuple[SmartConnect, int]:
    """
    Fetch all historical daily candles for a single symbol across date chunks.

    Handles session expiry by reconnecting once.

    Returns:
        (smart_api, rows_inserted)  — smart_api may be a fresh instance after reconnect.
    """
    symbol_ns = f"{symbol}.NS"
    all_rows: List[tuple] = []

    for from_str, to_str in date_chunks(from_date, to_date):
        try:
            candles = _fetch_with_retry(smart_api, token, from_str, to_str)
        except Exception as exc:
            err_str = str(exc)
            # Detect session expiry errors and reconnect once
            if any(kw in err_str for kw in ("Invalid Token", "Session Expired", "AG8001", "INVALID_TOKEN")):
                logger.warning("Session expired — reconnecting...")
                smart_api = _reconnect()
                try:
                    candles = _fetch_with_retry(smart_api, token, from_str, to_str)
                except Exception as inner_exc:
                    logger.error("Chunk %s→%s for %s failed after reconnect: %s", from_str, to_str, symbol, inner_exc)
                    candles = None
            else:
                logger.warning("Chunk %s→%s for %s failed: %s", from_str, to_str, symbol, exc)
                candles = None

        if candles:
            for candle in candles:
                all_rows.append(_parse_candle(candle, symbol_ns))

        time.sleep(RATE_LIMIT_SECS)

    inserted = 0
    if all_rows:
        inserted = insert_prices_batch(all_rows, sync=False)
        logger.debug("%s — inserted %d rows", symbol_ns, inserted)

    return smart_api, inserted


# ==========================================
# Token map loader
# ==========================================

def _load_token_map():
    """Load full Nifty 500 token map from angel_tokens.json."""
    with open(TOKENS_FILE) as f:
        return json.load(f)


# ==========================================
# Main
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap 5 years of daily OHLCV for all Nifty 500 stocks from Angel One."
    )
    parser.add_argument(
        "--from-date",
        default=DEFAULT_FROM_DATE,
        help=f"Start date for historical fetch (default: {DEFAULT_FROM_DATE}). Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip symbols that already have data in the DB.",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Bootstrap a single symbol, e.g. --symbol TCS",
    )
    args = parser.parse_args()

    init_database()

    token_map = _load_token_map()

    # Single-symbol mode
    if args.symbol:
        sym = args.symbol.upper()
        if sym not in token_map:
            print(f"Symbol {sym} not found in angel_tokens.json")
            sys.exit(1)
        token_map = {sym: token_map[sym]}

    to_date = datetime.now().strftime("%Y-%m-%d")
    total = len(token_map)

    print(f"\nHistorical Bootstrap — {total} symbol(s), {args.from_date} to {to_date}")
    if args.only_missing:
        print("Mode: --only-missing (skipping symbols already in DB)")
    print()

    smart_api = _angel_login()

    success_count = 0
    skip_count = 0
    fail_count = 0
    total_rows = 0
    failed_symbols: List[str] = []

    for idx, (symbol, info) in enumerate(token_map.items(), 1):
        # --only-missing check
        if args.only_missing:
            latest = get_latest_date(f"{symbol}.NS")
            if latest is not None:
                skip_count += 1
                logger.debug("Skipping %s — already has data up to %s", symbol, latest)
                continue

        try:
            smart_api, inserted = bootstrap_symbol(
                smart_api=smart_api,
                symbol=symbol,
                token=info["token"],
                from_date=args.from_date,
                to_date=to_date,
            )
            success_count += 1
            total_rows += inserted

        except Exception as exc:
            fail_count += 1
            failed_symbols.append(symbol)
            logger.warning("[%d/%d] %s FAILED: %s", idx, total, symbol, exc)

        # Progress log every PROGRESS_EVERY stocks
        if idx % PROGRESS_EVERY == 0:
            pct = (idx / total) * 100
            print(
                f"  Progress: {idx}/{total} ({pct:.0f}%) "
                f"| success={success_count} skip={skip_count} fail={fail_count} "
                f"| rows={total_rows}"
            )

    # Terminate session
    try:
        smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
    except Exception:
        pass

    # Final summary
    print(f"\n{'='*65}")
    print("Historical Bootstrap complete!")
    print(f"  Symbols processed : {success_count + fail_count} / {total}")
    print(f"  Successful        : {success_count}")
    print(f"  Skipped           : {skip_count}")
    print(f"  Failed            : {fail_count}")
    print(f"  Total rows saved  : {total_rows}")

    if failed_symbols:
        print(f"\n  Failed symbols ({len(failed_symbols)}):")
        for s in failed_symbols[:20]:
            print(f"    - {s}")
        if len(failed_symbols) > 20:
            print(f"    ... and {len(failed_symbols) - 20} more")


if __name__ == "__main__":
    main()
