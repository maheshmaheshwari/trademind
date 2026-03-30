"""
Nifty 500 AI — Intraday Data Collector

Fetches intraday OHLCV candles (5m, 15m, 30m) from Angel One SmartAPI for
stocks with open positions during market hours and stores them in the
local database.

Market hours: Monday–Friday, 09:15–15:30 IST.

Usage:
    cd backend
    python collectors/intraday_collector.py                 # 30m candles, open-position symbols
    python collectors/intraday_collector.py --interval 5m   # 5-minute candles
    python collectors/intraday_collector.py --interval 15m
    python collectors/intraday_collector.py --symbols TCS INFY RELIANCE
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import pyotp
from dotenv import load_dotenv
from SmartApi import SmartConnect

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import (
    get_connection,
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
TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "angel_tokens.json",
)
RATE_LIMIT_SECS = 0.4   # 0.4 s between requests
IST = ZoneInfo("Asia/Kolkata")

# Angel One interval string → DB interval label
INTERVAL_LABEL_MAP = {
    "ONE_MINUTE":      "1m",
    "THREE_MINUTE":    "3m",
    "FIVE_MINUTE":     "5m",
    "TEN_MINUTE":      "10m",
    "FIFTEEN_MINUTE":  "15m",
    "THIRTY_MINUTE":   "30m",
    "ONE_HOUR":        "1h",
    "ONE_DAY":         "1d",
}

# Short label → Angel One interval string
SHORT_TO_ANGEL = {
    "1m":  "ONE_MINUTE",
    "3m":  "THREE_MINUTE",
    "5m":  "FIVE_MINUTE",
    "10m": "TEN_MINUTE",
    "15m": "FIFTEEN_MINUTE",
    "30m": "THIRTY_MINUTE",
    "1h":  "ONE_HOUR",
    "1d":  "ONE_DAY",
}


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
# Market hours guard
# ==========================================

def _is_market_open() -> bool:
    """
    Return True if the current IST time is within NSE market hours:
    Monday–Friday, 09:15–15:30 IST.
    """
    now_ist = datetime.now(tz=IST)
    # Weekday: 0=Monday … 4=Friday
    if now_ist.weekday() >= 5:
        return False
    market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now_ist <= market_close


# ==========================================
# Token map loader
# ==========================================

def _load_token_map() -> dict:
    """Load full Nifty 500 token map from angel_tokens.json."""
    with open(TOKENS_FILE) as f:
        return json.load(f)


# ==========================================
# DB helpers
# ==========================================

def get_intraday_symbols() -> List[str]:
    """
    Return distinct symbols that have open positions in the DB.

    Queries: SELECT DISTINCT symbol FROM positions

    Returns:
        List of raw DB symbols, e.g. ["TCS.NS", "INFY.NS"]
    """
    conn = get_connection()
    try:
        rows = conn.execute("SELECT DISTINCT symbol FROM positions").fetchall()
        return [row[0] for row in rows]
    except Exception as exc:
        logger.error("Failed to query open positions: %s", exc)
        return []
    finally:
        conn.close()


def _strip_ns(symbol: str) -> str:
    """Convert 'TCS.NS' → 'TCS'."""
    return symbol.removesuffix(".NS")


# ==========================================
# Candle parsing
# ==========================================

def _parse_candle(candle, symbol_ns: str, interval_label: str) -> tuple:
    """
    Parse a single Angel One candle into a DB-ready tuple.

    Angel One candle format: [timestamp_str, open, high, low, close, volume]
    Timestamp example: "2024-01-15T09:15:00+0530"

    For intraday candles the time component is preserved.

    Returns:
        (symbol, exchange, date, time, open, high, low, close, volume, interval)
    """
    ts, o, h, l, c, v = candle
    dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
    date_str = dt.strftime("%Y-%m-%d")
    time_str = dt.strftime("%H:%M:%S")
    return (
        symbol_ns,
        "NSE",
        date_str,
        time_str,
        round(float(o), 2),
        round(float(h), 2),
        round(float(l), 2),
        round(float(c), 2),
        int(v),
        interval_label,
    )


# ==========================================
# Core collector
# ==========================================

def collect_intraday(
    symbols: Optional[List[str]] = None,
    interval: str = "THIRTY_MINUTE",
) -> dict:
    """
    Fetch intraday candles for a list of symbols and save to the DB.

    If `symbols` is None, the function queries the DB for all symbols
    that have open positions (via get_intraday_symbols()).

    Market hours are checked before fetching; if the market is not open,
    the function returns early without making any API calls.

    Args:
        symbols:  List of bare symbols like ["TCS", "INFY"] OR ".NS" suffixed
                  symbols like ["TCS.NS"]. If None, uses open-position symbols.
        interval: Angel One interval string, e.g. "THIRTY_MINUTE", "FIVE_MINUTE".
                  Also accepts short labels: "30m", "5m", "15m".

    Returns:
        Summary dict with keys: symbols_attempted, success_count, fail_count,
        total_rows, skipped_market_closed.
    """
    # Normalise interval — accept both short labels and Angel One strings
    if interval in SHORT_TO_ANGEL:
        interval = SHORT_TO_ANGEL[interval]

    interval_label = INTERVAL_LABEL_MAP.get(interval, interval.lower())

    # Market hours guard
    if not _is_market_open():
        now_ist = datetime.now(tz=IST).strftime("%Y-%m-%d %H:%M:%S IST")
        logger.info("Market is closed at %s — skipping intraday collection.", now_ist)
        return {
            "symbols_attempted": 0,
            "success_count": 0,
            "fail_count": 0,
            "total_rows": 0,
            "skipped_market_closed": True,
        }

    init_database()

    # Resolve symbols
    if symbols is None:
        db_symbols = get_intraday_symbols()
        if not db_symbols:
            logger.warning("No open positions found — nothing to collect.")
            return {
                "symbols_attempted": 0,
                "success_count": 0,
                "fail_count": 0,
                "total_rows": 0,
                "skipped_market_closed": False,
            }
        # db_symbols may be "TCS.NS"; strip suffix for token lookup
        bare_symbols = [_strip_ns(s) for s in db_symbols]
    else:
        # Accept both "TCS" and "TCS.NS"
        bare_symbols = [_strip_ns(s) for s in symbols]

    token_map = _load_token_map()

    # Build list of (bare_symbol, token) — skip unknowns
    resolved = []
    for sym in bare_symbols:
        info = token_map.get(sym.upper())
        if info is None:
            logger.warning("No token found for %s — skipping.", sym)
            continue
        resolved.append((sym.upper(), info["token"]))

    if not resolved:
        logger.warning("No resolvable symbols — aborting.")
        return {
            "symbols_attempted": 0,
            "success_count": 0,
            "fail_count": 0,
            "total_rows": 0,
            "skipped_market_closed": False,
        }

    # Date range: today from 09:15 to 15:30 IST
    today = datetime.now(tz=IST).strftime("%Y-%m-%d")
    from_str = f"{today} 09:15"
    to_str = f"{today} 15:30"

    logger.info(
        "Intraday collection — %d symbols, interval=%s (%s), %s to %s",
        len(resolved), interval, interval_label, from_str, to_str,
    )

    smart_api = _angel_login()

    success_count = 0
    fail_count = 0
    total_rows = 0
    failed_symbols: List[str] = []

    for symbol, token in resolved:
        symbol_ns = f"{symbol}.NS"
        params = {
            "exchange": "NSE",
            "symboltoken": token,
            "interval": interval,
            "fromdate": from_str,
            "todate": to_str,
        }

        try:
            data = smart_api.getCandleData(params)
            if not data.get("status") or not data.get("data"):
                logger.debug("No intraday data for %s: %s", symbol, data.get("message"))
                success_count += 1   # Not an error — could be no trades yet
                time.sleep(RATE_LIMIT_SECS)
                continue

            rows = [
                _parse_candle(candle, symbol_ns, interval_label)
                for candle in data["data"]
            ]

            if rows:
                inserted = insert_prices_batch(rows, sync=False)
                total_rows += inserted
                logger.info(
                    "%s — %d intraday candles (%s) saved", symbol_ns, inserted, interval_label
                )

            success_count += 1

        except Exception as exc:
            fail_count += 1
            failed_symbols.append(symbol)
            logger.warning("%s intraday fetch failed: %s", symbol, exc)

        time.sleep(RATE_LIMIT_SECS)

    # Terminate session
    try:
        smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID", ""))
    except Exception:
        pass

    summary = {
        "symbols_attempted": len(resolved),
        "success_count": success_count,
        "fail_count": fail_count,
        "total_rows": total_rows,
        "skipped_market_closed": False,
    }

    if failed_symbols:
        logger.warning("Failed symbols: %s", failed_symbols)

    logger.info(
        "Intraday collection done — %d/%d ok, %d rows",
        success_count, len(resolved), total_rows,
    )
    return summary


# ==========================================
# CLI entry point
# ==========================================

def main():
    parser = argparse.ArgumentParser(
        description="Fetch intraday candles for open-position stocks from Angel One."
    )
    parser.add_argument(
        "--interval",
        default="30m",
        choices=list(SHORT_TO_ANGEL.keys()) + list(INTERVAL_LABEL_MAP.keys()),
        help="Candle interval (default: 30m). Accepts short labels (5m, 15m, 30m) or Angel One strings.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        metavar="SYMBOL",
        help="Override symbol list, e.g. --symbols TCS INFY RELIANCE. "
             "If omitted, symbols are taken from open positions in the DB.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run even outside market hours (useful for testing).",
    )
    args = parser.parse_args()

    if args.force and not _is_market_open():
        print("Market is closed but --force flag set — proceeding anyway.")
        # Temporarily patch the guard so collect_intraday() won't bail early
        import collectors.intraday_collector as _self
        _self._is_market_open = lambda: True

    result = collect_intraday(
        symbols=args.symbols,
        interval=args.interval,
    )

    if result.get("skipped_market_closed"):
        print("Market is closed. Use --force to override.")
        sys.exit(0)

    print(f"\nIntraday collection summary:")
    print(f"  Symbols attempted : {result['symbols_attempted']}")
    print(f"  Successful        : {result['success_count']}")
    print(f"  Failed            : {result['fail_count']}")
    print(f"  Rows inserted     : {result['total_rows']}")


if __name__ == "__main__":
    main()
