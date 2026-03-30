"""
TradeMind AI — Index Collector

Fetches daily OHLCV for NIFTY50, NIFTY500, SENSEX, and INDIA VIX from
Angel One SmartAPI and populates the market_overview table.

Functions:
    collect_index_history(from_date)  — 5-year bootstrap for all 4 indices
    collect_index_daily()             — fetch yesterday + today (for scheduler)
    try_get_market_breadth()          — advances/declines from NSE India API

CLI:
    python index_collector.py             # daily update
    python index_collector.py --history   # full bootstrap from 2021-01-01
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pyotp
import requests
from dotenv import load_dotenv
from SmartApi import SmartConnect

# ---- path bootstrap so this file can be run directly ----
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import insert_market_overview, init_database

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index token map
# ---------------------------------------------------------------------------
INDEX_TOKENS = {
    "NIFTY50":  {"token": "99926000", "exchange": "NSE", "db_col": "nifty50"},
    "NIFTY500": {"token": "99926004", "exchange": "NSE", "db_col": "nifty500"},
    "SENSEX":   {"token": "99919000", "exchange": "BSE", "db_col": "sensex"},
    "INDIAVIX": {"token": "99919101", "exchange": "NSE", "db_col": "india_vix"},
}

# Max days Angel One allows per candle request
_CHUNK_DAYS = 390
# Rate-limit guard between API calls
_RATE_SLEEP = 0.4


# ---------------------------------------------------------------------------
# Angel One login helper
# ---------------------------------------------------------------------------
def _angel_login() -> SmartConnect:
    """Login to Angel One SmartAPI and return an authenticated client."""
    smart_api = SmartConnect(api_key=os.getenv("ANGEL_API_KEY", ""))
    totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET", "")).now()
    data = smart_api.generateSession(
        os.getenv("ANGEL_CLIENT_ID", ""),
        os.getenv("ANGEL_PASSWORD", ""),
        totp,
    )
    if not data.get("status"):
        raise RuntimeError(f"Angel One login failed: {data.get('message')}")
    logger.info("Angel One login successful")
    # Allow Angel One rate-limit window to reset after login
    time.sleep(3)
    return smart_api


# ---------------------------------------------------------------------------
# Low-level candle fetch for a single index + date range
# ---------------------------------------------------------------------------
def _fetch_candles(
    smart_api: SmartConnect,
    name: str,
    from_dt: datetime,
    to_dt: datetime,
) -> List[Dict[str, Any]]:
    """
    Fetch ONE_DAY candles for a single index between from_dt and to_dt.

    Returns a list of dicts:
        {"date": "2024-01-15", "close": 21500.0, "volume": 0}
    """
    token_info = INDEX_TOKENS[name]
    params = {
        "exchange": token_info["exchange"],
        "symboltoken": token_info["token"],
        "interval": "ONE_DAY",
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M"),
    }

    try:
        resp = smart_api.getCandleData(params)
    except Exception as exc:
        logger.warning(f"[{name}] getCandleData error: {exc}")
        return []

    if not resp.get("status") or not resp.get("data"):
        logger.warning(f"[{name}] no data for {from_dt.date()} – {to_dt.date()}: {resp.get('message')}")
        return []

    results = []
    for candle in resp["data"]:
        # candle format: [timestamp, open, high, low, close, volume]
        ts, _o, _h, _l, close, volume = candle
        # timestamp: "2024-01-15T09:15:00+0530"
        try:
            dt = datetime.strptime(ts[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            dt = datetime.strptime(ts[:10], "%Y-%m-%d")
        results.append({
            "date": dt.strftime("%Y-%m-%d"),
            "close": round(float(close), 2),
            "volume": int(volume),
        })

    return results


# ---------------------------------------------------------------------------
# yfinance fallback for INDIA VIX (Angel One returns no data for VIX token)
# ---------------------------------------------------------------------------
def _fetch_vix_yfinance(start_dt: datetime, end_dt: datetime) -> List[Dict[str, Any]]:
    """Fetch INDIA VIX history from Yahoo Finance (^INDIAVIX)."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("^INDIAVIX")
        df = ticker.history(
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
        )
        if df.empty:
            logger.warning("yfinance returned no data for ^INDIAVIX")
            return []
        results = []
        for ts, row in df.iterrows():
            results.append({
                "date":   ts.strftime("%Y-%m-%d"),
                "close":  round(float(row["Close"]), 4),
                "volume": 0,
            })
        logger.info(f"VIX yfinance: {len(results)} rows")
        return results
    except Exception as e:
        logger.warning(f"VIX yfinance fallback failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Merge helper: build per-date dicts from all 4 index series
# ---------------------------------------------------------------------------
def _merge_series(all_series: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Merge 4 index series (keyed by INDEX_TOKENS name) into per-date rows.

    Each output row has keys compatible with insert_market_overview():
        date, nifty50_close, nifty50_change_pct,
        nifty500_close, nifty500_change_pct,
        sensex_close, india_vix, total_volume
    """
    # Group each series by date
    by_date: Dict[str, Dict[str, Any]] = {}

    for idx_name, candles in all_series.items():
        col = INDEX_TOKENS[idx_name]["db_col"]
        prev_close: Optional[float] = None

        for c in sorted(candles, key=lambda x: x["date"]):
            date = c["date"]
            close = c["close"]

            if date not in by_date:
                by_date[date] = {"date": date}

            if col == "nifty50":
                by_date[date]["nifty50_close"] = close
                if prev_close:
                    by_date[date]["nifty50_change_pct"] = round(
                        (close - prev_close) / prev_close * 100, 4
                    )
            elif col == "nifty500":
                by_date[date]["nifty500_close"] = close
                if prev_close:
                    by_date[date]["nifty500_change_pct"] = round(
                        (close - prev_close) / prev_close * 100, 4
                    )
                # Use NIFTY500 volume as proxy for total_volume
                by_date[date]["total_volume"] = c["volume"]
            elif col == "sensex":
                by_date[date]["sensex_close"] = close
            elif col == "india_vix":
                by_date[date]["india_vix"] = close

            prev_close = close

    return sorted(by_date.values(), key=lambda r: r["date"])


# ---------------------------------------------------------------------------
# Public function: try_get_market_breadth
# ---------------------------------------------------------------------------
def try_get_market_breadth() -> Dict[str, int]:
    """
    Attempt to fetch NIFTY 500 advances/declines/unchanged from NSE India API.

    Returns a dict with keys {advances, declines, unchanged} on success,
    or an empty dict on any error (the endpoint is fragile — skip silently).
    """
    url = "https://www.nseindia.com/api/allIndices"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Referer": "https://www.nseindia.com",
        "Accept": "application/json",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        resp.raise_for_status()
        payload = resp.json()

        for entry in payload.get("data", []):
            if entry.get("index") == "NIFTY 500":
                return {
                    "advances":  int(entry.get("advances", 0)),
                    "declines":  int(entry.get("declines", 0)),
                    "unchanged": int(entry.get("unchanged", 0)),
                }
        logger.debug("NIFTY 500 entry not found in NSE allIndices response")
    except Exception as exc:
        logger.debug(f"try_get_market_breadth failed (non-critical): {exc}")

    return {}


# ---------------------------------------------------------------------------
# Public function: collect_index_history
# ---------------------------------------------------------------------------
def collect_index_history(from_date: str = "2021-01-01") -> int:
    """
    Bootstrap 5 years of daily index data for all 4 indices.

    Fetches in 390-day chunks to stay within Angel One limits, merges all
    index series by date, and calls insert_market_overview() for each date.

    Args:
        from_date: ISO date string to start from, e.g. "2021-01-01"

    Returns:
        Total number of rows inserted into market_overview.
    """
    init_database()
    smart_api = _angel_login()

    start_dt = datetime.strptime(from_date, "%Y-%m-%d")
    end_dt = datetime.now()

    # Build date chunks of up to _CHUNK_DAYS each
    chunks: List[tuple] = []
    cursor = start_dt
    while cursor < end_dt:
        chunk_end = min(cursor + timedelta(days=_CHUNK_DAYS), end_dt)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)

    logger.info(
        f"collect_index_history: {from_date} → {end_dt.date()} "
        f"({len(chunks)} chunks per index)"
    )
    print(
        f"Bootstrapping index history: {from_date} → {end_dt.date().isoformat()} "
        f"({len(chunks)} chunks × 4 indices)"
    )

    # Collect all candles per index across all chunks
    all_series: Dict[str, List[Dict[str, Any]]] = {name: [] for name in INDEX_TOKENS}

    for idx_name in INDEX_TOKENS:
        print(f"  Fetching {idx_name} ...")
        # INDIAVIX: Angel One getCandleData returns empty for VIX — use yfinance fallback
        if idx_name == "INDIAVIX":
            all_series[idx_name] = _fetch_vix_yfinance(start_dt, end_dt)
            logger.info(f"{idx_name}: {len(all_series[idx_name])} candles (yfinance)")
            continue
        for chunk_from, chunk_to in chunks:
            candles = _fetch_candles(smart_api, idx_name, chunk_from, chunk_to)
            all_series[idx_name].extend(candles)
            time.sleep(_RATE_SLEEP)
        logger.info(f"{idx_name}: {len(all_series[idx_name])} candles fetched")

    # Try to enrich today's row with market breadth
    breadth = try_get_market_breadth()

    # Merge and persist
    rows = _merge_series(all_series)
    today_str = datetime.now().strftime("%Y-%m-%d")
    inserted = 0

    for row in rows:
        # Attach breadth data only for today
        if row["date"] == today_str and breadth:
            row.update(breadth)
        ok = insert_market_overview(row)
        if ok:
            inserted += 1

    print(f"  Done — {inserted}/{len(rows)} rows inserted into market_overview")
    logger.info(f"collect_index_history complete: {inserted} rows")
    return inserted


# ---------------------------------------------------------------------------
# Public function: collect_index_daily
# ---------------------------------------------------------------------------
def collect_index_daily() -> int:
    """
    Fetch yesterday and today's candles for all 4 indices.
    Called by the scheduler after market close (typically ~16:00 IST).

    Returns:
        Number of rows inserted/updated.
    """
    init_database()
    smart_api = _angel_login()

    yesterday = datetime.now() - timedelta(days=2)
    today = datetime.now()

    logger.info(f"collect_index_daily: {yesterday.date()} → {today.date()}")
    print(f"Collecting daily index data ({yesterday.date()} – {today.date()}) ...")

    all_series: Dict[str, List[Dict[str, Any]]] = {name: [] for name in INDEX_TOKENS}

    for idx_name in INDEX_TOKENS:
        if idx_name == "INDIAVIX":
            all_series[idx_name] = _fetch_vix_yfinance(yesterday, today)
            continue
        candles = _fetch_candles(smart_api, idx_name, yesterday, today)
        all_series[idx_name].extend(candles)
        time.sleep(_RATE_SLEEP)

    breadth = try_get_market_breadth()
    rows = _merge_series(all_series)
    today_str = datetime.now().strftime("%Y-%m-%d")
    inserted = 0

    for row in rows:
        if row["date"] == today_str and breadth:
            row.update(breadth)
        ok = insert_market_overview(row)
        if ok:
            inserted += 1

    print(f"  Done — {inserted} row(s) upserted in market_overview")
    logger.info(f"collect_index_daily complete: {inserted} rows")
    return inserted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="TradeMind index data collector")
    parser.add_argument(
        "--history",
        action="store_true",
        help="Bootstrap full history from 2021-01-01 (default: daily update only)",
    )
    parser.add_argument(
        "--from-date",
        default="2021-01-01",
        help="Start date for history mode (YYYY-MM-DD, default: 2021-01-01)",
    )
    args = parser.parse_args()

    if args.history:
        print(f"Running full history bootstrap from {args.from_date} ...")
        count = collect_index_history(from_date=args.from_date)
        print(f"Bootstrap complete — {count} rows in market_overview.")
    else:
        print("Running daily index update ...")
        count = collect_index_daily()
        print(f"Daily update complete — {count} row(s) updated.")
