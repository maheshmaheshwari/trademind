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
TOKENS_FILE = os.path.join(os.path.dirname(__file__), "data", "angel_tokens.json")
RATE_LIMIT_SECS = 0.35  # Angel One allows ~3 req/sec


def load_token_map() -> Dict:
    """Load the full Nifty 500 token map from angel_tokens.json."""
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
) -> List[tuple]:
    """
    Fetch daily candles from Angel One for a single stock.

    Returns list of DB-ready tuples:
        (symbol_ns, exchange, date, time, open, high, low, close, volume, interval)
    """
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 09:15")
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


def main():
    parser = argparse.ArgumentParser(description="Update stock data via Angel One")
    parser.add_argument("--days", type=int, default=5, help="Days of history to fetch for new stocks (default: 5)")
    args = parser.parse_args()

    init_database()

    # Load tokens
    token_map = load_token_map()
    total = len(token_map)
    print(f"\n📊 Nifty 500 stocks in token map: {total}")
    print(f"📅 Fetching candles (smart gap detection, fallback: {args.days} days)...\n")

    # Login
    smart_api = angel_login()

    success = 0
    failed = 0
    total_rows = 0
    failed_symbols = []

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

    # Verify final state
    conn = get_connection()
    cur = _execute(conn, "SELECT MAX(date) as latest, COUNT(DISTINCT symbol) as symbols FROM prices WHERE interval = '1d'")
    final = cur.fetchone()
    print(f"\n📊 DB state: {final[1]} symbols, latest date: {final[0]}")
    conn.close()


if __name__ == "__main__":
    main()
