"""
TradeMind AI — NSE Delivery % Collector (Priority 2)

Downloads NSE bhavcopy (daily delivery data) for all stocks.
Delivery % = fraction of traded volume that was actual delivery (not squared off intraday).
High delivery % → institutional conviction → stronger price signal for ML model.

Source: https://archives.nseindia.com/products/content/sec_bhavdata_full_{DD-Mon-YYYY}.csv
Schedule: Daily after 6 PM IST (NSE uploads ~5:30 PM)

Usage:
    PYTHONPATH=. python collectors/delivery_collector.py              # today
    PYTHONPATH=. python collectors/delivery_collector.py --days 400  # backfill
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import get_connection, _execute, _executemany

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NSE_BHAVCOPY_URL = (
    "https://archives.nseindia.com/products/content/"
    "sec_bhavdata_full_{date}.csv"
)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nseindia.com",
}


def _ensure_table():
    """Create delivery_data table if not exists."""
    conn = get_connection()
    try:
        _execute(conn, """
            CREATE TABLE IF NOT EXISTS delivery_data (
                symbol       TEXT NOT NULL,
                date         DATE NOT NULL,
                delivery_pct FLOAT,
                total_volume BIGINT,
                PRIMARY KEY (symbol, date)
            )
        """)
        _execute(conn, "CREATE INDEX IF NOT EXISTS idx_delivery_symbol ON delivery_data(symbol, date DESC)")
        conn.commit()
    finally:
        conn.close()


def fetch_delivery_data(date: datetime) -> pd.DataFrame:
    """Fetch one day's bhavcopy from NSE. Returns DataFrame or empty."""
    date_str = date.strftime("%d-%b-%Y").upper()
    url = NSE_BHAVCOPY_URL.format(date=date_str)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        df.columns = df.columns.str.strip()
        if "SERIES" not in df.columns:
            return pd.DataFrame()
        df = df[df["SERIES"].str.strip() == "EQ"].copy()
        needed = {"SYMBOL", "DELIV_PER", "TTL_TRD_QNTY"}
        if not needed.issubset(df.columns):
            return pd.DataFrame()
        df = df[["SYMBOL", "DELIV_PER", "TTL_TRD_QNTY"]].copy()
        df.columns = ["symbol", "delivery_pct", "total_volume"]
        df["symbol"]       = df["symbol"].str.strip() + ".NS"
        df["delivery_pct"] = pd.to_numeric(df["delivery_pct"], errors="coerce")
        df["total_volume"] = pd.to_numeric(df["total_volume"],  errors="coerce")
        df["date"]         = date.strftime("%Y-%m-%d")
        return df.dropna(subset=["delivery_pct"])
    except Exception as e:
        logger.debug(f"{date_str}: {e}")
        return pd.DataFrame()


def store_delivery(df: pd.DataFrame) -> int:
    """Batch upsert delivery data for one day."""
    if df.empty:
        return 0
    conn = get_connection()
    try:
        rows = [
            (row["symbol"], row["date"], row["delivery_pct"], int(row["total_volume"] or 0))
            for _, row in df.iterrows()
        ]
        _executemany(conn, """
            INSERT INTO delivery_data (symbol, date, delivery_pct, total_volume)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (symbol, date) DO UPDATE SET
                delivery_pct = EXCLUDED.delivery_pct,
                total_volume = EXCLUDED.total_volume
        """, rows)
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def collect_today() -> int:
    _ensure_table()
    today = datetime.now()
    if today.weekday() >= 5:
        logger.info("Weekend — no NSE data")
        return 0
    df = fetch_delivery_data(today)
    if df.empty:
        logger.warning(f"No delivery data for {today.strftime('%Y-%m-%d')}")
        return 0
    n = store_delivery(df)
    logger.info(f"Stored {n} delivery records for {today.strftime('%Y-%m-%d')}")
    return n


def backfill(days: int = 400) -> int:
    _ensure_table()
    today = datetime.now()
    total = 0
    for i in range(days):
        dt = today - timedelta(days=i)
        if dt.weekday() >= 5:
            continue
        df = fetch_delivery_data(dt)
        if not df.empty:
            n = store_delivery(df)
            total += n
            logger.info(f"{dt.strftime('%Y-%m-%d')}: {n} records")
        time.sleep(0.5)
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=0, help="Backfill N days (0 = today only)")
    args = parser.parse_args()
    if args.days > 0:
        total = backfill(args.days)
        print(f"Backfill done: {total} records")
    else:
        collect_today()
