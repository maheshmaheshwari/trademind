"""
Nifty 500 AI — Price Collector

Collects historical and daily OHLCV data for Nifty 500 stocks using yfinance.
Supports both daily and intraday intervals with error handling, rate limiting,
duplicate detection, progress bars, and CSV backup creation.

Usage:
    from collectors.price_collector import collect_historical, collect_all_stocks
    
    # Collect 2 years of daily data for one stock
    collect_historical("TCS.NS", period="2y", interval="1d")
    
    # Collect data for all 50 stocks
    collect_all_stocks(interval="1d", period="2y")
"""

import csv
import logging
import os
import time
from datetime import datetime
from typing import List, Optional

import pandas as pd
import yfinance as yf
from tqdm import tqdm

from data.stocks_list import NIFTY_50_STOCKS, INDEX_SYMBOLS, get_all_symbols
from database.db import get_connection, init_database, insert_prices_batch, get_latest_date

# Configure logging
logger = logging.getLogger(__name__)

# Rate limit: wait 0.5 seconds between yfinance API calls
RATE_LIMIT_SECONDS = 0.5

# Backup directory
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backups")


def _ensure_backup_dir() -> None:
    """Create the backup directory if it doesn't exist."""
    os.makedirs(BACKUP_DIR, exist_ok=True)


def collect_historical(
    symbol: str,
    period: str = "2y",
    interval: str = "1d",
    save_csv: bool = True,
) -> int:
    """
    Collect historical price data for a single stock using yfinance.

    Args:
        symbol: yfinance symbol (e.g. "TCS.NS")
        period: Data period — "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"
        interval: Data interval — "1d", "1h", "5m", "15m", "30m"
        save_csv: Whether to save a CSV backup

    Returns:
        Number of rows inserted into the database.

    Example:
        rows = collect_historical("TCS.NS", period="2y")
        print(f"Inserted {rows} price records for TCS")
    """
    try:
        logger.info(f"Downloading {symbol} — period={period}, interval={interval}")

        # Download data from yfinance
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return 0

        # Prepare batch rows for insertion
        rows = []
        for idx, row in df.iterrows():
            # Handle both daily and intraday timestamps
            if interval == "1d":
                date_str = idx.strftime("%Y-%m-%d")
                time_str = None
            else:
                date_str = idx.strftime("%Y-%m-%d")
                time_str = idx.strftime("%H:%M:%S")

            rows.append((
                symbol,
                "NSE",
                date_str,
                time_str,
                round(row["Open"], 2),
                round(row["High"], 2),
                round(row["Low"], 2),
                round(row["Close"], 2),
                int(row["Volume"]),
                interval,
            ))

        # Batch insert (duplicates are skipped via INSERT OR IGNORE)
        inserted = insert_prices_batch(rows)

        # Save CSV backup
        if save_csv and rows:
            _save_csv_backup(symbol, df, interval)

        logger.info(f"✅ {symbol}: {inserted}/{len(rows)} new rows inserted")
        return inserted

    except Exception as e:
        logger.error(f"❌ Error collecting {symbol}: {e}")
        return 0


def collect_all_stocks(
    interval: str = "1d",
    period: str = "2y",
    save_csv: bool = True,
) -> dict:
    """
    Collect data for all Nifty 50 stocks with progress bar.

    Args:
        interval: Data interval ("1d", "1h", "5m")
        period: Data period ("2y", "1y", "6mo", etc.)
        save_csv: Whether to create CSV backups

    Returns:
        Summary dict with success_count, fail_count, total_rows, failed_symbols.

    Example:
        summary = collect_all_stocks(interval="1d", period="2y")
        print(f"Collected {summary['success_count']}/{len(NIFTY_50_STOCKS)} stocks")
    """
    # Ensure database is initialized
    init_database()
    _ensure_backup_dir()

    symbols = get_all_symbols()
    success_count = 0
    fail_count = 0
    total_rows = 0
    failed_symbols = []

    print(f"\n📊 Collecting {period} of {interval} data for {len(symbols)} stocks...\n")

    for stock in tqdm(NIFTY_50_STOCKS, desc="Downloading", unit="stock"):
        symbol = stock["symbol"]
        try:
            rows = collect_historical(symbol, period=period, interval=interval, save_csv=save_csv)
            if rows > 0:
                success_count += 1
                total_rows += rows
            else:
                fail_count += 1
                failed_symbols.append(symbol)
        except Exception as e:
            fail_count += 1
            failed_symbols.append(symbol)
            logger.error(f"Failed to collect {symbol}: {e}")

        # Rate limit to avoid being blocked by yfinance
        time.sleep(RATE_LIMIT_SECONDS)

    # Print summary
    print(f"\n{'='*50}")
    print(f"📈 Collection Summary")
    print(f"{'='*50}")
    print(f"✅ Successful: {success_count}/{len(symbols)} stocks")
    print(f"❌ Failed: {fail_count}/{len(symbols)} stocks")
    print(f"📊 Total rows inserted: {total_rows}")
    if failed_symbols:
        print(f"⚠️  Failed symbols: {', '.join(failed_symbols)}")
    print(f"{'='*50}\n")

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "total_rows": total_rows,
        "failed_symbols": failed_symbols,
    }


def collect_index_data(period: str = "2y") -> dict:
    """
    Collect historical data for major Indian indices.

    Downloads data for Nifty 500 (^CNX500), Nifty 50 (^NSEI),
    Sensex (^BSESN), and India VIX (^INDIAVIX).

    Args:
        period: Data period ("2y", "1y", etc.)

    Returns:
        Summary dict similar to collect_all_stocks().
    """
    _ensure_backup_dir()

    success_count = 0
    fail_count = 0
    total_rows = 0
    failed_indices = []

    print(f"\n📈 Collecting index data for {len(INDEX_SYMBOLS)} indices...\n")

    for name, symbol in tqdm(INDEX_SYMBOLS.items(), desc="Indices", unit="index"):
        try:
            rows = collect_historical(symbol, period=period, interval="1d", save_csv=True)
            if rows > 0:
                success_count += 1
                total_rows += rows
            else:
                fail_count += 1
                failed_indices.append(name)
        except Exception as e:
            fail_count += 1
            failed_indices.append(name)
            logger.error(f"Failed to collect index {name}: {e}")

        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n✅ Indices collected: {success_count}/{len(INDEX_SYMBOLS)}")
    if failed_indices:
        print(f"⚠️  Failed: {', '.join(failed_indices)}")

    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "total_rows": total_rows,
        "failed_indices": failed_indices,
    }


def collect_incremental(symbol: str) -> int:
    """
    Collect only new data since the last available date in the database.
    This is useful for daily updates — avoids re-downloading everything.

    Args:
        symbol: Stock symbol (e.g. "TCS.NS")

    Returns:
        Number of new rows inserted.
    """
    latest_date = get_latest_date(symbol)

    if latest_date:
        # We have data — download only from the last date onward
        # Use period="5d" for recent data (safer than start_date with yfinance)
        logger.info(f"Incremental update for {symbol} — last date: {latest_date}")
        return collect_historical(symbol, period="5d", interval="1d", save_csv=False)
    else:
        # No data at all — do a full download
        logger.info(f"No existing data for {symbol} — doing full 2y download")
        return collect_historical(symbol, period="2y", interval="1d", save_csv=True)


def collect_eod_data() -> dict:
    """
    End-of-day data collection: incremental update for all stocks.
    Meant to be run daily after market close (4 PM IST).

    Returns:
        Summary dict.
    """
    init_database()
    symbols = get_all_symbols()

    success_count = 0
    fail_count = 0
    total_rows = 0

    print(f"\n🌅 EOD Collection — updating {len(symbols)} stocks...\n")

    for symbol in tqdm(symbols, desc="EOD Update", unit="stock"):
        try:
            rows = collect_incremental(symbol)
            if rows >= 0:
                success_count += 1
                total_rows += rows
            else:
                fail_count += 1
        except Exception as e:
            fail_count += 1
            logger.error(f"EOD failed for {symbol}: {e}")

        time.sleep(RATE_LIMIT_SECONDS)

    print(f"\n✅ EOD Update: {success_count}/{len(symbols)} stocks, {total_rows} new rows")
    return {
        "success_count": success_count,
        "fail_count": fail_count,
        "total_rows": total_rows,
    }


def _save_csv_backup(symbol: str, df: pd.DataFrame, interval: str) -> None:
    """
    Save price data as a CSV backup file.

    Args:
        symbol: Stock symbol
        df: DataFrame with OHLCV data
        interval: Data interval for filename
    """
    try:
        _ensure_backup_dir()
        # Clean symbol for filename (remove .NS suffix, replace special chars)
        clean_symbol = symbol.replace(".NS", "").replace("&", "_").replace("-", "_")
        filename = f"{clean_symbol}_{interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        filepath = os.path.join(BACKUP_DIR, filename)

        df.to_csv(filepath)
        logger.debug(f"CSV backup saved: {filepath}")
    except Exception as e:
        logger.warning(f"Failed to save CSV backup for {symbol}: {e}")


# ==========================================
# Quick test when run directly
# ==========================================
if __name__ == "__main__":
    """Quick test: collect 1 month of TCS data."""
    logging.basicConfig(level=logging.INFO)
    init_database()

    print("Testing price collector with TCS.NS (1 month)...")
    rows = collect_historical("TCS.NS", period="1mo", interval="1d")
    print(f"Result: {rows} rows inserted for TCS.NS")
