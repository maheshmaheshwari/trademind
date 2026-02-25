#!/usr/bin/env python3
"""
yfinance OHLC Data Scraper

Fetches historical OHLC (Open, High, Low, Close, Volume) data from Yahoo Finance
for Nifty 500 stocks and stores it in the PostgreSQL database.

Usage:
    # Test mode (5 stocks, last 30 days)
    python scripts/yfinance_scraper.py --test
    
    # Full run (all Nifty 500 stocks, 2 years of data)
    python scripts/yfinance_scraper.py
    
    # Custom period
    python scripts/yfinance_scraper.py --period 5y
    
    # Specific stocks
    python scripts/yfinance_scraper.py --symbols RELIANCE TCS INFY
"""

import argparse
import csv
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yfinance as yf
from sqlalchemy.orm import Session, sessionmaker

from app.database import sync_engine
from app.models import Stock, OHLCData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('yfinance_scraper.log')
    ]
)
logger = logging.getLogger(__name__)


def convert_to_nse_symbol(symbol: str) -> str:
    """Convert stock symbol to NSE format for yfinance."""
    # yfinance uses .NS suffix for NSE stocks
    # Handle special cases
    symbol = symbol.upper().strip()
    
    # Some stocks have different symbols on yfinance
    symbol_mapping = {
        'M&M': 'M&M',
        'M&MFIN': 'M&MFIN',
        'L&TFH': 'L&TFH',
    }
    
    mapped = symbol_mapping.get(symbol, symbol)
    return f"{mapped}.NS"


def fetch_ohlc_data(symbol: str, period: str = "2y", max_retries: int = 3) -> Optional[dict]:
    """
    Fetch OHLC data from yfinance for a given symbol.
    
    Args:
        symbol: Stock symbol (without .NS suffix)
        period: Data period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with date -> OHLC data, or None if failed
    """
    nse_symbol = convert_to_nse_symbol(symbol)
    
    for attempt in range(max_retries):
        try:
            ticker = yf.Ticker(nse_symbol)
            
            # Suppress yfinance internal errors
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hist = ticker.history(period=period, timeout=30)
            
            if hist.empty:
                # Try with explicit date range
                from datetime import datetime, timedelta
                end_date = datetime.now()
                period_days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
                days = period_days.get(period, 730)
                start_date = end_date - timedelta(days=days)
                
                hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                      end=end_date.strftime('%Y-%m-%d'), 
                                      timeout=30)
            
            if hist.empty:
                if attempt < max_retries - 1:
                    logger.debug(f"Retry {attempt + 1} for {nse_symbol}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                logger.warning(f"No data found for {nse_symbol} after {max_retries} attempts")
                return None
            
            # Convert to dict format
            data = {}
            for idx, row in hist.iterrows():
                try:
                    data[idx.date()] = {
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if row['Volume'] > 0 else 0,
                        'adj_close': float(row['Close'])  # Using Close as Adj Close
                    }
                except (ValueError, KeyError) as e:
                    logger.debug(f"Skipping row for {symbol}: {e}")
                    continue
            
            if data:
                return data
            return None
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.debug(f"Retry {attempt + 1} for {symbol} due to: {e}")
                time.sleep(2 ** attempt)
            else:
                logger.error(f"Error fetching data for {symbol} after {max_retries} attempts: {e}")
                return None
    
    return None


def save_ohlc_to_database(symbol: str, ohlc_data: dict, session: Session) -> int:
    """
    Save OHLC data to database.
    
    Returns:
        Number of records saved/updated
    """
    # Find the stock
    stock = session.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        logger.warning(f"Stock {symbol} not found in database")
        return 0
    
    saved_count = 0
    
    for data_date, values in ohlc_data.items():
        # Check if record exists
        existing = session.query(OHLCData).filter(
            OHLCData.stock_id == stock.id,
            OHLCData.date == data_date
        ).first()
        
        if existing:
            # Update existing record
            existing.open = values['open']
            existing.high = values['high']
            existing.low = values['low']
            existing.close = values['close']
            existing.adj_close = values['adj_close']
            existing.volume = values['volume']
        else:
            # Create new record
            ohlc = OHLCData(
                stock_id=stock.id,
                date=data_date,
                open=values['open'],
                high=values['high'],
                low=values['low'],
                close=values['close'],
                adj_close=values['adj_close'],
                volume=values['volume']
            )
            session.add(ohlc)
        
        saved_count += 1
    
    return saved_count


def load_nifty500_list() -> list[tuple[str, str]]:
    """Load Nifty 500 stock list from CSV."""
    csv_path = Path(__file__).parent.parent / "data" / "nifty500_list.csv"
    
    stocks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row.get('Symbol', row.get('symbol', ''))
            name = row.get('Company Name', row.get('company_name', row.get('name', '')))
            if symbol:
                stocks.append((symbol.strip(), name.strip()))
    
    logger.info(f"Loaded {len(stocks)} stocks from {csv_path}")
    return stocks


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fetch OHLC data from yfinance')
    parser.add_argument('--test', action='store_true', help='Test mode: 5 stocks, 30 days')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of stocks')
    parser.add_argument('--period', type=str, default='2y', help='Data period (1mo, 3mo, 1y, 2y, 5y)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to fetch')
    parser.add_argument('--resume-from', type=str, help='Resume from a specific symbol')
    
    args = parser.parse_args()
    
    # Determine stocks to process
    if args.symbols:
        stocks = [(s, s) for s in args.symbols]
    else:
        stocks = load_nifty500_list()
    
    # Apply test mode settings
    period = args.period
    if args.test:
        stocks = stocks[:5]
        period = '1mo'
        logger.info("Test mode: 5 stocks, 1 month of data")
    
    if args.limit:
        stocks = stocks[:args.limit]
        logger.info(f"Limited to {args.limit} stocks")
    
    # Resume from specific stock if requested
    if args.resume_from:
        idx = next((i for i, (s, _) in enumerate(stocks) if s == args.resume_from), 0)
        stocks = stocks[idx:]
        logger.info(f"Resuming from {args.resume_from} ({len(stocks)} remaining)")
    
    # Create database session
    Session = sessionmaker(bind=sync_engine)
    
    success_count = 0
    fail_count = 0
    total_records = 0
    
    logger.info(f"Starting OHLC data fetch for {len(stocks)} stocks (period: {period})")
    logger.info("=" * 50)
    
    for i, (symbol, name) in enumerate(stocks):
        logger.info(f"[{i+1}/{len(stocks)}] Fetching {symbol}...")
        
        # Fetch data from yfinance
        ohlc_data = fetch_ohlc_data(symbol, period)
        
        if ohlc_data:
            # Save to database
            session = Session()
            try:
                saved = save_ohlc_to_database(symbol, ohlc_data, session)
                session.commit()
                success_count += 1
                total_records += saved
                logger.info(f"  ✓ Saved {saved} records for {symbol}")
            except Exception as e:
                session.rollback()
                logger.error(f"  ✗ Database error for {symbol}: {e}")
                fail_count += 1
            finally:
                session.close()
        else:
            fail_count += 1
        
        # Rate limiting - be nice to Yahoo Finance
        time.sleep(0.5)
    
    logger.info("")
    logger.info("=" * 50)
    logger.info("OHLC data fetch complete!")
    logger.info(f"Total: {len(stocks)} | Success: {success_count} | Failed: {fail_count}")
    logger.info(f"Total records saved: {total_records}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
