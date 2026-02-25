"""
Database Seeding Script

Populate the database with NIFTY 500 stocks and initial data.
"""

import asyncio
import csv
import logging
from datetime import date, timedelta
from pathlib import Path

from sqlalchemy import select

# Setup path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.database import async_session_maker, init_db
from app.models import Stock
from app.services.data_ingestion import StockDataFetcher

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# NIFTY 50 constituents (for flagging)
NIFTY_50_SYMBOLS = {
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "SBIN",
    "BHARTIARTL", "ITC", "KOTAKBANK", "LT", "AXISBANK", "ASIANPAINT", "MARUTI",
    "SUNPHARMA", "TITAN", "BAJFINANCE", "WIPRO", "ULTRACEMCO", "NESTLEIND",
    "HCLTECH", "TATAMOTORS", "POWERGRID", "NTPC", "M&M", "ONGC", "JSWSTEEL",
    "TATASTEEL", "ADANIENT", "COALINDIA", "BAJAJFINSV", "HDFCLIFE", "TECHM",
    "GRASIM", "INDUSINDBK", "SBILIFE", "DRREDDY", "DIVISLAB", "BRITANNIA",
    "CIPLA", "EICHERMOT", "APOLLOHOSP", "ADANIPORTS", "HEROMOTOCO", "BAJAJ-AUTO",
    "TATACONSUM", "HINDALCO", "BPCL", "UPL", "VEDL"
}

NIFTY_100_SYMBOLS = NIFTY_50_SYMBOLS | {
    "GODREJCP", "DABUR", "HAVELLS", "PIDILITIND", "SIEMENS", "AMBUJACEM",
    "MARICO", "BERGEPAINT", "DLF", "ICICIPRULI", "ICICIGI", "COLPAL",
    "TORNTPHARM", "MCDOWELL-N", "LUPIN", "HINDPETRO", "IOC", "GAIL",
    "TATAPOWER", "ADANIGREEN", "INDIGO", "ZOMATO", "NYKAA", "HAL",
    "LTIM", "COFORGE", "NAUKRI", "DMART", "VBL", "SHREECEM",
    "MOTHERSON", "BOSCHLTD", "SBICARD", "RECLTD", "PFC", "IRFC",
    "MCX", "CDSL", "BSE", "POLYCAB", "DIXON", "CROMPTON", "SRF",
    "PIIND", "ACC", "TRENT", "MUTHOOTFIN", "CHOLAFIN"
}


async def load_stocks_from_csv(filepath: str = None) -> list[dict]:
    """Load stock data from CSV file."""
    filepath = filepath or "data/nifty500_list.csv"
    
    # Use dict to deduplicate by symbol
    stocks_dict = {}
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbol = row["Symbol"].strip()
            # Only keep first occurrence
            if symbol not in stocks_dict:
                stocks_dict[symbol] = {
                    "symbol": symbol,
                    "name": row["Company Name"].strip(),
                    "industry": row["Industry"].strip(),
                    "market_cap_category": row["Market Cap Category"].strip(),
                }
    
    return list(stocks_dict.values())


async def seed_stocks(session):
    """Seed stocks into database."""
    stocks_data = await load_stocks_from_csv()
    
    created = 0
    updated = 0
    
    for stock_data in stocks_data:
        symbol = stock_data["symbol"]
        
        # Check if exists
        result = await session.execute(
            select(Stock).where(Stock.symbol == symbol)
        )
        existing = result.scalar_one_or_none()
        
        if existing:
            # Update
            existing.name = stock_data["name"]
            existing.sector = stock_data["industry"]
            existing.industry = stock_data["industry"]
            existing.market_cap_category = stock_data["market_cap_category"]
            existing.is_nifty50 = symbol in NIFTY_50_SYMBOLS
            existing.is_nifty100 = symbol in NIFTY_100_SYMBOLS
            existing.is_nifty500 = True
            existing.is_active = True
            updated += 1
        else:
            # Create
            stock = Stock(
                symbol=symbol,
                name=stock_data["name"],
                sector=stock_data["industry"],
                industry=stock_data["industry"],
                market_cap_category=stock_data["market_cap_category"],
                is_nifty50=symbol in NIFTY_50_SYMBOLS,
                is_nifty100=symbol in NIFTY_100_SYMBOLS,
                is_nifty500=True,
                is_active=True,
            )
            session.add(stock)
            created += 1
    
    await session.commit()
    
    return {"created": created, "updated": updated}


async def seed_historical_data(session, days: int = 365):
    """
    Seed historical OHLC data for all stocks.
    
    Args:
        session: Database session
        days: Number of days of history to fetch
    """
    start_date = date.today() - timedelta(days=days)
    
    # Get all active stocks
    result = await session.execute(
        select(Stock).where(Stock.is_active == True)
    )
    stocks = result.scalars().all()
    
    logger.info(f"Fetching {days} days of data for {len(stocks)} stocks")
    
    fetcher = StockDataFetcher(session)
    
    success = 0
    failed = 0
    
    for i, stock in enumerate(stocks):
        try:
            records = await fetcher.ingest_stock_data(stock, start_date)
            if records > 0:
                success += 1
                logger.info(f"[{i+1}/{len(stocks)}] {stock.symbol}: {records} records")
            else:
                logger.warning(f"[{i+1}/{len(stocks)}] {stock.symbol}: No data")
        except Exception as e:
            failed += 1
            logger.error(f"[{i+1}/{len(stocks)}] {stock.symbol}: Failed - {e}")
        
        # Rate limiting - avoid hitting API too fast
        if (i + 1) % 10 == 0:
            await asyncio.sleep(1)
    
    return {"success": success, "failed": failed, "total": len(stocks)}


async def main():
    """Main seeding function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seed TradeMind database")
    parser.add_argument("--stocks-only", action="store_true", help="Only seed stocks, no OHLC data")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data to fetch")
    args = parser.parse_args()
    
    logger.info("Initializing database...")
    await init_db()
    
    async with async_session_maker() as session:
        # Seed stocks
        logger.info("Seeding NIFTY 500 stocks...")
        stock_result = await seed_stocks(session)
        logger.info(f"Stocks: {stock_result['created']} created, {stock_result['updated']} updated")
        
        if not args.stocks_only:
            # Seed historical data
            logger.info(f"Fetching {args.days} days of historical data...")
            data_result = await seed_historical_data(session, args.days)
            logger.info(f"Data: {data_result['success']}/{data_result['total']} stocks loaded")
    
    logger.info("Seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())
