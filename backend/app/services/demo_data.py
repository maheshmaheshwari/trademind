"""
Demo Data Generator

Generate realistic demo OHLC data for testing when live data sources are unavailable.
"""

import asyncio
import logging
import random
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Stock, OHLCData

logger = logging.getLogger(__name__)


def generate_random_ohlc(
    base_price: float = 1000.0,
    num_days: int = 365,
    volatility: float = 0.02,
) -> pd.DataFrame:
    """
    Generate realistic random OHLC data.
    
    Uses geometric Brownian motion to simulate stock prices.
    
    Args:
        base_price: Starting price
        num_days: Number of trading days
        volatility: Daily volatility (default 2%)
        
    Returns:
        DataFrame with OHLC data
    """
    np.random.seed(None)  # Random seed each time
    
    # Generate daily returns with slight positive drift
    drift = 0.0002  # Small upward bias
    returns = np.random.normal(drift, volatility, num_days)
    
    # Calculate prices using cumulative returns
    price_multipliers = np.exp(np.cumsum(returns))
    close_prices = base_price * price_multipliers
    
    records = []
    trading_days = []
    
    # Generate trading days (exclude weekends)
    current_date = date.today() - timedelta(days=num_days * 1.5)
    while len(trading_days) < num_days:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    for i, trade_date in enumerate(trading_days):
        close = close_prices[i]
        
        # Generate realistic OHLC
        daily_range = close * random.uniform(0.01, 0.03)  # 1-3% daily range
        high = close + random.uniform(0, daily_range)
        low = close - random.uniform(0, daily_range)
        open_price = low + random.uniform(0, high - low)
        
        # Ensure consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Random volume (10K to 10M)
        volume = int(random.uniform(10000, 10000000))
        
        records.append({
            "date": trade_date,
            "open": round(open_price, 2),
            "high": round(high, 2),
            "low": round(low, 2),
            "close": round(close, 2),
            "adj_close": round(close, 2),
            "volume": volume,
        })
    
    df = pd.DataFrame(records)
    df.set_index("date", inplace=True)
    return df


def get_stock_base_price(symbol: str) -> float:
    """
    Get realistic base price for a stock based on typical price ranges.
    """
    # Some approximate price ranges for well-known stocks
    price_ranges = {
        # High price stocks
        "MRF": 100000, "PAGEIND": 40000, "BOSCHLTD": 25000, "HONAUT": 40000,
        # Large caps
        "RELIANCE": 2500, "TCS": 4000, "HDFCBANK": 1700, "INFY": 1800,
        "ICICIBANK": 1200, "HINDUNILVR": 2700, "SBIN": 700, "BHARTIARTL": 1500,
        "ITC": 450, "KOTAKBANK": 1800, "LT": 3500, "AXISBANK": 1100,
        # Mid price
        "MARUTI": 11000, "TITAN": 3500, "SUNPHARMA": 1700, "WIPRO": 500,
        "TATAMOTORS": 950, "NTPC": 350, "ONGC": 250, "COALINDIA": 400,
        # Lower price
        "YESBANK": 25, "IDEA": 10, "PNB": 100, "RPOWER": 20,
    }
    
    if symbol in price_ranges:
        return price_ranges[symbol]
    
    # Random base price for unknown stocks
    return random.uniform(50, 5000)


async def generate_demo_data_for_stock(
    db: AsyncSession,
    stock: Stock,
    num_days: int = 365,
) -> int:
    """
    Generate demo OHLC data for a stock.
    
    Args:
        db: Database session
        stock: Stock model instance
        num_days: Number of days of data
        
    Returns:
        Number of records inserted
    """
    base_price = get_stock_base_price(stock.symbol)
    volatility = 0.02 if stock.market_cap_category == "Large" else 0.03
    
    df = generate_random_ohlc(base_price, num_days, volatility)
    
    records_inserted = 0
    
    for trade_date, row in df.iterrows():
        # Check if record exists
        existing = await db.execute(
            select(OHLCData).where(
                OHLCData.stock_id == stock.id,
                OHLCData.date == trade_date,
            )
        )
        
        if existing.scalar_one_or_none():
            continue
        
        ohlc = OHLCData(
            stock_id=stock.id,
            date=trade_date,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            adj_close=float(row["adj_close"]),
            volume=int(row["volume"]),
        )
        db.add(ohlc)
        records_inserted += 1
    
    await db.commit()
    return records_inserted


async def generate_demo_data_all_stocks(
    db: AsyncSession,
    num_days: int = 365,
    limit: Optional[int] = None,
) -> dict:
    """
    Generate demo data for all stocks.
    
    Args:
        db: Database session
        num_days: Days of history
        limit: Optional limit on number of stocks
        
    Returns:
        Summary dict
    """
    # Get all active stocks
    query = select(Stock).where(Stock.is_active == True)
    if limit:
        query = query.limit(limit)
    
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    summary = {
        "total_stocks": len(stocks),
        "records_inserted": 0,
    }
    
    for i, stock in enumerate(stocks):
        logger.info(f"[{i+1}/{len(stocks)}] Generating data for {stock.symbol}")
        records = await generate_demo_data_for_stock(db, stock, num_days)
        summary["records_inserted"] += records
        
        # Progress checkpoint
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{len(stocks)} stocks, {summary['records_inserted']} total records")
    
    return summary


# CLI function for running from command line
async def main():
    """Generate demo data from command line."""
    import argparse
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from app.database import async_session_maker, init_db
    
    parser = argparse.ArgumentParser(description="Generate demo OHLC data")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--limit", type=int, help="Limit number of stocks")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    logger.info("Initializing database...")
    await init_db()
    
    async with async_session_maker() as session:
        logger.info(f"Generating {args.days} days of demo data...")
        result = await generate_demo_data_all_stocks(session, args.days, args.limit)
        logger.info(f"Done! Generated {result['records_inserted']} records for {result['total_stocks']} stocks")


if __name__ == "__main__":
    asyncio.run(main())
