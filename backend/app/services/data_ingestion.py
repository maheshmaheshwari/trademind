"""
Data Ingestion Service

Fetch and store OHLC data for NIFTY 500 stocks.
Uses NSE India API as primary source with yfinance as fallback.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import settings
from app.models import Stock, OHLCData

logger = logging.getLogger(__name__)

# NSE stock suffix for yfinance
NSE_SUFFIX = ".NS"

# Index symbols
INDEX_SYMBOLS = {
    "NIFTY50": "^NSEI",
    "NIFTYBANK": "^NSEBANK",
    "NIFTYIT": "^CNXIT",
    "INDIAVIX": "^INDIAVIX",
}


class StockDataFetcher:
    """Fetches OHLC data for NSE stocks with NSE API primary and yfinance fallback."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self._nse_fetcher = None
    
    async def _get_nse_fetcher(self):
        """Get or create NSE fetcher."""
        if self._nse_fetcher is None:
            from app.services.nse_fetcher import NSEIndiaFetcher
            self._nse_fetcher = NSEIndiaFetcher()
        return self._nse_fetcher
    
    async def close_nse(self):
        """Close NSE fetcher."""
        if self._nse_fetcher:
            await self._nse_fetcher.close()
            self._nse_fetcher = None
    
    async def fetch_ohlc_nse(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from NSE India API.
        
        Args:
            symbol: Stock symbol (without .NS suffix)
            start_date: Start date for data
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = date.today()
        
        logger.info(f"Fetching data for {symbol} from NSE ({start_date} to {end_date})")
        
        fetcher = await self._get_nse_fetcher()
        df = await fetcher.get_historical_data(symbol, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No NSE data found for {symbol}")
            return pd.DataFrame()
        
        # Add adjusted close (same as close for NSE data)
        df["adj_close"] = df["close"]
        
        return df
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def fetch_ohlc_yfinance(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data from yfinance as fallback.
        
        Args:
            symbol: Stock symbol (without .NS suffix)
            start_date: Start date for data
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = date.today()
        
        # Add NSE suffix
        ticker_symbol = f"{symbol}{NSE_SUFFIX}"
        
        logger.info(f"Fetching data for {ticker_symbol} from yfinance")
        
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=(end_date + timedelta(days=1)).isoformat(),
            auto_adjust=True,
        )
        
        if df.empty:
            logger.warning(f"No yfinance data found for {ticker_symbol}")
            return pd.DataFrame()
        
        # Rename columns to lowercase
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        
        # Add adjusted close (same as close when auto_adjust=True)
        df["adj_close"] = df["close"]
        
        return df
    
    async def fetch_ohlc(
        self,
        symbol: str,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLC data - tries NSE first, then yfinance as fallback.
        
        Args:
            symbol: Stock symbol (without .NS suffix)
            start_date: Start date for data
            end_date: End date (defaults to today)
            
        Returns:
            DataFrame with OHLC data
        """
        # Try NSE first
        try:
            df = await self.fetch_ohlc_nse(symbol, start_date, end_date)
            if not df.empty:
                return df
        except Exception as e:
            logger.warning(f"NSE fetch failed for {symbol}: {e}")
        
        # Fallback to yfinance
        try:
            df = self.fetch_ohlc_yfinance(symbol, start_date, end_date)
            return df
        except Exception as e:
            logger.error(f"yfinance fetch also failed for {symbol}: {e}")
            return pd.DataFrame()
    
    async def ingest_stock_data(
        self,
        stock: Stock,
        start_date: date,
        end_date: Optional[date] = None,
    ) -> int:
        """
        Ingest OHLC data for a stock into database.
        
        Args:
            stock: Stock model instance
            start_date: Start date
            end_date: End date
            
        Returns:
            Number of records inserted
        """
        df = await self.fetch_ohlc(stock.symbol, start_date, end_date)
        
        if df.empty:
            return 0
        
        records_inserted = 0
        
        for idx, row in df.iterrows():
            trade_date = idx.date() if hasattr(idx, 'date') else idx
            
            # Check if record exists
            existing = await self.db.execute(
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
            self.db.add(ohlc)
            records_inserted += 1
        
        await self.db.commit()
        logger.info(f"Inserted {records_inserted} records for {stock.symbol}")
        
        return records_inserted
    
    async def ingest_all_stocks(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> dict:
        """
        Ingest data for all active stocks.
        
        Returns:
            Summary of ingestion results
        """
        if start_date is None:
            start_date = date.today() - timedelta(days=1)
        
        # Get all active stocks
        result = await self.db.execute(
            select(Stock).where(Stock.is_active == True)
        )
        stocks = result.scalars().all()
        
        summary = {
            "total_stocks": len(stocks),
            "successful": 0,
            "failed": 0,
            "records_inserted": 0,
            "errors": [],
        }
        
        try:
            for i, stock in enumerate(stocks):
                try:
                    records = await self.ingest_stock_data(stock, start_date, end_date)
                    summary["records_inserted"] += records
                    if records > 0:
                        summary["successful"] += 1
                    
                    # Rate limiting for NSE API (1 request per second)
                    if (i + 1) % 5 == 0:
                        await asyncio.sleep(2)
                    else:
                        await asyncio.sleep(1)
                        
                except Exception as e:
                    logger.error(f"Failed to ingest {stock.symbol}: {e}")
                    summary["failed"] += 1
                    summary["errors"].append({"symbol": stock.symbol, "error": str(e)})
        finally:
            await self.close_nse()
        
        return summary


async def get_index_data(symbol: str) -> dict:
    """
    Fetch current index data from NSE.
    
    Args:
        symbol: Index name (e.g., 'NIFTY 50', 'INDIA VIX')
        
    Returns:
        Dict with close, change, change_percent
    """
    try:
        from app.services.nse_fetcher import NSEIndiaFetcher
        
        fetcher = NSEIndiaFetcher()
        try:
            data = await fetcher.get_index_data(symbol)
            if data:
                return {
                    "close": float(data.get("last", 0) or 0),
                    "change": float(data.get("change", 0) or 0),
                    "change_percent": float(data.get("pct_change", 0) or 0),
                }
        finally:
            await fetcher.close()
        
        # Fallback to yfinance
        ticker = yf.Ticker(INDEX_SYMBOLS.get(symbol.replace(" ", ""), symbol))
        hist = ticker.history(period="2d")
        
        if hist.empty:
            return {"close": 0, "change": 0, "change_percent": 0}
        
        current = hist["Close"].iloc[-1]
        previous = hist["Close"].iloc[-2] if len(hist) > 1 else current
        
        change = current - previous
        change_percent = (change / previous * 100) if previous != 0 else 0
        
        return {
            "close": float(current),
            "change": float(change),
            "change_percent": float(change_percent),
        }
    except Exception as e:
        logger.error(f"Failed to fetch index data for {symbol}: {e}")
        return {"close": 0, "change": 0, "change_percent": 0}


def load_nifty500_symbols(filepath: str = None) -> list[dict]:
    """
    Load NIFTY 500 stock symbols from CSV.
    
    Returns:
        List of stock dictionaries with symbol, name, sector, industry
    """
    filepath = filepath or settings.nifty500_symbols_file
    
    try:
        df = pd.read_csv(filepath)
        stocks = []
        
        for _, row in df.iterrows():
            stocks.append({
                "symbol": row.get("Symbol", row.get("symbol", "")),
                "name": row.get("Company Name", row.get("name", "")),
                "sector": row.get("Industry", row.get("sector", "")),
                "industry": row.get("Industry", row.get("industry", "")),
            })
        
        return stocks
    except FileNotFoundError:
        logger.warning(f"NIFTY 500 symbols file not found: {filepath}")
        return []
    except Exception as e:
        logger.error(f"Error loading NIFTY 500 symbols: {e}")
        return []
