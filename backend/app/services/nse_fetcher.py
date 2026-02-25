"""
NSE India Data Fetcher

Direct API integration with NSE India for stock data.
Note: NSE blocks automated requests, so we use proper headers.
"""

import asyncio
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import httpx
import pandas as pd

logger = logging.getLogger(__name__)


class NSEIndiaFetcher:
    """Fetch stock data directly from NSE India."""
    
    BASE_URL = "https://www.nseindia.com"
    
    # Headers to mimic browser request
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://www.nseindia.com/",
    }
    
    def __init__(self):
        self.cookies = None
        self._client = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with session cookies."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                headers=self.HEADERS,
                timeout=30.0,
                follow_redirects=True,
            )
            # First request to get cookies
            await self._client.get(self.BASE_URL)
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_quote(self, symbol: str) -> Optional[dict]:
        """
        Get current quote for a symbol.
        
        Args:
            symbol: NSE symbol (e.g., 'RELIANCE')
            
        Returns:
            Quote data dict or None
        """
        client = await self._get_client()
        
        try:
            url = f"{self.BASE_URL}/api/quote-equity?symbol={symbol}"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "symbol": symbol,
                    "last_price": data.get("priceInfo", {}).get("lastPrice"),
                    "open": data.get("priceInfo", {}).get("open"),
                    "high": data.get("priceInfo", {}).get("intraDayHighLow", {}).get("max"),
                    "low": data.get("priceInfo", {}).get("intraDayHighLow", {}).get("min"),
                    "close": data.get("priceInfo", {}).get("close"),
                    "prev_close": data.get("priceInfo", {}).get("previousClose"),
                    "volume": data.get("priceInfo", {}).get("totalTradedVolume"),
                    "change": data.get("priceInfo", {}).get("change"),
                    "pct_change": data.get("priceInfo", {}).get("pChange"),
                }
            else:
                logger.warning(f"Failed to get quote for {symbol}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date = None,
    ) -> pd.DataFrame:
        """
        Get historical OHLC data for a symbol.
        
        Note: NSE historical data API has limitations.
        For longer periods, we use the chart data API.
        
        Args:
            symbol: NSE symbol
            start_date: Start date
            end_date: End date (default: today)
            
        Returns:
            DataFrame with OHLC data
        """
        if end_date is None:
            end_date = date.today()
        
        client = await self._get_client()
        
        try:
            # Use chart data API for historical data
            # Format dates
            from_date = start_date.strftime("%d-%m-%Y")
            to_date = end_date.strftime("%d-%m-%Y")
            
            url = f"{self.BASE_URL}/api/historical/cm/equity?symbol={symbol}&series=[%22EQ%22]&from={from_date}&to={to_date}"
            
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if "data" in data and data["data"]:
                    records = []
                    for item in data["data"]:
                        records.append({
                            "date": datetime.strptime(item["CH_TIMESTAMP"], "%Y-%m-%d").date(),
                            "open": float(item["CH_OPENING_PRICE"]),
                            "high": float(item["CH_TRADE_HIGH_PRICE"]),
                            "low": float(item["CH_TRADE_LOW_PRICE"]),
                            "close": float(item["CH_CLOSING_PRICE"]),
                            "volume": int(item["CH_TOT_TRADED_QTY"]),
                            "value": float(item.get("CH_TOT_TRADED_VAL", 0)),
                        })
                    
                    df = pd.DataFrame(records)
                    df.set_index("date", inplace=True)
                    df.sort_index(inplace=True)
                    return df
                else:
                    logger.warning(f"No historical data for {symbol}")
                    return pd.DataFrame()
            else:
                logger.warning(f"Failed to get history for {symbol}: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_nifty500_list(self) -> list[dict]:
        """Get list of NIFTY 500 constituents from NSE."""
        client = await self._get_client()
        
        try:
            url = f"{self.BASE_URL}/api/equity-stockIndices?index=NIFTY%20500"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                stocks = []
                
                for item in data.get("data", []):
                    if item.get("symbol") and item["symbol"] != "NIFTY 500":
                        stocks.append({
                            "symbol": item["symbol"],
                            "name": item.get("meta", {}).get("companyName", item["symbol"]),
                            "industry": item.get("meta", {}).get("industry", ""),
                            "series": item.get("series", "EQ"),
                        })
                
                return stocks
            else:
                logger.error(f"Failed to get NIFTY 500 list: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching NIFTY 500 list: {e}")
            return []
    
    async def get_index_data(self, index: str = "NIFTY 50") -> Optional[dict]:
        """
        Get index data.
        
        Args:
            index: Index name (e.g., 'NIFTY 50', 'NIFTY 500', 'INDIA VIX')
        """
        client = await self._get_client()
        
        try:
            index_encoded = index.replace(" ", "%20")
            url = f"{self.BASE_URL}/api/equity-stockIndices?index={index_encoded}"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                metadata = data.get("metadata", {})
                
                return {
                    "index": index,
                    "last": metadata.get("last"),
                    "open": metadata.get("open"),
                    "high": metadata.get("high"),
                    "low": metadata.get("low"),
                    "prev_close": metadata.get("previousClose"),
                    "change": metadata.get("change"),
                    "pct_change": metadata.get("percChange"),
                    "timestamp": metadata.get("timeVal"),
                }
            return None
            
        except Exception as e:
            logger.error(f"Error fetching index {index}: {e}")
            return None


# Utility functions for easy use
async def fetch_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Convenience function to fetch stock data.
    
    Args:
        symbol: NSE symbol
        days: Number of days of history
        
    Returns:
        DataFrame with OHLC data
    """
    fetcher = NSEIndiaFetcher()
    try:
        start_date = date.today() - timedelta(days=days)
        return await fetcher.get_historical_data(symbol, start_date)
    finally:
        await fetcher.close()


async def fetch_multiple_stocks(symbols: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Fetch data for multiple stocks with rate limiting.
    
    Args:
        symbols: List of NSE symbols
        days: Number of days of history
        
    Returns:
        Dict of symbol -> DataFrame
    """
    fetcher = NSEIndiaFetcher()
    results = {}
    start_date = date.today() - timedelta(days=days)
    
    try:
        for i, symbol in enumerate(symbols):
            logger.info(f"[{i+1}/{len(symbols)}] Fetching {symbol}")
            df = await fetcher.get_historical_data(symbol, start_date)
            
            if not df.empty:
                results[symbol] = df
            
            # Rate limiting: 1 request per second
            await asyncio.sleep(1)
            
    finally:
        await fetcher.close()
    
    return results
