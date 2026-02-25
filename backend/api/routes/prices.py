"""
Nifty 500 AI — Prices API Routes

GET /api/prices/{symbol}?days=90&interval=1d
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from database.db import get_prices

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/prices/{symbol}")
async def get_price_history(
    symbol: str,
    days: int = Query(default=90, ge=1, le=3650, description="Number of days to look back"),
    interval: str = Query(default="1d", description="Data interval: 1d, 1h, 5m"),
):
    """
    Get OHLCV price history for a stock.

    Args:
        symbol: Stock symbol (e.g. "TCS.NS")
        days: Number of days of history (default 90, max 3650)
        interval: Data interval — "1d", "1h", "5m"

    Returns:
        List of {date, open, high, low, close, volume} objects.
    """
    try:
        prices = get_prices(symbol, days=days, interval=interval)

        if not prices:
            raise HTTPException(
                status_code=404,
                detail=f"No price data found for {symbol}. Run collector first.",
            )

        return {
            "symbol": symbol,
            "interval": interval,
            "count": len(prices),
            "data": prices,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching prices for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")
