"""
Nifty 500 AI — Signals API Routes

GET /api/signals/top-buys?limit=10
GET /api/signals/top-sells?limit=10
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from database.db import get_top_signals

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/signals/top-buys")
async def top_buys(
    limit: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Get top stocks with the strongest BUY signal today.

    Returns stocks sorted by signal confidence, including both
    "BUY" and "STRONG BUY" signals.

    Args:
        limit: Maximum number of results (default 10, max 50)
    """
    try:
        signals = get_top_signals(signal_type="BUY", limit=limit)

        if not signals:
            return {
                "count": 0,
                "signals": [],
                "message": "No BUY signals found. Run: python main.py collect",
            }

        return {
            "count": len(signals),
            "signals": signals,
        }

    except Exception as e:
        logger.error(f"Error fetching top buys: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/signals/top-sells")
async def top_sells(
    limit: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Get top stocks with the strongest SELL signal today.

    Returns stocks sorted by signal confidence, including both
    "SELL" and "STRONG SELL" signals.

    Args:
        limit: Maximum number of results (default 10, max 50)
    """
    try:
        signals = get_top_signals(signal_type="SELL", limit=limit)

        if not signals:
            return {
                "count": 0,
                "signals": [],
                "message": "No SELL signals found. Run: python main.py collect",
            }

        return {
            "count": len(signals),
            "signals": signals,
        }

    except Exception as e:
        logger.error(f"Error fetching top sells: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
