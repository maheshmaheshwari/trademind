"""
Nifty 500 AI — Indicators API Routes

GET /api/indicators/{symbol}
"""

import logging

from fastapi import APIRouter, HTTPException

from database.db import get_latest_indicators

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/indicators/{symbol}")
async def get_indicators(symbol: str):
    """
    Get the latest technical indicators for a stock.

    Returns RSI, MACD, Bollinger Bands, SMA, EMA, ATR, ADX,
    Stochastic, OBV, support/resistance levels, and the trading signal.

    Args:
        symbol: Stock symbol (e.g. "TCS.NS")

    Returns:
        Dict with all indicator values + signal + signal_strength.
    """
    try:
        data = get_latest_indicators(symbol)

        if not data:
            raise HTTPException(
                status_code=404,
                detail=f"No indicators found for {symbol}. Run: python main.py collect",
            )

        return {
            "symbol": symbol,
            "date": data.get("date"),
            "indicators": {
                "rsi_14": data.get("rsi_14"),
                "macd": data.get("macd"),
                "macd_signal": data.get("macd_signal"),
                "macd_hist": data.get("macd_hist"),
                "bb_upper": data.get("bb_upper"),
                "bb_middle": data.get("bb_middle"),
                "bb_lower": data.get("bb_lower"),
                "sma_20": data.get("sma_20"),
                "sma_50": data.get("sma_50"),
                "sma_200": data.get("sma_200"),
                "ema_9": data.get("ema_9"),
                "ema_21": data.get("ema_21"),
                "atr_14": data.get("atr_14"),
                "adx_14": data.get("adx_14"),
                "stoch_k": data.get("stoch_k"),
                "stoch_d": data.get("stoch_d"),
                "obv": data.get("obv"),
            },
            "support_resistance": {
                "support_1": data.get("support_1"),
                "support_2": data.get("support_2"),
                "support_3": data.get("support_3"),
                "resistance_1": data.get("resistance_1"),
                "resistance_2": data.get("resistance_2"),
                "resistance_3": data.get("resistance_3"),
            },
            "signal": data.get("signal"),
            "signal_strength": data.get("signal_strength"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching indicators for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
