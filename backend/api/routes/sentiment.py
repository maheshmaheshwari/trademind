"""
Nifty 500 AI — Sentiment API Routes

GET /api/sentiment/market
GET /api/sentiment/{symbol}
"""

import logging

from fastapi import APIRouter, HTTPException

from database.db import get_recent_news

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/sentiment/market")
async def market_sentiment():
    """
    Get overall market sentiment score and recent news.

    Returns Fear & Greed score (0-100), label, and last 20 news articles
    with their sentiment classifications.
    """
    try:
        # Get recent news from database
        news = get_recent_news(limit=20)

        if not news:
            return {
                "score": 50.0,
                "label": "Neutral",
                "article_count": 0,
                "news": [],
                "message": "No news data available. Run news collector first.",
            }

        # Calculate aggregate sentiment
        from analysis.sentiment import aggregate_sentiment
        agg = aggregate_sentiment(news)

        return {
            "score": agg["score"],
            "label": agg["label"],
            "article_count": agg["article_count"],
            "breakdown": agg["breakdown"],
            "news": news[:20],
        }

    except Exception as e:
        logger.error(f"Error fetching market sentiment: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/sentiment/{symbol}")
async def stock_sentiment(symbol: str):
    """
    Get sentiment analysis for a specific stock.

    Returns news articles about this stock with their sentiment scores.

    Args:
        symbol: Stock symbol (e.g. "TCS.NS")
    """
    try:
        news = get_recent_news(limit=20, symbol=symbol)

        if not news:
            return {
                "symbol": symbol,
                "score": 50.0,
                "label": "Neutral",
                "article_count": 0,
                "news": [],
                "message": f"No news found for {symbol}.",
            }

        from analysis.sentiment import aggregate_sentiment
        agg = aggregate_sentiment(news)

        return {
            "symbol": symbol,
            "score": agg["score"],
            "label": agg["label"],
            "article_count": agg["article_count"],
            "breakdown": agg["breakdown"],
            "news": news,
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
