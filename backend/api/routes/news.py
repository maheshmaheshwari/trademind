"""
TradeMind AI — User-Wise News & Sentiment Routes

GET /api/news/watchlist/{user_id}        — news feed for user's watchlist stocks
GET /api/news/watchlist/{user_id}/summary — sentiment summary per watchlist stock
GET /api/news/stock/{symbol}             — recent news for a specific stock
GET /api/news/market                     — market-wide news (symbol=NULL)
GET /api/signals/history/{user_id}      — AI signals the user has acted on
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from database.db import (
    get_news_for_user_watchlist,
    get_news_summary_for_user,
    get_recent_news,
    get_user_signal_history,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/news", tags=["News"])
signals_router = APIRouter(prefix="/api/signals", tags=["Signals"])


# ── Per-user news feed ────────────────────────────────────────────────────────

@router.get("/watchlist/{user_id}")
async def user_watchlist_news(
    user_id: int,
    limit: int = Query(default=50, le=200),
):
    """
    News feed for a user — articles for all stocks in their watchlist
    plus market-wide news, sorted by most recent.
    """
    try:
        news = get_news_for_user_watchlist(user_id, limit=limit)
        return {"data": news, "total": len(news), "user_id": user_id}
    except Exception as e:
        logger.error(f"user_watchlist_news {user_id}: {e}")
        return {"data": [], "total": 0, "user_id": user_id}


@router.get("/watchlist/{user_id}/summary")
async def user_watchlist_sentiment_summary(user_id: int):
    """
    Sentiment summary for a user's watchlist — per-stock avg sentiment
    over last 7 days + overall portfolio sentiment.
    """
    try:
        summary = get_news_summary_for_user(user_id)
        return {**summary, "user_id": user_id}
    except Exception as e:
        logger.error(f"user_watchlist_sentiment_summary {user_id}: {e}")
        return {"per_stock": [], "overall": {"total_articles": 0, "portfolio_sentiment": 0.0}, "user_id": user_id}


# ── Stock-level news ──────────────────────────────────────────────────────────

@router.get("/stock/{symbol}")
async def stock_news(
    symbol: str,
    limit: int = Query(default=20, le=100),
):
    """Recent news articles for a specific stock with sentiment scores."""
    try:
        news = get_recent_news(limit=limit, symbol=symbol)
        pos = sum(1 for n in news if float(n.get("sentiment") or 0) > 0)
        neg = sum(1 for n in news if float(n.get("sentiment") or 0) < 0)
        avg_sent = (sum(float(n.get("sentiment") or 0) for n in news) / len(news)) if news else 0
        return {
            "symbol":          symbol,
            "article_count":   len(news),
            "avg_sentiment":   round(avg_sent, 4),
            "positive_count":  pos,
            "negative_count":  neg,
            "neutral_count":   len(news) - pos - neg,
            "data":            news,
        }
    except Exception as e:
        logger.error(f"stock_news {symbol}: {e}")
        return {"symbol": symbol, "article_count": 0, "data": []}


# ── Market-wide news ──────────────────────────────────────────────────────────

@router.get("/market")
async def market_news(limit: int = Query(default=30, le=100)):
    """Market-wide news articles (not stock-specific)."""
    try:
        news = get_recent_news(limit=limit, symbol=None)
        # Filter to only market-wide (symbol IS NULL)
        market = [n for n in news if not n.get("symbol")]
        return {"data": market, "total": len(market)}
    except Exception as e:
        logger.error(f"market_news: {e}")
        return {"data": [], "total": 0}


# ── User-wise AI signal history ───────────────────────────────────────────────

@signals_router.get("/history/{user_id}")
async def user_signal_history(
    user_id: int,
    limit: int = Query(default=50, le=200),
):
    """
    AI trade signals this user has acted on.
    Shows which AI signals (with is_active status) the user executed,
    when they traded, at what price, and the current signal status.
    """
    try:
        history = get_user_signal_history(user_id, limit=limit)
        return {
            "data":    history,
            "total":   len(history),
            "user_id": user_id,
        }
    except Exception as e:
        logger.error(f"user_signal_history {user_id}: {e}")
        return {"data": [], "total": 0, "user_id": user_id}
