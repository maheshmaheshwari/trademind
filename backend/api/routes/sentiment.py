"""
Nifty 500 AI — Sentiment API Routes

GET /api/sentiment/market
GET /api/sentiment/health
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
        raise HTTPException(status_code=500, detail="Error fetching market sentiment")


@router.get("/sentiment/health")
async def sentiment_health():
    """
    Sentiment pipeline health dashboard.

    Returns collector throughput, coverage metrics, scoring backlog,
    and market_overview completeness — all in one call.
    """
    try:
        from database.db import get_connection, release_connection, _execute, _rows_to_dicts

        conn = get_connection()
        try:
            # --- Scoring backlog ---
            backlog = _execute(conn,
                "SELECT COUNT(*) FROM news_sentiment WHERE sentiment IS NULL"
            ).fetchone()[0]

            # --- Articles last 24h by source ---
            cur = _execute(conn, """
                SELECT source, COUNT(*) as cnt
                FROM news_sentiment
                WHERE published_at >= NOW() - INTERVAL '24 hours'
                GROUP BY source ORDER BY cnt DESC
            """)
            sources_24h = {r[0]: r[1] for r in cur.fetchall()}

            # --- 7-day stock coverage ---
            total_stocks = _execute(conn,
                "SELECT COUNT(DISTINCT symbol) FROM prices WHERE interval='1d' AND symbol LIKE '%.NS'"
            ).fetchone()[0]

            covered_7d = _execute(conn, """
                SELECT COUNT(DISTINCT symbol) FROM news_sentiment
                WHERE published_at >= NOW() - INTERVAL '7 days'
                  AND symbol IS NOT NULL AND symbol != ''
            """).fetchone()[0]

            missing_7d = _execute(conn, """
                SELECT COUNT(DISTINCT p.symbol)
                FROM prices p
                WHERE p.interval='1d' AND p.symbol LIKE '%.NS'
                  AND p.symbol NOT IN (
                      SELECT DISTINCT symbol FROM news_sentiment
                      WHERE published_at >= NOW() - INTERVAL '7 days'
                        AND symbol IS NOT NULL AND symbol != ''
                  )
            """).fetchone()[0]

            # --- market_overview completeness (last 30 days) ---
            mo = _execute(conn, """
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN nifty50_close  IS NOT NULL THEN 1 ELSE 0 END) as has_n50,
                       SUM(CASE WHEN nifty500_close IS NOT NULL THEN 1 ELSE 0 END) as has_n500,
                       SUM(CASE WHEN india_vix      IS NOT NULL THEN 1 ELSE 0 END) as has_vix,
                       SUM(CASE WHEN fii_net        IS NOT NULL THEN 1 ELSE 0 END) as has_fii,
                       MAX(date) as latest_date
                FROM market_overview
                WHERE date >= NOW() - INTERVAL '30 days'
            """).fetchone()

            # --- Collector job health (last 7 days) ---
            cur = _execute(conn, """
                SELECT job_name,
                       SUM(CASE WHEN status='done'   THEN 1 ELSE 0 END) as done,
                       SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) as failed,
                       MAX(completed_at) as last_run
                FROM scheduler_log
                WHERE scheduled_at >= NOW() - INTERVAL '7 days'
                  AND (job_name ILIKE '%news%' OR job_name ILIKE '%sentiment%' OR job_name ILIKE '%index%')
                GROUP BY job_name ORDER BY job_name
            """)
            collector_health = _rows_to_dicts(cur)

        finally:
            release_connection(conn)

        total_24h = sum(sources_24h.values())
        mo_total  = mo[0] or 0

        return {
            "scoring": {
                "backlog_unscored":      backlog,
                "articles_last_24h":     total_24h,
                "by_source_24h":         sources_24h,
            },
            "coverage": {
                "total_stocks":          total_stocks,
                "covered_7d":            covered_7d,
                "missing_7d":            missing_7d,
                "coverage_pct":          round(covered_7d / total_stocks * 100, 1) if total_stocks else 0,
            },
            "market_overview": {
                "records_last_30d":      mo_total,
                "nifty50_filled_pct":    round((mo[1] or 0) / mo_total * 100, 1) if mo_total else 0,
                "nifty500_filled_pct":   round((mo[2] or 0) / mo_total * 100, 1) if mo_total else 0,
                "vix_filled_pct":        round((mo[3] or 0) / mo_total * 100, 1) if mo_total else 0,
                "fii_filled_pct":        round((mo[4] or 0) / mo_total * 100, 1) if mo_total else 0,
                "latest_date":           str(mo[5]) if mo[5] else None,
            },
            "collector_health":          collector_health,
        }

    except Exception as e:
        logger.error(f"sentiment_health error: {e}")
        raise HTTPException(status_code=500, detail="Error fetching sentiment health")


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
        raise HTTPException(status_code=500, detail="Error fetching sentiment")
