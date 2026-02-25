"""
Nifty 500 AI — FastAPI Server

Main application server with CORS, error handling, and response caching.
Serves stock data, indicators, sentiment, and signals via REST API.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python main.py server
"""

import logging
import os
import time
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import prices, indicators, sentiment, signals
from database.db import init_database

load_dotenv()
logger = logging.getLogger(__name__)

# ==========================================
# Create FastAPI app
# ==========================================
app = FastAPI(
    title="Nifty 500 AI Trading API",
    description=(
        "AI-powered stock market data pipeline for the Nifty 500 index.\n\n"
        "Provides real-time prices, technical indicators, news sentiment,\n"
        "and AI-generated trading signals for Indian stocks.\n\n"
        "**Disclaimer**: This is not financial advice. Always do your own research."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ==========================================
# CORS Middleware — allow all origins for dashboard
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Include route modules
# ==========================================
app.include_router(prices.router, prefix="/api", tags=["Prices"])
app.include_router(indicators.router, prefix="/api", tags=["Indicators"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment"])
app.include_router(signals.router, prefix="/api", tags=["Signals"])

from api.routes import portfolio as portfolio_routes
from api.routes import trades as trades_routes
app.include_router(portfolio_routes.router)
app.include_router(trades_routes.router)


# ==========================================
# Simple in-memory cache
# ==========================================
_cache: Dict[str, Dict[str, Any]] = {}

# Cache TTLs in seconds
CACHE_TTLS = {
    "prices": 60,       # 1 minute
    "indicators": 300,  # 5 minutes
    "sentiment": 600,   # 10 minutes
    "overview": 120,    # 2 minutes
    "signals": 300,     # 5 minutes
}


def get_cached(key: str, category: str = "prices"):
    """Get a cached value if it hasn't expired."""
    if key in _cache:
        entry = _cache[key]
        ttl = CACHE_TTLS.get(category, 60)
        if time.time() - entry["timestamp"] < ttl:
            return entry["data"]
    return None


def set_cached(key: str, data: Any, category: str = "prices"):
    """Set a cached value with timestamp."""
    _cache[key] = {"data": data, "timestamp": time.time()}


# ==========================================
# Startup / Shutdown events
# ==========================================
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    try:
        init_database()
        logger.info("Database initialized on startup")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


# ==========================================
# Health Check Endpoint
# ==========================================
@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns server status, whether market is open, and timestamp.
    """
    now = datetime.now()

    # IST market hours: 9:15 AM to 3:30 PM, Monday-Friday
    ist_hour = now.hour
    ist_minute = now.minute
    is_weekday = now.weekday() < 5  # Mon-Fri

    market_open = (
        is_weekday
        and (ist_hour > 9 or (ist_hour == 9 and ist_minute >= 15))
        and (ist_hour < 15 or (ist_hour == 15 and ist_minute <= 30))
    )

    return {
        "status": "ok",
        "market_open": market_open,
        "timestamp": now.isoformat(),
        "version": "1.0.0",
    }


# ==========================================
# Market Overview Endpoint
# ==========================================
@app.get("/api/market/overview", tags=["Market"])
async def market_overview():
    """
    Get today's market overview.

    Returns Nifty 50/500, Sensex, VIX, advances/declines, FII/DII flows.
    """
    from database.db import get_market_overview

    # Check cache first
    cached = get_cached("market_overview", "overview")
    if cached:
        return cached

    overview_list = get_market_overview(days=1)
    if overview_list:
        data = overview_list[0]
        set_cached("market_overview", data, "overview")
        return data
    else:
        return {"message": "No market overview data available. Run collector first."}


# ==========================================
# Watchlist Endpoint (combined data)
# ==========================================
@app.get("/api/watchlist/{symbol}", tags=["Watchlist"])
async def watchlist(symbol: str):
    """
    Get combined data for a stock: price + indicators + sentiment + signal.

    This is a convenience endpoint that aggregates data from multiple tables.
    """
    from database.db import get_prices, get_latest_indicators, get_recent_news

    cache_key = f"watchlist_{symbol}"
    cached = get_cached(cache_key, "prices")
    if cached:
        return cached

    # Get latest price data
    price_data = get_prices(symbol, days=5)
    latest_price = price_data[-1] if price_data else None

    # Get indicators
    indicator_data = get_latest_indicators(symbol)

    # Get news (stock-specific)
    news = get_recent_news(limit=5, symbol=symbol)

    result = {
        "symbol": symbol,
        "latest_price": latest_price,
        "indicators": indicator_data,
        "news": news,
        "signal": indicator_data.get("signal") if indicator_data else None,
        "signal_strength": indicator_data.get("signal_strength") if indicator_data else None,
    }

    set_cached(cache_key, result, "prices")
    return result


# ==========================================
# Heatmap Endpoint
# ==========================================
@app.get("/api/heatmap/sectors", tags=["Market"])
async def heatmap_sectors():
    """
    Returns sector-wise performance data for the heatmap visualization.
    """
    from data.stocks_list import NIFTY_50_STOCKS, get_all_sectors
    from database.db import get_prices

    cached = get_cached("heatmap_sectors", "overview")
    if cached:
        return cached

    sectors = {}

    for stock in NIFTY_50_STOCKS:
        sector = stock["sector"]
        symbol = stock["symbol"]

        if sector not in sectors:
            sectors[sector] = {"stocks": [], "total_change": 0, "count": 0}

        # Get last 2 days of price data
        prices = get_prices(symbol, days=5)
        if len(prices) >= 2:
            prev_close = prices[-2]["close"]
            curr_close = prices[-1]["close"]
            change_pct = ((curr_close - prev_close) / prev_close) * 100

            sectors[sector]["stocks"].append({
                "symbol": symbol,
                "name": stock["name"],
                "close": curr_close,
                "change_pct": round(change_pct, 2),
            })
            sectors[sector]["total_change"] += change_pct
            sectors[sector]["count"] += 1

    # Calculate sector averages
    result = []
    for sector_name, data in sectors.items():
        avg_change = data["total_change"] / data["count"] if data["count"] > 0 else 0
        result.append({
            "sector": sector_name,
            "avg_change_pct": round(avg_change, 2),
            "stock_count": data["count"],
            "stocks": sorted(data["stocks"], key=lambda x: x["change_pct"], reverse=True),
        })

    result.sort(key=lambda x: x["avg_change_pct"], reverse=True)

    set_cached("heatmap_sectors", result, "overview")
    return result


# ==========================================
# Global Error Handler
# ==========================================
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions and return a clean error response.
    """
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )


# ==========================================
# Root endpoint
# ==========================================
@app.get("/", tags=["Root"])
async def root():
    """API root — shows available endpoints."""
    return {
        "name": "Nifty 500 AI Trading API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "market_overview": "/api/market/overview",
            "prices": "/api/prices/{symbol}?days=90&interval=1d",
            "indicators": "/api/indicators/{symbol}",
            "sentiment_market": "/api/sentiment/market",
            "sentiment_stock": "/api/sentiment/{symbol}",
            "top_buys": "/api/signals/top-buys?limit=10",
            "top_sells": "/api/signals/top-sells?limit=10",
            "watchlist": "/api/watchlist/{symbol}",
            "heatmap": "/api/heatmap/sectors",
        },
        "docs": "/docs",
    }
