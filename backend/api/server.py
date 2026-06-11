"""
Nifty 500 AI — FastAPI Server

Main application server with CORS, error handling, and response caching.
Serves stock data, indicators, sentiment, and signals via REST API.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

Or via CLI:
    python main.py server
"""

import asyncio
import calendar
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from typing import Any, Dict
from zoneinfo import ZoneInfo

# Initialise date-rotating file logging as early as possible.
# Called here (module level) so it runs whether uvicorn imports us directly
# OR via main.py.  The startup event below calls it again as a fallback for
# the case where uvicorn's dictConfig() fires AFTER this import and wipes
# our handlers.
from api.logging_setup import setup_logging as _setup_logging
_LOG_DIR   = "logs"
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
_setup_logging(log_dir=_LOG_DIR, level=_LOG_LEVEL)

# B4: Thread pool for offloading synchronous psycopg2 calls from the async event loop.
# All blocking DB calls inside async handlers should use:
#   await run_in_thread(some_sync_db_function, arg1, arg2)
_DB_THREAD_POOL = ThreadPoolExecutor(max_workers=10, thread_name_prefix="db-worker")

async def run_in_thread(func, *args, **kwargs):
    """Run a blocking function in the DB thread pool without blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_DB_THREAD_POOL, partial(func, *args, **kwargs))

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
# CORS Middleware
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Request / Response logging middleware
# ==========================================

_SKIP_LOG_PATHS = {"/api/health", "/docs", "/redoc", "/openapi.json"}
_SENSITIVE_FIELDS = {"password", "token", "secret", "password_hash", "totp"}

_req_logger = logging.getLogger("trademind.access")


def _mask(body_bytes: bytes) -> str:
    """Return a loggable JSON string with sensitive fields masked."""
    try:
        d = json.loads(body_bytes)
        for k in list(d.keys()):
            if any(s in k.lower() for s in _SENSITIVE_FIELDS):
                d[k] = "***"
        return json.dumps(d, ensure_ascii=False)[:500]
    except Exception:
        return body_bytes.decode(errors="replace")[:200]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.url.path in _SKIP_LOG_PATHS:
        return await call_next(request)

    start = time.perf_counter()
    method = request.method
    path = request.url.path
    query = request.url.query

    # Log request — include body for mutating methods
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        raw = await request.body()
        body_str = _mask(raw) if raw else ""
        _req_logger.info(
            f"→ {method} {path}{'?' + query if query else ''} "
            f"| body={body_str}"
        )
        # Rebuild the request so the route handler can still read the body
        from starlette.datastructures import Headers
        from starlette.requests import Request as StarletteRequest

        async def receive():
            return {"type": "http.request", "body": raw, "more_body": False}

        request = StarletteRequest(request.scope, receive=receive)
    else:
        _req_logger.info(
            f"→ {method} {path}{'?' + query if query else ''}"
        )

    # Call the actual route handler
    try:
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        _req_logger.log(
            level,
            f"← {method} {path} | {response.status_code} | {elapsed_ms:.1f}ms",
        )
        return response
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _req_logger.error(
            f"✗ {method} {path} | EXCEPTION | {elapsed_ms:.1f}ms | {exc}",
            exc_info=True,
        )
        raise


# ==========================================
# Include route modules
# ==========================================
app.include_router(prices.router, prefix="/api", tags=["Prices"])
app.include_router(indicators.router, prefix="/api", tags=["Indicators"])
app.include_router(sentiment.router, prefix="/api", tags=["Sentiment"])
app.include_router(signals.router, prefix="/api", tags=["Signals"])

from api.routes import stocks as stocks_routes
app.include_router(stocks_routes.router, prefix="/api", tags=["Stocks"])

from api.routes import portfolio as portfolio_routes
from api.routes import trades as trades_routes
from api.routes.trading import router as trading_router
from api.routes.watchlist import router as watchlist_router
from api.routes.notifications import router as notifications_router
from api.routes.orders import router as orders_router
from api.routes.news import router as news_router, signals_router as user_signals_router
from api.routes.autopilot import router as autopilot_router
from api.routes.backtest import router as backtest_router
app.include_router(portfolio_routes.router)
app.include_router(trades_routes.router)
app.include_router(trading_router)
app.include_router(watchlist_router)
app.include_router(notifications_router)
app.include_router(orders_router)
app.include_router(news_router)
app.include_router(user_signals_router)
app.include_router(autopilot_router)
app.include_router(backtest_router)


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
    """Initialize database and start background scheduler on startup."""
    _setup_logging(log_dir=_LOG_DIR, level=_LOG_LEVEL)

    # With --workers N, each worker process runs this event independently.
    # Use the worker's OS PID to elect exactly one scheduler owner so cron
    # jobs don't fire N times per interval.
    import os
    worker_pid = os.getpid()

    # Write own PID to a lock file; the lowest-PID worker wins the election.
    _lock_path = os.path.join("logs", ".scheduler_owner.pid")
    os.makedirs("logs", exist_ok=True)
    is_scheduler_worker = False
    try:
        # Atomic O_EXCL create — only ONE worker succeeds, rest get FileExistsError
        try:
            fd = os.open(_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(worker_pid).encode())
            os.close(fd)
            is_scheduler_worker = True   # we created the file — we own the scheduler
        except FileExistsError:
            # File already exists — check if the owner is still alive
            try:
                existing = int(open(_lock_path).read().strip())
                os.kill(existing, 0)   # signal 0 = alive check
                is_scheduler_worker = False   # owner alive
            except (ProcessLookupError, PermissionError):
                # Owner dead — take over atomically
                os.remove(_lock_path)
                fd = os.open(_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(worker_pid).encode())
                os.close(fd)
                is_scheduler_worker = True
    except Exception:
        is_scheduler_worker = False   # safe default — don't double-run

    # Only the scheduler worker runs DB init to avoid concurrent-update races
    # when all 4 workers start simultaneously.
    if is_scheduler_worker:
        try:
            init_database()
            logger.info("Database initialized (worker %d)", worker_pid)
        except Exception as e:
            logger.error("Database initialization failed: %s", e)

    if is_scheduler_worker:
        try:
            from scheduler.jobs import start_background_scheduler
            sched = start_background_scheduler()
            if sched:
                logger.info("✅ Background scheduler started (worker %d)", worker_pid)
        except Exception as e:
            logger.error("Background scheduler failed to start: %s", e)

        # Run missed-job recovery queue 30s after startup (FIFO, single worker)
        import asyncio

        async def _recovery_task():
            await asyncio.sleep(30)
            try:
                from scheduler.jobs import run_recovery_queue
                logger.info("🔄 Running missed-job recovery queue...")
                run_recovery_queue()
            except Exception as exc:
                logger.error("Recovery queue error: %s", exc)

        asyncio.create_task(_recovery_task())
    else:
        logger.info("Scheduler already running in another worker — skipping (worker %d)", worker_pid)


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background scheduler on shutdown."""
    try:
        from scheduler.jobs import stop_background_scheduler
        stop_background_scheduler()
        logger.info("Background scheduler stopped")
    except Exception as e:
        logger.error(f"Error stopping scheduler: {e}")


# ==========================================
# Health Check Endpoint
# ==========================================
@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns server status, whether market is open, and timestamp.
    """
    now = datetime.now(tz=ZoneInfo("Asia/Kolkata"))

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
    Returns structured market overview: indices, breadth, FII/DII history,
    VIX, top gainers, top losers, and sector heatmap.
    """
    cached = get_cached("market_overview_v2", "overview")
    if cached:
        return cached

    from database.db import get_market_overview, get_top_signals, _execute, _rows_to_dicts, get_connection

    rows = get_market_overview(days=14)

    # Coalesce: for each field, find the most recent row that has a non-null value
    def _coalesce(field, rows):
        for r in rows:
            v = r.get(field)
            if v is not None:
                return v, r
        return None, {}

    nifty50_val,  nifty50_row  = _coalesce("nifty50_close",  rows)
    nifty500_val, nifty500_row = _coalesce("nifty500_close", rows)
    sensex_val,   sensex_row   = _coalesce("sensex_close",   rows)
    vix_val,      vix_row      = _coalesce("india_vix",      rows)
    advances_val, breadth_row  = _coalesce("advances",       rows)

    # For "previous day" comparison use the next row that has a value for that field
    def _prev_val(field, current_row, rows):
        found_current = False
        for r in rows:
            if r is current_row:
                found_current = True
                continue
            if found_current and r.get(field) is not None:
                return r.get(field)
        return None

    # ── Fetch live index data + spark from yfinance ───────────────────────────
    INDEX_MAP = [
        ("NIFTY 50",  "^NSEI",     nifty50_val,  "nifty50_close"),
        ("NIFTY 500", "^CRSLDX",   nifty500_val, "nifty500_close"),
        ("SENSEX",    "^BSESN",    sensex_val,   "sensex_close"),
        ("INDIA VIX", "^INDIAVIX", vix_val,      "india_vix"),
    ]

    def _fetch_index_spark(ticker: str):
        """Return (latest_close, prev_close, intraday_spark[]) for one index."""
        try:
            import yfinance as yf

            def _squeeze_close(df):
                """Extract Close as a flat Series regardless of column level."""
                if df.empty:
                    return None
                c = df["Close"]
                # Multi-level columns → squeeze to Series
                if hasattr(c, "squeeze"):
                    c = c.squeeze()
                return c.dropna()

            # Intraday 5-min candles for the main chart spark
            intra = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True)
            spark = []
            if not intra.empty:
                closes = _squeeze_close(intra)
                if closes is not None and len(closes):
                    spark = [round(float(v), 2) for v in closes.values]

            # Daily for current + previous close
            daily = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=True)
            latest = prev = None
            if not daily.empty:
                closes = _squeeze_close(daily)
                if closes is not None and len(closes) >= 2:
                    latest = round(float(closes.iloc[-1]), 2)
                    prev   = round(float(closes.iloc[-2]), 2)
                elif closes is not None and len(closes) == 1:
                    latest = round(float(closes.iloc[-1]), 2)
            return latest, prev, spark
        except Exception:
            return None, None, []

    indices = []
    for name, ticker, db_val, db_field in INDEX_MAP:
        live_val, live_prev, spark = _fetch_index_spark(ticker)
        val  = live_val  if live_val  is not None else db_val
        prev = live_prev if live_prev is not None else _prev_val(db_field, nifty50_row, rows)
        if val is None:
            continue
        change = round(val - (prev or val), 2)
        pct    = round(change / (prev or val) * 100, 2) if prev else 0.0
        # For NIFTY 50 specifically update the live values for vix fix
        if name == "NIFTY 50":
            nifty50_val = val
        elif name == "INDIA VIX":
            vix_val = val
        indices.append({"name": name, "value": val, "change": change, "pct": pct, "spark": spark})

    # ── Breadth — derive from prices table if market_overview lacks it ────────
    db_advances  = breadth_row.get("advances")
    db_declines  = breadth_row.get("declines")
    if db_advances is not None and db_declines is not None:
        breadth = {
            "advances":  db_advances  or 0,
            "declines":  db_declines  or 0,
            "unchanged": breadth_row.get("unchanged") or 0,
        }
    else:
        try:
            conn = get_connection()
            from database.db import release_connection
            cur = _execute(conn, """
                SELECT
                    SUM(CASE WHEN close > open THEN 1 ELSE 0 END) AS advances,
                    SUM(CASE WHEN close < open THEN 1 ELSE 0 END) AS declines,
                    SUM(CASE WHEN close = open THEN 1 ELSE 0 END) AS unchanged
                FROM prices
                WHERE date = (SELECT MAX(date) FROM prices)
            """, ())
            r = _rows_to_dicts(cur)
            release_connection(conn)
            b = r[0] if r else {}
            breadth = {
                "advances":  b.get("advances")  or 0,
                "declines":  b.get("declines")  or 0,
                "unchanged": b.get("unchanged") or 0,
            }
        except Exception:
            breadth = {"advances": 0, "declines": 0, "unchanged": 0}

    # Use the most recent row that has index/sentiment data
    today = nifty50_row or vix_row or (rows[0] if rows else {})

    # ── FII/DII — last 5 trading days ─────────────────────────────────────────
    fii_dii = []
    for r in reversed(rows[:5]):
        d = r.get("date")
        day_label = calendar.day_abbr[d.weekday()] if hasattr(d, "weekday") else str(d)
        fii_dii.append({
            "day": day_label,
            "fii": round(r.get("fii_net") or 0, 2),
            "dii": round(r.get("dii_net") or 0, 2),
        })

    # ── Gainers / Losers from latest trade signals ────────────────────────────
    def _sig_to_stock(s):
        return {
            "symbol":    s.get("symbol", ""),
            "name":      s.get("name", ""),
            "sector":    "",
            "price":     s.get("current_price") or s.get("buy_price") or 0,
            "change":    round(s.get("expected_return_pct") or 0, 2),
            "signal":    s.get("signal", ""),
            "confidence": round(s.get("confidence") or 0, 4),
        }

    gainers = [_sig_to_stock(s) for s in get_top_signals("BUY",  limit=5)]
    losers  = [_sig_to_stock(s) for s in get_top_signals("SELL", limit=5)]

    # ── Sector heatmap (reuse cached result from /api/market/sectors) ─────────
    heatmap = get_cached("market_sectors", "overview") or []

    sentiment_val, _ = _coalesce("overall_sentiment_score", rows)
    fear_greed_val, _ = _coalesce("fear_greed_label", rows)
    result = {
        "indices":  indices,
        "breadth":  breadth,
        "fii_dii":  fii_dii,
        "vix":      vix_val or 0,
        "gainers":  gainers,
        "losers":   losers,
        "heatmap":  heatmap,
        "sentiment_score": sentiment_val,
        "sentiment":       sentiment_val or 50,
        "fear_greed":      fear_greed_val,
    }

    set_cached("market_overview_v2", result, "overview")
    return result


# ==========================================
# Sector Performance Endpoint
# ==========================================
@app.get("/api/market/sectors", tags=["Market"])
async def market_sectors():
    """
    Returns today's sector performance aggregated from latest trade signals.
    Uses Nifty 500 sector mapping — covers all 500 stocks.
    Response shape per sector: { sector, change, signal_dist, stock_count, stocks }
    """
    cached = get_cached("market_sectors", "overview")
    if cached:
        return cached

    from database.db import get_trade_signals_formatted
    from data.nifty500_full import NIFTY_500_STOCKS

    # Build symbol → sector lookup
    sym_to_sector = {s["symbol"]: s["sector"] for s in NIFTY_500_STOCKS}

    raw = get_trade_signals_formatted()
    all_signals = raw.get("actionable_trades", []) + raw.get("hold_list", []) + raw.get("avoid_list", [])

    sectors: dict = {}
    for sig in all_signals:
        symbol  = sig.get("symbol", "")
        sector  = sym_to_sector.get(symbol, "Other")
        signal  = sig.get("signal", "HOLD")
        conf    = sig.get("confidence") or 0
        exp_ret = (sig.get("trade") or {}).get("expected_return_pct") or 0
        price   = (sig.get("price") or {}).get("current") or 0

        if sector not in sectors:
            sectors[sector] = {
                "sector": sector,
                "total_exp_ret": 0.0,
                "total_conf": 0.0,
                "count": 0,
                "buy": 0, "sell": 0, "hold": 0,
                "stocks": [],
            }

        g = sectors[sector]
        g["total_exp_ret"] += exp_ret
        g["total_conf"]    += conf
        g["count"]         += 1
        if "BUY"  in signal: g["buy"]  += 1
        elif "SELL" in signal: g["sell"] += 1
        else: g["hold"] += 1

        g["stocks"].append({
            "symbol":     symbol,
            "name":       sig.get("name", ""),
            "price":      round(price, 2),
            "change":     round(exp_ret, 2),
            "signal":     signal,
            "confidence": round(conf, 4),
        })

    result = []
    for g in sectors.values():
        n = g["count"] or 1
        result.append({
            "sector":      g["sector"],
            "change":      round(g["total_exp_ret"] / n, 2),
            "avg_conf":    round(g["total_conf"] / n, 4),
            "stock_count": g["count"],
            "buy_count":   g["buy"],
            "sell_count":  g["sell"],
            "hold_count":  g["hold"],
            "stocks":      sorted(g["stocks"], key=lambda x: x["change"], reverse=True)[:10],
        })

    result.sort(key=lambda x: x["change"], reverse=True)

    set_cached("market_sectors", result, "overview")
    # also warm the overview cache's heatmap slice
    overview_cached = get_cached("market_overview_v2", "overview")
    if overview_cached:
        overview_cached["heatmap"] = result
        set_cached("market_overview_v2", overview_cached, "overview")

    return result


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
    detail = str(exc) if os.getenv("DEBUG", "").lower() == "true" else "An internal error occurred"
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": detail,
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
