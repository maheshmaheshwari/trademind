"""
Nifty 500 AI — Database Connection & Helper Functions

Database setup, CRUD operations, and query helpers.
Uses libsql with Turso cloud sync (embedded replicas).
Falls back to local-only SQLite if Turso credentials are not set.

Usage:
    from database.db import init_database, get_connection, get_prices
    init_database()  # Creates tables if they don't exist
"""

import json
import logging
import os
import libsql_experimental as libsql
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from database.models import ALL_TABLES, CREATE_INDEXES

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Database file path (local embedded replica)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nifty500.db")

# Turso cloud credentials
TURSO_DATABASE_URL = os.getenv("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = os.getenv("TURSO_AUTH_TOKEN", "")

# Detect if Turso is configured
USE_TURSO = bool(TURSO_DATABASE_URL and TURSO_AUTH_TOKEN)

# Environment: 'local' writes to local DB only, 'production' syncs with Turso
ENV = os.getenv("ENV", "local").lower()


def get_connection(sync_on_connect: bool = True):
    """
    Get a database connection.

    If Turso credentials are configured in .env, connects to the cloud
    database with a local embedded replica (reads are instant, writes sync).
    Otherwise, uses a local-only SQLite file.

    Args:
        sync_on_connect: Whether to sync from cloud on connection.
            Set False during bulk inserts for better performance.

    Returns:
        Database connection (libsql). Compatible with sqlite3 API.

    Example:
        conn = get_connection()
        cursor = conn.execute("SELECT * FROM prices LIMIT 5")
        rows = cursor.fetchall()
        conn.close()
    """
    if USE_TURSO:
        conn = libsql.connect(
            DB_PATH,
            sync_url=TURSO_DATABASE_URL,
            auth_token=TURSO_AUTH_TOKEN,
        )
        if sync_on_connect:
            conn.sync()
    else:
        conn = libsql.connect(DB_PATH)
    return conn


def get_local_connection():
    """
    Get a local-only database connection (no Turso sync).
    Used when ENV=local to avoid syncing on every write.
    """
    return libsql.connect(DB_PATH)


def get_turso_connection(sync_on_connect: bool = True):
    """
    Get a Turso-synced database connection.
    Used for EOD sync to push local data to Turso cloud.
    """
    if not USE_TURSO:
        logger.warning("Turso not configured, returning local connection")
        return libsql.connect(DB_PATH)
    conn = libsql.connect(
        DB_PATH,
        sync_url=TURSO_DATABASE_URL,
        auth_token=TURSO_AUTH_TOKEN,
    )
    if sync_on_connect:
        conn.sync()
    return conn


def _rows_to_dicts(cursor) -> List[Dict]:
    """
    Convert cursor results to a list of dicts using cursor.description.
    libsql returns tuples, not sqlite3.Row objects.
    """
    rows = cursor.fetchall()
    if not rows or not cursor.description:
        return []
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


def _row_to_dict(cursor) -> Optional[Dict]:
    """
    Convert a single cursor result to a dict.
    Returns None if no row found.
    """
    row = cursor.fetchone()
    if not row or not cursor.description:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def init_database() -> None:
    """
    Initialize the database: create all tables and indexes if they don't exist.

    This is safe to call multiple times — uses CREATE TABLE IF NOT EXISTS.

    Example:
        from database.db import init_database
        init_database()
        print("Database ready!")
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()

        # Create all tables
        for table_sql in ALL_TABLES:
            cursor.execute(table_sql)

        # Create indexes
        for index_sql in CREATE_INDEXES:
            cursor.execute(index_sql)

        conn.commit()
        if USE_TURSO:
            conn.sync()
            logger.info(f"Database initialized — synced to Turso cloud")
            print(f"☁️  Database initialized — synced to Turso ({TURSO_DATABASE_URL})")
        else:
            logger.info(f"Database initialized at {DB_PATH}")
            print(f"✅ Database initialized at {DB_PATH} (local only)")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    finally:
        conn.close()


# ==========================================
# INSERT / UPSERT FUNCTIONS
# ==========================================


def insert_price(
    symbol: str,
    date: str,
    open_price: float,
    high: float,
    low: float,
    close: float,
    volume: int,
    interval: str = "1d",
    time_val: Optional[str] = None,
    exchange: str = "NSE",
) -> bool:
    """
    Insert a price row. Skips if duplicate (same symbol + date + time + interval).

    Args:
        symbol: Stock symbol, e.g. "TCS.NS"
        date: ISO date string, e.g. "2024-01-15"
        open_price: Opening price
        high: Day's high
        low: Day's low
        close: Closing price
        volume: Trading volume
        interval: Data interval ("1d", "1h", "5m")
        time_val: Time string for intraday, None for daily
        exchange: Exchange name, default "NSE"

    Returns:
        True if inserted, False if duplicate skipped.
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR IGNORE INTO prices
            (symbol, exchange, date, time, open, high, low, close, volume, interval)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, exchange, date, time_val, open_price, high, low, close, volume, interval),
        )
        conn.commit()
        if USE_TURSO:
            conn.sync()
        return True
    except Exception as e:
        logger.error(f"Error inserting price for {symbol} on {date}: {e}")
        return False
    finally:
        conn.close()


def insert_prices_batch(rows: List[Tuple], sync: bool = True) -> int:
    """
    Insert multiple price rows at once for better performance.
    Only syncs to Turso ONCE at the end (not per-row).

    Args:
        rows: List of tuples (symbol, exchange, date, time, open, high, low, close, volume, interval)
        sync: Whether to sync to Turso after inserting (set False for mid-pipeline calls)

    Returns:
        Number of rows inserted (duplicates skipped).
    """
    if not rows:
        return 0

    conn = get_connection(sync_on_connect=False)
    try:
        for row in rows:
            conn.execute(
                """INSERT OR IGNORE INTO prices
                (symbol, exchange, date, time, open, high, low, close, volume, interval)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                row,
            )
        conn.commit()
        if sync and USE_TURSO:
            conn.sync()
        logger.info(f"Batch inserted {len(rows)} price rows")
        return len(rows)
    except Exception as e:
        logger.error(f"Error in batch insert: {e}")
        return 0
    finally:
        conn.close()


def insert_indicators(
    symbol: str,
    date: str,
    indicators: Dict[str, Any],
    conn: Optional[Any] = None,
) -> bool:
    """
    Insert or replace technical indicators for a stock on a date.

    Args:
        symbol: Stock symbol
        date: ISO date string
        indicators: Dict with keys matching column names (rsi_14, macd, etc.)
        conn: Optional existing DB connection to reuse

    Returns:
        True if successful.
    """
    db_conn = conn or get_connection()
    try:
        db_conn.execute(
            """INSERT OR REPLACE INTO technical_indicators
            (symbol, date, rsi_14, macd, macd_signal, macd_hist,
             bb_upper, bb_middle, bb_lower,
             sma_20, sma_50, sma_200, ema_9, ema_21,
             atr_14, adx_14, stoch_k, stoch_d, obv,
             support_1, support_2, support_3,
             resistance_1, resistance_2, resistance_3,
             signal, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol, date,
                indicators.get("rsi_14"),
                indicators.get("macd"),
                indicators.get("macd_signal"),
                indicators.get("macd_hist"),
                indicators.get("bb_upper"),
                indicators.get("bb_middle"),
                indicators.get("bb_lower"),
                indicators.get("sma_20"),
                indicators.get("sma_50"),
                indicators.get("sma_200"),
                indicators.get("ema_9"),
                indicators.get("ema_21"),
                indicators.get("atr_14"),
                indicators.get("adx_14"),
                indicators.get("stoch_k"),
                indicators.get("stoch_d"),
                indicators.get("obv"),
                indicators.get("support_1"),
                indicators.get("support_2"),
                indicators.get("support_3"),
                indicators.get("resistance_1"),
                indicators.get("resistance_2"),
                indicators.get("resistance_3"),
                indicators.get("signal"),
                indicators.get("signal_strength"),
            ),
        )
        db_conn.commit()
        if not conn and USE_TURSO:
            db_conn.sync()
        return True
    except Exception as e:
        logger.error(f"Error inserting indicators for {symbol} on {date}: {e}")
        return False
    finally:
        if not conn:
            db_conn.close()


def insert_news(
    headline: str,
    source: Optional[str] = None,
    published_at: Optional[str] = None,
    symbol: Optional[str] = None,
    sentiment: Optional[str] = None,
    confidence: Optional[float] = None,
    url: Optional[str] = None,
) -> bool:
    """
    Insert a news headline with sentiment score.

    Args:
        headline: News headline text
        source: News source name
        published_at: Publication datetime string
        symbol: Stock symbol (None for market-wide news)
        sentiment: "positive", "negative", or "neutral"
        confidence: Confidence score 0.0 to 1.0
        url: URL of the article

    Returns:
        True if successful.
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO news_sentiment
            (headline, source, published_at, symbol, sentiment, confidence, url)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (headline, source, published_at, symbol, sentiment, confidence, url),
        )
        conn.commit()
        if USE_TURSO:
            conn.sync()
        return True
    except Exception as e:
        logger.error(f"Error inserting news: {e}")
        return False
    finally:
        conn.close()


def insert_market_overview(data: Dict[str, Any]) -> bool:
    """
    Insert or replace daily market overview data.

    Args:
        data: Dict with keys matching column names (date, nifty500_close, etc.)

    Returns:
        True if successful.
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO market_overview
            (date, nifty500_close, nifty500_change_pct,
             nifty50_close, nifty50_change_pct,
             sensex_close, india_vix,
             advances, declines, unchanged,
             total_volume, fii_net, dii_net,
             overall_sentiment_score, fear_greed_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                data["date"],
                data.get("nifty500_close"),
                data.get("nifty500_change_pct"),
                data.get("nifty50_close"),
                data.get("nifty50_change_pct"),
                data.get("sensex_close"),
                data.get("india_vix"),
                data.get("advances"),
                data.get("declines"),
                data.get("unchanged"),
                data.get("total_volume"),
                data.get("fii_net"),
                data.get("dii_net"),
                data.get("overall_sentiment_score"),
                data.get("fear_greed_label"),
            ),
        )
        conn.commit()
        if USE_TURSO:
            conn.sync()
        return True
    except Exception as e:
        logger.error(f"Error inserting market overview: {e}")
        return False
    finally:
        conn.close()


def insert_ai_signal(
    symbol: str,
    signal: str,
    confidence: float,
    model_version: str = "v1.0.0",
    target_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    reasoning: Optional[List[str]] = None,
    features_used: Optional[Dict] = None,
    conn: Optional[Any] = None,
) -> bool:
    """
    Insert an AI-generated trading signal.

    Args:
        symbol: Stock symbol
        signal: "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
        confidence: Confidence score 0 to 100
        model_version: Version string of the model
        target_price: Suggested target price
        stop_loss: Suggested stop loss price
        reasoning: List of reason strings
        features_used: Dict of feature values used
        conn: Optional existing DB connection to reuse

    Returns:
        True if successful.
    """
    db_conn = conn or get_connection()
    try:
        db_conn.execute(
            """INSERT INTO ai_signals
            (symbol, signal, confidence, model_version,
             target_price, stop_loss, reasoning, features_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                symbol,
                signal,
                confidence,
                model_version,
                target_price,
                stop_loss,
                json.dumps(reasoning) if reasoning else None,
                json.dumps(features_used) if features_used else None,
            ),
        )
        db_conn.commit()
        if not conn and USE_TURSO:
            db_conn.sync()
        return True
    except Exception as e:
        logger.error(f"Error inserting AI signal for {symbol}: {e}")
        return False
    finally:
        if not conn:
            db_conn.close()


def insert_trade_signals_batch(
    trades: List[Dict],
    generated_date: str,
    generated_at: str,
    sync: bool = True,
) -> int:
    """
    Bulk-upsert trade signals into the trade_signals table.
    Uses INSERT OR REPLACE to deduplicate on (symbol, generated_date).

    Args:
        trades: List of trade dicts from generate_trades.py
        generated_date: YYYY-MM-DD date string
        generated_at: Full timestamp string
        sync: Whether to sync to Turso after inserting

    Returns:
        Number of trades stored.
    """
    if not trades:
        return 0

    # Use local-only connection in local env to avoid per-write Turso sync
    if ENV == "local":
        conn = get_local_connection()
    else:
        conn = get_connection(sync_on_connect=False)

    try:
        count = 0
        for t in trades:
            conn.execute(
                """INSERT OR REPLACE INTO trade_signals
                (symbol, name, signal, confidence, trade_type,
                 buy_price, target_price, stop_loss, risk_reward, expected_return_pct,
                 current_price, atr_14, atr_pct,
                 avg_daily_volume, daily_turnover_cr, liquidity,
                 max_safe_qty, max_qty_per_user, max_investment_per_user, min_qty,
                 model_name, model_horizon, model_accuracy, model_precision,
                 top_drivers, sentiment,
                 generated_date, generated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    t["symbol"],
                    t.get("name", ""),
                    t["signal"],
                    t.get("confidence"),
                    t.get("trade", {}).get("type"),
                    t.get("trade", {}).get("buy_price"),
                    t.get("trade", {}).get("target_price"),
                    t.get("trade", {}).get("stop_loss"),
                    t.get("trade", {}).get("risk_reward"),
                    t.get("trade", {}).get("expected_return_pct"),
                    t.get("price", {}).get("current"),
                    t.get("price", {}).get("atr_14"),
                    t.get("price", {}).get("atr_pct"),
                    t.get("position", {}).get("avg_daily_volume"),
                    t.get("position", {}).get("daily_turnover_cr"),
                    t.get("position", {}).get("liquidity"),
                    t.get("position", {}).get("max_safe_qty"),
                    t.get("position", {}).get("max_qty_per_user"),
                    t.get("position", {}).get("max_investment_per_user"),
                    t.get("position", {}).get("min_qty"),
                    t.get("model", {}).get("name"),
                    t.get("model", {}).get("horizon"),
                    t.get("model", {}).get("accuracy"),
                    t.get("model", {}).get("precision"),
                    json.dumps(t.get("top_drivers", [])),
                    json.dumps(t.get("sentiment", {})),
                    generated_date,
                    generated_at,
                ),
            )
            count += 1

        conn.commit()
        if sync and ENV != "local" and USE_TURSO:
            conn.sync()
        logger.info(f"Stored {count} trade signals for {generated_date}")
        return count
    except Exception as e:
        logger.error(f"Error inserting trade signals: {e}")
        return 0
    finally:
        conn.close()


def get_trade_signals(
    date: Optional[str] = None,
    signal_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    """
    Query trade signals with optional filters.

    Args:
        date: Filter by generated_date (YYYY-MM-DD). None = latest date.
        signal_type: Filter by signal (e.g. "BUY", "STRONG BUY").
        limit: Max number of results.

    Returns:
        List of trade signal dicts sorted by confidence descending.
    """
    conn = get_connection(sync_on_connect=False)
    try:
        if date is None:
            # Get the latest date
            row = conn.execute(
                "SELECT MAX(generated_date) FROM trade_signals"
            ).fetchone()
            date = row[0] if row and row[0] else datetime.now().strftime("%Y-%m-%d")

        if signal_type:
            cur = conn.execute(
                """SELECT * FROM trade_signals
                WHERE generated_date = ? AND signal LIKE ?
                ORDER BY confidence DESC
                LIMIT ?""",
                (date, f"%{signal_type}%", limit),
            )
        else:
            cur = conn.execute(
                """SELECT * FROM trade_signals
                WHERE generated_date = ?
                ORDER BY confidence DESC
                LIMIT ?""",
                (date, limit),
            )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def sync_trade_signals_to_turso() -> int:
    """
    Sync trade_signals from local DB to Turso cloud.
    Called by the EOD sync scheduler job.

    Reads unsent trade signals from local DB and pushes them
    to Turso via a synced connection.

    Returns:
        Number of rows synced.
    """
    if not USE_TURSO:
        logger.warning("Turso not configured — skipping sync")
        return 0

    # Read from local
    local_conn = get_local_connection()
    try:
        rows = local_conn.execute(
            "SELECT * FROM trade_signals"
        ).fetchall()
        cols = [d[0] for d in local_conn.execute(
            "SELECT * FROM trade_signals LIMIT 1"
        ).description]
    finally:
        local_conn.close()

    if not rows:
        logger.info("No trade signals to sync")
        return 0

    # Push to Turso
    turso_conn = get_turso_connection(sync_on_connect=True)
    try:
        # Ensure table exists
        from database.models import CREATE_TRADE_SIGNALS_TABLE
        turso_conn.execute(CREATE_TRADE_SIGNALS_TABLE)
        turso_conn.commit()

        cols_no_id = [c for c in cols if c != "id"]
        id_idx = cols.index("id") if "id" in cols else None

        count = 0
        for row in rows:
            values = [row[i] for i in range(len(cols)) if cols[i] != "id"]
            placeholders = ",".join(["?" for _ in cols_no_id])
            turso_conn.execute(
                f"INSERT OR REPLACE INTO trade_signals ({','.join(cols_no_id)}) VALUES ({placeholders})",
                values,
            )
            count += 1

        turso_conn.commit()
        turso_conn.sync()
        logger.info(f"Synced {count} trade signals to Turso")
        return count
    except Exception as e:
        logger.error(f"Error syncing trade signals to Turso: {e}")
        return 0
    finally:
        turso_conn.close()


# ==========================================
# QUERY HELPER FUNCTIONS
# ==========================================


def get_prices(
    symbol: str,
    days: int = 90,
    interval: str = "1d",
) -> List[Dict]:
    """
    Get price history for a stock.

    Args:
        symbol: Stock symbol, e.g. "TCS.NS"
        days: Number of days to look back (default 90)
        interval: Data interval ("1d", "1h", "5m")

    Returns:
        List of dicts with date, open, high, low, close, volume keys.

    Example:
        prices = get_prices("TCS.NS", days=30)
        for p in prices:
            print(f"{p['date']}: Close={p['close']}")
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        cur = conn.execute(
            """SELECT date, open, high, low, close, volume
            FROM prices
            WHERE symbol = ? AND interval = ? AND date >= ?
            ORDER BY date ASC""",
            (symbol, interval, start_date),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def get_all_prices_df(symbol: str, days: int = 365) -> List[Dict]:
    """
    Get all OHLCV data for a symbol as a list of dicts (for DataFrame creation).

    Args:
        symbol: Stock symbol
        days: Number of days to look back

    Returns:
        List of dicts with all price columns.
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        cur = conn.execute(
            """SELECT date, open, high, low, close, volume
            FROM prices
            WHERE symbol = ? AND interval = '1d' AND date >= ?
            ORDER BY date ASC""",
            (symbol, start_date),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def get_latest_indicators(symbol: str) -> Optional[Dict]:
    """
    Get the most recent technical indicators for a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Dict with all indicator values, or None if not found.

    Example:
        indicators = get_latest_indicators("TCS.NS")
        if indicators:
            print(f"RSI: {indicators['rsi_14']}, Signal: {indicators['signal']}")
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """SELECT * FROM technical_indicators
            WHERE symbol = ?
            ORDER BY date DESC
            LIMIT 1""",
            (symbol,),
        )
        return _row_to_dict(cur)
    finally:
        conn.close()


def get_recent_news(
    limit: int = 20,
    symbol: Optional[str] = None,
) -> List[Dict]:
    """
    Get recent news headlines with sentiment.

    Args:
        limit: Maximum number of articles to return
        symbol: Filter by stock symbol (None for all news)

    Returns:
        List of dicts with headline, source, sentiment, confidence, etc.
    """
    conn = get_connection()
    try:
        if symbol:
            cur = conn.execute(
                """SELECT * FROM news_sentiment
                WHERE symbol = ?
                ORDER BY published_at DESC
                LIMIT ?""",
                (symbol, limit),
            )
        else:
            cur = conn.execute(
                """SELECT * FROM news_sentiment
                ORDER BY published_at DESC
                LIMIT ?""",
                (limit,),
            )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def get_market_overview(days: int = 30) -> List[Dict]:
    """
    Get market overview history.

    Args:
        days: Number of days to look back

    Returns:
        List of dicts with daily market overview data.
    """
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    conn = get_connection()
    try:
        cur = conn.execute(
            """SELECT * FROM market_overview
            WHERE date >= ?
            ORDER BY date DESC""",
            (start_date,),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


def get_top_signals(
    signal_type: str = "BUY",
    limit: int = 10,
) -> List[Dict]:
    """
    Get top AI signals sorted by confidence.

    Args:
        signal_type: Filter by signal type ("BUY", "STRONG BUY", "SELL", etc.)
        limit: Maximum number of results

    Returns:
        List of dicts with signal data.
    """
    conn = get_connection()
    try:
        # Get today's date
        today = datetime.now().strftime("%Y-%m-%d")

        cur = conn.execute(
            """SELECT * FROM ai_signals
            WHERE signal LIKE ?
            AND date(generated_at) = ?
            ORDER BY confidence DESC
            LIMIT ?""",
            (f"%{signal_type}%", today, limit),
        )
        results = _rows_to_dicts(cur)

        # If no signals today, get the most recent ones
        if not results:
            cur = conn.execute(
                """SELECT * FROM ai_signals
                WHERE signal LIKE ?
                ORDER BY generated_at DESC, confidence DESC
                LIMIT ?""",
                (f"%{signal_type}%", limit),
            )
            results = _rows_to_dicts(cur)

        return results
    finally:
        conn.close()


def get_db_stats() -> Dict[str, int]:
    """
    Get row counts for all tables — useful for status checks.

    Returns:
        Dict mapping table name to row count.

    Example:
        stats = get_db_stats()
        print(f"Prices: {stats['prices']} rows")
        print(f"Indicators: {stats['technical_indicators']} rows")
    """
    tables = ["prices", "technical_indicators", "news_sentiment", "market_overview", "ai_signals", "trade_signals"]
    conn = get_connection()
    try:
        stats = {}
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[table] = count
        return stats
    finally:
        conn.close()


def get_all_symbols() -> List[str]:
    """
    Get all unique stock symbols that have price data in the database.

    Returns:
        List of symbol strings.
    """
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM prices WHERE interval = '1d' ORDER BY symbol"
        ).fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def get_latest_date(symbol: str) -> Optional[str]:
    """
    Get the most recent date for which we have price data for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Date string or None if no data exists.
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(date) as latest FROM prices WHERE symbol = ? AND interval = '1d'",
            (symbol,),
        ).fetchone()
        return row[0] if row and row[0] else None
    finally:
        conn.close()
