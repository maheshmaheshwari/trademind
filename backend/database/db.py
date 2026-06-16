"""
TradeMind AI — Database Connection & Helper Functions

Uses TimescaleDB (PostgreSQL via psycopg2) exclusively.

Usage:
    from database.db import init_database, get_connection, get_prices
    init_database()
"""

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PGHOST     = os.getenv("PGHOST", "localhost")
PGPORT     = int(os.getenv("PGPORT", "5433"))
PGDATABASE = os.getenv("PGDATABASE", "trademind")
PGUSER     = os.getenv("PGUSER", "trademind")
PGPASSWORD = os.getenv("PGPASSWORD", "trademind")
PGSSLMODE  = os.getenv("PGSSLMODE", "prefer")


# ---------------------------------------------------------------------------
# Connection pool
# ---------------------------------------------------------------------------

_pool = None


def _get_pool():
    """Return (and lazily initialize) the shared connection pool."""
    global _pool
    if _pool is None:
        from psycopg2 import pool as pg_pool
        _pool = pg_pool.ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=PGHOST,
            port=PGPORT,
            dbname=PGDATABASE,
            user=PGUSER,
            password=PGPASSWORD,
            sslmode=PGSSLMODE,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=10,
            keepalives_count=5,
            connect_timeout=10,
            # Hard cap on query runtime — prevents 50-second hangs that exhaust
            # the connection pool and kill the APScheduler thread pool.
            options="-c statement_timeout=30000",
        )
    return _pool


def get_connection():
    """Return a healthy psycopg2 connection drawn from the pool.

    Validates each connection with a lightweight ping and discards dead ones
    (stale SSL/TCP connections that Docker dropped silently) before returning.
    """
    pool = _get_pool()
    for attempt in range(3):
        conn = pool.getconn()
        try:
            conn.autocommit = False
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
            return conn
        except Exception:
            try:
                pool.putconn(conn, close=True)
            except Exception:
                pass
            if attempt == 2:
                raise RuntimeError("Could not obtain a healthy database connection after 3 attempts")
    # unreachable, but satisfies type checkers
    raise RuntimeError("Could not obtain a healthy database connection")


def release_connection(conn) -> None:
    """Return a connection to the pool, discarding it if it is broken."""
    try:
        pool = _get_pool()
        if getattr(conn, "closed", 0):
            pool.putconn(conn, close=True)
        else:
            pool.putconn(conn)
    except Exception:
        pass


def _rows_to_dicts(cursor) -> List[Dict]:
    """Convert cursor results to a list of dicts."""
    rows = cursor.fetchall()
    if not rows:
        return []
    if cursor.description:
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]
    return []


def _row_to_dict(cursor) -> Optional[Dict]:
    """Convert a single cursor result to a dict."""
    row = cursor.fetchone()
    if not row or not cursor.description:
        return None
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


def _execute(conn, sql: str, params: tuple = ()):
    """
    Execute a SQL statement via psycopg2 cursor.
    Translates ? → %s for convenience so callers can use either style.
    Returns the cursor.
    """
    sql = sql.replace("?", "%s")
    sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
    sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    cur = conn.cursor()
    cur.execute(sql, params)
    return cur


def _executemany(conn, sql: str, params_list):
    """Execute a statement for multiple parameter sets via psycopg2."""
    import psycopg2.extras
    sql = sql.replace("?", "%s")
    sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO")
    sql = sql.replace("INSERT OR REPLACE INTO", "INSERT INTO")
    cur = conn.cursor()
    psycopg2.extras.execute_batch(cur, sql, params_list, page_size=2000)
    return cur


def _on_conflict_ignore(sql: str, unique_cols: List[str]) -> str:
    """Append ON CONFLICT DO NOTHING for upsert-ignore semantics."""
    return sql + " ON CONFLICT DO NOTHING"


def _on_conflict_replace(sql: str, unique_cols: List[str], update_cols: List[str]) -> str:
    """Append ON CONFLICT (...) DO UPDATE SET ... for upsert-replace semantics."""
    conflict = ", ".join(unique_cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    return sql + f" ON CONFLICT ({conflict}) DO UPDATE SET {updates}"


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_database() -> None:
    """Create all tables, hypertables, indexes (idempotent)."""
    from database.schema_pg import init_timescale
    conn = get_connection()
    try:
        init_timescale(conn)
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# INSERT helpers
# ---------------------------------------------------------------------------

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
    conn = get_connection()
    try:
        if time_val is None:
            sql = (
                "INSERT INTO prices"
                " (symbol, exchange, date, time, open, high, low, close, volume, interval)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (symbol, date, interval) WHERE time IS NULL"
                " DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high,"
                " low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume"
            )
        else:
            sql = _on_conflict_ignore(
                "INSERT INTO prices"
                " (symbol, exchange, date, time, open, high, low, close, volume, interval)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ["symbol", "date", "time", "interval"],
            )
        _execute(conn, sql, (symbol, exchange, date, time_val, open_price, high, low, close, volume, interval))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_price {symbol} {date}: {e}")
        return False
    finally:
        release_connection(conn)


def _sanitize_row(row: Tuple) -> Tuple:
    """Convert numpy scalars / NaN floats to Python-native types for psycopg2."""
    out = []
    for v in row:
        if v is None:
            out.append(None)
        elif hasattr(v, "item"):          # numpy scalar → Python native
            native = v.item()
            out.append(None if (isinstance(native, float) and native != native) else native)
        elif isinstance(v, float) and v != v:  # Python float NaN → NULL
            out.append(None)
        else:
            out.append(v)
    return tuple(out)


def insert_prices_batch(rows: List[Tuple], sync: bool = True) -> int:
    if not rows:
        return 0
    rows = [_sanitize_row(r) for r in rows]
    conn = get_connection()
    try:
        # Split into daily (time=NULL) and intraday rows
        daily_rows = [r for r in rows if r[3] is None]
        intraday_rows = [r for r in rows if r[3] is not None]

        inserted = 0
        base_sql = """INSERT INTO prices
               (symbol, exchange, date, time, open, high, low, close, volume, interval)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""

        if daily_rows:
            # DO UPDATE so EOD data always overwrites incomplete intraday candles
            sql_daily = (
                base_sql
                + " ON CONFLICT (symbol, date, interval) WHERE time IS NULL"
                + " DO UPDATE SET open=EXCLUDED.open, high=EXCLUDED.high,"
                + " low=EXCLUDED.low, close=EXCLUDED.close, volume=EXCLUDED.volume"
            )
            _executemany(conn, sql_daily, daily_rows)
            inserted += len(daily_rows)

        if intraday_rows:
            sql_intra = _on_conflict_ignore(base_sql, ["symbol", "date", "time", "interval"])
            _executemany(conn, sql_intra, intraday_rows)
            inserted += len(intraday_rows)

        conn.commit()
        logger.info(f"Batch inserted {inserted} price rows")
        return inserted
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_prices_batch: {e}")
        return 0
    finally:
        release_connection(conn)


def insert_indicators(
    symbol: str,
    date: str,
    indicators: Dict[str, Any],
    conn: Optional[Any] = None,
) -> bool:
    db_conn = conn or get_connection()
    try:
        base_sql = """INSERT INTO technical_indicators
            (symbol, date, rsi_14, macd, macd_signal, macd_hist,
             bb_upper, bb_middle, bb_lower,
             sma_20, sma_50, sma_200, ema_9, ema_21,
             atr_14, adx_14, stoch_k, stoch_d, obv,
             support_1, support_2, support_3,
             resistance_1, resistance_2, resistance_3,
             signal, signal_strength)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        sql = _on_conflict_replace(
            base_sql, ["symbol", "date"],
            ["rsi_14", "macd", "macd_signal", "macd_hist",
             "bb_upper", "bb_middle", "bb_lower",
             "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
             "atr_14", "adx_14", "stoch_k", "stoch_d", "obv",
             "support_1", "support_2", "support_3",
             "resistance_1", "resistance_2", "resistance_3",
             "signal", "signal_strength"],
        )
        params = (
            symbol, date,
            indicators.get("rsi_14"), indicators.get("macd"),
            indicators.get("macd_signal"), indicators.get("macd_hist"),
            indicators.get("bb_upper"), indicators.get("bb_middle"), indicators.get("bb_lower"),
            indicators.get("sma_20"), indicators.get("sma_50"), indicators.get("sma_200"),
            indicators.get("ema_9"), indicators.get("ema_21"),
            indicators.get("atr_14"), indicators.get("adx_14"),
            indicators.get("stoch_k"), indicators.get("stoch_d"), indicators.get("obv"),
            indicators.get("support_1"), indicators.get("support_2"), indicators.get("support_3"),
            indicators.get("resistance_1"), indicators.get("resistance_2"), indicators.get("resistance_3"),
            indicators.get("signal"), indicators.get("signal_strength"),
        )
        _execute(db_conn, sql, params)
        if not conn:
            db_conn.commit()
        return True
    except Exception as e:
        if not conn:
            db_conn.rollback()
        logger.error(f"insert_indicators {symbol} {date}: {e}")
        return False
    finally:
        if not conn:
            release_connection(db_conn)


def insert_news(
    headline: str,
    source: Optional[str] = None,
    published_at: Optional[str] = None,
    symbol: Optional[str] = None,
    sentiment: Optional[str] = None,
    confidence: Optional[float] = None,
    url: Optional[str] = None,
) -> bool:
    conn = get_connection()
    try:
        _execute(conn,
            """INSERT INTO news_sentiment
               (headline, source, published_at, symbol, sentiment, confidence, url)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            (headline, source, published_at, symbol, sentiment, confidence, url),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_news: {e}")
        return False
    finally:
        release_connection(conn)


def insert_market_overview(data: Dict[str, Any]) -> bool:
    conn = get_connection()
    try:
        base_sql = """INSERT INTO market_overview
            (date, nifty500_close, nifty500_change_pct,
             nifty50_close, nifty50_change_pct, sensex_close, india_vix,
             advances, declines, unchanged, total_volume, fii_net, dii_net,
             overall_sentiment_score, fear_greed_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        sql = _on_conflict_replace(
            base_sql, ["date"],
            ["nifty500_close", "nifty500_change_pct", "nifty50_close",
             "nifty50_change_pct", "sensex_close", "india_vix",
             "advances", "declines", "unchanged", "total_volume",
             "fii_net", "dii_net", "overall_sentiment_score", "fear_greed_label"],
        )
        _execute(conn, sql, (
            data["date"], data.get("nifty500_close"), data.get("nifty500_change_pct"),
            data.get("nifty50_close"), data.get("nifty50_change_pct"),
            data.get("sensex_close"), data.get("india_vix"),
            data.get("advances"), data.get("declines"), data.get("unchanged"),
            data.get("total_volume"), data.get("fii_net"), data.get("dii_net"),
            data.get("overall_sentiment_score"), data.get("fear_greed_label"),
        ))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_market_overview: {e}")
        return False
    finally:
        release_connection(conn)




def insert_trade_signals_batch(
    trades: List[Dict],
    generated_date: str,
    generated_at: str,
    sync: bool = True,
) -> int:
    if not trades:
        return 0
    conn = get_connection()
    try:
        base_sql = """INSERT INTO trade_signals
            (symbol, name, signal, confidence, trade_type,
             buy_price, target_price, stop_loss, risk_reward, expected_return_pct,
             current_price, atr_14, atr_pct,
             avg_daily_volume, daily_turnover_cr, liquidity,
             max_safe_qty, max_qty_per_user, max_investment_per_user, min_qty,
             recommended_volume, consumed_volume,
             model_name, model_horizon, model_accuracy, model_precision,
             top_drivers, sentiment, generated_date, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        sql = _on_conflict_replace(
            base_sql, ["symbol", "generated_date"],
            ["signal", "confidence", "trade_type", "buy_price", "target_price",
             "stop_loss", "risk_reward", "expected_return_pct", "current_price",
             "atr_14", "atr_pct", "model_name", "model_horizon",
             "model_accuracy", "model_precision", "top_drivers", "sentiment", "generated_at"],
        )
        # Deactivate all previous signals for each symbol before inserting new ones
        symbols = [t["symbol"] for t in trades]
        if symbols:
            placeholders = ",".join(["?" ] * len(symbols))
            _execute(conn,
                f"UPDATE trade_signals SET is_active = FALSE WHERE symbol IN ({placeholders}) AND generated_date < ?",
                (*symbols, generated_date),
            )

        count = 0
        for t in trades:
            symbol = t["symbol"]
            # Carry forward consumed_volume from the current active signal so
            # capacity tracking survives signal refreshes (EOD overwriting intraday).
            try:
                cv_row = _execute(conn,
                    "SELECT consumed_volume FROM trade_signals WHERE symbol = ? AND is_active = TRUE ORDER BY generated_date DESC LIMIT 1",
                    (symbol,)
                ).fetchone()
                carried_consumed = int(cv_row[0] or 0) if cv_row else 0
            except Exception:
                carried_consumed = 0

            _execute(conn, sql, (
                symbol, t.get("name", ""), t["signal"], t.get("confidence"),
                t.get("trade", {}).get("type"),
                t.get("trade", {}).get("buy_price"), t.get("trade", {}).get("target_price"),
                t.get("trade", {}).get("stop_loss"), t.get("trade", {}).get("risk_reward"),
                t.get("trade", {}).get("expected_return_pct"),
                t.get("price", {}).get("current"),
                t.get("price", {}).get("atr_14"), t.get("price", {}).get("atr_pct"),
                t.get("position", {}).get("avg_daily_volume"),
                t.get("position", {}).get("daily_turnover_cr"),
                t.get("position", {}).get("liquidity"),
                t.get("position", {}).get("max_safe_qty"),
                t.get("position", {}).get("max_qty_per_user"),
                t.get("position", {}).get("max_investment_per_user"),
                t.get("position", {}).get("min_qty"),
                t.get("position", {}).get("recommended_volume"), carried_consumed,
                t.get("model", {}).get("name"), t.get("model", {}).get("horizon"),
                t.get("model", {}).get("accuracy"), t.get("model", {}).get("precision"),
                json.dumps(t.get("top_drivers", [])),
                json.dumps(t.get("sentiment", {})),
                generated_date, generated_at,
            ))
            count += 1
        conn.commit()
        logger.info(f"Stored {count} trade signals for {generated_date}")
        return count
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_trade_signals_batch: {e}")
        return 0
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# QUERY helpers
# ---------------------------------------------------------------------------

def get_prices(symbol: str, days: int = 90, interval: str = "1d") -> List[Dict]:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT date, open, high, low, close, volume FROM prices
               WHERE symbol = ? AND interval = ? AND date >= ?
               ORDER BY date ASC""",
            (symbol, interval, start_date),
        )
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_all_prices_df(symbol: str, days: int = 365) -> List[Dict]:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT p.date, p.open, p.high, p.low, p.close, p.volume,
                      COALESCE(d.delivery_pct, 50.0) AS delivery_pct
               FROM prices p
               LEFT JOIN delivery_data d ON d.symbol = p.symbol AND d.date = p.date
               WHERE p.symbol = ? AND p.interval = '1d' AND p.date >= ?
               ORDER BY p.date ASC""",
            (symbol, start_date),
        )
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_latest_indicators(symbol: str) -> Optional[Dict]:
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT * FROM technical_indicators
               WHERE symbol = ? ORDER BY date DESC LIMIT 1""",
            (symbol,),
        )
        return _row_to_dict(cur)
    finally:
        release_connection(conn)


def get_recent_news(limit: int = 20, symbol: Optional[str] = None) -> List[Dict]:
    conn = get_connection()
    try:
        if symbol:
            cur = _execute(conn,
                "SELECT * FROM news_sentiment WHERE symbol = ? ORDER BY published_at DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cur = _execute(conn,
                "SELECT * FROM news_sentiment ORDER BY published_at DESC LIMIT ?",
                (limit,),
            )
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_news_for_user_watchlist(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Return recent news for all stocks in a user's watchlist + market-wide news.
    Used for per-user news feed.
    """
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT n.*
            FROM news_sentiment n
            WHERE n.symbol IN (
                SELECT symbol FROM watchlist WHERE user_id = ?
            )
            OR n.symbol IS NULL
            ORDER BY n.published_at DESC
            LIMIT ?
        """, (user_id, limit))
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_news_summary_for_user(user_id: int) -> Dict:
    """
    Aggregate sentiment summary for a user's watchlist stocks over last 7 days.
    Returns per-stock sentiment + overall portfolio sentiment.
    """
    conn = get_connection()
    try:
        # Per-stock sentiment for watchlist
        cur = _execute(conn, """
            SELECT
                n.symbol,
                COUNT(*)                                    AS article_count,
                AVG(CAST(n.sentiment AS FLOAT))             AS avg_sentiment,
                SUM(CASE WHEN CAST(n.sentiment AS FLOAT) > 0 THEN 1 ELSE 0 END) AS positive,
                SUM(CASE WHEN CAST(n.sentiment AS FLOAT) < 0 THEN 1 ELSE 0 END) AS negative,
                MAX(n.published_at)                         AS latest_article
            FROM news_sentiment n
            JOIN watchlist w ON w.symbol = n.symbol AND w.user_id = ?
            WHERE n.published_at >= NOW() - INTERVAL '7 days'
            GROUP BY n.symbol
            ORDER BY AVG(CAST(n.sentiment AS FLOAT)) DESC
        """, (user_id,))
        per_stock = _rows_to_dicts(cur)

        # Overall portfolio sentiment
        cur2 = _execute(conn, """
            SELECT
                COUNT(*)                                    AS total_articles,
                AVG(CAST(n.sentiment AS FLOAT))             AS portfolio_sentiment
            FROM news_sentiment n
            JOIN watchlist w ON w.symbol = n.symbol AND w.user_id = ?
            WHERE n.published_at >= NOW() - INTERVAL '7 days'
        """, (user_id,))
        row = cur2.fetchone()
        overall = {
            "total_articles": row[0] or 0,
            "portfolio_sentiment": round(float(row[1] or 0), 4),
        }

        return {"per_stock": per_stock, "overall": overall}
    finally:
        release_connection(conn)


def get_user_signal_history(user_id: int, limit: int = 50) -> List[Dict]:
    """
    Return AI trade signals that a user has acted on (linked via trade_signal_id in orders).
    User-wise classification of which AI signals were used.
    """
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT DISTINCT
                ts.symbol, ts.signal, ts.confidence, ts.model_horizon,
                ts.buy_price, ts.target_price, ts.stop_loss,
                ts.generated_date, ts.is_active,
                o.created_at AS traded_at,
                o.status     AS order_status,
                o.fill_price
            FROM orders o
            JOIN trade_signals ts ON ts.id = o.trade_signal_id
            WHERE o.user_id = ? AND o.order_purpose = 'ENTRY'
            ORDER BY o.created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_user_analytics(user_id: int) -> Dict:
    """
    Comprehensive per-user trading performance analytics.
    Covers: P&L breakdown, win/loss by signal type, horizon, confidence band,
    AI signal accuracy, volume consumed, and best/worst trades.
    """
    conn = get_connection()
    try:
        # ── Overall summary ───────────────────────────────────────────────
        cur = _execute(conn, """
            SELECT
                COUNT(*)                                        AS total_orders,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)       AS wins,
                SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END)      AS losses,
                SUM(COALESCE(pnl, 0))                           AS total_realized_pnl,
                AVG(COALESCE(pnl, 0))                           AS avg_pnl_per_trade,
                MAX(COALESCE(pnl, 0))                           AS best_trade_pnl,
                MIN(COALESCE(pnl, 0))                           AS worst_trade_pnl,
                SUM(price * quantity)                           AS total_invested
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
        """, (user_id,))
        summary_row = cur.fetchone()
        summary = {
            "total_trades":      summary_row[0] or 0,
            "wins":              summary_row[1] or 0,
            "losses":            summary_row[2] or 0,
            "win_rate":          round((summary_row[1] or 0) / max(summary_row[0] or 1, 1) * 100, 1),
            "total_realized_pnl": round(float(summary_row[3] or 0), 2),
            "avg_pnl_per_trade": round(float(summary_row[4] or 0), 2),
            "best_trade_pnl":    round(float(summary_row[5] or 0), 2),
            "worst_trade_pnl":   round(float(summary_row[6] or 0), 2),
            "total_invested":    round(float(summary_row[7] or 0), 2),
        }

        # ── P&L by signal type (BUY/SELL) ────────────────────────────────
        cur = _execute(conn, """
            SELECT signal,
                COUNT(*)                                AS trade_count,
                SUM(COALESCE(pnl, 0))                  AS total_pnl,
                AVG(COALESCE(pnl, 0))                  AS avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
            GROUP BY signal
        """, (user_id,))
        by_signal = _rows_to_dicts(cur)

        # ── P&L by horizon ────────────────────────────────────────────────
        cur = _execute(conn, """
            SELECT horizon,
                COUNT(*)                                AS trade_count,
                SUM(COALESCE(pnl, 0))                  AS total_pnl,
                AVG(COALESCE(pnl, 0))                  AS avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
              AND horizon IS NOT NULL
            GROUP BY horizon ORDER BY total_pnl DESC
        """, (user_id,))
        by_horizon = _rows_to_dicts(cur)

        # ── P&L by confidence band ────────────────────────────────────────
        cur = _execute(conn, """
            SELECT
                CASE
                    WHEN confidence >= 90 THEN '90-100%'
                    WHEN confidence >= 80 THEN '80-90%'
                    WHEN confidence >= 70 THEN '70-80%'
                    WHEN confidence >= 60 THEN '60-70%'
                    ELSE '<60%'
                END AS confidence_band,
                COUNT(*)                                AS trade_count,
                SUM(COALESCE(pnl, 0))                  AS total_pnl,
                AVG(COALESCE(pnl, 0))                  AS avg_pnl,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
              AND confidence IS NOT NULL
            GROUP BY confidence_band ORDER BY confidence_band DESC
        """, (user_id,))
        by_confidence = _rows_to_dicts(cur)

        # ── AI signal accuracy (acted signals vs outcome) ─────────────────
        cur = _execute(conn, """
            SELECT
                ts.signal          AS ai_signal,
                ts.is_active       AS signal_still_active,
                ts.model_horizon   AS horizon,
                o.pnl              AS realized_pnl,
                o.symbol,
                o.created_at       AS traded_at
            FROM orders o
            JOIN trade_signals ts ON ts.id = o.trade_signal_id
            WHERE o.user_id = ? AND o.order_purpose = 'ENTRY' AND o.status = 'EXECUTED'
            ORDER BY o.created_at DESC
            LIMIT 20
        """, (user_id,))
        signal_accuracy = _rows_to_dicts(cur)

        # ── Volume consumed per signal ────────────────────────────────────
        cur = _execute(conn, """
            SELECT usv.symbol, usv.quantity_consumed, usv.investment_amount,
                   ts.signal, ts.is_active, ts.confidence, ts.model_horizon,
                   usv.created_at
            FROM user_signal_volume usv
            JOIN trade_signals ts ON ts.id = usv.trade_signal_id
            WHERE usv.user_id = ?
            ORDER BY usv.created_at DESC
        """, (user_id,))
        volume_consumed = _rows_to_dicts(cur)

        # ── Best and worst trades ─────────────────────────────────────────
        cur = _execute(conn, """
            SELECT symbol, signal, pnl, price, quantity, created_at
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
              AND pnl IS NOT NULL
            ORDER BY pnl DESC LIMIT 5
        """, (user_id,))
        best_trades = _rows_to_dicts(cur)

        cur = _execute(conn, """
            SELECT symbol, signal, pnl, price, quantity, created_at
            FROM orders
            WHERE user_id = ? AND order_purpose = 'ENTRY' AND status = 'EXECUTED'
              AND pnl IS NOT NULL
            ORDER BY pnl ASC LIMIT 5
        """, (user_id,))
        worst_trades = _rows_to_dicts(cur)

        return {
            "user_id":        user_id,
            "summary":        summary,
            "by_signal":      by_signal,
            "by_horizon":     by_horizon,
            "by_confidence":  by_confidence,
            "signal_accuracy": signal_accuracy,
            "volume_consumed": volume_consumed,
            "best_trades":    best_trades,
            "worst_trades":   worst_trades,
        }
    finally:
        release_connection(conn)


def get_user_signal_volume(user_id: int) -> List[Dict]:
    """How much of each AI signal this user has consumed."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT usv.*, ts.signal, ts.confidence, ts.is_active,
                   ts.recommended_volume, ts.max_qty_per_user
            FROM user_signal_volume usv
            JOIN trade_signals ts ON ts.id = usv.trade_signal_id
            WHERE usv.user_id = ?
            ORDER BY usv.created_at DESC
        """, (user_id,))
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_market_overview(days: int = 30) -> List[Dict]:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT * FROM market_overview WHERE date >= ? ORDER BY date DESC",
            (start_date,),
        )
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def get_top_signals(signal_type: str = "BUY", limit: int = 10) -> List[Dict]:
    conn = get_connection()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        like_val = f"%{signal_type}%"
        # Filter by is_active=TRUE — only current signals, not superseded ones
        cur = _execute(conn,
            """SELECT * FROM trade_signals
               WHERE signal LIKE ? AND generated_date = ? AND is_active = TRUE
               ORDER BY confidence DESC LIMIT ?""",
            (like_val, today, limit),
        )
        results = _rows_to_dicts(cur)
        if not results:
            cur = _execute(conn,
                """SELECT * FROM trade_signals WHERE signal LIKE ? AND is_active = TRUE
                   ORDER BY generated_date DESC, confidence DESC LIMIT ?""",
                (like_val, limit),
            )
            results = _rows_to_dicts(cur)
        return results
    finally:
        release_connection(conn)


def get_active_signal_id(symbol: str) -> Optional[int]:
    """Return the id of the current active trade_signal for a symbol, or None."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT id FROM trade_signals WHERE symbol = ? AND is_active = TRUE ORDER BY generated_date DESC LIMIT 1",
            (symbol,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        release_connection(conn)


def get_trade_signals(
    date: Optional[str] = None,
    signal_type: Optional[str] = None,
    limit: int = 100,
) -> List[Dict]:
    conn = get_connection()
    try:
        if date is None:
            cur = _execute(conn, "SELECT MAX(generated_date) FROM trade_signals", ())
            row = cur.fetchone()
            date = row[0] if row and row[0] else datetime.now().strftime("%Y-%m-%d")

        vol_filter = "AND (consumed_volume < recommended_volume OR recommended_volume IS NULL OR consumed_volume IS NULL)"

        if signal_type:
            cur = _execute(conn,
                f"SELECT * FROM trade_signals WHERE generated_date = ? AND signal LIKE ? AND is_active = TRUE {vol_filter} ORDER BY confidence DESC LIMIT ?",
                (date, f"%{signal_type}%", limit),
            )
        else:
            cur = _execute(conn,
                f"SELECT * FROM trade_signals WHERE generated_date = ? AND is_active = TRUE {vol_filter} ORDER BY confidence DESC LIMIT ?",
                (date, limit),
            )
        return _rows_to_dicts(cur)
    except Exception as e:
        logger.error(f"get_trade_signals: {e}")
        return []
    finally:
        release_connection(conn)


def get_db_stats() -> Dict[str, int]:
    # approximate_row_count() is O(1) and works correctly for TimescaleDB hypertables
    # (pg_class.reltuples returns 0 for hypertable parents because data lives in chunks).
    tables = ["prices", "technical_indicators", "news_sentiment",
              "market_overview", "trade_signals"]
    conn = get_connection()
    try:
        cur = conn.cursor()
        stats: Dict[str, int] = {}
        for t in tables:
            cur.execute("SELECT approximate_row_count(%s)::bigint", (t,))
            stats[t] = max(0, cur.fetchone()[0])
        return stats
    finally:
        release_connection(conn)


def get_all_symbols() -> List[str]:
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT DISTINCT symbol FROM prices WHERE interval = '1d' ORDER BY symbol", (),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        release_connection(conn)


def get_latest_date(symbol: str) -> Optional[str]:
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT MAX(date) FROM prices WHERE symbol = ? AND interval = '1d'",
            (symbol,),
        )
        row = cur.fetchone()
        val = row[0] if row and row[0] else None
        return str(val) if val else None
    finally:
        release_connection(conn)


def get_trade_signals_formatted(
    signal_filter: Optional[List[str]] = None,
    date: Optional[str] = None,
) -> Dict:
    conn = get_connection()
    try:
        if date is None:
            cur = _execute(conn, "SELECT MAX(generated_date) FROM trade_signals", ())
            row = cur.fetchone()
            date = str(row[0]) if row and row[0] else datetime.now().strftime("%Y-%m-%d")

        cur = _execute(conn,
            "SELECT * FROM trade_signals WHERE generated_date = ? AND is_active = TRUE ORDER BY confidence DESC",
            (date,),
        )
        rows = _rows_to_dicts(cur)
    finally:
        release_connection(conn)

    formatted = [_format_trade_signal(r) for r in rows]
    if signal_filter:
        formatted = [t for t in formatted if t.get("signal") in signal_filter]

    actionable, avoid, hold = [], [], []
    for t in formatted:
        sig = t.get("signal", "")
        if "BUY" in sig:
            actionable.append(t)
        elif "SELL" in sig:
            avoid.append(t)
        else:
            hold.append(t)

    return {
        "date": str(date),
        "trades": formatted,
        "actionable_trades": actionable,
        "avoid_list": avoid,
        "hold_list": hold,
        "summary": {
            "total": len(formatted),
            "actionable": len(actionable),
            "avoid": len(avoid),
            "hold": len(hold),
            "generated_date": str(date),
        },
    }


def get_signal_history(limit: int = 30) -> List[Dict]:
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT generated_date, MAX(generated_at) as generated_at,
               COUNT(*) as total_signals,
               SUM(CASE WHEN signal LIKE '%%BUY%%' THEN 1 ELSE 0 END) as buy_count,
               SUM(CASE WHEN signal LIKE '%%SELL%%' THEN 1 ELSE 0 END) as sell_count
               FROM trade_signals
               GROUP BY generated_date
               ORDER BY generated_date DESC LIMIT ?""",
            (limit,),
        )
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# Watchlist helpers
# ---------------------------------------------------------------------------

def get_watchlist(user_id: int) -> List[Dict]:
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT * FROM watchlist WHERE user_id = ? ORDER BY added_at DESC", (user_id,))
        return _rows_to_dicts(cur)
    finally:
        release_connection(conn)


def add_to_watchlist(user_id: int, symbol: str) -> None:
    conn = get_connection()
    try:
        sql = _on_conflict_ignore(
            "INSERT INTO watchlist (user_id, symbol) VALUES (?, ?)",
            ["user_id", "symbol"],
        )
        _execute(conn, sql, (user_id, symbol))
        conn.commit()
    finally:
        release_connection(conn)


def remove_from_watchlist(user_id: int, symbol: str) -> None:
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM watchlist WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        conn.commit()
    finally:
        release_connection(conn)


def update_watchlist_alerts(user_id: int, symbol: str, alert_above: float = None, alert_below: float = None) -> None:
    conn = get_connection()
    try:
        _execute(conn,
            "UPDATE watchlist SET alert_above = ?, alert_below = ? WHERE user_id = ? AND symbol = ?",
            (alert_above, alert_below, user_id, symbol))
        conn.commit()
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# Notification helpers
# ---------------------------------------------------------------------------

def get_notifications(user_id: int, limit: int = 50) -> Dict:
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT * FROM notifications WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit))
        rows = _rows_to_dicts(cur)
        unread = sum(1 for r in rows if not r.get("is_read"))
        return {"data": rows, "unread": unread}
    finally:
        release_connection(conn)


def mark_notifications_read(user_id: int) -> None:
    conn = get_connection()
    try:
        _execute(conn,
            "UPDATE notifications SET is_read = TRUE WHERE user_id = ? AND is_read = FALSE",
            (user_id,))
        conn.commit()
    finally:
        release_connection(conn)


def delete_notification(notif_id: int, user_id: int) -> None:
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM notifications WHERE id = ? AND user_id = ?", (notif_id, user_id))
        conn.commit()
    finally:
        release_connection(conn)


def insert_notification(user_id: int, type: str, title: str, message: str = None, icon: str = None, color: str = None) -> None:
    conn = get_connection()
    try:
        _execute(conn,
            "INSERT INTO notifications (user_id, type, title, message, icon, color) VALUES (?,?,?,?,?,?)",
            (user_id, type, title, message, icon, color))
        conn.commit()
    finally:
        release_connection(conn)


def _format_trade_signal(row: Dict) -> Dict:
    return {
        "symbol": row.get("symbol"),
        "name": row.get("name"),
        "signal": row.get("signal"),
        "confidence": row.get("confidence"),
        "trade": {
            "type": row.get("trade_type"),
            "buy_price": row.get("buy_price"),
            "target_price": row.get("target_price"),
            "stop_loss": row.get("stop_loss"),
            "risk_reward": row.get("risk_reward"),
            "expected_return_pct": row.get("expected_return_pct"),
        },
        "price": {
            "current": row.get("current_price"),
            "atr_14": row.get("atr_14"),
            "atr_pct": row.get("atr_pct"),
        },
        "position": {
            "avg_daily_volume": row.get("avg_daily_volume"),
            "daily_turnover_cr": row.get("daily_turnover_cr"),
            "liquidity": row.get("liquidity"),
            "max_safe_qty": row.get("max_safe_qty"),
            "max_qty_per_user": row.get("max_qty_per_user"),
            "max_investment_per_user": row.get("max_investment_per_user"),
            "min_qty": row.get("min_qty"),
            "recommended_volume": row.get("recommended_volume"),
            "consumed_volume": row.get("consumed_volume"),
        },
        "model": {
            "name": row.get("model_name"),
            "horizon": row.get("model_horizon"),
            "accuracy": row.get("model_accuracy"),
            "precision": row.get("model_precision"),
        },
        "top_drivers": json.loads(row.get("top_drivers") or "[]"),
        "sentiment": json.loads(row.get("sentiment") or "{}"),
        "generated_date": str(row.get("generated_date") or ""),
        "generated_at": str(row.get("generated_at") or ""),
    }
