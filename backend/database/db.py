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
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# APP_ENV=test routes every PG* var at a separate Timescale Cloud test instance,
# overriding whatever .env already set — regardless of import order, this module
# owns the final connection params. See CLAUDE.md "Testing & Test Database".
if os.getenv("APP_ENV") == "test":
    load_dotenv(".env.test", override=True)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PGHOST     = os.getenv("PGHOST", "localhost")
PGPORT     = int(os.getenv("PGPORT", "5433"))
PGDATABASE = os.getenv("PGDATABASE", "trademind")
PGUSER     = os.getenv("PGUSER", "trademind")
PGPASSWORD = os.getenv("PGPASSWORD", "trademind")  # set in .env; override with .env.test for test DB
# Defaults to "prefer" for local-dev flexibility (e.g. a local Postgres with
# no SSL configured). Both backend/.env (prod) and backend/.env.test set
# PGSSLMODE=require explicitly, so the real Timescale Cloud instances always
# negotiate TLS regardless of this default.
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
    import re as _re
    _safe = _re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
    for col in unique_cols + update_cols:
        if not _safe.match(col):
            raise ValueError(f"Unsafe column name rejected: {col!r}")
    conflict = ", ".join(unique_cols)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)
    return sql + f" ON CONFLICT ({conflict}) DO UPDATE SET {updates}"


# ---------------------------------------------------------------------------
# Schema init
# ---------------------------------------------------------------------------

def init_database() -> None:
    """Create all tables, hypertables, indexes (idempotent).

    Applies the FULL schema, migrations included. This is the deliberate
    migration entry point — use it from a script or a shell one-liner, not from
    a process that restarts on its own. See ensure_schema() for the boot path.
    """
    from database.schema_pg import init_timescale
    conn = get_connection()
    try:
        init_timescale(conn)
    finally:
        release_connection(conn)


def ensure_schema() -> bool:
    """Bootstrap the schema only if it is actually missing. Returns True if it ran.

    The boot-time counterpart to init_database(). A fresh database gets built;
    a database that already has every declared table is left completely alone.

    The distinction matters because init_database() is not read-only against an
    existing DB — beyond CREATE TABLE IF NOT EXISTS (a genuine no-op) it runs
    ALTER TABLE ADD COLUMN, CREATE INDEX, and the trade_signals constraint
    migration. Under `bash dev.sh`, watchfiles restarts uvicorn on every save,
    so calling init_database() at startup meant a half-finished edit to
    schema_pg.py was applied to PROD the moment it hit disk, and raced
    deliberate migrations ("tuple concurrently updated").

    Checking for missing tables rather than a single sentinel means a newly
    declared table still gets created on the next boot.
    """
    from database.schema_pg import expected_tables, init_timescale, missing_tables
    conn = get_connection()
    try:
        missing = missing_tables(conn)
        if not missing:
            logger.info("Schema present (%d tables) — nothing to create",
                        len(expected_tables()))
            return False
        logger.info("Schema incomplete — creating %d missing table(s): %s",
                    len(missing), ", ".join(missing))
        init_timescale(conn)
        return True
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
             top_drivers, sentiment, generated_date, generated_at, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        # is_active is in the update list because a conflicting row may already be
        # is_active = FALSE — a third run of the day re-emitting a horizon that the
        # second run dropped collides with the first run's row, which the second
        # run retired. Without this it would stay retired and the fresh signal
        # would never be served.
        sql = _on_conflict_replace(
            base_sql, ["symbol", "generated_date", "model_horizon"],
            ["signal", "confidence", "trade_type", "buy_price", "target_price",
             "stop_loss", "risk_reward", "expected_return_pct", "current_price",
             "atr_14", "atr_pct", "model_name",
             "model_accuracy", "model_precision", "top_drivers", "sentiment",
             "generated_at", "is_active"],
        )

        count = 0
        for t in trades:
            symbol = t["symbol"]
            # Carry forward consumed_volume from the current active signal so
            # capacity tracking survives signal refreshes (EOD overwriting intraday).
            try:
                model_horizon = t.get("model", {}).get("horizon", "")
                cv_row = _execute(conn,
                    "SELECT consumed_volume FROM trade_signals WHERE symbol = ? AND model_horizon = ? AND is_active = TRUE ORDER BY generated_date DESC LIMIT 1",
                    (symbol, model_horizon)
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
                t.get("position", {}).get("max_qty_per_user") or t.get("position", {}).get("suggested_qty_per_user"),
                t.get("position", {}).get("max_investment_per_user") or t.get("position", {}).get("suggested_investment_per_user"),
                t.get("position", {}).get("min_qty"),
                t.get("position", {}).get("recommended_volume"), carried_consumed,
                t.get("model", {}).get("name"), t.get("model", {}).get("horizon"),
                t.get("model", {}).get("accuracy"), t.get("model", {}).get("precision"),
                json.dumps(t.get("top_drivers", [])),
                json.dumps(t.get("sentiment", {})),
                generated_date, generated_at, True,
            ))
            count += 1

        # Retire everything this run did not just write.
        #
        # Runs AFTER the inserts, and is scoped by run rather than by symbol.
        # Both details are load-bearing:
        #
        #  - After, because the consumed_volume carry-forward above reads
        #    is_active = TRUE; deactivating first would blank capacity tracking.
        #  - By run (generated_date + generated_at), because the old
        #    "symbol IN (batch) AND generated_date < ?" left two classes of row
        #    stranded at is_active = TRUE, and the API serves on is_active:
        #      1. Same-date re-runs. The upsert key includes model_horizon, so a
        #         re-run overwrote only the horizons it re-emitted; horizons its
        #         models no longer picked survived. The 2026-08-07 weekly retrain
        #         left 144 such rows from that morning's EOD run live alongside
        #         its own 1,867 (123 symbols).
        #      2. Symbols absent from the batch entirely — never touched by a
        #         symbol-scoped UPDATE, so their last signals stayed active
        #         forever (ANURAS.NS was serving 2026-08-04 signals on 08-10).
        #    Scoping by run covers both: the newest run defines the live set.
        _execute(conn,
            "UPDATE trade_signals SET is_active = FALSE "
            "WHERE is_active = TRUE AND NOT (generated_date = ? AND generated_at = ?)",
            (generated_date, generated_at),
        )

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
        # Literal % in these string labels must be escaped as %% — psycopg2's
        # parameter substitution treats a lone % as the start of a %s/%(name)s
        # placeholder, which previously broke this query with a spurious
        # "IndexError: tuple index out of range" (only 1 bound param, but the
        # unescaped %9/%8/%7/%6/%< sequences looked like more placeholders).
        cur = _execute(conn, """
            SELECT
                CASE
                    WHEN confidence >= 90 THEN '90-100%%'
                    WHEN confidence >= 80 THEN '80-90%%'
                    WHEN confidence >= 70 THEN '70-80%%'
                    WHEN confidence >= 60 THEN '60-70%%'
                    ELSE '<60%%'
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


def get_all_corporate_actions() -> Dict[str, List[Dict]]:
    """
    Load all rows from corporate_actions grouped by nse_symbol.
    Returns {nse_symbol: [{"ex_date": date, "adj_factor": float, "event_type": str}, ...]}
    Sorted ex_date ASC per symbol so compound adjustment applies oldest→newest.
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT nse_symbol, ex_date, adj_factor, event_type "
            "FROM corporate_actions "
            "WHERE adj_factor IS NOT NULL "
            "ORDER BY nse_symbol, ex_date ASC"
        )
        rows = cur.fetchall()
    finally:
        release_connection(conn)

    result: Dict[str, List[Dict]] = {}
    for nse_symbol, ex_date, adj_factor, event_type in rows:
        result.setdefault(nse_symbol, []).append({
            "ex_date":    ex_date,
            "adj_factor": float(adj_factor),
            "event_type": event_type,
        })
    return result


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


def _to_native(o):
    """json.dumps default: coerce numpy scalars (np.float64 etc.) to native
    Python so backtest payloads serialize (pandas/numpy leak np.float64)."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON-serializable: {type(o)}")


def _f(v):
    """Coerce a numpy/None scalar to a plain float (or None) for a DB column."""
    if v is None:
        return None
    return float(v.item() if hasattr(v, "item") else v)


def insert_strategy_backtest(payload: Dict) -> None:
    """Store one strategy-backtest run in strategy_backtest_results.

    `payload` is the full result dict (benchmarks, configs, headline, curves)
    produced by scripts/backtest_signals.py. Headline metrics are also written
    to flat columns; the full document goes in `payload` as JSON text. One row
    per run — the API reads the newest via get_latest_strategy_backtest()."""
    h = payload.get("headline") or {}
    conn = get_connection()
    try:
        _execute(conn,
            """INSERT INTO strategy_backtest_results
                 (window_start, universe_symbols, cost_pct, min_rr, capital, risk_frac,
                  headline_label, headline_cagr, headline_maxdd, payload)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (payload.get("window_start"), payload.get("universe_symbols"),
             _f(payload.get("cost_pct")), _f(payload.get("min_rr")), _f(payload.get("capital")),
             _f(payload.get("risk_frac")), h.get("label"), _f(h.get("cagr_pct")),
             _f(h.get("max_drawdown_pct")), json.dumps(payload, default=_to_native)))
        conn.commit()
    finally:
        release_connection(conn)


def get_nifty_constituents() -> List[Dict]:
    """Nifty 500 constituent list (symbol/name/sector) from the DB.

    Replaces the undeployed data.nifty500_full module — seed with
    scripts/seed_nifty_constituents.py. Returns [] if the table isn't seeded.

    Active constituents only: de-indexed names are kept as is_active = FALSE
    rows for the audit trail, not to be served."""
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT symbol, name, sector FROM nifty_constituents "
                             "WHERE is_active = TRUE ORDER BY symbol")
        return [{"symbol": r[0], "name": r[1], "sector": r[2]} for r in cur.fetchall()]
    finally:
        release_connection(conn)


_sector_map_cache: Dict[str, str] = {}


def get_sector_map() -> Dict[str, str]:
    """symbol (".NS"-suffixed) → sector, from nifty_constituents. Cached in-process.

    Sourced from the DB rather than data/angel_tokens.json: data/ is excluded
    from the HF Space deploy, so the file-backed map was empty in production and
    every signal came back sector="Unknown". Empty dict if the table isn't
    seeded — see scripts/seed_nifty_constituents.py."""
    global _sector_map_cache
    if not _sector_map_cache:
        _sector_map_cache = {
            c["symbol"]: c["sector"]
            for c in get_nifty_constituents() if c.get("sector")
        }
    return _sector_map_cache


def upsert_nifty_constituents(rows: List[Dict]) -> int:
    """Upsert constituent rows [{symbol,name,sector}]. Returns count written."""
    conn = get_connection()
    try:
        for r in rows:
            _execute(conn,
                # is_active/removed_at are reset on conflict so a name that
                # rejoins the index after a review is reactivated rather than
                # staying dormant — the caller passes the current NSE list, so
                # anything in `rows` is by definition a live constituent.
                """INSERT INTO nifty_constituents (symbol, name, sector) VALUES (?,?,?)
                   ON CONFLICT (symbol) DO UPDATE SET
                       name = EXCLUDED.name, sector = EXCLUDED.sector, updated_at = NOW(),
                       is_active = TRUE, removed_at = NULL""",
                (r.get("symbol"), r.get("name"), r.get("sector")))
        conn.commit()
        _sector_map_cache.clear()   # reseeded list must not be masked by the cache
        return len(rows)
    finally:
        release_connection(conn)


def get_active_universe() -> List[str]:
    """The tradable universe: current Nifty 500 constituents only.

    Use this anywhere a job needs "every stock we trade". The obvious
    alternative, `SELECT DISTINCT symbol FROM prices`, is wrong and was the
    source of the weekly retrain training ~537 models instead of 500 — prices
    also holds de-indexed names (their history is kept on purpose for
    backtests) and the 4 index tickers (^NSEI, ^BSESN, ^INDIAVIX, ^CRSLDX),
    which are benchmarks, not stocks. Market features read those benchmarks
    from market_overview, never from a per-symbol prices row, so excluding
    them here costs nothing.

    Falls back to the prices enumeration only if nifty_constituents is empty
    (a fresh DB that has not run the universe sync yet), so a brand-new
    environment still bootstraps instead of silently training nothing.
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT symbol FROM nifty_constituents WHERE is_active = TRUE ORDER BY symbol")
        symbols = [r[0] for r in cur.fetchall()]
        if symbols:
            return symbols

        logger.warning("nifty_constituents is empty — falling back to prices enumeration. "
                       "Run scripts/sync_nifty500_universe.py to populate the universe.")
        cur = _execute(conn,
            "SELECT DISTINCT symbol FROM prices WHERE interval='1d' ORDER BY symbol")
        return [r[0] for r in cur.fetchall()
                if not r[0].startswith("^") and not r[0].startswith("MARKET:")]
    finally:
        release_connection(conn)


def get_recent_fii_dii(days: int = 5) -> List[Dict]:
    """Last `days` trading days of FII/DII net flows from fii_dii_daily (the
    complete source — market_overview's fii/dii columns are sparsely filled)."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT date, fii_net, dii_net FROM fii_dii_daily ORDER BY date DESC LIMIT ?",
            (days,))
        return [{"date": r[0], "fii_net": r[1], "dii_net": r[2]} for r in cur.fetchall()]
    finally:
        release_connection(conn)


def get_market_sentiment() -> Optional[float]:
    """Latest market-wide daily sentiment from news_daily_sentiment (symbol IS
    NULL) — market_overview.overall_sentiment_score is never populated."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT avg_sentiment FROM news_daily_sentiment "
            "WHERE symbol IS NULL ORDER BY date DESC LIMIT 1")
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None
    finally:
        release_connection(conn)


def insert_model_training_stats(run_id: str, rows: List[Dict]) -> int:
    """Upsert per-symbol training metrics for one retrain run into
    model_training_stats. Called by scripts/retrain_walk_forward.py (each shard
    upserts its own symbols under the shared run_id). Replaces the append-only
    data/retrain_results.csv. Idempotent on (run_id, symbol). Returns row count."""
    if not rows:
        return 0
    conn = get_connection()
    try:
        for r in rows:
            _execute(conn,
                """INSERT INTO model_training_stats
                     (run_id, symbol, status, best_model, horizon,
                      accuracy, precision, recall, f1, quality_tier, corp_adj, error)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT (run_id, symbol) DO UPDATE SET
                     status=EXCLUDED.status, best_model=EXCLUDED.best_model,
                     horizon=EXCLUDED.horizon, accuracy=EXCLUDED.accuracy,
                     precision=EXCLUDED.precision, recall=EXCLUDED.recall,
                     f1=EXCLUDED.f1, quality_tier=EXCLUDED.quality_tier,
                     corp_adj=EXCLUDED.corp_adj, error=EXCLUDED.error,
                     created_at=NOW()""",
                (run_id, r.get("symbol"), r.get("status"), r.get("best_model"),
                 r.get("horizon"), _f(r.get("accuracy")), _f(r.get("precision")),
                 _f(r.get("recall")), _f(r.get("f1")), r.get("quality_tier"),
                 bool(r.get("corp_adj")), str(r.get("error") or "")))
        conn.commit()
        return len(rows)
    finally:
        release_connection(conn)


def get_latest_model_training_stats() -> List[Dict]:
    """Per-symbol training metrics from the most recent retrain run only.

    Finds the newest run_id (by created_at) and returns every row for it, so
    /api/backtest/summary reflects a single run — not the CSV's append-only
    blend of all historical runs. Returns [] if no run has been recorded."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT run_id FROM model_training_stats ORDER BY created_at DESC LIMIT 1")
        row = cur.fetchone()
        if not row:
            return []
        run_id = row[0]
        cur = _execute(conn,
            "SELECT symbol, status, best_model, horizon, accuracy, precision, "
            "recall, f1, quality_tier, corp_adj FROM model_training_stats "
            "WHERE run_id = ? ORDER BY symbol", (run_id,))
        return [
            {"symbol": r[0], "status": r[1], "best_model": r[2], "horizon": r[3],
             "accuracy": r[4], "precision": r[5], "recall": r[6], "f1": r[7],
             "quality_tier": r[8], "corp_adj": r[9]}
            for r in cur.fetchall()
        ]
    finally:
        release_connection(conn)


def get_latest_strategy_backtest() -> Optional[Dict]:
    """Return the most recent strategy-backtest result, or None if none run yet."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT payload, generated_at FROM strategy_backtest_results "
            "ORDER BY generated_at DESC LIMIT 1")
        row = cur.fetchone()
    finally:
        release_connection(conn)
    if not row:
        return None
    data = json.loads(row[0])
    data["generated_at"] = str(row[1])
    return data


# ---------------------------------------------------------------------------
# Market holidays / trading calendar
# ---------------------------------------------------------------------------
#
# The NSE trading calendar lives in `market_holidays` (seeded by
# collectors/nse_holidays_collector.py). Everything below reads from that
# table — a trading day is a weekday that is not in it. The whole table is
# tiny (~20 rows/year) so it's cached wholesale in-process, invalidated by
# upsert_market_holidays() and expired by a TTL for the workers that didn't
# do the writing.

_holiday_cache: Optional[Dict[Any, str]] = None
_holiday_cache_at: float = 0.0
# The weekly refresh job runs in one worker only, so the other workers' caches
# would never see an unscheduled closure NSE adds mid-year. A short TTL makes
# every worker self-heal without adding cross-process invalidation.
_HOLIDAY_CACHE_TTL_SEC = 3600


def upsert_market_holidays(rows: List[Dict], segment: str = "CM",
                           exchange: str = "NSE", source: str = "nseindia") -> int:
    """Upsert holiday rows [{date, weekday, description}]. Returns count written.

    `date` may be a `datetime.date` or an ISO 'YYYY-MM-DD' string. Idempotent —
    re-running the collector refreshes descriptions without losing prior years.
    """
    if not rows:
        return 0
    conn = get_connection()
    try:
        for r in rows:
            _execute(conn,
                """INSERT INTO market_holidays
                       (holiday_date, segment, exchange, weekday, description, source)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT (exchange, segment, holiday_date) DO UPDATE SET
                       weekday     = EXCLUDED.weekday,
                       description = EXCLUDED.description,
                       source      = EXCLUDED.source,
                       updated_at  = NOW()""",
                (r.get("date"), segment, exchange, r.get("weekday"), r.get("description"), source))
        conn.commit()
    finally:
        release_connection(conn)
    global _holiday_cache
    _holiday_cache = None
    return len(rows)


def get_market_holidays(start: Optional[str] = None, end: Optional[str] = None,
                        segment: str = "CM", exchange: str = "NSE") -> List[Dict]:
    """Holidays in [start, end] (ISO dates, both optional) ordered ascending."""
    sql = ("SELECT holiday_date, weekday, description FROM market_holidays "
           "WHERE exchange = ? AND segment = ?")
    params: List[Any] = [exchange, segment]
    if start:
        sql += " AND holiday_date >= ?"
        params.append(start)
    if end:
        sql += " AND holiday_date <= ?"
        params.append(end)
    sql += " ORDER BY holiday_date"

    conn = get_connection()
    try:
        cur = _execute(conn, sql, tuple(params))
        return [{"date": str(r[0]), "weekday": r[1], "description": r[2]}
                for r in cur.fetchall()]
    finally:
        release_connection(conn)


def get_holiday_map(segment: str = "CM", exchange: str = "NSE") -> Dict[Any, str]:
    """date → description for every stored holiday. Cached in-process.

    Returns {} when the table has never been seeded — callers must treat an
    empty map as "no calendar", not as "no holidays" (see
    analysis/trading_calendar.py, which refuses to report gaps for years it
    has no holiday coverage for).
    """
    global _holiday_cache, _holiday_cache_at
    if _holiday_cache is None or (time.time() - _holiday_cache_at) > _HOLIDAY_CACHE_TTL_SEC:
        conn = get_connection()
        try:
            cur = _execute(conn,
                "SELECT holiday_date, description FROM market_holidays "
                "WHERE exchange = ? AND segment = ?", (exchange, segment))
            _holiday_cache = {r[0]: (r[1] or "Holiday") for r in cur.fetchall()}
            _holiday_cache_at = time.time()
        finally:
            release_connection(conn)
    return _holiday_cache


def clear_holiday_cache() -> None:
    """Drop the in-process holiday cache — call after changing the table."""
    global _holiday_cache
    _holiday_cache = None


def get_holiday_years(segment: str = "CM", exchange: str = "NSE") -> List[int]:
    """Calendar years the holiday table actually covers (ascending).

    NSE's API serves only the current year, so coverage builds up over time;
    verification must not claim a missing trading day in an uncovered year.
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT DISTINCT EXTRACT(YEAR FROM holiday_date)::int FROM market_holidays "
            "WHERE exchange = ? AND segment = ? ORDER BY 1", (exchange, segment))
        return [int(r[0]) for r in cur.fetchall()]
    finally:
        release_connection(conn)


def get_price_dates(start: str, end: str, interval: str = "1d") -> Dict[Any, int]:
    """date → number of symbols with a daily bar, for dates in [start, end].

    Drives the price-date verification: a date present for only a handful of
    symbols is a partial collection, which is as much a gap as a missing date.
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT date, COUNT(DISTINCT symbol) FROM prices "
            "WHERE interval = ? AND date >= ? AND date <= ? GROUP BY date",
            (interval, start, end))
        return {r[0]: int(r[1]) for r in cur.fetchall()}
    finally:
        release_connection(conn)
