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


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def get_connection():
    """Return a psycopg2 connection to TimescaleDB."""
    import psycopg2
    conn = psycopg2.connect(
        host=PGHOST, port=PGPORT, dbname=PGDATABASE,
        user=PGUSER, password=PGPASSWORD,
        sslmode="require",
    )
    conn.autocommit = False
    return conn


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
        conn.close()


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
        sql = _on_conflict_ignore(
            """INSERT INTO prices
               (symbol, exchange, date, time, open, high, low, close, volume, interval)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
        conn.close()


def insert_prices_batch(rows: List[Tuple], sync: bool = True) -> int:
    if not rows:
        return 0
    conn = get_connection()
    try:
        sql = _on_conflict_ignore(
            """INSERT INTO prices
               (symbol, exchange, date, time, open, high, low, close, volume, interval)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ["symbol", "date", "time", "interval"],
        )
        _executemany(conn, sql, rows)
        conn.commit()
        logger.info(f"Batch inserted {len(rows)} price rows")
        return len(rows)
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_prices_batch: {e}")
        return 0
    finally:
        conn.close()


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
    conn = get_connection()
    try:
        _execute(conn,
            """INSERT INTO news_sentiment
               (headline, source, published_at, symbol, sentiment, confidence, url)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (headline, source, published_at, symbol, sentiment, confidence, url),
        )
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        logger.error(f"insert_news: {e}")
        return False
    finally:
        conn.close()


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
    db_conn = conn or get_connection()
    try:
        _execute(db_conn,
            """INSERT INTO ai_signals
               (symbol, signal, confidence, model_version,
                target_price, stop_loss, reasoning, features_used)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (symbol, signal, confidence, model_version, target_price, stop_loss,
             json.dumps(reasoning) if reasoning else None,
             json.dumps(features_used) if features_used else None),
        )
        if not conn:
            db_conn.commit()
        return True
    except Exception as e:
        if not conn:
            db_conn.rollback()
        logger.error(f"insert_ai_signal {symbol}: {e}")
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
        count = 0
        for t in trades:
            _execute(conn, sql, (
                t["symbol"], t.get("name", ""), t["signal"], t.get("confidence"),
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
                t.get("position", {}).get("recommended_volume"), 0,
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
        conn.close()


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
        conn.close()


def get_all_prices_df(symbol: str, days: int = 365) -> List[Dict]:
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT date, open, high, low, close, volume FROM prices
               WHERE symbol = ? AND interval = '1d' AND date >= ?
               ORDER BY date ASC""",
            (symbol, start_date),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


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
        conn.close()


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
        conn.close()


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
        conn.close()


def get_top_signals(signal_type: str = "BUY", limit: int = 10) -> List[Dict]:
    conn = get_connection()
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        like_val = f"%{signal_type}%"
        cur = _execute(conn,
            """SELECT * FROM trade_signals WHERE signal LIKE ? AND generated_date = ?
               ORDER BY confidence DESC LIMIT ?""",
            (like_val, today, limit),
        )
        results = _rows_to_dicts(cur)
        if not results:
            cur = _execute(conn,
                """SELECT * FROM trade_signals WHERE signal LIKE ?
                   ORDER BY generated_date DESC, confidence DESC LIMIT ?""",
                (like_val, limit),
            )
            results = _rows_to_dicts(cur)
        return results
    finally:
        conn.close()


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
                f"SELECT * FROM trade_signals WHERE generated_date = ? AND signal LIKE ? {vol_filter} ORDER BY confidence DESC LIMIT ?",
                (date, f"%{signal_type}%", limit),
            )
        else:
            cur = _execute(conn,
                f"SELECT * FROM trade_signals WHERE generated_date = ? {vol_filter} ORDER BY confidence DESC LIMIT ?",
                (date, limit),
            )
        return _rows_to_dicts(cur)
    except Exception as e:
        logger.error(f"get_trade_signals: {e}")
        return []
    finally:
        conn.close()


def get_db_stats() -> Dict[str, int]:
    tables = ["prices", "technical_indicators", "news_sentiment",
              "market_overview", "ai_signals", "trade_signals"]
    conn = get_connection()
    try:
        stats = {}
        for table in tables:
            cur = _execute(conn, f"SELECT COUNT(*) FROM {table}", ())
            stats[table] = cur.fetchone()[0]
        return stats
    finally:
        conn.close()


def get_all_symbols() -> List[str]:
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT DISTINCT symbol FROM prices WHERE interval = '1d' ORDER BY symbol", (),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


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
        conn.close()


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
            "SELECT * FROM trade_signals WHERE generated_date = ? ORDER BY confidence DESC",
            (date,),
        )
        rows = _rows_to_dicts(cur)
    finally:
        conn.close()

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
               SUM(CASE WHEN signal LIKE '%BUY%' THEN 1 ELSE 0 END) as buy_count,
               SUM(CASE WHEN signal LIKE '%SELL%' THEN 1 ELSE 0 END) as sell_count
               FROM trade_signals
               GROUP BY generated_date
               ORDER BY generated_date DESC LIMIT ?""",
            (limit,),
        )
        return _rows_to_dicts(cur)
    finally:
        conn.close()


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
