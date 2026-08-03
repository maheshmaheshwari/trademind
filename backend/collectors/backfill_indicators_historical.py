"""
TradeMind — Historical Technical Indicator Backfill

Computes RSI, MACD, Bollinger Bands, SMA, EMA, ATR, ADX, Stochastic, OBV,
Support/Resistance for EVERY trading day in the prices table and stores one
row per (symbol, date) in technical_indicators.

The current process_all_stocks() only stores the latest date per stock.
This script fills all historical dates so the ML model has complete
indicator history for retraining.

Usage:
    PYTHONPATH=. python collectors/backfill_indicators_historical.py
    PYTHONPATH=. python collectors/backfill_indicators_historical.py --symbol HDFCBANK.NS
"""

import argparse
import logging
import math
import sys
import os

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.indicators import calculate_all
from database.db import (
    get_connection,
    release_connection,
    init_database,
    _execute,
    _executemany,
    _on_conflict_replace,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(value):
    if value is None:
        return None
    try:
        f = float(value)
        return round(f, 4) if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


def find_missing_indicator_dates(symbols=None, conn=None):
    """Map symbol -> sorted price dates that have NO technical_indicators row.

    Every price date is expected to have one: indicators are stored from a
    symbol's very first bar (NULL-valued where the lookback isn't satisfied
    yet), so a price date without a row is a genuine gap, never warmup.

    The gaps are not what they look like from row counts alone. RELIANCE and TCS
    were each short by 15 rows, and the missing dates are identical between them
    (2026-06-09, 06-12, 06-29 ... 07-14, 07-15, 07-20) — universe-wide days when
    calculate_indicators_job did not run, not anything symbol-specific. A count
    or ratio comparison would call a symbol "close enough" and leave those days
    permanently empty, which is why this compares dates.
    """
    own = conn is None
    if own:
        conn = get_connection()
    try:
        # Deliberately NOT ordered in SQL. This anti-join used to end in
        # `ORDER BY p.symbol, p.date`, and ten concurrent shards each sorting it
        # exhausted the Timescale instance —
        #   psycopg2.errors.OutOfMemory
        #   DETAIL: Failed on request of size 32 in memory context "Caller tuples"
        # which killed 4 of 10 shards in run 30808272331. It is the same sort
        # node that had to be removed from model_training.prefetch_all_data for
        # the same reason. Grouping into a dict does not care about row order,
        # and sorting each symbol's handful of dates in Python is free.
        sql = """SELECT p.symbol, p.date
                   FROM prices p
                   LEFT JOIN technical_indicators t
                          ON t.symbol = p.symbol AND t.date = p.date
                  WHERE p.interval = '1d' AND t.date IS NULL"""
        params = ()
        if symbols:
            sql += " AND p.symbol = ANY(?)"
            params = (list(symbols),)

        out = {}
        for sym, d in _execute(conn, sql, params).fetchall():
            out.setdefault(sym, []).append(d)
        for dates in out.values():
            dates.sort()
        return out
    finally:
        if own:
            release_connection(conn)


def backfill_missing_indicators(symbols=None, conn=None) -> dict:
    """Compute indicators for exactly the dates that are missing them.

    Symbols already complete are skipped without computing anything — the point
    is that a routine run costs nothing when there is nothing wrong.

    Indicators are still computed over the symbol's FULL price history, because
    every one of them is a lookback (sma_200 needs 200 prior bars); only the
    missing dates are then written back. Computing just the gap dates in
    isolation would produce wrong values.

    Returns symbol -> rows written.
    """
    own = conn is None
    if own:
        conn = get_connection()
    try:
        missing = find_missing_indicator_dates(symbols, conn=conn)
        if not missing:
            logger.info("Indicators: every price date already has a row — nothing to do")
            return {}

        total_dates = sum(len(v) for v in missing.values())
        logger.info(f"Indicators: {len(missing)} symbol(s) with gaps, "
                    f"{total_dates} missing date(s) — backfilling")

        written = {}
        for sym, dates in missing.items():
            want = {d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)
                    for d in dates}
            n = backfill_symbol(sym, conn=conn, only_dates=want)
            if n:
                written[sym] = n
                logger.info(f"   {sym:16} filled {n}/{len(want)} missing date(s)")
        return written
    finally:
        if own:
            release_connection(conn)


def backfill_symbol(symbol: str, conn=None, only_dates=None) -> int:
    """
    Compute and store indicators for every date in prices for one symbol.
    Returns number of rows inserted/updated.

    only_dates: optional set of 'YYYY-MM-DD' strings. Indicators are still
    computed across the symbol's full history (they are all lookbacks), but only
    these dates are written. Used by backfill_missing_indicators() to repair
    gaps without rewriting rows that are already correct.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()

    try:
        cur = _execute(conn,
            """SELECT date, open, high, low, close, volume
               FROM prices
               WHERE symbol = ? AND interval = '1d'
               ORDER BY date ASC""",
            (symbol,)
        )
        rows = cur.fetchall()
        if len(rows) < 14:
            logger.warning(f"{symbol}: only {len(rows)} rows — skipping")
            return 0

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        df["date"] = pd.to_datetime(df["date"])

        # Compute all indicators across the full history
        df = calculate_all(df)

        # Build all rows as tuples for a single batch insert
        batch = []
        for _, row in df.iterrows():
            date_str = row["date"].strftime("%Y-%m-%d")
            if only_dates is not None and date_str not in only_dates:
                continue
            batch.append((
                symbol,
                date_str,
                _safe_float(row.get("rsi_14")),
                _safe_float(row.get("macd")),
                _safe_float(row.get("macd_signal")),
                _safe_float(row.get("macd_hist")),
                _safe_float(row.get("bb_upper")),
                _safe_float(row.get("bb_middle")),
                _safe_float(row.get("bb_lower")),
                _safe_float(row.get("sma_20")),
                _safe_float(row.get("sma_50")),
                _safe_float(row.get("sma_200")),
                _safe_float(row.get("ema_9")),
                _safe_float(row.get("ema_21")),
                _safe_float(row.get("atr_14")),
                _safe_float(row.get("adx_14")),
                _safe_float(row.get("stoch_k")),
                _safe_float(row.get("stoch_d")),
                _safe_float(row.get("obv")),
                _safe_float(row.get("support_1")),
                _safe_float(row.get("support_2")),
                _safe_float(row.get("support_3")),
                _safe_float(row.get("resistance_1")),
                _safe_float(row.get("resistance_2")),
                _safe_float(row.get("resistance_3")),
                None,   # signal
                None,   # signal_strength
            ))

        if not batch:
            return 0

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

        # Single batch insert — all rows in one network round-trip
        _executemany(conn, sql, batch)
        conn.commit()

        return len(batch)

    except Exception as e:
        logger.error(f"{symbol}: {e}")
        return 0
    finally:
        if own_conn:
            release_connection(conn)


def backfill_all(symbol_filter: str = None):
    init_database()

    # Active constituents only — de-indexed names and index tickers still have
    # price history, but computing indicators for them is wasted work.
    from database.db import get_active_universe
    all_symbols = get_active_universe()

    if symbol_filter:
        all_symbols = [s for s in all_symbols if s == symbol_filter]
        if not all_symbols:
            print(f"Symbol {symbol_filter} not found in prices table.")
            return

    total_symbols = len(all_symbols)
    total_rows = 0
    failed = []

    print(f"\n📊 Backfilling indicators for {total_symbols} stocks...\n")

    for symbol in tqdm(all_symbols, desc="Backfilling", unit="stock"):
        rows = backfill_symbol(symbol)
        if rows > 0:
            total_rows += rows
        else:
            failed.append(symbol)

    print(f"\n{'='*60}")
    print(f"✅ Done: {total_symbols - len(failed)}/{total_symbols} stocks")
    print(f"📊 Total rows inserted/updated: {total_rows:,}")
    if failed:
        print(f"❌ Failed ({len(failed)}): {', '.join(failed[:20])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", help="Run for a single symbol only (e.g. HDFCBANK.NS)")
    args = parser.parse_args()
    backfill_all(symbol_filter=args.symbol)
