"""
Migrate all data from nifty500.db (SQLite) → TimescaleDB (PostgreSQL).

Tables migrated in order (respecting FK constraints):
  1. users
  2. portfolios → portfolio_sectors, portfolio_stocks
  3. risk_settings
  4. trade_signals
  5. ai_signals
  6. orders
  7. positions
  8. market_overview
  9. prices            (largest — ~558K rows, chunked)
  10. technical_indicators (~387K rows, chunked)
  11. news_sentiment    (~250 rows)

Usage:
    python migrate_sqlite_to_pg.py              # full migration
    python migrate_sqlite_to_pg.py --table prices  # single table
    python migrate_sqlite_to_pg.py --dry-run    # count rows, no writes
"""

import argparse
import sqlite3
import sys
import os
import time

import psycopg2
import psycopg2.extras

SQLITE_PATH = "nifty500.db"

PG_DSN = {
    "host":     os.getenv("PGHOST", "localhost"),
    "port":     int(os.getenv("PGPORT", "5433")),
    "dbname":   os.getenv("PGDATABASE", "trademind"),
    "user":     os.getenv("PGUSER", "trademind"),
    "password": os.getenv("PGPASSWORD", "trademind"),
}

CHUNK = 5_000  # rows per INSERT batch


def sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def pg_conn():
    conn = psycopg2.connect(**PG_DSN)
    conn.autocommit = False
    return conn


# ---------------------------------------------------------------------------
# Column definitions — only the columns that exist in BOTH DBs
# (SQLite schema may differ slightly from schema_pg.py)
# ---------------------------------------------------------------------------

TABLE_DEFS = {
    "users": {
        "cols": ["id", "username", "email", "password_hash", "display_name",
                 "virtual_balance", "virtual_invested", "total_pnl",
                 "win_count", "loss_count", "mode", "angel_client_id", "created_at"],
        "conflict": "ON CONFLICT (username) DO NOTHING",
    },
    "portfolios": {
        "cols": ["id", "name", "investment_amount", "time_horizon", "risk_profile",
                 "created_at", "updated_at"],
        "conflict": "ON CONFLICT (id) DO NOTHING",
    },
    "portfolio_sectors": {
        "cols": ["id", "portfolio_id", "sector", "allocation_pct",
                 "ai_suggested_pct", "num_stocks"],
        "conflict": "ON CONFLICT (id) DO NOTHING",
    },
    "portfolio_stocks": {
        "cols": ["id", "portfolio_id", "symbol", "sector", "signal",
                 "confidence", "buy_price", "target_price", "stop_loss",
                 "allocated_amount", "quantity", "status", "added_at"],
        "conflict": "ON CONFLICT (id) DO NOTHING",
    },
    "risk_settings": {
        "cols": ["id", "user_id", "max_daily_loss", "max_daily_trades",
                 "max_position_pct", "auto_stop_loss", "auto_target"],
        "conflict": "ON CONFLICT (user_id) DO NOTHING",
    },
    "trade_signals": {
        "cols": ["id", "symbol", "name", "signal", "confidence", "trade_type",
                 "buy_price", "target_price", "stop_loss", "risk_reward",
                 "expected_return_pct", "current_price", "atr_14", "atr_pct",
                 "avg_daily_volume", "daily_turnover_cr", "liquidity",
                 "max_safe_qty", "max_qty_per_user", "max_investment_per_user",
                 "min_qty", "recommended_volume", "consumed_volume",
                 "model_name", "model_horizon", "model_accuracy",
                 "model_precision", "top_drivers", "sentiment",
                 "generated_date", "generated_at"],
        "conflict": "ON CONFLICT (symbol, generated_date) DO NOTHING",
    },
    "ai_signals": {
        "cols": ["id", "symbol", "generated_at", "signal", "confidence",
                 "model_version", "target_price", "stop_loss", "reasoning",
                 "features_used"],
        "conflict": "ON CONFLICT (symbol, generated_at, model_version) DO NOTHING",
    },
    "orders": {
        "cols": ["id", "user_id", "bracket_id", "order_id", "symbol", "name",
                 "exchange", "order_type", "order_purpose", "quantity", "price",
                 "trigger_price", "status", "mode", "signal", "confidence",
                 "horizon", "fill_price", "fees", "pnl", "created_at", "updated_at"],
        "conflict": "ON CONFLICT (id) DO NOTHING",
    },
    "positions": {
        "cols": ["id", "user_id", "symbol", "name", "quantity", "avg_buy_price",
                 "current_price", "target_price", "stop_loss", "unrealized_pnl",
                 "unrealized_pnl_pct", "invested_amount", "current_value",
                 "mode", "bracket_id", "updated_at"],
        "conflict": "ON CONFLICT (user_id, symbol) DO NOTHING",
    },
    "market_overview": {
        "cols": ["date", "nifty500_close", "nifty500_change_pct",
                 "nifty50_close", "nifty50_change_pct", "sensex_close",
                 "india_vix", "advances", "declines", "unchanged",
                 "total_volume", "fii_net", "dii_net",
                 "overall_sentiment_score", "fear_greed_label"],
        "conflict": "ON CONFLICT (date) DO NOTHING",
    },
    "prices": {
        "cols": ["symbol", "exchange", "date", "time", "open", "high",
                 "low", "close", "volume", "interval"],
        "conflict": "ON CONFLICT (symbol, date, time, interval) DO NOTHING",
    },
    "technical_indicators": {
        "cols": ["symbol", "date", "rsi_14", "macd", "macd_signal",
                 "macd_hist", "bb_upper", "bb_middle", "bb_lower",
                 "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
                 "atr_14", "adx_14", "stoch_k", "stoch_d", "obv",
                 "support_1", "support_2", "support_3",
                 "resistance_1", "resistance_2", "resistance_3",
                 "signal", "signal_strength"],
        "conflict": "ON CONFLICT (symbol, date) DO NOTHING",
    },
    "news_sentiment": {
        "cols": ["headline", "source", "published_at", "symbol",
                 "sentiment", "confidence", "url", "created_at"],
        "conflict": "",  # no unique constraint — insert all
    },
}

# Ordered so FK parents come before children
MIGRATION_ORDER = [
    "users",
    "portfolios",
    "portfolio_sectors",
    "portfolio_stocks",
    "risk_settings",
    "trade_signals",
    "ai_signals",
    "orders",
    "positions",
    "market_overview",
    "prices",
    "technical_indicators",
    "news_sentiment",
]


def get_sqlite_cols(sqlite, table: str) -> set[str]:
    """Return the set of column names that actually exist in the SQLite table."""
    rows = sqlite.execute(f"PRAGMA table_info({table})").fetchall()
    return {r["name"] for r in rows}


def migrate_table(sqlite, pg, table: str, dry_run: bool) -> int:
    defn = TABLE_DEFS[table]
    available = get_sqlite_cols(sqlite, table)

    # Only migrate columns that exist in both SQLite and PG schema
    cols = [c for c in defn["cols"] if c in available]
    if not cols:
        print(f"  {table}: no matching columns — skipped")
        return 0

    total = sqlite.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {total:,} rows → ", end="", flush=True)

    if dry_run or total == 0:
        print("(dry-run)" if dry_run else "empty")
        return total

    col_list = ", ".join(cols)
    placeholders = ", ".join(["%s"] * len(cols))
    conflict = defn["conflict"]
    sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) {conflict}"

    cur = pg.cursor()
    migrated = 0
    offset = 0

    while offset < total:
        rows = sqlite.execute(
            f"SELECT {col_list} FROM {table} LIMIT {CHUNK} OFFSET {offset}"
        ).fetchall()
        if not rows:
            break

        batch = [tuple(r[c] for c in cols) for r in rows]
        psycopg2.extras.execute_batch(cur, sql, batch, page_size=1000)
        migrated += len(batch)
        offset += CHUNK
        print(f"{migrated:,}...", end="", flush=True)

    pg.commit()
    print(f"done ({migrated:,})")
    return migrated


def reset_sequences(pg):
    """Reset all BIGSERIAL sequences to max(id)+1 after bulk insert."""
    tables_with_id = [
        "users", "portfolios", "portfolio_sectors", "portfolio_stocks",
        "risk_settings", "trade_signals", "ai_signals", "orders", "positions",
        "prices", "technical_indicators", "news_sentiment",
    ]
    cur = pg.cursor()
    for t in tables_with_id:
        try:
            cur.execute(f"""
                SELECT setval(pg_get_serial_sequence('{t}', 'id'),
                              COALESCE(MAX(id), 1))
                FROM {t}
            """)
        except Exception:
            pg.rollback()
    pg.commit()
    cur.close()
    print("  sequences reset")


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite → TimescaleDB")
    parser.add_argument("--table", help="Migrate only this table")
    parser.add_argument("--dry-run", action="store_true", help="Count rows only, no writes")
    args = parser.parse_args()

    if not os.path.exists(SQLITE_PATH):
        print(f"ERROR: {SQLITE_PATH} not found")
        sys.exit(1)

    tables = [args.table] if args.table else MIGRATION_ORDER

    # Validate
    for t in tables:
        if t not in TABLE_DEFS:
            print(f"Unknown table: {t}. Valid: {list(TABLE_DEFS)}")
            sys.exit(1)

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Migrating {SQLITE_PATH} → TimescaleDB {PG_DSN['host']}:{PG_DSN['port']}/{PG_DSN['dbname']}")
    print()

    sqlite = sqlite_conn()
    pg = pg_conn() if not args.dry_run else None

    t0 = time.time()
    total_rows = 0

    for table in tables:
        rows = migrate_table(sqlite, pg, table, args.dry_run)
        total_rows += rows

    if not args.dry_run:
        print("\nResetting sequences...")
        reset_sequences(pg)
        pg.close()

    sqlite.close()
    elapsed = time.time() - t0
    print(f"\nDone. {total_rows:,} rows in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
