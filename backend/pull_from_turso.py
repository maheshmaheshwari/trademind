"""
Pull data FROM Turso cloud INTO local nifty500.db.

Reads all tables from Turso and writes them into the local SQLite database.
"""
import sys
import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.getcwd())

from database.db import get_remote_turso_connection

LOCAL_DB = "nifty500.db"

# Tables to pull — columns must match the TURSO schema exactly
TABLES = {
    "prices": {
        "columns": "symbol, exchange, date, time, open, high, low, close, volume, interval",
        "conflict": "ON CONFLICT(symbol, date, time, interval) DO NOTHING",
    },
    "technical_indicators": {
        "columns": "symbol, date, rsi_14, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower, sma_20, sma_50, sma_200, ema_9, ema_21, atr_14, adx_14, stoch_k, stoch_d, obv, support_1, support_2, support_3, resistance_1, resistance_2, resistance_3, signal, signal_strength",
        "conflict": "ON CONFLICT(symbol, date) DO NOTHING",
    },
    "news_sentiment": {
        "columns": "headline, source, published_at, symbol, sentiment, confidence, url",
        "conflict": "",
    },
    "news_daily_sentiment": {
        "columns": "date, symbol, avg_sentiment, news_count, positive_count, negative_count, neutral_count, max_positive, max_negative, avg_confidence, source",
        "conflict": "ON CONFLICT(date, symbol) DO NOTHING",
    },
    "ai_signals": {
        "columns": "symbol, generated_at, signal, confidence, model_version, target_price, stop_loss, reasoning, features_used",
        "conflict": "",
    },
    "trade_signals": {
        "columns": "symbol, name, signal, confidence, trade_type, buy_price, target_price, stop_loss, risk_reward, expected_return_pct, current_price, atr_14, atr_pct, avg_daily_volume, daily_turnover_cr, liquidity, max_safe_qty, max_qty_per_user, max_investment_per_user, min_qty, recommended_volume, consumed_volume, model_name, model_horizon, model_accuracy, model_precision, top_drivers, sentiment, generated_date, generated_at",
        "conflict": "",
    },
    "market_overview": {
        "columns": "*",
        "conflict": "",
        "use_star": True,
    },
}


def pull_table(table_name, config, remote_conn, local_conn):
    """Pull a single table from Turso into local DB."""
    columns = config["columns"]
    conflict = config.get("conflict", "")
    use_star = config.get("use_star", False)

    print(f"\n📥 Pulling '{table_name}'...")

    # Read from Turso
    try:
        remote_cursor = remote_conn.cursor()
        remote_cursor.execute(f"SELECT {columns} FROM {table_name}")
        rows = remote_cursor.fetchall()
        
        # If using *, get column names from cursor description (excluding id)
        if use_star and remote_cursor.description:
            col_names = [desc[0] for desc in remote_cursor.description if desc[0] != 'id']
            # Filter rows to exclude id column
            id_idx = next((i for i, desc in enumerate(remote_cursor.description) if desc[0] == 'id'), None)
            if id_idx is not None:
                rows = [tuple(v for i, v in enumerate(row) if i != id_idx) for row in rows]
            columns = ", ".join(col_names)
    except Exception as e:
        print(f"   ⚠️  Error reading '{table_name}' from Turso: {e}")
        return 0

    if not rows:
        print(f"   ℹ️  No data in Turso for '{table_name}'")
        return 0

    print(f"   Found {len(rows)} rows in Turso")

    col_list = [c.strip() for c in columns.split(",")]
    placeholders = ", ".join(["?" for _ in col_list])

    # Write to local DB
    local_cursor = local_conn.cursor()
    written = 0

    for i, row in enumerate(rows):
        try:
            sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders}) {conflict}"
            local_cursor.execute(sql, row)
            written += 1
        except sqlite3.IntegrityError:
            pass
        except Exception:
            pass

        if (i + 1) % 10000 == 0:
            local_conn.commit()
            print(f"   ...written {i + 1}/{len(rows)}")

    local_conn.commit()
    print(f"   ✅ Wrote {written} new rows to local '{table_name}'")
    return written


def main():
    print("=" * 60)
    print("📦 PULLING DATA FROM TURSO → LOCAL nifty500.db")
    print("=" * 60)

    print("\n1. Connecting to Turso cloud...")
    remote_conn = get_remote_turso_connection()

    print(f"2. Opening local database: {LOCAL_DB}")
    local_conn = sqlite3.connect(LOCAL_DB, timeout=30)
    local_conn.execute("PRAGMA journal_mode=WAL")
    local_conn.execute("PRAGMA synchronous=NORMAL")

    total = 0
    for table_name, config in TABLES.items():
        try:
            count = pull_table(table_name, config, remote_conn, local_conn)
            total += count
        except Exception as e:
            print(f"   ❌ Failed to pull '{table_name}': {e}")

    local_conn.close()

    print(f"\n{'=' * 60}")
    print(f"✅ DONE! Wrote {total} total new rows to local DB.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
