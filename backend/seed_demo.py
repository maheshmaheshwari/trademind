"""
TradeMind AI — Demo Data Seeder

Seeds a demo portfolio, positions, and paper trade orders
for investor prototype demonstration.

Usage:
    PYTHONPATH=. python seed_demo.py
"""
import os
import sys
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from database.db import get_connection, release_connection, _execute

USER_ID = 1  # maheshmaheshwari983@gmail.com
INVESTMENT = 1_000_000  # ₹10 lakh demo portfolio

SECTOR_MAP = {
    "HDFCBANK.NS": "Banking",    "ICICIBANK.NS": "Banking",   "SBIN.NS": "Banking",
    "BAJAJFINSV.NS": "Finance",  "CHOLAFIN.NS": "Finance",    "CHOLAHLDNG.NS": "Finance",
    "ERIS.NS": "Pharma",         "CONCORDBIO.NS": "Pharma",   "DABUR.NS": "FMCG",
    "AFFLE.NS": "Technology",    "DIXON.NS": "Technology",
    "CHAMBLFERT.NS": "Chemicals","DEEPAKFERT.NS": "Chemicals","DCMSHRIRAM.NS": "Chemicals",
    "DBREALTY.NS": "Real Estate",
}


def seed():
    conn = get_connection()
    try:
        # ── 1. Load top buy signals from DB ───────────────────────────────────
        cur = _execute(conn, """
            SELECT symbol, name, signal, confidence, buy_price, target_price,
                   stop_loss, risk_reward, model_horizon
            FROM trade_signals
            WHERE signal IN ('STRONG BUY', 'BUY') AND is_active = TRUE
            ORDER BY confidence DESC
            LIMIT 15
        """, ())
        cols = [d[0] for d in cur.description]
        signals = [dict(zip(cols, row)) for row in cur.fetchall()]

        if not signals:
            print("No active signals found — run generate_trades.py first")
            return

        print(f"Loaded {len(signals)} buy signals from DB")

        # ── 2. Clear existing demo data ───────────────────────────────────────
        # Get existing TradeMind portfolio IDs first
        cur = _execute(conn, "SELECT id FROM portfolios WHERE name LIKE ?", ("TradeMind%",))
        existing_ids = [row[0] for row in cur.fetchall()]
        if existing_ids:
            for pid in existing_ids:
                _execute(conn, "DELETE FROM portfolio_stocks WHERE portfolio_id = ?", (pid,))
                _execute(conn, "DELETE FROM portfolio_sectors WHERE portfolio_id = ?", (pid,))
            _execute(conn, "DELETE FROM portfolios WHERE name LIKE ?", ("TradeMind%",))

        _execute(conn, "DELETE FROM positions WHERE user_id = ?", (USER_ID,))
        _execute(conn, "DELETE FROM orders WHERE user_id = ?", (USER_ID,))
        _execute(conn, "DELETE FROM risk_settings WHERE user_id = ?", (USER_ID,))

        # ── 3. Create portfolio ───────────────────────────────────────────────
        cur = _execute(conn, """
            INSERT INTO portfolios (name, investment_amount, time_horizon, risk_profile)
            VALUES (?, ?, ?, ?) RETURNING id
        """, ("TradeMind AI Portfolio", INVESTMENT, "medium", "moderate"))
        portfolio_id = cur.fetchone()[0]
        print(f"Created portfolio id={portfolio_id}")

        # ── 4. Sector allocation ──────────────────────────────────────────────
        sector_alloc = {}
        for s in signals:
            sector = SECTOR_MAP.get(s["symbol"], "Others")
            sector_alloc[sector] = sector_alloc.get(sector, 0) + 1

        for sector, count in sector_alloc.items():
            pct = round(count / len(signals) * 100, 1)
            _execute(conn, """
                INSERT INTO portfolio_sectors
                (portfolio_id, sector, allocation_pct, ai_suggested_pct, num_stocks)
                VALUES (?, ?, ?, ?, ?)
            """, (portfolio_id, sector, pct, pct, count))

        # ── 5. Portfolio stocks ───────────────────────────────────────────────
        per_stock = INVESTMENT / len(signals)
        for s in signals:
            buy = s["buy_price"] or 100
            qty = max(1, int(per_stock / buy))
            sector = SECTOR_MAP.get(s["symbol"], "Others")
            _execute(conn, """
                INSERT INTO portfolio_stocks
                (portfolio_id, symbol, sector, signal, confidence, buy_price,
                 target_price, stop_loss, allocated_amount, quantity, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (portfolio_id, s["symbol"], sector, s["signal"], s["confidence"],
                  s["buy_price"], s["target_price"], s["stop_loss"],
                  round(per_stock, 2), qty, "recommended"))
        print(f"Seeded {len(signals)} portfolio stocks")

        # ── 6. Paper trade orders (mix of executed + pending) ─────────────────
        now = datetime.now()
        order_count = 0
        for i, s in enumerate(signals[:10]):
            buy = s["buy_price"] or 100
            qty = max(1, int(per_stock / buy))
            days_ago = random.randint(1, 14)
            created = now - timedelta(days=days_ago)
            status = "COMPLETE" if i < 7 else "PENDING"
            fill = round(buy * random.uniform(0.995, 1.005), 2) if status == "COMPLETE" else None
            pnl = None
            if fill:
                current = buy * random.uniform(0.97, 1.08)
                pnl = round((current - fill) * qty, 2)

            _execute(conn, """
                INSERT INTO orders
                (user_id, symbol, name, exchange, order_type, order_purpose, quantity,
                 price, status, mode, signal, confidence, horizon, fill_price, pnl, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (USER_ID, s["symbol"], s.get("name") or "", "NSE", "LIMIT", "ENTRY",
                  qty, buy, status, "PAPER", s["signal"], s["confidence"],
                  s.get("model_horizon", ""), fill, pnl, created))
            order_count += 1
        print(f"Seeded {order_count} paper orders")

        # ── 7. Open positions from completed orders ───────────────────────────
        pos_count = 0
        for s in signals[:7]:
            buy = s["buy_price"] or 100
            qty = max(1, int(per_stock / buy))
            avg_price = round(buy * random.uniform(0.995, 1.005), 2)
            current = round(avg_price * random.uniform(0.97, 1.09), 2)
            invested = round(avg_price * qty, 2)
            current_val = round(current * qty, 2)
            pnl = round(current_val - invested, 2)
            pnl_pct = round((pnl / invested) * 100, 2)

            _execute(conn, """
                INSERT INTO positions
                (user_id, symbol, name, quantity, avg_buy_price, current_price,
                 target_price, stop_loss, unrealized_pnl, unrealized_pnl_pct,
                 invested_amount, current_value, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (user_id, symbol) DO UPDATE SET
                  current_price = EXCLUDED.current_price,
                  unrealized_pnl = EXCLUDED.unrealized_pnl,
                  unrealized_pnl_pct = EXCLUDED.unrealized_pnl_pct,
                  current_value = EXCLUDED.current_value,
                  updated_at = NOW()
            """, (USER_ID, s["symbol"], s.get("name") or "", qty,
                  avg_price, current, s["target_price"], s["stop_loss"],
                  pnl, pnl_pct, invested, current_val, "PAPER"))
            pos_count += 1
        print(f"Seeded {pos_count} open positions")

        # ── 8. Risk settings ──────────────────────────────────────────────────
        _execute(conn, """
            INSERT INTO risk_settings
            (user_id, max_daily_loss, max_daily_trades, max_position_pct,
             max_position_size, stop_loss_pct, target_pct, auto_stop_loss, auto_target, mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (USER_ID, 15000, 5, 15, 75000, 7, 15, True, True, "PAPER"))
        print("Seeded risk settings")

        conn.commit()
        print("\n✅ Demo data seeded successfully!")
        print(f"   Portfolio: ₹{INVESTMENT:,.0f} across {len(signals)} AI-selected stocks")
        print(f"   Orders: {order_count} paper trades (7 complete, 3 pending)")
        print(f"   Positions: {pos_count} open positions with simulated P&L")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
    finally:
        release_connection(conn)


if __name__ == "__main__":
    seed()
