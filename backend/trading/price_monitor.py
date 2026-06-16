"""
TradeMind AI — Price Monitor (Enhanced)

Checks latest prices against SL/Target for paper positions.
Auto-closes positions when SL or Target is hit.

Uses Angel One LTP (live) during market hours, falls back to DB prices.
Run every 5 minutes during market hours (9:15 AM – 3:30 PM IST).
"""
import logging
from datetime import datetime
from typing import List, Dict
from database.db import get_connection, release_connection, _execute
from trading.trading_engine import square_off

logger = logging.getLogger(__name__)


def _is_market_open() -> bool:
    """Check if Indian stock market is currently open."""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=15, second=0)
    market_close = now.replace(hour=15, minute=30, second=0)
    return market_open <= now <= market_close


def _fetch_live_prices(symbols: List[str]) -> Dict[str, float]:
    """
    Try to fetch live LTP from Angel One.
    Returns dict of {symbol: price}. Falls back to empty dict on failure.
    """
    if not _is_market_open():
        logger.info("Market closed — skipping live price fetch")
        return {}

    try:
        from collectors.ltp_fetcher import fetch_ltp_batch
        prices = fetch_ltp_batch(symbols)
        return prices
    except Exception as e:
        logger.error(f"Live price fetch failed: {e}")
        return {}


def _get_db_price(conn, symbol: str) -> float:
    """Get latest close price from DB as fallback."""
    cur = _execute(conn,
        "SELECT close FROM prices WHERE symbol = ? ORDER BY date DESC, time DESC LIMIT 1",
        (symbol,)
    )
    latest = cur.fetchone()
    return float(latest[0]) if latest else 0.0


def _col_names(conn, table: str) -> List[str]:
    cur = _execute(conn, f"SELECT * FROM {table} LIMIT 0")
    return [d[0] for d in cur.description]


def update_position_prices(user_id: int = None) -> List[Dict]:
    """
    Update current prices for all open positions and check SL/Target triggers.

    Flow:
      1. Get all open positions
      2. Fetch live LTP from Angel One (market hours) or DB fallback
      3. Update position's current_price, current_value, unrealized P&L
      4. Check SL trigger → auto square-off at SL price
      5. Check Target trigger → auto square-off at target price

    Returns list of triggered (auto-closed) positions.
    """
    conn = get_connection()
    triggered = []
    try:
        # Get all open positions
        if user_id:
            positions = _execute(conn,
                "SELECT * FROM positions WHERE user_id = ?", (user_id,)
            ).fetchall()
        else:
            positions = _execute(conn, "SELECT * FROM positions").fetchall()

        if not positions:
            return []

        pos_cols = _col_names(conn, "positions")

        # Collect unique symbols for batch LTP fetch
        symbols = list(set(
            dict(zip(pos_cols, pos))["symbol"] for pos in positions
        ))

        # Try live prices first, then fall back to DB
        live_prices = _fetch_live_prices(symbols)
        if live_prices:
            logger.info(f"Using live LTP for {len(live_prices)} symbols")
        else:
            logger.info("Using DB prices (fallback)")

        for pos in positions:
            pos_dict = dict(zip(pos_cols, pos))
            symbol = pos_dict["symbol"]
            uid = pos_dict["user_id"]

            # Get price: live LTP > DB fallback
            current_price = live_prices.get(symbol)
            if not current_price:
                current_price = _get_db_price(conn, symbol)

            if not current_price or current_price <= 0:
                continue

            qty = pos_dict["quantity"]
            invested = pos_dict["invested_amount"]
            current_value = round(qty * current_price, 2)
            pnl = round(current_value - invested, 2)
            pnl_pct = round(pnl / invested * 100, 2) if invested > 0 else 0

            price_source = "LTP" if symbol in live_prices else "DB"

            # Update position with latest price
            _execute(conn, """
                UPDATE positions SET
                    current_price = ?, current_value = ?,
                    unrealized_pnl = ?, unrealized_pnl_pct = ?,
                    updated_at = ?
                WHERE id = ?
            """, (current_price, current_value, pnl, pnl_pct,
                  datetime.now().strftime("%Y-%m-%d %H:%M:%S"), pos_dict["id"]))

            # Check SL trigger
            sl = pos_dict.get("stop_loss")
            target = pos_dict.get("target_price")

            if sl and current_price <= sl:
                conn.commit()
                release_connection(conn)
                conn = None  # mark released so finally doesn't double-release
                logger.warning(
                    f"🛑 STOP LOSS triggered: {symbol} @ ₹{current_price:.2f} "
                    f"(SL: ₹{sl:.2f}) [{price_source}]"
                )
                result = square_off(uid, symbol, sell_price=sl)
                result["trigger"] = "STOP_LOSS"
                result["trigger_price"] = current_price
                result["price_source"] = price_source
                triggered.append(result)
                conn = get_connection()
                continue

            # Check Target trigger
            if target and current_price >= target:
                conn.commit()
                release_connection(conn)
                conn = None
                logger.warning(
                    f"🎯 TARGET triggered: {symbol} @ ₹{current_price:.2f} "
                    f"(Target: ₹{target:.2f}) [{price_source}]"
                )
                result = square_off(uid, symbol, sell_price=target)
                result["trigger"] = "TARGET"
                result["trigger_price"] = current_price
                result["price_source"] = price_source
                triggered.append(result)
                conn = get_connection()
                continue

        if conn is not None:
            conn.commit()
        return triggered

    except Exception:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        raise
    finally:
        if conn is not None:
            release_connection(conn)


def run_monitor():
    """Run the price monitor for all users. Call this every 5 min during market hours."""
    now = datetime.now()
    print(f"\n⏰ Price monitor running at {now.strftime('%H:%M:%S')}...")

    if not _is_market_open():
        print(f"   Market closed (weekday={now.weekday()}, time={now.strftime('%H:%M')})")
        print("   Checking with DB prices for pending triggers...")

    triggered = update_position_prices()

    if triggered:
        for t in triggered:
            emoji = "🎯" if t.get("trigger") == "TARGET" else "🛑"
            src = t.get("price_source", "?")
            print(
                f"   {emoji} {t['symbol']}: {t['trigger']} hit @ ₹{t.get('trigger_price', 0):.2f} "
                f"→ P&L: ₹{t['pnl']:+,.2f} ({t['pnl_pct']:+.1f}%) [{src}]"
            )
    else:
        conn = get_connection()
        positions = _execute(conn,
            "SELECT symbol, current_price, stop_loss, target_price, unrealized_pnl_pct FROM positions"
        ).fetchall()
        release_connection(conn)
        if positions:
            print(f"   {len(positions)} open positions monitored:")
            for p in positions[:10]:
                sym, cp, sl, tp, pnl_pct = p
                status = "✅" if (pnl_pct or 0) >= 0 else "🔴"
                print(f"     {status} {sym}: ₹{cp or 0:.2f} (SL: ₹{sl or 0:.2f} | T: ₹{tp or 0:.2f} | P&L: {pnl_pct or 0:+.1f}%)")
        else:
            print("   No open positions to monitor")

    return triggered


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    run_monitor()
