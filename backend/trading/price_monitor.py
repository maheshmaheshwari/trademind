"""
TradeMind AI — Price Monitor (Enhanced)

Checks latest prices against SL/Target for paper positions.
Auto-closes positions when SL or Target is hit.

Uses Angel One LTP (live) during market hours, falls back to DB prices.
Run every 5 minutes during market hours (9:15 AM – 3:30 PM IST).
"""
import logging
import os
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
    # Tests must never hit the live Angel One API — a real LTP against a
    # fabricated test position can trigger a genuine SL/Target square-off
    # mid-test. DB fallback is safe: the test DB's prices table is truncated.
    if os.getenv("APP_ENV") == "test":
        return {}

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


_MAX_DB_PRICE_STALENESS_DAYS = 4  # covers a long weekend; older than this and we refuse to drive SL/Target off it


def _get_db_price(conn, symbol: str) -> float:
    """Get latest close price from DB as fallback. Refuses stale (multi-day-old) data."""
    cur = _execute(conn,
        "SELECT close, date FROM prices WHERE symbol = ? ORDER BY date DESC, time DESC LIMIT 1",
        (symbol,)
    )
    latest = cur.fetchone()
    if not latest:
        return 0.0
    close, price_date = latest
    age_days = (datetime.now().date() - price_date).days
    if age_days > _MAX_DB_PRICE_STALENESS_DAYS:
        logger.warning(f"DB price for {symbol} is {age_days} days old — refusing to use it for SL/Target")
        return 0.0
    return float(close)


_ALLOWED_TABLES = frozenset({"positions", "orders", "users"})


def _col_names(conn, table: str) -> List[str]:
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not in the allowed list")
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

            # Keep the autopilot mandates' cmp in step with the position.
            #
            # authorized_trades.cmp was written once at authorisation and never
            # touched again, so it always equalled `entry`. The autopilot page
            # renders it as the live market price, which meant every row showed
            # a P&L of exactly +₹0 no matter how far the stock had moved — the
            # screen looked frozen because the number genuinely was. Observed on
            # all 8 of user 2's mandates (e.g. BAJAJFINSV cmp=1764.60 while the
            # stock traded at 2080).
            #
            # Updated here rather than on read: this loop already has a live LTP
            # (or the DB fallback), so it is the one place that knows the price.
            _execute(conn, """
                UPDATE authorized_trades SET cmp = ?, updated_at = NOW()
                 WHERE user_id = ? AND symbol = ? AND status = 'EXECUTED'
            """, (current_price, pos_dict["user_id"], symbol))

            # Commit now so this UPDATE's row lock is released before
            # square_off() (on its own pooled connection) takes its own
            # FOR UPDATE lock on the same row below.
            conn.commit()

            # Check SL trigger
            sl = pos_dict.get("stop_loss")
            target = pos_dict.get("target_price")

            if sl and current_price <= sl:
                logger.warning(
                    f"🛑 STOP LOSS triggered: {symbol} @ ₹{current_price:.2f} "
                    f"(SL: ₹{sl:.2f}) [{price_source}]"
                )
                # square_off() is now self-contained and atomic (FOR UPDATE +
                # its own try/finally) — a concurrent sweep or synchronous
                # request may have already closed this position, in which
                # case it cleanly raises ValueError. Isolate that per
                # position so one already-closed position doesn't abort the
                # rest of the batch (audit finding C2).
                try:
                    # Exit at the market, not at the SL price. A stop triggers
                    # AT the stop and fills at whatever the market is offering —
                    # on a gap that is materially worse. Booking every stop at
                    # `sl` regardless invented the same fill trading_engine's
                    # entry path used to: a stock that opens at ₹240 against an
                    # SL of ₹264 was squared off at ₹264, hiding ₹24 a share of
                    # real loss and understating every drawdown stat built on it.
                    #
                    # min(), so the ordinary case is unchanged: when price drifts
                    # down through the stop, current_price is a hair below `sl`
                    # and that is what fills. Only a gap moves the number.
                    #
                    # The TARGET branch below deliberately does NOT do this — see
                    # the comment there.
                    exit_price = min(current_price, sl)
                    result = square_off(uid, symbol, sell_price=exit_price, trigger="STOP_LOSS")
                    result["trigger"] = "STOP_LOSS"
                    result["trigger_price"] = current_price
                    result["price_source"] = price_source
                    triggered.append(result)
                except Exception as e:
                    logger.warning(f"square_off failed for {symbol} (SL trigger), skipping: {e}")
                continue

            # Check Target trigger
            if target and current_price >= target:
                logger.warning(
                    f"🎯 TARGET triggered: {symbol} @ ₹{current_price:.2f} "
                    f"(Target: ₹{target:.2f}) [{price_source}]"
                )
                try:
                    # Sells at `target`, NOT at current_price — the opposite of
                    # the SL branch above, and deliberately so. The target leg is
                    # a LIMIT sell resting at `target`, so it fills at `target`
                    # the moment the stock trades through it. This loop only
                    # samples every 5 minutes, so current_price here can be well
                    # past the target after a run-up the resting order would have
                    # been filled long before — booking that price would credit a
                    # gain the user could never have captured.
                    result = square_off(uid, symbol, sell_price=target, trigger="TARGET")
                    result["trigger"] = "TARGET"
                    result["trigger_price"] = current_price
                    result["price_source"] = price_source
                    triggered.append(result)
                except Exception as e:
                    logger.warning(f"square_off failed for {symbol} (target trigger), skipping: {e}")
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
