"""
TradeMind AI — Trading Engine

Handles virtual (paper) and live trading:
- User account creation with ₹10L virtual balance
- Auto bracket orders: BUY + STOP_LOSS + TARGET
- Position tracking with P&L
- Square-off (sell) positions
"""
import json
import logging
import os
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from database.db import get_connection, release_connection, _execute, get_active_signal_id
from trading.risk_manager import check_order

_angel_log = logging.getLogger(__name__)


class PartialCapacityError(Exception):
    """Raised when a user's requested qty exceeds remaining platform capacity for a signal."""
    def __init__(self, symbol: str, requested: int, available: int):
        self.symbol    = symbol
        self.requested = requested
        self.available = available
        super().__init__(
            f"{symbol}: requested {requested} shares but only {available} platform capacity remains."
        )


class RiskCheckFailed(Exception):
    """Raised when check_order() rejects a trade — carries the same shape the API previously
    returned directly. Enforced inside execute_signal() itself (audit H8) under the same
    advisory lock as the rest of the trade (audit H9), so no caller can bypass risk checks
    and no concurrent request can act on a stale pre-check snapshot."""
    def __init__(self, reason: str, checks: list):
        self.reason = reason
        self.checks = checks
        super().__init__(reason)

# ── B9: Cached Angel One session ────────────────────────────────────────────
# One session is created and reused across all LIVE order calls.
# Refreshed automatically when expired (every 6 hours) or on 401.
class _AngelSessionCache:
    def __init__(self):
        self._lock       = threading.Lock()
        self._api        = None
        self._expires_at = datetime.min

    def get(self):
        """Return a live SmartConnect session, re-authenticating if needed."""
        with self._lock:
            if self._api is not None and datetime.now() < self._expires_at:
                return self._api
            return self._refresh()

    def _refresh(self):
        try:
            from SmartApi import SmartConnect
            import pyotp
            api_key      = os.getenv("ANGEL_API_KEY", "")
            client_id    = os.getenv("ANGEL_CLIENT_ID", "")
            password     = os.getenv("ANGEL_MPIN", "") or os.getenv("ANGEL_PASSWORD", "")
            totp_secret  = os.getenv("ANGEL_TOTP_SECRET", "")
            if not all([api_key, client_id, password, totp_secret]):
                _angel_log.error("Angel One credentials incomplete — check .env")
                return None
            api = SmartConnect(api_key=api_key)
            totp = pyotp.TOTP(totp_secret).now()
            session = api.generateSession(client_id, password, totp)
            if not session or session.get("status") is False:
                _angel_log.error(f"Angel One login failed: {session.get('message')}")
                return None
            self._api        = api
            self._expires_at = datetime.now() + timedelta(hours=6)
            _angel_log.info(f"Angel One session refreshed — valid until {self._expires_at:%H:%M}")
            return api
        except Exception as e:
            _angel_log.error(f"Angel One session refresh error: {e}")
            self._api = None
            return None

    def invalidate(self):
        """Force re-login on next call (e.g. after receiving 401)."""
        with self._lock:
            self._api        = None
            self._expires_at = datetime.min

_angel_cache = _AngelSessionCache()


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today():
    return datetime.now().strftime("%Y-%m-%d")


def _fetchone(conn, sql: str, params: tuple = ()) -> Optional[tuple]:
    """Execute a query and return one row."""
    cur = _execute(conn, sql, params)
    return cur.fetchone()


def _fetchall(conn, sql: str, params: tuple = ()) -> List[tuple]:
    """Execute a query and return all rows."""
    cur = _execute(conn, sql, params)
    return cur.fetchall()


_ALLOWED_TABLES = frozenset({
    "users", "orders", "positions", "risk_settings", "trade_signals",
    "authorized_trades", "autopilot_settings", "user_signal_volume",
    "user_sessions", "broker_connections", "watchlist",
})


def _col_names(conn, table: str) -> List[str]:
    """Return column names for a table."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not in the allowed list")
    cur = _execute(conn, f"SELECT * FROM {table} LIMIT 0")
    return [d[0] for d in cur.description]


# ==========================================
# USER MANAGEMENT
# ==========================================

def _safe_user(user_dict: Dict) -> Dict:
    """Strip sensitive fields from user dict before returning to API."""
    return {k: v for k, v in user_dict.items() if k != "password_hash"}


def create_user(username: str, password_hash: str, display_name: str = None, email: str = None) -> Dict:
    """Create a virtual trading account with ₹10,00,000 starting balance."""
    conn = get_connection()
    try:
        _execute(
            conn,
            "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, display_name or username)
        )
        conn.commit()
        user = _fetchone(conn, "SELECT * FROM users WHERE username = ?", (username,))
        cols = _col_names(conn, "users")
        release_connection(conn)
        return dict(zip(cols, user))
    except Exception as e:
        release_connection(conn)
        try:
            import psycopg2.errors
            if isinstance(e, psycopg2.errors.UniqueViolation):
                raise ValueError(f"Username '{username}' already exists") from e
        except ImportError:
            pass
        if "UNIQUE" in str(e) or "unique" in str(e):
            raise ValueError(f"Username '{username}' already exists") from e
        raise


def get_user(user_id: int) -> Optional[Dict]:
    """Get user account details."""
    conn = get_connection()
    try:
        row = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
        if not row:
            return None
        cols = _col_names(conn, "users")
        return dict(zip(cols, row))
    finally:
        release_connection(conn)


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username."""
    conn = get_connection()
    try:
        row = _fetchone(conn, "SELECT * FROM users WHERE username = ?", (username,))
        if not row:
            return None
        cols = _col_names(conn, "users")
        return dict(zip(cols, row))
    finally:
        release_connection(conn)


# ==========================================
# ANGEL ONE LIVE ORDER HELPERS
# ==========================================

def _place_angel_buy(symbol: str, quantity: int, price: float) -> Optional[str]:
    """
    Place a real BUY LIMIT order on Angel One using the cached session.
    Returns the Angel One order_id on success, None on failure.
    B9: Session is cached and reused — no fresh login per call.
    """
    tokens_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "angel_tokens.json"
    )
    try:
        with open(tokens_file) as f:
            token_map = json.load(f)
    except Exception:
        _angel_log.error("angel_tokens.json not found")
        return None

    short = symbol.replace(".NS", "").upper()
    token_info = token_map.get(short)
    if not token_info:
        _angel_log.error(f"No Angel One token for {short}")
        return None

    try:
        smart_api = _angel_cache.get()
        if smart_api is None:
            _angel_log.error("Angel One session unavailable")
            return None

        order_params = {
            "variety": "NORMAL",
            "tradingsymbol": token_info["trading_symbol"],
            "symboltoken": token_info["token"],
            "transactiontype": "BUY",
            "exchange": "NSE",
            "ordertype": "LIMIT",
            "producttype": "DELIVERY",
            "duration": "DAY",
            "price": str(price),
            "quantity": str(quantity),
        }

        order_id = smart_api.placeOrder(order_params)
        _angel_log.info(f"✅ Angel One BUY placed: {symbol} qty={quantity} price=₹{price} → order_id={order_id}")
        return str(order_id)

    except Exception as e:
        err = str(e).lower()
        if "401" in err or "unauthorized" in err or "session" in err or "token" in err:
            _angel_log.warning("Angel One session expired — invalidating cache for next call")
            _angel_cache.invalidate()
        _angel_log.error(f"❌ Failed to place BUY on Angel One for {symbol}: {e}")
        return None


# ==========================================
# BRACKET ORDER EXECUTION
# ==========================================

def execute_signal(
    user_id: int,
    symbol: str,
    name: str,
    investment_amount: float,
    buy_price: float,
    target_price: float,
    stop_loss: float,
    signal: str = "BUY",
    confidence: float = 0,
    horizon: str = "Unknown",
    mode: str = "PAPER",
) -> Dict:
    """
    Execute an AI trade signal as a bracket order.

    Creates 3 orders:
      1. BUY (ENTRY) — executed immediately (paper) or via Angel One (live)
      2. STOP_LOSS — pending, triggers if price drops to SL
      3. TARGET — pending, triggers if price reaches target

    mode: "PAPER" (virtual balance) or "LIVE" (real Angel One orders + GTT)
    """
    if mode == "LIVE":
        raise ValueError("Live trading is not yet available. Please use PAPER mode.")

    conn = get_connection()
    angel_order_id = None

    try:
        # Advisory transaction lock scoped to this user — auto-released at
        # commit/rollback. Closes two real races (audit findings H5, H9):
        # (1) SELECT...FOR UPDATE on the new-position check below only locks
        # an EXISTING row, so two concurrent first-time-position requests for
        # the same symbol would both see no row and both proceed; (2) risk
        # checks (daily-loss/trade-count/concentration, now run below) and
        # this function's own balance/capacity checks run against a snapshot
        # that a concurrent request — even for a different symbol, since
        # daily-trade-count/concentration are account-wide — could invalidate
        # between the check and the write. Serializing per-user means the
        # second concurrent call only proceeds once the first has fully
        # committed (or rolled back), so it always sees true post-commit
        # state.
        _execute(conn, "SELECT pg_advisory_xact_lock(hashtext(?))", (f"execute_signal:{user_id}",))

        # Risk checks (audit H8: enforced here, inside execute_signal itself,
        # so no caller — autopilot, a future script, anything — can bypass
        # them by skipping the one HTTP route that used to be the only place
        # this ran).
        quantity_for_check = int(investment_amount / buy_price) if buy_price > 0 else 0
        approved, reason, checks = check_order(
            user_id=user_id, symbol=symbol, investment_amount=investment_amount,
            quantity=quantity_for_check, mode=mode,
        )
        if not approved:
            raise RiskCheckFailed(reason, checks)

        # Get user
        user = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
        if not user:
            raise ValueError("User not found")

        user_cols = _col_names(conn, "users")
        user_dict = dict(zip(user_cols, user))

        # Check balance
        available = user_dict["virtual_balance"]
        if investment_amount > available:
            raise ValueError(f"Insufficient balance: ₹{available:.2f} available, ₹{investment_amount:.2f} requested")

        # Lock the existing position row (if any) to prevent concurrent duplicate inserts.
        # A second buy for the same symbol merges via ON CONFLICT DO UPDATE below.
        _fetchone(conn, "SELECT id FROM positions WHERE user_id = ? AND symbol = ? FOR UPDATE", (user_id, symbol))

        # Calculate quantity BEFORE capacity check so both use the same value.
        quantity = round(investment_amount / buy_price)
        if quantity < 1:
            raise ValueError(f"Investment amount ₹{investment_amount:.2f} too small for {symbol} at ₹{buy_price:.2f}")

        # Hard platform-wide capacity check — the only quantity blocker.
        sig_cap = _fetchone(conn, """
            SELECT consumed_volume, recommended_volume,
                   GREATEST(0, COALESCE(recommended_volume, 0) - COALESCE(consumed_volume, 0)) AS remaining
            FROM trade_signals WHERE symbol = ? AND is_active = TRUE
            ORDER BY generated_date DESC LIMIT 1
        """, (symbol,))
        if sig_cap:
            consumed  = sig_cap[0] or 0
            rec_vol   = sig_cap[1] or 0
            remaining = sig_cap[2] or 0
            if rec_vol > 0 and consumed >= rec_vol:
                raise ValueError(
                    f"{symbol} has reached full platform capacity ({consumed:,}/{rec_vol:,} shares). "
                    f"No more users can buy this stock until the signal refreshes."
                )
            if rec_vol > 0 and quantity > remaining:
                raise PartialCapacityError(symbol, quantity, remaining)

        actual_investment = round(quantity * buy_price, 2)
        bracket_id = f"BRK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = _now()

        # Estimate fees: brokerage 0.05% + STT 0.1% on sell + SEBI 0.0001% + stamp duty 0.015% on buy
        brokerage = round(actual_investment * 0.0005, 2)
        stt       = round(actual_investment * 0.001, 2)
        sebi      = round(actual_investment * 0.000001, 2)
        stamp     = round(actual_investment * 0.00015, 2)
        fees      = round(brokerage + stt + sebi + stamp, 2)

        # Link to the active AI trade signal for this symbol (for user-wise traceability)
        trade_signal_id = get_active_signal_id(symbol)

        # 1. BUY order — PAPER: immediately EXECUTED; LIVE: PLACED until Angel One confirms
        entry_status = "EXECUTED" if mode == "PAPER" else "PLACED"
        _execute(conn, """
            INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                quantity, price, status, mode, signal, confidence, horizon, fill_price, fees,
                order_id, trade_signal_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'BUY', 'ENTRY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bracket_id, symbol, name, quantity, buy_price,
              entry_status, mode, signal, confidence, horizon, buy_price, fees,
              None, trade_signal_id, now, now))

        # ---- GTT or PAPER pending orders ----
        sl_gtt_id = None
        target_gtt_id = None
        gtt_placement_failed = False

        if mode == "LIVE":
            # Place GTT orders on Angel One
            from trading.gtt_manager import place_bracket_gtts
            gtt_result = place_bracket_gtts(symbol, quantity, stop_loss, target_price)
            sl_gtt_id = gtt_result.get("sl_rule_id")
            target_gtt_id = gtt_result.get("target_rule_id")

            if not gtt_result["success"]:
                gtt_placement_failed = True
                _angel_log.error(
                    f"GTT placement failed for {symbol}. BUY executed but SL/Target NOT placed!"
                )

        # 2. STOP_LOSS order — pending
        _execute(conn, """
            INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                quantity, price, trigger_price, status, mode, signal, confidence, horizon,
                gtt_rule_id, gtt_status, trade_signal_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'SELL', 'STOP_LOSS', ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bracket_id, symbol, name, quantity, stop_loss, stop_loss,
              mode, signal, confidence, horizon,
              str(sl_gtt_id) if sl_gtt_id else None,
              'PENDING' if sl_gtt_id else None,
              trade_signal_id, now, now))

        # 3. TARGET order — pending
        _execute(conn, """
            INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                quantity, price, trigger_price, status, mode, signal, confidence, horizon,
                gtt_rule_id, gtt_status, trade_signal_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'SELL', 'TARGET', ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bracket_id, symbol, name, quantity, target_price, target_price,
              mode, signal, confidence, horizon,
              str(target_gtt_id) if target_gtt_id else None,
              'PENDING' if target_gtt_id else None,
              trade_signal_id, now, now))

        # Create or merge into existing position for this symbol.
        # ON CONFLICT merges: weighted avg_buy_price, combined qty + invested_amount.
        # Keeps existing bracket_id/SL/target so ongoing price-monitor tracking isn't disrupted.
        _execute(conn, """
            INSERT INTO positions (user_id, symbol, name, quantity, avg_buy_price, current_price,
                target_price, stop_loss, unrealized_pnl, unrealized_pnl_pct, invested_amount,
                current_value, mode, bracket_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, symbol) DO UPDATE SET
                quantity       = positions.quantity + EXCLUDED.quantity,
                avg_buy_price  = (positions.invested_amount + EXCLUDED.invested_amount)
                                 / (positions.quantity + EXCLUDED.quantity),
                invested_amount = positions.invested_amount + EXCLUDED.invested_amount,
                current_value  = positions.current_value  + EXCLUDED.current_value,
                current_price  = EXCLUDED.current_price,
                updated_at     = EXCLUDED.updated_at
        """, (user_id, symbol, name, quantity, buy_price, buy_price,
              target_price, stop_loss, actual_investment, actual_investment, mode, bracket_id, now))

        # Deduct from virtual balance
        _execute(conn, """
            UPDATE users SET
                virtual_balance = virtual_balance - ?,
                virtual_invested = virtual_invested + ?
            WHERE id = ?
        """, (actual_investment + fees, actual_investment, user_id))

        # Atomically increment consumed_volume and close signal when capacity is exhausted.
        # SELECT FOR UPDATE locks the row so concurrent trades on the same stock
        # can't both read the same consumed_volume and double-count.
        # columns: 0=id, 1=consumed_volume, 2=recommended_volume
        sig_row = _fetchone(conn, """
            SELECT id, consumed_volume, recommended_volume
            FROM trade_signals
            WHERE symbol = ? AND is_active = TRUE
            ORDER BY generated_date DESC LIMIT 1
            FOR UPDATE
        """, (symbol,))

        if sig_row:
            new_consumed = (sig_row[1] or 0) + quantity
            rec_vol      = sig_row[2] or 0
            at_capacity  = rec_vol > 0 and new_consumed >= rec_vol
            _execute(conn, """
                UPDATE trade_signals
                SET consumed_volume = ?,
                    is_active = CASE WHEN ? THEN FALSE ELSE is_active END
                WHERE id = ?
            """, (new_consumed, at_capacity, sig_row[0]))

        # Per-user volume tracking — record how much of this signal each user consumed
        if trade_signal_id:
            _execute(conn, """
                INSERT INTO user_signal_volume (user_id, trade_signal_id, symbol, quantity_consumed, investment_amount)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (user_id, trade_signal_id)
                DO UPDATE SET
                    quantity_consumed = user_signal_volume.quantity_consumed + EXCLUDED.quantity_consumed,
                    investment_amount = user_signal_volume.investment_amount + EXCLUDED.investment_amount
            """, (user_id, trade_signal_id, symbol, quantity, actual_investment))

        conn.commit()

        # ---- LIVE MODE: Place real BUY order on Angel One AFTER DB commit ----
        if mode == "LIVE":
            angel_order_id = _place_angel_buy(symbol, quantity, buy_price)
            if angel_order_id:
                # Upgrade BUY order status from PLACED to EXECUTED and record Angel One order_id
                _execute(conn, """
                    UPDATE orders SET status = 'EXECUTED', order_id = ?, updated_at = ?
                    WHERE bracket_id = ? AND order_purpose = 'ENTRY'
                """, (angel_order_id, _now(), bracket_id))
                conn.commit()
            else:
                # Compensating rollback (audit H6): the BUY never actually executed
                # on the broker, so reverse the position/balance/order writes that
                # already committed above rather than leaving a "ghost" position
                # with no real holding behind it. Currently unreachable — LIVE mode
                # is blocked at the top of this function — but guarded now so this
                # can't silently corrupt account state whenever LIVE is enabled.
                _angel_log.error(
                    f"Angel One BUY failed for {symbol} after DB commit — rolling back position/balance/orders"
                )
                try:
                    _execute(conn, "DELETE FROM positions WHERE user_id = ? AND symbol = ? AND bracket_id = ?",
                             (user_id, symbol, bracket_id))
                    _execute(conn, """
                        UPDATE orders SET status = 'CANCELLED', updated_at = ?
                        WHERE bracket_id = ?
                    """, (_now(), bracket_id))
                    _execute(conn, """
                        UPDATE users SET
                            virtual_balance = virtual_balance + ?,
                            virtual_invested = virtual_invested - ?
                        WHERE id = ?
                    """, (actual_investment + fees, actual_investment, user_id))
                    conn.commit()
                except Exception as rollback_err:
                    conn.rollback()
                    _angel_log.error(f"Compensating rollback for {symbol} FAILED — manual reconciliation needed: {rollback_err}")
                raise ValueError(f"Angel One BUY order failed for {symbol} — trade reversed")

        # Fetch all orders for this bracket
        orders = _fetchall(conn, "SELECT * FROM orders WHERE bracket_id = ? ORDER BY id", (bracket_id,))
        order_cols = _col_names(conn, "orders")
        orders_list = [dict(zip(order_cols, o)) for o in orders]

        # Get updated user
        updated_user = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
        updated_user_dict = dict(zip(user_cols, updated_user))

    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)

    return {
        "bracket_id": bracket_id,
        "mode": mode,
        "orders": orders_list,
        "gtt": {
            "sl_rule_id": sl_gtt_id,
            "target_rule_id": target_gtt_id,
            "placement_failed": gtt_placement_failed,
        } if mode == "LIVE" else None,
        "position": {
            "symbol": symbol,
            "name": name,
            "quantity": quantity,
            "buy_price": buy_price,
            "invested": actual_investment,
            "target": target_price,
            "stop_loss": stop_loss,
            "fees": fees,
        },
        "account": {
            "balance_before": available,
            "balance_after": updated_user_dict["virtual_balance"],
            "total_invested": updated_user_dict["virtual_invested"],
        }
    }


# ==========================================
# POSITION MANAGEMENT
# ==========================================

def get_positions(user_id: int) -> List[Dict]:
    """Get all open positions for a user with current P&L."""
    conn = get_connection()
    rows = _fetchall(
        conn,
        "SELECT * FROM positions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    )
    cols = _col_names(conn, "positions")
    release_connection(conn)
    return [dict(zip(cols, r)) for r in rows]


def get_orders(user_id: int, limit: int = 50) -> List[Dict]:
    """Get order history for a user."""
    conn = get_connection()
    rows = _fetchall(
        conn,
        "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    )
    cols = _col_names(conn, "orders")
    release_connection(conn)
    return [dict(zip(cols, r)) for r in rows]


def square_off(user_id: int, symbol: str, sell_price: float = None, trigger: str = "MANUAL") -> Dict:
    """trigger: 'STOP_LOSS' | 'TARGET' | 'MANUAL'"""
    """
    Sell an entire position at given price (or current price from DB).
    Cancels pending SL/Target orders, books P&L.

    `SELECT ... FOR UPDATE` on the position row makes this atomic against
    concurrent callers (e.g. an overlapping price-monitor sweep and a
    synchronous square-off request for the same position): the second
    caller blocks until the first commits, then cleanly sees no row and
    raises ValueError, instead of both racing off the same stale read
    (audit findings C2/H7). The whole body runs in one try/finally so the
    connection is always released exactly once and rolled back on error
    (audit finding M6).
    """
    conn = get_connection()
    try:
        # Get position — locked until this transaction commits/rolls back
        pos = _fetchone(
            conn,
            "SELECT * FROM positions WHERE user_id = ? AND symbol = ? FOR UPDATE",
            (user_id, symbol)
        )
        if not pos:
            raise ValueError(f"No open position in {symbol}")

        pos_cols = _col_names(conn, "positions")
        pos_dict = dict(zip(pos_cols, pos))

        # Use current_price if sell_price not provided; never silently fall
        # through to a missing/zero price (audit finding M7 — that would
        # book a fake 100%-loss P&L with no validation).
        if not sell_price:
            sell_price = pos_dict["current_price"] or pos_dict["avg_buy_price"]
        if not sell_price or sell_price <= 0:
            raise ValueError(f"No valid price available to square off {symbol}")

        qty = pos_dict["quantity"]
        buy_price = pos_dict["avg_buy_price"]
        invested = pos_dict["invested_amount"]
        sell_value = round(qty * sell_price, 2)
        pnl = round(sell_value - invested, 2)
        fees = round(sell_value * 0.0005, 2)
        net_pnl = round(pnl - fees, 2)

        now = _now()
        bracket_id = pos_dict.get("bracket_id")
        position_mode = pos_dict.get("mode", "PAPER")

        # Create SELL order
        _execute(conn, """
            INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                quantity, price, status, mode, fill_price, fees, pnl, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'SELL', 'SQUARE_OFF', ?, ?, 'EXECUTED', ?, ?, ?, ?, ?, ?)
        """, (user_id, bracket_id, symbol, pos_dict.get("name"), qty, sell_price,
              position_mode, sell_price, fees, net_pnl, now, now))

        # Cancel pending SL and TARGET orders for this bracket
        if bracket_id:
            # If LIVE mode, cancel GTT rules on Angel One first
            if position_mode == "LIVE":
                pending_gtts = _fetchall(conn, """
                    SELECT gtt_rule_id FROM orders
                    WHERE bracket_id = ? AND status = 'PENDING' AND gtt_rule_id IS NOT NULL
                """, (bracket_id,))

                from trading.gtt_manager import cancel_gtt
                for row in pending_gtts:
                    if row[0]:
                        cancel_gtt(int(row[0]))

            _execute(conn, """
                UPDATE orders SET status = 'CANCELLED', gtt_status = CASE
                    WHEN gtt_rule_id IS NOT NULL THEN 'CANCELLED' ELSE gtt_status END,
                    updated_at = ?
                WHERE bracket_id = ? AND status = 'PENDING'
            """, (now, bracket_id))

        # Update user balance
        is_win = 1 if net_pnl > 0 else 0
        _execute(conn, """
            UPDATE users SET
                virtual_balance = virtual_balance + ?,
                virtual_invested = virtual_invested - ?,
                total_pnl = total_pnl + ?,
                win_count = win_count + ?,
                loss_count = loss_count + ?
            WHERE id = ?
        """, (sell_value - fees, invested, net_pnl, is_win, 1 - is_win, user_id))

        # Delete position
        _execute(conn, "DELETE FROM positions WHERE user_id = ? AND symbol = ?", (user_id, symbol))

        conn.commit()

        # Settle the autopilot mandate if this position was AI-managed
        if bracket_id:
            try:
                mandate_status = "COMPLETED" if trigger == "TARGET" else "STOPPED"
                _execute(conn,
                    """UPDATE authorized_trades
                       SET status = ?, actual_pnl = ?, updated_at = NOW()
                       WHERE bracket_id = ? AND status = 'EXECUTED'""",
                    (mandate_status, net_pnl, bracket_id))
                conn.commit()
            except Exception as _e:
                logger.warning(f"Could not settle authorized_trade for bracket {bracket_id}: {_e}")

        # Get updated user
        user = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
        user_cols = _col_names(conn, "users")
        user_dict = dict(zip(user_cols, user))

        return {
            "symbol": symbol,
            "quantity": qty,
            "buy_price": buy_price,
            "sell_price": sell_price,
            "invested": invested,
            "sell_value": sell_value,
            "pnl": net_pnl,
            "pnl_pct": round(net_pnl / invested * 100, 2) if invested > 0 else 0,
            "fees": fees,
            "result": "PROFIT" if net_pnl > 0 else "LOSS",
            "account": {
                "balance": user_dict["virtual_balance"],
                "total_invested": user_dict["virtual_invested"],
                "total_pnl": user_dict["total_pnl"],
                "win_count": user_dict["win_count"],
                "loss_count": user_dict["loss_count"],
            }
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        release_connection(conn)


def square_off_all(user_id: int) -> Dict:
    """Emergency kill switch: sell all positions at current price."""
    positions = get_positions(user_id)
    results = []
    for pos in positions:
        try:
            r = square_off(user_id, pos["symbol"])
            results.append(r)
        except Exception as e:
            results.append({"symbol": pos["symbol"], "error": str(e)})

    user = get_user(user_id)
    return {
        "positions_closed": len(results),
        "results": results,
        "account": {
            "balance": user["virtual_balance"],
            "total_invested": user["virtual_invested"],
            "total_pnl": user["total_pnl"],
            "win_count": user["win_count"],
            "loss_count": user["loss_count"],
        }
    }


def get_portfolio_summary(user_id: int) -> Dict:
    """Get full portfolio summary: balance, invested, P&L, win rate."""
    user = get_user(user_id)
    if not user:
        raise ValueError("User not found")

    positions = get_positions(user_id)
    total_unrealized = sum(p.get("unrealized_pnl", 0) or 0 for p in positions)

    wins = user["win_count"]
    losses = user["loss_count"]
    total_trades = wins + losses
    win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0

    return {
        "user": {
            "id": user["id"],
            "username": user["username"],
            "display_name": user["display_name"],
        },
        "balance": user["virtual_balance"],
        "invested": user["virtual_invested"],
        "total_value": round(user["virtual_balance"] + user["virtual_invested"] + total_unrealized, 2),
        "realized_pnl": user["total_pnl"],
        "unrealized_pnl": round(total_unrealized, 2),
        "total_pnl": round(user["total_pnl"] + total_unrealized, 2),
        "open_positions": len(positions),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "positions": positions,
    }
