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
        return dict(zip(cols, user))
    except Exception as e:
        conn.rollback()
        # users has UNIQUE on BOTH username and email — report which one collided,
        # otherwise a duplicate email surfaces as a bogus "username taken" error.
        def _dupe_error(exc) -> ValueError:
            constraint = ""
            try:
                constraint = (exc.diag.constraint_name or "")
            except Exception:
                pass
            haystack = f"{constraint} {exc}".lower()
            if "email" in haystack:
                return ValueError(f"Email '{email}' is already registered")
            return ValueError(f"Username '{username}' already exists")

        try:
            import psycopg2.errors
            if isinstance(e, psycopg2.errors.UniqueViolation):
                raise _dupe_error(e) from e
        except ImportError:
            pass
        if "UNIQUE" in str(e) or "unique" in str(e):
            raise _dupe_error(e) from e
        raise
    finally:
        release_connection(conn)


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

_TOKEN_MAP: Optional[Dict] = None
_TOKEN_MAP_LOCK = threading.Lock()


def _angel_token(symbol: str) -> Optional[Dict]:
    """Angel One instrument-token entry for a symbol, or None. Map is read once and cached."""
    global _TOKEN_MAP
    if _TOKEN_MAP is None:
        with _TOKEN_MAP_LOCK:
            if _TOKEN_MAP is None:
                tokens_file = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "angel_tokens.json"
                )
                try:
                    with open(tokens_file) as f:
                        _TOKEN_MAP = json.load(f)
                except Exception:
                    _angel_log.error("angel_tokens.json not found")
                    _TOKEN_MAP = {}

    short = symbol.replace(".NS", "").upper()
    token_info = _TOKEN_MAP.get(short)
    if not token_info:
        _angel_log.error(f"No Angel One token for {short}")
    return token_info


def _place_angel_buy(symbol: str, quantity: int, price: float) -> Optional[str]:
    """
    Place a real BUY LIMIT order on Angel One using the cached session.
    Returns the Angel One order_id on success, None on failure.
    B9: Session is cached and reused — no fresh login per call.
    """
    token_info = _angel_token(symbol)
    if not token_info:
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
# PAPER FILL PRICE
# ==========================================

def _live_ltp(symbol: str) -> float:
    """
    Last traded price from Angel One via the cached session, or 0.0.

    Deliberately not collectors.ltp_fetcher.fetch_ltp_batch: that opens a fresh
    SmartConnect session per call, and autopilot fires pending mandates in a
    loop — one full login per mandate would be both slow on a user-facing
    request and a good way to get rate-limited. This reuses the same 6-hour
    session the LIVE order path uses.
    """
    # Tests must never reach the live Angel One API (same rule as
    # price_monitor._fetch_live_prices).
    if os.getenv("APP_ENV") == "test":
        return 0.0

    from trading.price_monitor import _is_market_open
    if not _is_market_open():
        return 0.0

    token_info = _angel_token(symbol)
    if not token_info:
        return 0.0

    try:
        smart_api = _angel_cache.get()
        if smart_api is None:
            return 0.0
        short = symbol.replace(".NS", "").upper()
        data = smart_api.ltpData(
            exchange="NSE",
            tradingsymbol=token_info.get("trading_symbol", f"{short}-EQ"),
            symboltoken=token_info["token"],
        )
        if data and data.get("status") and data.get("data"):
            return float(data["data"].get("ltp") or 0.0)
    except Exception as e:
        err = str(e).lower()
        if "401" in err or "unauthorized" in err or "session" in err or "token" in err:
            _angel_cache.invalidate()
        _angel_log.warning(f"Live LTP lookup failed for {symbol}: {e}")
    return 0.0


def _market_price(symbol: str) -> float:
    """
    Best available tradable price for `symbol` right now: live Angel One LTP
    during market hours, else the most recent stored close. Returns 0.0 when
    neither is available (no session, and the stored close missing or too stale).

    The DB fallback is price_monitor's, which already refuses multi-day-old
    closes — the same rule that keeps stale data from driving SL/Target should
    keep it from pricing an entry. Imported inside the function because
    price_monitor imports square_off from this module at import time.
    """
    try:
        ltp = _live_ltp(symbol)
        if ltp > 0:
            return ltp
    except Exception as e:
        _angel_log.warning(f"Live LTP unavailable for {symbol}: {e}")

    conn = get_connection()
    try:
        from trading.price_monitor import _get_db_price
        return float(_get_db_price(conn, symbol) or 0.0)
    except Exception as e:
        _angel_log.warning(f"DB price lookup failed for {symbol}: {e}")
        return 0.0
    finally:
        release_connection(conn)


def _paper_fill_price(symbol: str, buy_price: float) -> float:
    """
    Price a PAPER entry actually fills at, given the signal's entry price.

    A real entry is a BUY LIMIT at `buy_price`, so it can never fill *worse*
    than that: with the market below the limit it fills at the market price,
    and above the limit it does not fill at that moment at all. Paper used to
    book every entry at `buy_price` regardless, which invented fills that live
    trading would never produce — JINDALSAW.NS was filled at ₹272 while the
    stock traded at ₹264.75, overstating the cost basis by ₹7.25 a share and
    understating the position's P&L from the first tick.

    So: fill at min(market, entry).

    The market > entry case is still booked at `entry` rather than rejected —
    paper deliberately assumes the limit eventually fills at the price the user
    authorised, since a mandate that silently never executes is worse feedback
    than one filled at its stated entry. It is the optimistic half of the
    simulation; the pessimistic half (never filling better than the market) is
    what this function adds.

    Falls back to `buy_price` when no market price is available at all —
    without a reference the signal's entry is the only number we have.
    """
    market = _market_price(symbol)
    if market <= 0:
        _angel_log.info(f"No market price for {symbol} — filling paper entry at signal entry ₹{buy_price:,.2f}")
        return buy_price
    fill = round(min(market, buy_price), 2)
    if fill < buy_price:
        _angel_log.info(
            f"{symbol}: market ₹{market:,.2f} below entry ₹{buy_price:,.2f} — filling paper entry at ₹{fill:,.2f}"
        )
    return fill


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

    # Resolve the paper fill price BEFORE taking the connection and the per-user
    # advisory lock — it may do an Angel One LTP round-trip, and holding the lock
    # across a network call would serialise every one of this user's trades behind it.
    #
    # LIVE keeps buy_price untouched: there the entry really is a LIMIT at that
    # price (_place_angel_buy) and the true fill comes back from the broker.
    fill_price = _paper_fill_price(symbol, buy_price) if mode == "PAPER" else buy_price

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
        # ONE quantity, used by the risk checks, the orders and the position.
        #
        # This used to be int() here and round() at order time, so the checks
        # validated a quantity the order did not use: with 11.6 shares' worth of
        # funding, check_order saw 11 and the order placed 12. That is a
        # risk-control bypass (position-size and exposure limits approved for a
        # smaller trade than executed) and it overspends the allocation, since
        # actual_investment = quantity * buy_price then exceeds investment_amount.
        #
        # Floor, not round: never buy more than the user funded. It also lines
        # the order up with authorized_trades.qty, whose disagreement left user 2
        # with a position of 68 shares against entry orders totalling 67.
        #
        # Sized off buy_price (the authorised entry), NOT fill_price: an order is
        # placed for a quantity, so a cheaper fill buys the same shares for less
        # money, it does not silently buy more of them. Sizing off the fill would
        # also re-open the authorized_trades.qty disagreement above, and since
        # fill_price <= buy_price the spend stays within investment_amount either way.
        quantity = int(investment_amount / buy_price) if buy_price > 0 else 0
        approved, reason, checks = check_order(
            user_id=user_id, symbol=symbol, investment_amount=investment_amount,
            quantity=quantity, mode=mode,
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

        # quantity was computed above, before check_order, so the risk checks,
        # the capacity check, the orders and the position all use one value.
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

        actual_investment = round(quantity * fill_price, 2)
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
        # price = the limit (the authorised entry); fill_price = what it actually filled at.
        entry_status = "EXECUTED" if mode == "PAPER" else "PLACED"
        _execute(conn, """
            INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                quantity, price, status, mode, signal, confidence, horizon, fill_price, fees,
                order_id, trade_signal_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'BUY', 'ENTRY', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bracket_id, symbol, name, quantity, buy_price,
              entry_status, mode, signal, confidence, horizon, fill_price, fees,
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
                current_value, mode, bracket_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id, symbol) DO UPDATE SET
                quantity       = positions.quantity + EXCLUDED.quantity,
                avg_buy_price  = (positions.invested_amount + EXCLUDED.invested_amount)
                                 / (positions.quantity + EXCLUDED.quantity),
                invested_amount = positions.invested_amount + EXCLUDED.invested_amount,
                current_value  = positions.current_value  + EXCLUDED.current_value,
                current_price  = EXCLUDED.current_price,
                updated_at     = EXCLUDED.updated_at
        """, (user_id, symbol, name, quantity, fill_price, fill_price,
              target_price, stop_loss, actual_investment, actual_investment, mode, bracket_id, now, now))

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

    try:
        from database.db import insert_notification
        insert_notification(
            user_id=user_id,
            type="trade",
            title=f"Order placed — {symbol}",
            message=f"{mode} BUY · {quantity} shares @ ₹{fill_price:,.2f} · "
                    f"Target ₹{target_price:,.2f} · SL ₹{stop_loss:,.2f}",
            icon="ShoppingCart",
            color="#3B82F6",
        )
    except Exception:
        pass

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
            "buy_price": fill_price,
            "signal_price": buy_price,
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
    """Get all open positions for a user with current P&L.

    `sector` is joined on rather than stored: it is reference data about the
    symbol, not about the holding. Every consumer needs it — the holdings table
    labels each row with it and the allocation donut groups by it — so it is
    attached here instead of at each call site.
    """
    conn = get_connection()
    try:
        rows = _fetchall(conn, "SELECT * FROM positions WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
        cols = _col_names(conn, "positions")
        positions = [dict(zip(cols, r)) for r in rows]
        if positions:
            symbols = sorted({p["symbol"] for p in positions})
            ph = ",".join("?" for _ in symbols)
            sectors = dict(_fetchall(conn,
                f"SELECT symbol, sector FROM nifty_constituents WHERE symbol IN ({ph})",
                tuple(symbols)))
            for p in positions:
                p["sector"] = sectors.get(p["symbol"]) or "Unclassified"
        return positions
    finally:
        release_connection(conn)


def get_orders(user_id: int, limit: int = 50) -> List[Dict]:
    """Get order history for a user."""
    conn = get_connection()
    try:
        rows = _fetchall(conn, "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?", (user_id, limit))
        cols = _col_names(conn, "orders")
        return [dict(zip(cols, r)) for r in rows]
    finally:
        release_connection(conn)


def get_bracket_levels(bracket_ids: List[str]) -> Dict[str, Dict]:
    """
    The four prices that describe a trade, per bracket_id.

    An `orders` row on its own cannot answer "what was this trade?" — a bracket
    is three or four rows (ENTRY, STOP_LOSS, TARGET, and a SQUARE_OFF once it
    closes), so a SELL leg carries a trigger price and no entry, and the ENTRY
    leg carries no target. Callers rendering one row per order need the whole
    bracket's levels on every row.

    Returns per bracket:
      entry_price  — what the ENTRY leg filled at (its limit if it predates fill_price)
      stop_loss    — the STOP_LOSS leg's trigger
      target_price — the TARGET leg's trigger
      sell_price   — what it ACTUALLY sold for, or None while still open
      sold         — True once sell_price is a real execution rather than a projection

    `sell_price` is deliberately None rather than falling back to the target
    here: "sold at ₹291" and "expected to sell at ₹291" are different claims,
    and only the caller knows how to show that difference. Collapsing them in
    the payload would make a projection indistinguishable from a fill.
    """
    if not bracket_ids:
        return {}

    conn = get_connection()
    try:
        placeholders = ",".join("?" for _ in bracket_ids)
        rows = _fetchall(conn, f"""
            SELECT bracket_id, order_purpose, order_type, status, price, trigger_price, fill_price
            FROM orders
            WHERE bracket_id IN ({placeholders})
        """, tuple(bracket_ids))
    finally:
        release_connection(conn)

    levels: Dict[str, Dict] = {
        b: {"entry_price": None, "stop_loss": None, "target_price": None,
            "sell_price": None, "sold": False}
        for b in bracket_ids
    }

    for bracket_id, purpose, _otype, status, price, trigger_price, fill_price in rows:
        lv = levels.get(bracket_id)
        if lv is None:
            continue
        if purpose == "ENTRY":
            lv["entry_price"] = fill_price if fill_price is not None else price
        elif purpose == "STOP_LOSS":
            lv["stop_loss"] = trigger_price if trigger_price is not None else price
        elif purpose == "TARGET":
            lv["target_price"] = trigger_price if trigger_price is not None else price
        elif purpose == "SQUARE_OFF" and status == "EXECUTED":
            # The only leg that represents a real sale. A PENDING SL/TARGET row
            # is an instruction, not a fill, which is why neither sets this.
            lv["sell_price"] = fill_price if fill_price is not None else price
            lv["sold"] = True

    return levels


def get_trades(user_id: int, limit: int = 200) -> List[Dict]:
    """
    One row per TRADE, not per order leg.

    `orders` stores legs — a bracket is ENTRY + STOP_LOSS + TARGET, plus a
    SQUARE_OFF once it closes. Nothing in the schema represents the trade those
    legs belong to; `bracket_id` is only a correlation string. So "Trade History"
    was rendering raw legs, which meant one ITI.NS trade appeared as three rows,
    two of which (a PENDING stop and a PENDING target) had never happened at all.

    This collapses each bracket into the thing a trader means by "a trade":
    bought this many at this price, protected here, targeting there, and either
    still open or closed at a known price for a known reason.

    Status is the TRADE's state, deliberately using the same vocabulary the
    autopilot screen shows, so the two pages can no longer disagree:
      OPEN | TARGET_HIT | STOPPED | CLOSED
    """
    conn = get_connection()
    try:
        rows = _fetchall(conn, """
            SELECT bracket_id, symbol, name, mode, signal, order_purpose, status,
                   quantity, price, trigger_price, fill_price, pnl, exit_reason,
                   created_at
            FROM orders
            WHERE user_id = ? AND bracket_id IS NOT NULL
            ORDER BY created_at ASC
        """, (user_id,))
    finally:
        release_connection(conn)

    trades: Dict[str, Dict] = {}
    for (bracket_id, symbol, name, mode, signal, purpose, status, qty, price,
         trigger_price, fill_price, pnl, exit_reason, created_at) in rows:
        t = trades.setdefault(bracket_id, {
            "bracket_id": bracket_id, "symbol": symbol, "name": name,
            "mode": mode, "signal": signal, "quantity": qty,
            "entry_price": None, "entry_at": None,
            "target_price": None, "stop_loss": None,
            "exit_price": None, "exit_at": None, "exit_reason": None,
            "realized_pnl": None, "status": "OPEN",
        })
        if purpose == "ENTRY" and status == "EXECUTED":
            t["entry_price"] = fill_price if fill_price is not None else price
            t["entry_at"]    = created_at
            t["quantity"]    = qty
        elif purpose == "STOP_LOSS":
            t["stop_loss"] = trigger_price if trigger_price is not None else price
        elif purpose == "TARGET":
            t["target_price"] = trigger_price if trigger_price is not None else price
        elif purpose == "SQUARE_OFF" and status == "EXECUTED":
            t["exit_price"]   = fill_price if fill_price is not None else price
            t["exit_at"]      = created_at
            t["realized_pnl"] = pnl
            t["exit_reason"]  = exit_reason

    out = []
    for t in trades.values():
        if t["entry_price"] is None:
            # A bracket whose entry never filled is not a trade that happened.
            continue
        if t["exit_price"] is not None:
            reason = t["exit_reason"]
            if not reason:
                # Legacy rows predate exit_reason (square_off used to discard the
                # trigger). Infer from where it closed relative to the levels.
                if t["target_price"] and t["exit_price"] >= t["target_price"]:
                    reason = "TARGET"
                elif t["stop_loss"] and t["exit_price"] <= t["stop_loss"]:
                    reason = "STOP_LOSS"
                else:
                    reason = "MANUAL"
                t["exit_reason"] = reason
            t["status"] = {"TARGET": "TARGET_HIT",
                           "STOP_LOSS": "STOPPED"}.get(reason, "CLOSED")
        out.append(t)

    out.sort(key=lambda r: (r["exit_at"] or r["entry_at"]), reverse=True)
    return out[:limit]


def enrich_orders(rows: List[Dict]) -> List[Dict]:
    """
    Attach the four prices every order table renders to each row, in place.

    Adds `current_price` (latest close) plus the row's bracket levels —
    entry_price / stop_loss / target_price / sell_price / sold. Shared by every
    route that serves order rows so the columns cannot drift apart between the
    order history and the GTT list.

    Call this AFTER pagination: it looks up only the symbols and brackets on the
    rows handed to it, not the user's whole history.
    """
    if not rows:
        return rows

    from database.db import get_latest_close_map

    close_map = get_latest_close_map(sorted({r["symbol"] for r in rows if r.get("symbol")}))
    levels = get_bracket_levels(sorted({r["bracket_id"] for r in rows if r.get("bracket_id")}))

    for r in rows:
        r["current_price"] = close_map.get(r.get("symbol"))
        lv = levels.get(r.get("bracket_id")) or {}
        r["entry_price"]  = lv.get("entry_price")
        r["stop_loss"]    = lv.get("stop_loss")
        r["target_price"] = lv.get("target_price")
        r["sell_price"]   = lv.get("sell_price")   # None while still open
        r["sold"]         = lv.get("sold", False)
    return rows


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

        # ---- Exit legs, one per bracket ----------------------------------
        #
        # `positions` is UNIQUE(user_id, symbol), so a position merges every buy
        # of that symbol and can span several brackets. This used to write ONE
        # SQUARE_OFF row against the position's own bracket_id, which left every
        # other contributing bracket with an entry and no exit — permanently
        # reporting as still-open in the trade view even though the shares were
        # sold. Real case (user 2, BAJAJFINSV.NS): bracket 4b3f33 got a 68-share
        # exit against its 56-share entry, and bracket d1fdf7's 11 shares got no
        # exit row at all.
        #
        # The position-level totals below (sell_value, invested, net_pnl) stay
        # authoritative and still drive the balance — only the record of *which
        # bracket sold what* becomes per-bracket.
        open_brackets = _fetchall(conn, """
            SELECT o.bracket_id,
                   SUM(CASE WHEN o.order_purpose = 'ENTRY' AND o.status = 'EXECUTED'
                            THEN o.quantity ELSE 0 END)                        AS entry_qty,
                   MAX(CASE WHEN o.order_purpose = 'ENTRY' AND o.status = 'EXECUTED'
                            THEN COALESCE(o.fill_price, o.price) END)          AS entry_price
            FROM orders o
            WHERE o.user_id = ? AND o.symbol = ? AND o.bracket_id IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM orders s
                  WHERE s.bracket_id = o.bracket_id
                    AND s.order_purpose = 'SQUARE_OFF' AND s.status = 'EXECUTED')
            GROUP BY o.bracket_id
            HAVING SUM(CASE WHEN o.order_purpose = 'ENTRY' AND o.status = 'EXECUTED'
                            THEN o.quantity ELSE 0 END) > 0
            ORDER BY MIN(o.created_at)
        """, (user_id, symbol))

        # Defensive: a position with no identifiable open bracket still has to be
        # recorded, so fall back to the old single-row behaviour.
        if not open_brackets:
            open_brackets = [(bracket_id, qty, buy_price)]

        entry_total = sum(int(b[1] or 0) for b in open_brackets) or qty

        # Each bracket exits exactly what it entered; any drift between the
        # position and the sum of its entries goes to the FIRST (earliest)
        # bracket. Proportional apportionment would be worse here: with a
        # position of 68 against entries of 56+11, splitting by ratio hands the
        # 11-share bracket a 12-share exit — selling more than it ever bought.
        exit_legs = []
        drift = qty - entry_total
        assigned_fees = 0
        for i, (b_id, b_qty, b_entry) in enumerate(open_brackets):
            leg_qty = int(b_qty or 0) + (drift if i == 0 else 0)
            if leg_qty <= 0:
                continue
            last = i == len(open_brackets) - 1
            if last:
                leg_fees = round(fees - assigned_fees, 2)
            else:
                leg_fees = round(fees * (leg_qty / qty), 2) if qty else 0.0
                assigned_fees += leg_fees
            basis   = b_entry if b_entry is not None else buy_price
            leg_pnl = round(leg_qty * (sell_price - basis) - leg_fees, 2)
            exit_legs.append((b_id, leg_qty, leg_fees, leg_pnl))

        legs_pnl = round(sum(l[3] for l in exit_legs), 2)
        if abs(legs_pnl - net_pnl) > 0.05:
            # Not forced to match: a gap means the position's cost basis and its
            # entry legs genuinely disagree, and hiding that would make the books
            # look consistent when they are not.
            _angel_log.warning(
                "square_off %s user=%s: per-bracket P&L %.2f != position P&L %.2f "
                "(cost-basis drift across %d brackets)",
                symbol, user_id, legs_pnl, net_pnl, len(exit_legs))

        for b_id, leg_qty, leg_fees, leg_pnl in exit_legs:
            _execute(conn, """
                INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
                    quantity, price, status, mode, fill_price, fees, pnl, exit_reason,
                    created_at, updated_at)
                VALUES (?, ?, ?, ?, 'SELL', 'SQUARE_OFF', ?, ?, 'EXECUTED', ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, b_id, symbol, pos_dict.get("name"), leg_qty, sell_price,
                  position_mode, sell_price, leg_fees, leg_pnl, trigger, now, now))

        # Cancel the pending SL/TARGET legs of EVERY bracket that just exited —
        # not only the position's own, or the others keep resting forever.
        exited_brackets = [l[0] for l in exit_legs if l[0]]
        if exited_brackets:
            placeholders = ",".join("?" for _ in exited_brackets)
            if position_mode == "LIVE":
                pending_gtts = _fetchall(conn, f"""
                    SELECT gtt_rule_id FROM orders
                    WHERE bracket_id IN ({placeholders})
                      AND status = 'PENDING' AND gtt_rule_id IS NOT NULL
                """, tuple(exited_brackets))

                from trading.gtt_manager import cancel_gtt
                for row in pending_gtts:
                    if row[0]:
                        cancel_gtt(int(row[0]))

            _execute(conn, f"""
                UPDATE orders SET status = 'CANCELLED', gtt_status = CASE
                    WHEN gtt_rule_id IS NOT NULL THEN 'CANCELLED' ELSE gtt_status END,
                    updated_at = ?
                WHERE bracket_id IN ({placeholders}) AND status = 'PENDING'
            """, (now, *exited_brackets))

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

        # Settle EVERY autopilot mandate this position covered — not just the
        # one whose bracket_id the position happens to carry.
        #
        # positions is UNIQUE(user_id, symbol) and execute_signal merges on
        # conflict, deliberately keeping the FIRST bracket_id so price-monitor
        # tracking is not disrupted. So a second mandate on the same symbol is
        # folded into the same row and its bracket_id is discarded. Settling by
        # bracket_id alone therefore closed only the first mandate and orphaned
        # the rest at status='OPEN' forever, with no position behind them.
        #
        # Real case (user 2, BAJAJFINSV.NS): mandate 1 (56 @ 1765) created the
        # position; mandate 2 (11 @ 1764.6) merged in; on 2026-07-31 the target
        # hit and all 67 shares were sold, but only mandate 1 was settled — and
        # it was credited the full 67-share P&L of 14,339.99 rather than its own
        # 11,860.80. Mandate 2 still showed "Running" six days later against a
        # position that no longer existed.
        #
        # So: settle every EXECUTED mandate for this user+symbol, and apportion
        # the realised P&L by quantity so each mandate reports its own share.
        try:
            cur = _execute(conn,
                """SELECT id, bracket_id, qty FROM authorized_trades
                    WHERE user_id = ? AND symbol = ? AND status = 'OPEN'
                    ORDER BY id""",
                (user_id, symbol))
            mandates = cur.fetchall()

            if mandates:
                mandate_status = "COMPLETED" if trigger == "TARGET" else "STOPPED"
                total_qty = sum((m[2] or 0) for m in mandates) or 0
                for m_id, m_bracket, m_qty in mandates:
                    # Proportional split; falls back to an equal share if the
                    # quantities are missing so nothing is left unsettled.
                    share = ((m_qty or 0) / total_qty) if total_qty else (1.0 / len(mandates))
                    # cmp is set to the EXIT price, not left at entry. The
                    # autopilot page renders cmp as the live market price, and
                    # price_monitor only refreshes it while status='OPEN';
                    # once settled nothing touches it again, so a closed mandate
                    # would display its entry price forever and show a P&L of
                    # +₹0 next to a realised profit.
                    _execute(conn,
                        """UPDATE authorized_trades
                           SET status = ?, actual_pnl = ?, cmp = ?, updated_at = NOW()
                           WHERE id = ? AND status = 'OPEN'""",
                        (mandate_status, round(net_pnl * share, 2), sell_price, m_id))
                conn.commit()
                if len(mandates) > 1:
                    _angel_log.info(
                        f"Settled {len(mandates)} mandates sharing {symbol} "
                        f"(qty {total_qty}) — P&L {net_pnl:+,.2f} apportioned by quantity"
                    )
        except Exception as _e:
            _angel_log.warning(f"Could not settle authorized_trades for {symbol}: {_e}")

        # Get updated user
        user = _fetchone(conn, "SELECT * FROM users WHERE id = ?", (user_id,))
        user_cols = _col_names(conn, "users")
        user_dict = dict(zip(user_cols, user))

        try:
            from database.db import insert_notification
            is_profit = net_pnl >= 0
            trigger_label = {"TARGET": "Target hit", "STOP_LOSS": "Stop-loss hit"}.get(trigger, "Closed manually")
            insert_notification(
                user_id=user_id,
                type="trade",
                title=f"{trigger_label} — {symbol}",
                message=f"{'PROFIT' if is_profit else 'LOSS'} · {qty} shares · "
                        f"₹{sell_price:,.2f} exit · P&L {'+'if is_profit else ''}{net_pnl:,.0f} ({round(net_pnl/invested*100,1) if invested else 0:+.1f}%)",
                icon="TrendingUp" if is_profit else "TrendingDown",
                color="#16A34A" if is_profit else "#EF4444",
            )
        except Exception:
            pass

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
        # Served, not left to the client to divide. PortfolioPage derived this
        # as (total_pnl / invested) * 100 and then picked its +/− glyph from the
        # percentage while taking the magnitude from total_pnl — two numbers, one
        # sign, and nothing guaranteeing they agree.
        "total_pnl_pct": round(
            (user["total_pnl"] + total_unrealized) / user["virtual_invested"] * 100, 2
        ) if user["virtual_invested"] else 0.0,
        "open_positions": len(positions),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "positions": positions,
        # Sector mix of the open book. Served here rather than derived in the
        # browser because `positions` carries no sector — it lives in
        # nifty_constituents, which the client has no reason to fetch.
        "allocation": get_portfolio_allocation(user_id),
    }


# ---------------------------------------------------------------------------
# Portfolio composition & value history
# ---------------------------------------------------------------------------

def get_portfolio_allocation(user_id: int, conn=None) -> List[Dict]:
    """Open positions grouped by sector, largest first.

    `positions` carries no sector of its own — sector is reference data, so it
    is joined from `nifty_constituents` rather than copied onto every row. A
    symbol missing from that table (or holding a NULL sector) is reported as
    "Unclassified" instead of being dropped: a slice silently missing from the
    donut makes the percentages lie.

    Grouped on current_value, i.e. what the holding is worth NOW, so the donut
    matches the "Current Value" card above it rather than cost basis.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        rows = _fetchall(conn, """
            SELECT COALESCE(NULLIF(n.sector, ''), 'Unclassified') AS sector,
                   SUM(COALESCE(p.current_value, p.invested_amount, 0))     AS val,
                   COUNT(*)                                                 AS holdings
            FROM positions p
            LEFT JOIN nifty_constituents n ON n.symbol = p.symbol
            WHERE p.user_id = ?
            GROUP BY 1
            ORDER BY 2 DESC
        """, (user_id,))
    finally:
        if own_conn:
            release_connection(conn)

    total = sum(float(r[1] or 0) for r in rows)
    return [
        {
            "sector": r[0],
            "val": round(float(r[1] or 0), 2),
            "pct": round(float(r[1] or 0) / total * 100, 2) if total else 0.0,
            "holdings": int(r[2]),
        }
        for r in rows
    ]


# How many points each range is drawn with, and how far back it reaches.
# 1Y samples weekly — 250 daily points on a 230px chart is noise, and it is the
# shape the range is being asked about.
_HISTORY_RANGES: Dict[str, Tuple[int, int]] = {
    #        days back, sample every N trading days
    "30D": (30,  1),
    "90D": (90,  1),
    "1Y":  (365, 5),
}

# Order rows that actually moved cash. Statuses are not uniform across the
# table's history — the paper engine writes 'EXECUTED', older/seeded rows use
# 'COMPLETE' — so both count as filled. PENDING/CANCELLED never moved anything.
_FILLED_STATUSES = ("EXECUTED", "COMPLETE", "COMPLETED", "FILLED")


def get_portfolio_value_history(user_id: int, range_key: str = "90D") -> Dict:
    """Total portfolio value (cash + holdings at market) per trading day.

    Reconstructed backwards from the state we know is correct — today's cash
    balance and today's open positions — by un-applying each filled order in
    reverse chronological order:

        an ENTRY  took  qty*fill + fees  out of cash and put qty shares in
        a SQUARE_OFF put qty*fill - fees back into cash and took qty shares out

    Replaying forwards from a guessed opening balance would drift, because the
    starting balance is not recorded anywhere; walking backwards from the
    current row is anchored to a number the ledger already agrees on.

    Each sampled day is then valued with that day's close for every symbol held
    on it (last close on or before the day, so a holiday or a missing bar
    carries the previous price forward rather than valuing the stock at zero).
    """
    days_back, step = _HISTORY_RANGES.get(range_key, _HISTORY_RANGES["90D"])
    start = (datetime.now() - timedelta(days=days_back)).date()

    conn = get_connection()
    try:
        user = _fetchone(conn, "SELECT virtual_balance FROM users WHERE id = ?", (user_id,))
        if not user:
            raise ValueError("User not found")
        cash = float(user[0] or 0)

        holdings: Dict[str, float] = {}
        live_value = 0.0
        for sym, qty, cur_val in _fetchall(conn,
                "SELECT symbol, quantity, COALESCE(current_value, invested_amount, 0) "
                "FROM positions WHERE user_id = ?", (user_id,)):
            holdings[sym] = holdings.get(sym, 0) + float(qty or 0)
            live_value += float(cur_val or 0)

        placeholders = ",".join("?" for _ in _FILLED_STATUSES)
        events = _fetchall(conn, f"""
            SELECT created_at, symbol, order_purpose, quantity,
                   COALESCE(fill_price, price), COALESCE(fees, 0)
            FROM orders
            WHERE user_id = ?
              AND order_purpose IN ('ENTRY', 'SQUARE_OFF')
              AND status IN ({placeholders})
            ORDER BY created_at DESC
        """, (user_id, *_FILLED_STATUSES))

        # Every symbol that appears anywhere in the window — currently held or
        # since sold. A position closed 20 days ago still has to be valued on
        # the days before it closed.
        symbols = sorted(set(holdings) | {e[1] for e in events})
        closes: Dict[str, List[Tuple] ] = {}
        trading_days: List = []
        if symbols:
            sym_ph = ",".join("?" for _ in symbols)
            price_rows = _fetchall(conn, f"""
                SELECT date, symbol, close
                FROM prices
                WHERE interval = '1d' AND date >= ? AND symbol IN ({sym_ph})
                ORDER BY date
            """, (start, *symbols))
            seen_days = set()
            for d, sym, close in price_rows:
                d = d.date() if isinstance(d, datetime) else d
                closes.setdefault(sym, []).append((d, float(close or 0)))
                seen_days.add(d)
            trading_days = sorted(seen_days)
    finally:
        release_connection(conn)

    if not trading_days:
        # An account that has never traded still has a portfolio — it is all
        # cash. Fall back to the exchange calendar (the single source of truth
        # for "was the market open"), so a new user sees a flat line at their
        # balance rather than an empty panel that reads as a failed request.
        from analysis.trading_calendar import trading_days_between, today_ist
        trading_days = trading_days_between(start, today_ist())
        if not trading_days:
            return {"range": range_key, "dates": [], "series": []}

    # Sliced from the newest end so the most recent close is always the last
    # point — a chart whose right edge is four days stale reads as a flat week.
    sampled = trading_days[::-1][::step][::-1]

    # Last close on or before each sampled day, per symbol — forward-filled.
    price_at: Dict[str, Dict] = {}
    for sym, series in closes.items():
        by_day, idx, last = {}, 0, None
        for day in sampled:
            while idx < len(series) and series[idx][0] <= day:
                last = series[idx][1]
                idx += 1
            by_day[day] = last
        price_at[sym] = by_day

    # Today's true state, before the walk backwards starts mutating it. `prices`
    # only has a bar once the EOD collection runs, so outside that window the
    # newest trading day is a day or two old — and a trade placed since then
    # would be invisible on a chart that stops there.
    today = datetime.now().date()
    current_point = (today, round(cash + live_value, 2))

    dates, values = [], []
    ev = 0
    for day in reversed(sampled):
        # Undo everything that happened after this day, so `cash`/`holdings`
        # describe the close of `day`.
        while ev < len(events):
            ts = events[ev][0]
            ts_date = ts.date() if isinstance(ts, datetime) else ts
            if ts_date <= day:
                break
            _, sym, purpose, qty, fill, fees = events[ev]
            qty, fill, fees = float(qty or 0), float(fill or 0), float(fees or 0)
            if purpose == "ENTRY":
                cash += qty * fill + fees
                holdings[sym] = holdings.get(sym, 0) - qty
            else:  # SQUARE_OFF
                cash -= qty * fill - fees
                holdings[sym] = holdings.get(sym, 0) + qty
            ev += 1

        if ev == 0 and holdings:
            # Newest point, and nothing has traded since it: the open positions
            # still carry a live price that is fresher than any stored close
            # (prices lags by a day or two outside market hours). Using it makes
            # the right edge of the chart equal the "Current Value" card, which
            # is computed the same way; a close-derived last point would sit a
            # few thousand rupees below it for no visible reason.
            market = live_value
        else:
            market = 0.0
            for sym, qty in holdings.items():
                if qty <= 0:
                    continue
                px = price_at.get(sym, {}).get(day)
                if px is not None:
                    market += qty * px
        dates.append(day.isoformat())
        values.append(round(cash + market, 2))

    dates.reverse()
    values.reverse()
    if current_point[0] > sampled[-1]:
        dates.append(current_point[0].isoformat())
        values.append(current_point[1])
    return {"range": range_key, "dates": dates, "series": values}
