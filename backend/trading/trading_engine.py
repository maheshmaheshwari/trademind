"""
TradeMind AI — Trading Engine

Handles virtual (paper) and live trading:
- User account creation with ₹10L virtual balance
- Auto bracket orders: BUY + STOP_LOSS + TARGET
- Position tracking with P&L
- Square-off (sell) positions
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from database.db import get_connection


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today():
    return datetime.now().strftime("%Y-%m-%d")


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
        conn.execute(
            "INSERT INTO users (username, email, password_hash, display_name) VALUES (?, ?, ?, ?)",
            (username, email, password_hash, display_name or username)
        )
        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
        conn.close()
        return dict(zip(cols, user))
    except Exception as e:
        conn.close()
        if "UNIQUE" in str(e):
            raise ValueError(f"Username '{username}' already exists")
        raise


def get_user(user_id: int) -> Optional[Dict]:
    """Get user account details."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not row:
        conn.close()
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
    conn.close()
    return dict(zip(cols, row))


def get_user_by_username(username: str) -> Optional[Dict]:
    """Get user by username."""
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        conn.close()
        return None
    cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
    conn.close()
    return dict(zip(cols, row))


# ==========================================
# ANGEL ONE LIVE ORDER HELPERS
# ==========================================

def _place_angel_buy(symbol: str, quantity: int, price: float) -> Optional[str]:
    """
    Place a real BUY LIMIT order on Angel One.
    Returns the Angel One order_id on success, None on failure.
    """
    import os
    import json
    import logging
    log = logging.getLogger(__name__)

    tokens_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "angel_tokens.json"
    )
    try:
        with open(tokens_file) as f:
            token_map = json.load(f)
    except Exception:
        log.error("angel_tokens.json not found")
        return None

    short = symbol.replace(".NS", "").upper()
    token_info = token_map.get(short)
    if not token_info:
        log.error(f"No Angel One token for {short}")
        return None

    try:
        from SmartApi import SmartConnect
        import pyotp

        api_key = os.getenv("ANGEL_API_KEY")
        client_id = os.getenv("ANGEL_CLIENT_ID")
        password = os.getenv("ANGEL_PASSWORD")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_id, password, totp)

        if not session or session.get("status") is False:
            log.error(f"Angel One login failed: {session}")
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
        log.info(f"✅ Angel One BUY placed: {symbol} qty={quantity} price=₹{price} → order_id={order_id}")
        return str(order_id)

    except Exception as e:
        log.error(f"❌ Failed to place BUY on Angel One for {symbol}: {e}")
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
    conn = get_connection()
    
    # Get user
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not user:
        conn.close()
        raise ValueError("User not found")
    
    user_cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
    user_dict = dict(zip(user_cols, user))
    
    # Check balance
    available = user_dict["virtual_balance"]
    if investment_amount > available:
        conn.close()
        raise ValueError(f"Insufficient balance: ₹{available:.2f} available, ₹{investment_amount:.2f} requested")
    
    # Check if already have a position in this stock
    existing = conn.execute(
        "SELECT * FROM positions WHERE user_id = ? AND symbol = ?",
        (user_id, symbol)
    ).fetchone()
    if existing:
        conn.close()
        raise ValueError(f"Already have an open position in {symbol}. Square off first.")
    
    # Calculate quantity
    quantity = int(investment_amount / buy_price)
    if quantity < 1:
        conn.close()
        raise ValueError(f"Investment amount ₹{investment_amount:.2f} too small for {symbol} at ₹{buy_price:.2f}")
    
    actual_investment = round(quantity * buy_price, 2)
    bracket_id = f"BRK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    now = _now()
    
    # Estimate fees (approx 0.05% brokerage + taxes)
    fees = round(actual_investment * 0.0005, 2)
    
    # ---- LIVE MODE: Place real BUY order on Angel One ----
    angel_order_id = None
    if mode == "LIVE":
        angel_order_id = _place_angel_buy(symbol, quantity, buy_price)
        if not angel_order_id:
            conn.close()
            raise ValueError(f"Failed to place BUY order on Angel One for {symbol}")
    
    # 1. BUY order — immediately executed
    conn.execute("""
        INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
            quantity, price, status, mode, signal, confidence, horizon, fill_price, fees,
            order_id, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'BUY', 'ENTRY', ?, ?, 'EXECUTED', ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, bracket_id, symbol, name, quantity, buy_price,
          mode, signal, confidence, horizon, buy_price, fees,
          angel_order_id, now, now))
    
    # ---- GTT or PAPER pending orders ----
    sl_gtt_id = None
    target_gtt_id = None
    
    if mode == "LIVE":
        # Place GTT orders on Angel One
        from trading.gtt_manager import place_bracket_gtts
        gtt_result = place_bracket_gtts(symbol, quantity, stop_loss, target_price)
        sl_gtt_id = gtt_result.get("sl_rule_id")
        target_gtt_id = gtt_result.get("target_rule_id")
        
        if not gtt_result["success"]:
            # GTT failed — rollback: we still keep the BUY but warn
            import logging
            logging.getLogger(__name__).error(
                f"GTT placement failed for {symbol}. BUY executed but SL/Target NOT placed!"
            )
    
    # 2. STOP_LOSS order — pending
    conn.execute("""
        INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
            quantity, price, trigger_price, status, mode, signal, confidence, horizon,
            gtt_rule_id, gtt_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'SELL', 'STOP_LOSS', ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, bracket_id, symbol, name, quantity, stop_loss, stop_loss,
          mode, signal, confidence, horizon,
          str(sl_gtt_id) if sl_gtt_id else None,
          'PENDING' if sl_gtt_id else None,
          now, now))
    
    # 3. TARGET order — pending
    conn.execute("""
        INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
            quantity, price, trigger_price, status, mode, signal, confidence, horizon,
            gtt_rule_id, gtt_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'SELL', 'TARGET', ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, bracket_id, symbol, name, quantity, target_price, target_price,
          mode, signal, confidence, horizon,
          str(target_gtt_id) if target_gtt_id else None,
          'PENDING' if target_gtt_id else None,
          now, now))
    
    # Create position
    conn.execute("""
        INSERT INTO positions (user_id, symbol, name, quantity, avg_buy_price, current_price,
            target_price, stop_loss, unrealized_pnl, unrealized_pnl_pct, invested_amount,
            current_value, mode, bracket_id, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, ?, ?)
    """, (user_id, symbol, name, quantity, buy_price, buy_price,
          target_price, stop_loss, actual_investment, actual_investment, mode, bracket_id, now))
    
    # Deduct from virtual balance
    conn.execute("""
        UPDATE users SET 
            virtual_balance = virtual_balance - ?,
            virtual_invested = virtual_invested + ?
        WHERE id = ?
    """, (actual_investment + fees, actual_investment, user_id))
    
    # Increment consumed_volume on the trade signal (platform-wide volume tracking)
    conn.execute("""
        UPDATE trade_signals SET consumed_volume = COALESCE(consumed_volume, 0) + ?
        WHERE symbol = ? AND generated_date = (
            SELECT MAX(generated_date) FROM trade_signals WHERE symbol = ?
        )
    """, (quantity, symbol, symbol))
    
    conn.commit()
    
    # Fetch all orders for this bracket
    orders = conn.execute(
        "SELECT * FROM orders WHERE bracket_id = ? ORDER BY id", (bracket_id,)
    ).fetchall()
    order_cols = [d[0] for d in conn.execute("SELECT * FROM orders LIMIT 0").description]
    orders_list = [dict(zip(order_cols, o)) for o in orders]
    
    # Get updated user
    updated_user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    updated_user_dict = dict(zip(user_cols, updated_user))
    
    conn.close()
    
    return {
        "bracket_id": bracket_id,
        "mode": mode,
        "orders": orders_list,
        "gtt": {
            "sl_rule_id": sl_gtt_id,
            "target_rule_id": target_gtt_id,
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
    rows = conn.execute(
        "SELECT * FROM positions WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,)
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM positions LIMIT 0").description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def get_orders(user_id: int, limit: int = 50) -> List[Dict]:
    """Get order history for a user."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM orders WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit)
    ).fetchall()
    cols = [d[0] for d in conn.execute("SELECT * FROM orders LIMIT 0").description]
    conn.close()
    return [dict(zip(cols, r)) for r in rows]


def square_off(user_id: int, symbol: str, sell_price: float = None) -> Dict:
    """
    Sell an entire position at given price (or current price from DB).
    Cancels pending SL/Target orders, books P&L.
    """
    conn = get_connection()
    
    # Get position
    pos = conn.execute(
        "SELECT * FROM positions WHERE user_id = ? AND symbol = ?",
        (user_id, symbol)
    ).fetchone()
    if not pos:
        conn.close()
        raise ValueError(f"No open position in {symbol}")
    
    pos_cols = [d[0] for d in conn.execute("SELECT * FROM positions LIMIT 0").description]
    pos_dict = dict(zip(pos_cols, pos))
    
    # Use current_price if sell_price not provided
    if not sell_price:
        sell_price = pos_dict["current_price"] or pos_dict["avg_buy_price"]
    
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
    conn.execute("""
        INSERT INTO orders (user_id, bracket_id, symbol, name, order_type, order_purpose,
            quantity, price, status, mode, fill_price, fees, pnl, created_at, updated_at)
        VALUES (?, ?, ?, ?, 'SELL', 'SQUARE_OFF', ?, ?, 'EXECUTED', ?, ?, ?, ?, ?, ?)
    """, (user_id, bracket_id, symbol, pos_dict.get("name"), qty, sell_price,
          position_mode, sell_price, fees, net_pnl, now, now))
    
    # Cancel pending SL and TARGET orders for this bracket
    if bracket_id:
        # If LIVE mode, cancel GTT rules on Angel One first
        if position_mode == "LIVE":
            pending_gtts = conn.execute("""
                SELECT gtt_rule_id FROM orders
                WHERE bracket_id = ? AND status = 'PENDING' AND gtt_rule_id IS NOT NULL
            """, (bracket_id,)).fetchall()
            
            from trading.gtt_manager import cancel_gtt
            for row in pending_gtts:
                if row[0]:
                    cancel_gtt(int(row[0]))
        
        conn.execute("""
            UPDATE orders SET status = 'CANCELLED', gtt_status = CASE
                WHEN gtt_rule_id IS NOT NULL THEN 'CANCELLED' ELSE gtt_status END,
                updated_at = ?
            WHERE bracket_id = ? AND status = 'PENDING'
        """, (now, bracket_id))
    
    # Update user balance
    is_win = 1 if net_pnl > 0 else 0
    conn.execute("""
        UPDATE users SET 
            virtual_balance = virtual_balance + ?,
            virtual_invested = virtual_invested - ?,
            total_pnl = total_pnl + ?,
            win_count = win_count + ?,
            loss_count = loss_count + ?
        WHERE id = ?
    """, (sell_value - fees, invested, net_pnl, is_win, 1 - is_win, user_id))
    
    # Delete position
    conn.execute(
        "DELETE FROM positions WHERE user_id = ? AND symbol = ?",
        (user_id, symbol)
    )
    
    conn.commit()
    
    # Get updated user
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    user_cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
    user_dict = dict(zip(user_cols, user))
    conn.close()
    
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
            "total_pnl": user["total_pnl"],
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
