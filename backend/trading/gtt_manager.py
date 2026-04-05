"""
TradeMind AI — GTT Manager (Angel One)

Manages Good Till Triggered (GTT) orders on Angel One SmartAPI.
GTT orders persist on the broker side for up to 365 days and trigger
automatically when price conditions are met — no polling needed.

Used for placing SL (Stop Loss) and Target exit orders after a BUY entry.

Usage:
    from trading.gtt_manager import create_sl_gtt, create_target_gtt, cancel_gtt
"""

import os
import json
import logging
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)

# Token map for symbol → Angel One token
_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "angel_tokens.json"
)
_TOKEN_MAP = {}
try:
    with open(_TOKENS_FILE) as f:
        _TOKEN_MAP = json.load(f)
except FileNotFoundError:
    logger.warning("angel_tokens.json not found for GTT manager")


def _get_angel_session():
    """Create authenticated Angel One SmartConnect session using .env credentials."""
    try:
        from SmartApi import SmartConnect
        import pyotp

        api_key = os.getenv("ANGEL_API_KEY")
        client_id = os.getenv("ANGEL_CLIENT_ID")
        password = os.getenv("ANGEL_PASSWORD")
        totp_secret = os.getenv("ANGEL_TOTP_SECRET")

        if not all([api_key, client_id, password, totp_secret]):
            logger.error("Missing Angel One credentials in .env")
            return None

        smart_api = SmartConnect(api_key=api_key)
        totp = pyotp.TOTP(totp_secret).now()
        session = smart_api.generateSession(client_id, password, totp)

        if not session or session.get("status") is False:
            logger.error(f"Angel One login failed: {session}")
            return None

        logger.info("Angel One GTT session created")
        return smart_api

    except Exception as e:
        logger.error(f"Error creating Angel One session: {e}")
        return None


def _get_token_info(symbol: str) -> Optional[Dict]:
    """Get Angel One token info for a symbol (e.g., 'RELIANCE.NS' → token data)."""
    short = symbol.replace(".NS", "").upper()
    info = _TOKEN_MAP.get(short)
    if not info:
        logger.warning(f"No Angel One token found for {short}")
    return info


# ==========================================
# GTT OPERATIONS
# ==========================================

def create_sl_gtt(
    symbol: str,
    qty: int,
    trigger_price: float,
    limit_price: float = None,
) -> Optional[int]:
    """
    Create a Stop Loss GTT rule (SELL when price drops to trigger_price).

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"
        qty: Number of shares to sell
        trigger_price: SL trigger price (when LTP <= this, order activates)
        limit_price: Limit price to sell at. Default: trigger_price * 0.99 (1% buffer)

    Returns:
        GTT rule ID on success, None on failure
    """
    if not limit_price:
        # Set limit slightly below trigger to ensure fill
        limit_price = round(trigger_price * 0.99, 2)

    token_info = _get_token_info(symbol)
    if not token_info:
        return None

    smart_api = _get_angel_session()
    if not smart_api:
        return None

    try:
        params = {
            "tradingsymbol": token_info["trading_symbol"],
            "symboltoken": token_info["token"],
            "exchange": "NSE",
            "producttype": "DELIVERY",
            "transactiontype": "SELL",
            "price": str(limit_price),
            "qty": str(qty),
            "triggerprice": str(trigger_price),
            "timeperiod": 365,
        }

        rule_id = smart_api.gttCreateRule(params)
        logger.info(
            f"✅ SL GTT created: {symbol} qty={qty} trigger=₹{trigger_price} "
            f"limit=₹{limit_price} → rule_id={rule_id}"
        )
        return rule_id

    except Exception as e:
        logger.error(f"❌ Failed to create SL GTT for {symbol}: {e}")
        return None


def create_target_gtt(
    symbol: str,
    qty: int,
    trigger_price: float,
    limit_price: float = None,
) -> Optional[int]:
    """
    Create a Target GTT rule (SELL when price rises to trigger_price).

    Args:
        symbol: Stock symbol e.g. "RELIANCE.NS"
        qty: Number of shares to sell
        trigger_price: Target trigger price (when LTP >= this, order activates)
        limit_price: Limit price to sell at. Default: trigger_price * 0.99 (small buffer)

    Returns:
        GTT rule ID on success, None on failure
    """
    if not limit_price:
        # Set limit slightly below trigger to ensure fill
        limit_price = round(trigger_price * 0.99, 2)

    token_info = _get_token_info(symbol)
    if not token_info:
        return None

    smart_api = _get_angel_session()
    if not smart_api:
        return None

    try:
        params = {
            "tradingsymbol": token_info["trading_symbol"],
            "symboltoken": token_info["token"],
            "exchange": "NSE",
            "producttype": "DELIVERY",
            "transactiontype": "SELL",
            "price": str(limit_price),
            "qty": str(qty),
            "triggerprice": str(trigger_price),
            "timeperiod": 365,
        }

        rule_id = smart_api.gttCreateRule(params)
        logger.info(
            f"✅ Target GTT created: {symbol} qty={qty} trigger=₹{trigger_price} "
            f"limit=₹{limit_price} → rule_id={rule_id}"
        )
        return rule_id

    except Exception as e:
        logger.error(f"❌ Failed to create Target GTT for {symbol}: {e}")
        return None


def cancel_gtt(rule_id: int) -> bool:
    """Cancel a GTT rule by its ID."""
    smart_api = _get_angel_session()
    if not smart_api:
        return False

    try:
        params = {
            "id": str(rule_id),
            "symboltoken": "0",   # required but ignored for cancel
            "exchange": "NSE",
        }
        result = smart_api.gttCancelRule(params)
        logger.info(f"🗑️ GTT rule {rule_id} cancelled: {result}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to cancel GTT rule {rule_id}: {e}")
        return False


def get_gtt_details(rule_id: int) -> Optional[Dict]:
    """Get details/satus of a specific GTT rule."""
    smart_api = _get_angel_session()
    if not smart_api:
        return None

    try:
        details = smart_api.gttDetails(rule_id)
        return details
    except Exception as e:
        logger.error(f"Error getting GTT details for rule {rule_id}: {e}")
        return None


def list_gtt_rules(status: str = "NEW", page: int = 1, count: int = 50) -> List[Dict]:
    """
    List GTT rules by status.

    Args:
        status: "NEW" (pending), "TRIGGERED", "CANCELLED", "ACTIVE", "SENTTOEXCHANGE"
        page: Page number (1-based)
        count: Results per page

    Returns:
        List of GTT rule dicts
    """
    smart_api = _get_angel_session()
    if not smart_api:
        return []

    try:
        rules = smart_api.gttLists(status, page, count)
        return rules if isinstance(rules, list) else []
    except Exception as e:
        logger.error(f"Error listing GTT rules (status={status}): {e}")
        return []


def sync_gtt_statuses() -> List[Dict]:
    """
    Sync GTT rule statuses from Angel One to our local DB.

    Checks all orders with a gtt_rule_id that are still PENDING.
    If the GTT has been triggered on Angel One side, updates our DB:
      - Mark the order as EXECUTED
      - Square off the position with realized P&L

    Returns list of triggered GTT rules that were synced.
    """
    from database.db import get_connection, _execute
    from trading.trading_engine import square_off

    conn = get_connection()

    # Find all orders with pending GTT rules
    pending_gtts = _execute(conn, """
        SELECT o.id, o.user_id, o.symbol, o.order_purpose, o.gtt_rule_id,
               o.quantity, o.price, o.bracket_id
        FROM orders o
        WHERE o.gtt_rule_id IS NOT NULL
          AND o.gtt_status = 'PENDING'
          AND o.status = 'PENDING'
    """).fetchall()

    if not pending_gtts:
        conn.close()
        logger.debug("No pending GTT rules to sync")
        return []

    cols = ["id", "user_id", "symbol", "order_purpose", "gtt_rule_id",
            "quantity", "price", "bracket_id"]

    smart_api = _get_angel_session()
    if not smart_api:
        conn.close()
        return []

    triggered = []

    for row in pending_gtts:
        gtt = dict(zip(cols, row))
        rule_id = gtt["gtt_rule_id"]

        try:
            details = smart_api.gttDetails(int(rule_id))

            if not details:
                continue

            # Check if rule status indicates triggered
            rule_status = None
            if isinstance(details, dict):
                rule_status = details.get("status", "").upper()
            elif isinstance(details, list) and len(details) > 0:
                rule_status = details[0].get("status", "").upper()

            if rule_status in ("TRIGGERED", "SENTTOEXCHANGE", "FORALL"):
                logger.warning(
                    f"⚡ GTT TRIGGERED: {gtt['symbol']} {gtt['order_purpose']} "
                    f"rule_id={rule_id}"
                )

                # Update GTT status in our DB
                from datetime import datetime
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                _execute(conn, """
                    UPDATE orders SET gtt_status = 'TRIGGERED', status = 'EXECUTED',
                        fill_price = price, updated_at = ?
                    WHERE id = ?
                """, (now, gtt["id"]))

                # Cancel the other leg (if SL triggered, cancel Target and vice versa)
                other_purpose = "TARGET" if gtt["order_purpose"] == "STOP_LOSS" else "STOP_LOSS"
                other_order = _execute(conn, """
                    SELECT id, gtt_rule_id FROM orders
                    WHERE bracket_id = ? AND order_purpose = ? AND status = 'PENDING'
                """, (gtt["bracket_id"], other_purpose)).fetchone()

                if other_order and other_order[1]:
                    cancel_gtt(int(other_order[1]))
                    _execute(conn, """
                        UPDATE orders SET gtt_status = 'CANCELLED', status = 'CANCELLED',
                            updated_at = ?
                        WHERE id = ?
                    """, (now, other_order[0]))

                conn.commit()

                # Square off the position in our trading engine
                try:
                    result = square_off(gtt["user_id"], gtt["symbol"], sell_price=gtt["price"])
                    result["trigger"] = gtt["order_purpose"]
                    result["gtt_rule_id"] = rule_id
                    triggered.append(result)
                except Exception as e:
                    logger.error(f"Error squaring off after GTT trigger: {e}")

            elif rule_status == "CANCELLED":
                _execute(conn, """
                    UPDATE orders SET gtt_status = 'CANCELLED', updated_at = ?
                    WHERE id = ?
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gtt["id"]))
                conn.commit()

        except Exception as e:
            logger.error(f"Error checking GTT rule {rule_id}: {e}")

    conn.close()
    return triggered


# ==========================================
# CONVENIENCE: Place both SL + Target GTTs
# ==========================================

def place_bracket_gtts(
    symbol: str,
    qty: int,
    stop_loss: float,
    target_price: float,
) -> Dict:
    """
    Place both SL and Target GTT rules for a position.

    Returns:
        {
            "sl_rule_id": int or None,
            "target_rule_id": int or None,
            "success": bool
        }
    """
    sl_id = create_sl_gtt(symbol, qty, trigger_price=stop_loss)
    target_id = create_target_gtt(symbol, qty, trigger_price=target_price)

    success = sl_id is not None and target_id is not None

    if not success:
        # Rollback: cancel any GTT that was created
        if sl_id:
            cancel_gtt(sl_id)
        if target_id:
            cancel_gtt(target_id)
        logger.error(f"❌ Failed to place bracket GTTs for {symbol} — rolled back")

    return {
        "sl_rule_id": sl_id,
        "target_rule_id": target_id,
        "success": success,
    }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Test: list active GTT rules
    print("\n📋 Active GTT rules:")
    rules = list_gtt_rules(status="NEW")
    if rules:
        for r in rules[:5]:
            print(f"  {r}")
    else:
        print("  No active rules")
