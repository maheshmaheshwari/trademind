"""
TradeMind AI — Risk Manager

6 pre-trade safety checks before every order:
1. Sufficient virtual balance
2. Daily loss limit not breached
3. Max trades per day not exceeded
4. Position concentration check (max % in one stock)
5. Quantity within volume safety cap
6. Market hours check (9:15-15:30 IST)
"""
from datetime import datetime
from typing import Dict, Tuple
from database.db import get_connection


def get_risk_settings(user_id: int) -> Dict:
    """Get risk settings for a user. Creates defaults if not exist."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)
    ).fetchone()
    
    if not row:
        conn.execute(
            "INSERT INTO risk_settings (user_id) VALUES (?)", (user_id,)
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)
        ).fetchone()
    
    cols = [d[0] for d in conn.execute("SELECT * FROM risk_settings LIMIT 0").description]
    conn.close()
    return dict(zip(cols, row))


def update_risk_settings(user_id: int, settings: Dict) -> Dict:
    """Update risk settings for a user."""
    conn = get_connection()
    allowed = ["max_daily_loss", "max_daily_trades", "max_position_pct",
               "auto_stop_loss", "auto_target"]
    
    for key, value in settings.items():
        if key in allowed:
            conn.execute(
                f"UPDATE risk_settings SET {key} = ? WHERE user_id = ?",
                (value, user_id)
            )
    conn.commit()
    conn.close()
    return get_risk_settings(user_id)


def check_order(
    user_id: int,
    symbol: str,
    investment_amount: float,
    quantity: int,
    max_safe_qty: int = None,
) -> Tuple[bool, str, list]:
    """
    Run all 6 risk checks. Returns (approved, reason, checks).
    
    Each check is: {"name": str, "passed": bool, "detail": str}
    """
    conn = get_connection()
    checks = []
    
    # Get user + settings
    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    user_cols = [d[0] for d in conn.execute("SELECT * FROM users LIMIT 0").description]
    user_dict = dict(zip(user_cols, user))
    
    settings = get_risk_settings(user_id)
    today = datetime.now().strftime("%Y-%m-%d")
    
    # 1. Balance check
    available = user_dict["virtual_balance"]
    has_balance = investment_amount <= available
    checks.append({
        "name": "Balance",
        "passed": has_balance,
        "detail": f"₹{available:,.2f} available, ₹{investment_amount:,.2f} needed"
    })
    
    # 2. Daily loss limit
    daily_loss = conn.execute("""
        SELECT COALESCE(SUM(pnl), 0) FROM orders 
        WHERE user_id = ? AND DATE(created_at) = ? AND pnl < 0
    """, (user_id, today)).fetchone()[0]
    loss_ok = abs(daily_loss) < settings["max_daily_loss"]
    checks.append({
        "name": "Daily Loss Limit",
        "passed": loss_ok,
        "detail": f"Today's loss: ₹{abs(daily_loss):,.2f} / ₹{settings['max_daily_loss']:,.2f} max"
    })
    
    # 3. Daily trade count
    trade_count = conn.execute("""
        SELECT COUNT(*) FROM orders 
        WHERE user_id = ? AND DATE(created_at) = ? AND order_purpose = 'ENTRY'
    """, (user_id, today)).fetchone()[0]
    trades_ok = trade_count < settings["max_daily_trades"]
    checks.append({
        "name": "Daily Trade Limit",
        "passed": trades_ok,
        "detail": f"{trade_count} / {settings['max_daily_trades']} trades today"
    })
    
    # 4. Position concentration
    total_capital = user_dict["virtual_balance"] + user_dict["virtual_invested"]
    position_pct = (investment_amount / total_capital * 100) if total_capital > 0 else 100
    conc_ok = position_pct <= settings["max_position_pct"]
    checks.append({
        "name": "Position Concentration",
        "passed": conc_ok,
        "detail": f"{position_pct:.1f}% of capital / {settings['max_position_pct']}% max"
    })
    
    # 5. Volume safety
    if max_safe_qty and max_safe_qty > 0:
        vol_ok = quantity <= max_safe_qty
        checks.append({
            "name": "Volume Safety",
            "passed": vol_ok,
            "detail": f"{quantity} qty / {max_safe_qty} max safe qty (2% ADV)"
        })
    else:
        checks.append({
            "name": "Volume Safety",
            "passed": True,
            "detail": "No volume data — skipped"
        })
    
    # 6. Market hours (IST: 9:15 - 15:30)
    # For paper trading, we allow anytime but flag it
    now = datetime.now()
    hour, minute = now.hour, now.minute
    market_open = (hour > 9 or (hour == 9 and minute >= 15)) and \
                  (hour < 15 or (hour == 15 and minute <= 30))
    checks.append({
        "name": "Market Hours",
        "passed": True,  # Always pass for paper trading
        "detail": "Market OPEN" if market_open else "Market CLOSED (paper trade OK)"
    })
    
    conn.close()
    
    # Overall result
    all_passed = all(c["passed"] for c in checks)
    failed = [c for c in checks if not c["passed"]]
    reason = failed[0]["detail"] if failed else "All checks passed"
    
    return all_passed, reason, checks
