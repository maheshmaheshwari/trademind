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
from database.db import get_connection, release_connection, _execute


_ALLOWED_TABLES = frozenset({"users", "orders", "positions", "risk_settings", "trade_signals"})


def _col_names(conn, table: str):
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Table '{table}' is not in the allowed list")
    cur = _execute(conn, f"SELECT * FROM {table} LIMIT 0")
    return [d[0] for d in cur.description]


def get_risk_settings(user_id: int) -> Dict:
    """Get risk settings for a user. Creates defaults if not exist."""
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            _execute(conn, "INSERT INTO risk_settings (user_id) VALUES (?)", (user_id,))
            conn.commit()
            row = _execute(conn, "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)).fetchone()
        cols = _col_names(conn, "risk_settings")
        return dict(zip(cols, row))
    finally:
        release_connection(conn)


def update_risk_settings(user_id: int, settings: Dict) -> Dict:
    """Update risk settings for a user."""
    get_risk_settings(user_id)  # ensure a default row exists before UPDATE — first-ever
                                 # update for a user would otherwise affect 0 rows
    conn = get_connection()
    allowed = ["max_daily_loss", "max_daily_trades", "max_position_pct",
               "max_position_size", "stop_loss_pct", "target_pct",
               "auto_stop_loss", "auto_target", "mode"]
    try:
        for key, value in settings.items():
            if key in allowed:
                _execute(
                    conn,
                    f"UPDATE risk_settings SET {key} = ? WHERE user_id = ?",
                    (value, user_id)
                )
        conn.commit()
    finally:
        release_connection(conn)
    return get_risk_settings(user_id)


def check_order(
    user_id: int,
    symbol: str,
    investment_amount: float,
    quantity: int,
    max_safe_qty: int = None,  # deprecated/ignored — see audit M9; kept only for call-site compatibility
    mode: str = "PAPER",
) -> Tuple[bool, str, list]:
    """
    Run all 6 risk checks. Returns (approved, reason, checks).

    Each check is: {"name": str, "passed": bool, "detail": str}
    """
    conn = get_connection()
    checks = []

    try:
        today = datetime.now().strftime("%Y-%m-%d")

        # Get user
        user = _execute(conn, "SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        user_cols = _col_names(conn, "users")
        user_dict = dict(zip(user_cols, user))

        # Get settings inline (avoids opening a second connection)
        row = _execute(conn, "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            _execute(conn, "INSERT INTO risk_settings (user_id) VALUES (?)", (user_id,))
            conn.commit()
            row = _execute(conn, "SELECT * FROM risk_settings WHERE user_id = ?", (user_id,)).fetchone()
        settings = dict(zip(_col_names(conn, "risk_settings"), row))

        # 1. Balance check
        available = user_dict["virtual_balance"]
        has_balance = investment_amount <= available
        checks.append({
            "name": "Balance",
            "passed": has_balance,
            "detail": f"₹{available:,.2f} available, ₹{investment_amount:,.2f} needed"
        })

        # 2. Daily loss limit
        daily_loss = _execute(conn, """
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
        trade_count = _execute(conn, """
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

        # 5. Volume safety — derived server-side from trade_signals.recommended_volume
        # / consumed_volume, never trusted from the caller (audit M9: a client could
        # previously omit max_safe_qty entirely to disable this check).
        sig_row = _execute(conn, """
            SELECT recommended_volume, consumed_volume FROM trade_signals
            WHERE symbol = ? AND is_active = TRUE
            ORDER BY generated_date DESC LIMIT 1
        """, (symbol,)).fetchone()
        server_max_safe_qty = None
        if sig_row and sig_row[0]:
            server_max_safe_qty = max(0, (sig_row[0] or 0) - (sig_row[1] or 0))

        if server_max_safe_qty is not None:
            vol_ok = quantity <= server_max_safe_qty
            checks.append({
                "name": "Volume Safety",
                "passed": vol_ok,
                "detail": f"{quantity} qty / {server_max_safe_qty} max safe qty (platform capacity remaining)"
            })
        else:
            checks.append({
                "name": "Volume Safety",
                "passed": True,
                "detail": "No volume data — skipped"
            })

        # 6. Market hours (IST: 9:15 - 15:30) — only enforced for LIVE mode
        now_dt = datetime.now()
        hour, minute = now_dt.hour, now_dt.minute
        is_weekday = now_dt.weekday() < 5
        market_open = is_weekday and (hour > 9 or (hour == 9 and minute >= 15)) and \
                      (hour < 15 or (hour == 15 and minute <= 30))
        market_hours_pass = True if mode == "PAPER" else market_open
        checks.append({
            "name": "Market Hours",
            "passed": market_hours_pass,
            "detail": "Market OPEN" if market_open else (
                "Market CLOSED — LIVE orders rejected outside market hours"
                if mode == "LIVE" else "Market CLOSED (paper trade OK)"
            ),
        })

    finally:
        release_connection(conn)

    # Overall result
    all_passed = all(c["passed"] for c in checks)
    failed = [c for c in checks if not c["passed"]]
    reason = failed[0]["detail"] if failed else "All checks passed"

    return all_passed, reason, checks
