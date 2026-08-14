"""
Price-monitor exit tests: what price a PAPER position is actually squared off at
when SL or target triggers.

The two branches deliberately price their exits differently, because the two
legs are different order types:

  * STOP_LOSS is a stop — it triggers AT the stop and fills at whatever the
    market is offering, which on a gap is materially worse. Exits at
    min(market, sl).
  * TARGET is a resting LIMIT sell — it fills at `target` the moment the stock
    trades through it. The monitor only samples every 5 minutes, so the market
    price it sees can be well past the target after a run-up the resting order
    would have been filled long before. Exits at `target`.

APP_ENV=test blocks the live-LTP path (price_monitor._fetch_live_prices), so the
seeded `prices` row is the market price these tests exercise.
"""
from datetime import datetime

import pytest

from database.db import get_connection, release_connection, _execute

_TODAY = datetime.now().strftime("%Y-%m-%d")


def _register(api_client, username):
    resp = api_client.post(
        "/api/trading/register",
        json={"username": username, "password": "Sup3rSecret!", "display_name": "Monitor Tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["token"], body["user"]["id"]


def _set_price(symbol, close):
    """Set today's bar for a symbol — the DB fallback the monitor reads."""
    conn = get_connection()
    try:
        _execute(conn,
            "INSERT INTO prices (symbol, date, open, high, low, close, volume, interval) "
            "VALUES (?,?,?,?,?,?,?,'1d') "
            "ON CONFLICT (symbol, date, interval) WHERE time IS NULL "
            "DO UPDATE SET close = EXCLUDED.close",
            (symbol, _TODAY, close, close, close, close, 10000))
        conn.commit()
    finally:
        release_connection(conn)


def _buy(api_client, token, user_id, symbol, price, sl, target, amount=30000.0):
    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": symbol, "name": symbol.replace(".NS", ""),
            "investment_amount": amount, "buy_price": price,
            "target_price": target, "stop_loss": sl, "mode": "PAPER",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "executed"
    return resp.json()


def _exit_order(user_id, symbol):
    conn = get_connection()
    try:
        return _execute(conn,
            "SELECT price, fill_price, pnl FROM orders "
            "WHERE user_id = ? AND symbol = ? AND order_purpose = 'SQUARE_OFF'",
            (user_id, symbol)).fetchone()
    finally:
        release_connection(conn)


def test_stop_loss_gap_exits_at_market_not_at_the_stop(api_client):
    """A stock that gaps clean through the stop fills below it. Booking every
    stop at `sl` regardless hid the gap portion of the loss entirely."""
    from trading.price_monitor import update_position_prices

    token, user_id = _register(api_client, "slgaptest")
    _set_price("GAPDOWN.NS", 300.0)
    _buy(api_client, token, user_id, "GAPDOWN.NS", price=300.0, sl=280.0, target=330.0)

    # Gaps to 240 — far below the 280 stop.
    _set_price("GAPDOWN.NS", 240.0)
    triggered = update_position_prices(user_id)

    assert len(triggered) == 1
    assert triggered[0]["trigger"] == "STOP_LOSS"

    price, fill_price, pnl = _exit_order(user_id, "GAPDOWN.NS")
    assert fill_price == pytest.approx(240.0), "stop must fill at the market, not at the stop price"

    # The whole ₹60/share loss is booked, not the ₹20 the stop level implies.
    qty = int(30000.0 / 300.0)
    gross = (240.0 - 300.0) * qty
    assert pnl == pytest.approx(gross - round(240.0 * qty * 0.0005, 2), abs=0.01)


def test_stop_loss_without_a_gap_still_exits_at_the_stop(api_client):
    """The ordinary case is unchanged: price drifting down through the stop
    fills essentially at the stop, so min() must not move the number."""
    from trading.price_monitor import update_position_prices

    token, user_id = _register(api_client, "sldrifttest")
    _set_price("DRIFT.NS", 300.0)
    _buy(api_client, token, user_id, "DRIFT.NS", price=300.0, sl=280.0, target=330.0)

    _set_price("DRIFT.NS", 279.5)   # a hair through the stop
    triggered = update_position_prices(user_id)

    assert len(triggered) == 1
    assert triggered[0]["trigger"] == "STOP_LOSS"
    _, fill_price, _ = _exit_order(user_id, "DRIFT.NS")
    assert fill_price == pytest.approx(279.5)


def test_target_exits_at_the_target_not_at_a_higher_market(api_client):
    """The opposite rule for the target leg: a resting limit sell fills at the
    limit. The monitor's 5-minute sample can be well past it, and crediting that
    price would book a gain the order could never have captured."""
    from trading.price_monitor import update_position_prices

    token, user_id = _register(api_client, "targetruntest")
    _set_price("RUNUP.NS", 300.0)
    _buy(api_client, token, user_id, "RUNUP.NS", price=300.0, sl=280.0, target=330.0)

    # Ran to 360 between two samples — the limit at 330 would have gone first.
    _set_price("RUNUP.NS", 360.0)
    triggered = update_position_prices(user_id)

    assert len(triggered) == 1
    assert triggered[0]["trigger"] == "TARGET"
    _, fill_price, _ = _exit_order(user_id, "RUNUP.NS")
    assert fill_price == pytest.approx(330.0), "target must fill at the limit, not at the run-up price"


def test_position_untouched_between_stop_and_target(api_client):
    """No trigger, no exit order — the position just re-prices."""
    from trading.price_monitor import update_position_prices

    token, user_id = _register(api_client, "notriggertest")
    _set_price("QUIET.NS", 300.0)
    _buy(api_client, token, user_id, "QUIET.NS", price=300.0, sl=280.0, target=330.0)

    _set_price("QUIET.NS", 305.0)
    assert update_position_prices(user_id) == []

    conn = get_connection()
    try:
        pos = _execute(conn,
            "SELECT current_price FROM positions WHERE user_id = ? AND symbol = ?",
            (user_id, "QUIET.NS")).fetchone()
    finally:
        release_connection(conn)
    assert pos is not None, "position must still be open"
    assert pos[0] == pytest.approx(305.0)
    assert _exit_order(user_id, "QUIET.NS") is None
