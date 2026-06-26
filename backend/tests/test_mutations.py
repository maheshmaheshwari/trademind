"""
Write-path (POST / PATCH / PUT / DELETE) tests against the TEST database.

Each test registers a fresh user (real DB write, same path the app itself
uses), authenticates with the returned JWT, then exercises the mutating
endpoint and checks both the HTTP response and the resulting DB state —
the same "enter values like how it will be entered, then verify" pattern
as the GET-route tests in test_api_routes.py.

Endpoints that require a real external broker/Angel One session
(execute-signal, angel-one/connect, totp/*, signal refresh, autopilot) are
out of scope here — see test_external_api_contracts.py for how those
external dependencies are tested via fixtures instead.
"""
from datetime import datetime

from database.db import get_connection, release_connection, _execute

_TODAY = datetime.now().strftime("%Y-%m-%d")


def _register(api_client, username="mutationtester"):
    resp = api_client.post(
        "/api/trading/register",
        json={"username": username, "password": "Sup3rSecret!", "display_name": "Mutation Tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["token"], body["user"]["id"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------------------
# Auth: register / PATCH /auth/me / PUT /auth/preferences
# ---------------------------------------------------------------------------

def test_register_creates_user_and_returns_token(api_client):
    token, user_id = _register(api_client, "registertest")
    assert token
    resp = api_client.get("/api/trading/me", headers=_auth_headers(token))
    assert resp.status_code == 200
    assert resp.json()["username"] == "registertest"


def test_register_rejects_short_password(api_client):
    resp = api_client.post(
        "/api/trading/register",
        json={"username": "shortpw", "password": "short", "display_name": "X"},
    )
    assert resp.status_code == 400


def test_patch_auth_me_updates_profile(api_client):
    token, user_id = _register(api_client, "patchmetest")

    resp = api_client.patch(
        "/auth/me",
        json={"display_name": "Updated Name", "email": "updated@example.com"},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["display_name"] == "Updated Name"
    assert body["email"] == "updated@example.com"


def test_patch_auth_me_rejects_empty_update(api_client):
    token, _ = _register(api_client, "patchemptytest")
    resp = api_client.patch("/auth/me", json={}, headers=_auth_headers(token))
    assert resp.status_code == 400


def test_put_auth_preferences_updates_and_persists(api_client):
    token, user_id = _register(api_client, "prefstest")

    resp = api_client.put(
        "/auth/preferences",
        json={"default_account": "LIVE", "currency": "USD"},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200
    assert resp.json()["default_account"] == "LIVE"

    resp = api_client.get("/auth/preferences", headers=_auth_headers(token))
    assert resp.json()["default_account"] == "LIVE"
    assert resp.json()["currency"] == "USD"


def test_put_auth_preferences_rejects_invalid_account_type(api_client):
    token, _ = _register(api_client, "prefsinvalidtest")
    resp = api_client.put(
        "/auth/preferences", json={"default_account": "BOGUS"}, headers=_auth_headers(token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Watchlist: POST / PUT / DELETE
# ---------------------------------------------------------------------------

def test_watchlist_add_update_alerts_and_remove(api_client):
    token, user_id = _register(api_client, "watchlisttest")
    headers = _auth_headers(token)

    resp = api_client.post(f"/api/users/{user_id}/watchlist/reliance", headers=headers)
    assert resp.status_code == 201
    assert resp.json()["data"]["symbol"] == "RELIANCE"

    resp = api_client.get(f"/api/users/{user_id}/watchlist", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["total"] == 1

    resp = api_client.put(
        f"/api/users/{user_id}/watchlist/reliance/alerts",
        json={"alert_above": 3000.0, "alert_below": 2500.0},
        headers=headers,
    )
    assert resp.status_code == 200

    resp = api_client.delete(f"/api/users/{user_id}/watchlist/reliance", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get(f"/api/users/{user_id}/watchlist", headers=headers)
    assert resp.json()["total"] == 0


def test_watchlist_rejects_other_users_access(api_client):
    token, user_id = _register(api_client, "watchlistowner")
    _, other_user_id = _register(api_client, "watchlistintruder")
    headers = _auth_headers(token)

    resp = api_client.post(f"/api/users/{other_user_id}/watchlist/tcs", headers=headers)
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Notifications: POST mark-read / PUT preferences
# ---------------------------------------------------------------------------

def test_notification_preferences_put_then_get(api_client):
    token, user_id = _register(api_client, "notifprefstest")
    headers = _auth_headers(token)

    resp = api_client.put(
        "/api/notifications/preferences",
        json={"signal_change": False, "ch_sms": True},
        headers=headers,
    )
    assert resp.status_code == 200

    resp = api_client.get("/api/notifications/preferences", headers=headers)
    body = resp.json()
    assert body["signal_change"] is False
    assert body["ch_sms"] is True
    assert body["price_alert"] is True  # untouched default preserved


def test_notifications_mark_read(api_client):
    token, user_id = _register(api_client, "marknotifstest")
    headers = _auth_headers(token)
    from database.db import insert_notification

    insert_notification(user_id, "signal", "Test alert", "A test notification")

    resp = api_client.post("/api/notifications/mark-read", headers=headers)
    assert resp.status_code == 200

    conn = get_connection()
    try:
        row = _execute(
            conn, "SELECT is_read FROM notifications WHERE user_id = ?", (user_id,)
        ).fetchone()
    finally:
        release_connection(conn)
    assert row[0] is True


# ---------------------------------------------------------------------------
# Risk settings: PUT
# ---------------------------------------------------------------------------

def test_put_risk_settings_updates_values(api_client):
    token, user_id = _register(api_client, "risksettingstest")
    headers = _auth_headers(token)

    resp = api_client.put(
        f"/api/trading/risk-settings/{user_id}",
        json={"max_daily_loss": 5000.0, "max_daily_trades": 3},
        headers=headers,
    )
    assert resp.status_code == 200
    settings = resp.json()["settings"]
    assert settings["max_daily_loss"] == 5000.0
    assert settings["max_daily_trades"] == 3


def test_put_risk_settings_rejects_empty_update(api_client):
    token, user_id = _register(api_client, "riskemptytest")
    resp = api_client.put(
        f"/api/trading/risk-settings/{user_id}", json={}, headers=_auth_headers(token),
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Portfolio: POST create / PUT sectors / DELETE
# ---------------------------------------------------------------------------

def _seed_trade_signal_for_portfolio(conn, symbol, name, sector_signal="STRONG BUY", horizon="1 Month"):
    _execute(
        conn,
        """INSERT INTO trade_signals
           (symbol, name, signal, confidence, trade_type, buy_price, target_price,
            stop_loss, risk_reward, expected_return_pct, model_horizon,
            generated_date, generated_at, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, name, sector_signal, 90.0, "LONG", 100.0, 110.0, 95.0, 2.0, 10.0,
         horizon, _TODAY, f"{_TODAY} 09:00:00", True),
    )


def test_portfolio_create_update_sectors_and_delete(api_client):
    token, user_id = _register(api_client, "portfoliotest")
    headers = _auth_headers(token)

    conn = get_connection()
    try:
        _seed_trade_signal_for_portfolio(conn, "RELIANCE.NS", "Reliance Industries Ltd.")
        _seed_trade_signal_for_portfolio(conn, "TCS.NS", "Tata Consultancy Services Ltd.")
        _seed_trade_signal_for_portfolio(conn, "HDFCBANK.NS", "HDFC Bank Ltd.")
        conn.commit()
    finally:
        release_connection(conn)

    resp = api_client.post(
        "/api/portfolio/create",
        json={
            "name": "Test Portfolio", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    portfolio_id = resp.json()["data"]["portfolio_id"] if "data" in resp.json() else resp.json().get("portfolio_id")
    assert portfolio_id

    resp = api_client.get("/api/portfolio", headers=headers)
    assert resp.status_code == 200
    assert any(p["id"] == portfolio_id for p in resp.json()["data"])

    conn = get_connection()
    try:
        sectors = _execute(
            conn, "SELECT sector, allocation_pct FROM portfolio_sectors WHERE portfolio_id = ?",
            (portfolio_id,),
        ).fetchall()
    finally:
        release_connection(conn)
    assert sectors, "expected AI-allocated sectors to be saved"

    # Rescale to exactly 100% — this test exercises the PUT /sectors endpoint
    # itself, not the AI allocator's own rounding (which may cap+redistribute
    # to slightly under 100 when every sector hits the risk-profile cap).
    raw_total = sum(s[1] for s in sectors)
    sector_update = [
        {"sector": s[0], "allocation_pct": round(s[1] / raw_total * 100, 1)} for s in sectors
    ]
    resp = api_client.put(f"/api/portfolio/{portfolio_id}/sectors", json={"sectors": sector_update}, headers=headers)
    assert resp.status_code == 200

    resp = api_client.delete(f"/api/portfolio/{portfolio_id}", headers=headers)
    assert resp.status_code == 200

    conn = get_connection()
    try:
        remaining = _execute(
            conn, "SELECT id FROM portfolios WHERE id = ?", (portfolio_id,)
        ).fetchone()
    finally:
        release_connection(conn)
    assert remaining is None


def test_portfolio_create_fails_without_signals(api_client):
    token, _ = _register(api_client, "emptyportfoliotest")
    resp = api_client.post(
        "/api/portfolio/create",
        json={
            "name": "Empty Portfolio", "investment_amount": 50000.0,
            "time_horizon": "short", "risk_profile": "conservative",
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 400


def test_put_portfolio_sectors_rejects_nonexistent_portfolio(api_client):
    token, _ = _register(api_client, "nonexistentportfoliotest")
    resp = api_client.put(
        "/api/portfolio/999999/sectors",
        json={"sectors": [{"sector": "IT", "allocation_pct": 100}]},
        headers=_auth_headers(token),
    )
    assert resp.status_code == 404


def test_portfolio_create_requires_auth(api_client):
    resp = api_client.post(
        "/api/portfolio/create",
        json={"name": "No Auth", "investment_amount": 1000.0, "time_horizon": "short", "risk_profile": "moderate"},
    )
    assert resp.status_code == 401


def test_portfolio_cannot_be_accessed_by_other_user(api_client):
    """Regression test for audit finding C1 (IDOR) — portfolios are now user-scoped."""
    owner_token, _ = _register(api_client, "portfolioowner")
    intruder_token, _ = _register(api_client, "portfoliointruder")

    conn = get_connection()
    try:
        _seed_trade_signal_for_portfolio(conn, "RELIANCE.NS", "Reliance Industries Ltd.")
        conn.commit()
    finally:
        release_connection(conn)

    resp = api_client.post(
        "/api/portfolio/create",
        json={"name": "Owner Portfolio", "investment_amount": 50000.0, "time_horizon": "medium", "risk_profile": "moderate"},
        headers=_auth_headers(owner_token),
    )
    assert resp.status_code == 200, resp.text
    portfolio_id = resp.json()["data"]["portfolio_id"]

    # Intruder can't see it via GET, can't delete it, and the owner's list excludes nothing.
    resp = api_client.get(f"/api/portfolio/{portfolio_id}", headers=_auth_headers(intruder_token))
    assert resp.status_code == 404

    resp = api_client.delete(f"/api/portfolio/{portfolio_id}", headers=_auth_headers(intruder_token))
    assert resp.status_code == 404

    resp = api_client.get("/api/portfolio", headers=_auth_headers(intruder_token))
    assert resp.status_code == 200
    assert resp.json()["data"] == []

    # Unauthenticated callers are rejected outright.
    resp = api_client.get(f"/api/portfolio/{portfolio_id}")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Trading engine: execute-signal -> position -> square-off (PAPER mode)
#
# Exercises the concurrency-redesign work from the security audit (C2, H5-H9,
# M6, M7) end-to-end through the real API: the per-user advisory lock in
# execute_signal(), risk checks now enforced inside it, and square_off()'s
# new FOR UPDATE + try/finally.
# ---------------------------------------------------------------------------

def test_execute_signal_creates_position_and_square_off_closes_it(api_client):
    token, user_id = _register(api_client, "tradingenginetest")
    headers = _auth_headers(token)

    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": "WIPRO.NS", "name": "Wipro Ltd.",
            "investment_amount": 10000.0, "buy_price": 250.0,
            "target_price": 275.0, "stop_loss": 235.0, "mode": "PAPER",
        },
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "executed"
    assert body["position"]["symbol"] == "WIPRO.NS"

    resp = api_client.get(f"/api/trading/positions/{user_id}", headers=headers)
    assert resp.status_code == 200
    symbols = [p["symbol"] for p in resp.json()["data"]]
    assert "WIPRO.NS" in symbols

    # Can't open a second position in the same symbol without squaring off first.
    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": "WIPRO.NS", "name": "Wipro Ltd.",
            "investment_amount": 5000.0, "buy_price": 250.0,
            "target_price": 275.0, "stop_loss": 235.0, "mode": "PAPER",
        },
        headers=headers,
    )
    assert resp.status_code == 400

    resp = api_client.post(f"/api/trading/square-off/{user_id}/WIPRO.NS", json={}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "WIPRO.NS"

    # Position is gone; squaring off again cleanly reports "no position" rather than crashing.
    resp = api_client.get(f"/api/trading/positions/{user_id}", headers=headers)
    assert resp.json()["data"] == []

    resp = api_client.post(f"/api/trading/square-off/{user_id}/WIPRO.NS", json={}, headers=headers)
    assert resp.status_code == 400


def test_execute_signal_rejects_zero_buy_price(api_client):
    """Regression test for audit finding M11 — buy_price=0 used to reach
    trading_engine.py and raise a raw ZeroDivisionError; now rejected at the
    request-validation layer."""
    token, user_id = _register(api_client, "zeropricetest")
    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": "INFY.NS", "name": "Infosys Ltd.",
            "investment_amount": 10000.0, "buy_price": 0,
            "target_price": 1700.0, "stop_loss": 1500.0, "mode": "PAPER",
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 422


def test_execute_signal_rejects_insufficient_balance(api_client):
    token, user_id = _register(api_client, "insufficientbalancetest")
    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": "TATAMOTORS.NS", "name": "Tata Motors Ltd.",
            "investment_amount": 99999999.0, "buy_price": 500.0,
            "target_price": 550.0, "stop_loss": 470.0, "mode": "PAPER",
        },
        headers=_auth_headers(token),
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert any(not c["passed"] for c in resp.json()["risk_checks"])


def test_execute_signal_concurrent_same_symbol_only_one_wins(api_client):
    """
    Best-effort regression test for audit finding H5 (new-position TOCTOU
    race). Before the per-user advisory lock, two concurrent first-time
    execute_signal() calls for the same symbol could both pass the
    no-existing-position check and both proceed, with only the positions
    table's UNIQUE(user_id, symbol) constraint catching the second one
    AFTER its money movement had already committed. This is probabilistic
    by nature of testing real concurrency in a single process — the actual
    guarantee comes from the DB-level advisory lock, not from this test —
    but it reliably reproduces the race shape often enough to catch a
    regression if the lock is ever removed. What must always hold
    regardless of timing: exactly one call succeeds, the rest fail cleanly
    (a caught exception/error dict), and the account is left in a
    consistent state (exactly one open position, balance debited once).
    """
    from concurrent.futures import ThreadPoolExecutor
    from trading.trading_engine import execute_signal

    _, user_id = _register(api_client, "concurrencytest")

    def attempt():
        try:
            return execute_signal(
                user_id=user_id, symbol="ITC.NS", name="ITC Ltd.",
                investment_amount=5000.0, buy_price=400.0,
                target_price=440.0, stop_loss=380.0, mode="PAPER",
            )
        except Exception as e:
            return {"error": str(e)}

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda _: attempt(), range(5)))

    successes = [r for r in results if "error" not in r]
    failures = [r for r in results if "error" in r]
    assert len(successes) == 1, f"expected exactly 1 success, got {len(successes)}: {results}"
    assert len(failures) == 4

    conn = get_connection()
    try:
        positions = _execute(conn, "SELECT symbol FROM positions WHERE user_id = ? AND symbol = ?", (user_id, "ITC.NS")).fetchall()
        balance = _execute(conn, "SELECT virtual_balance FROM users WHERE id = ?", (user_id,)).fetchone()[0]
    finally:
        release_connection(conn)
    assert len(positions) == 1, "exactly one position should exist, not duplicated or zero"
    quantity = int(5000.0 / 400.0)
    actual_investment = round(quantity * 400.0, 2)
    fees = round(actual_investment * (0.0005 + 0.001 + 0.000001 + 0.00015), 2)
    assert balance == 1000000.0 - actual_investment - fees, \
        "balance should be debited exactly once, not once per attempt"
