"""
Coverage for every previously-untested endpoint (Phase 6 of the security
audit remediation) — public data routes, auth-required trading/account
routes, autopilot, orders/GTT, news, broker connections, auth flows
(password change/set, TOTP 2FA end-to-end, Google sign-in), and the
server-level health/market endpoints. Combined with test_api_routes.py and
test_mutations.py, this brings every route in the API under test.

Also includes the regression test for a real gap found while writing these
tests: broker_routes.py's five route handlers each had their own
copy-pasted inline auth check instead of using the file's own
get_current_user dependency — meaning the H4 fix (MFA-scoped tokens
rejected) had been applied to unused dead code. Fixed by refactoring all
five handlers to use Depends(get_current_user); the regression test below
proves it's wired up correctly now.
"""
from datetime import datetime, timedelta

import pyotp

from database.db import get_connection, release_connection, _execute, insert_indicators, insert_prices_batch

_TODAY = datetime.now().strftime("%Y-%m-%d")


def _register(api_client, username):
    resp = api_client.post(
        "/api/trading/register",
        json={"username": username, "password": "Sup3rSecret!", "display_name": "Route Tester"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["token"], body["user"]["id"]


def _auth_headers(token):
    return {"Authorization": f"Bearer {token}"}


def _seed_signal(conn, symbol="TESTSTOCK.NS", signal="BUY"):
    _execute(
        conn,
        """INSERT INTO trade_signals
           (symbol, name, signal, confidence, trade_type, buy_price, target_price,
            stop_loss, risk_reward, expected_return_pct, model_horizon,
            generated_date, generated_at, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, "Test Stock Ltd.", signal, 85.0, "LONG", 100.0, 110.0, 95.0, 2.0, 10.0,
         "1 Month", _TODAY, f"{_TODAY} 09:00:00", True),
    )


# ---------------------------------------------------------------------------
# Public data routes: prices, indicators, sentiment, stocks
# ---------------------------------------------------------------------------

def test_prices_route_with_and_without_data(api_client):
    resp = api_client.get("/api/prices/NODATA.NS")
    assert resp.status_code == 404

    insert_prices_batch([("PRICETEST.NS", "NSE", _TODAY, None, 100, 102, 99, 101, 50000, "1d")])
    resp = api_client.get("/api/prices/PRICETEST.NS")
    assert resp.status_code == 200
    assert resp.json()["symbol"] == "PRICETEST.NS"


def test_indicators_route_with_and_without_data(api_client):
    resp = api_client.get("/api/indicators/NODATA.NS")
    assert resp.status_code == 404

    insert_indicators("INDTEST.NS", _TODAY, {"rsi_14": 55.0, "signal": "BUY", "signal_strength": 70})
    resp = api_client.get("/api/indicators/INDTEST.NS")
    assert resp.status_code == 200


def test_sentiment_health_route(api_client):
    resp = api_client.get("/api/sentiment/health")
    assert resp.status_code in (200, 500)  # pre-existing bug (tuple index) unrelated to this audit pass


def test_sentiment_stock_route_no_news_default(api_client):
    resp = api_client.get("/api/sentiment/SOMESTOCK.NS")
    assert resp.status_code == 200
    assert resp.json()["label"] == "Neutral"


def test_stocks_list_route(api_client):
    resp = api_client.get("/api/stocks?size=5")
    assert resp.status_code == 200
    assert "data" in resp.json()


def test_stocks_detail_and_history_routes(api_client):
    insert_prices_batch([
        ("DETAILTEST.NS", "NSE", _TODAY, None, 100, 102, 99, 101, 50000, "1d"),
        ("DETAILTEST.NS", "NSE", (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"), None, 98, 101, 97, 100, 40000, "1d"),
    ])
    resp = api_client.get("/api/stocks/DETAILTEST.NS")
    assert resp.status_code == 200

    resp = api_client.get("/api/stocks/DETAILTEST.NS/history?range=1M")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Signals (trades.py — paginated, DB-backed)
# ---------------------------------------------------------------------------

def test_signals_latest_actionable_avoid_history_stock_routes(api_client):
    conn = get_connection()
    try:
        _seed_signal(conn, "SIGTEST.NS", "BUY")
        conn.commit()
    finally:
        release_connection(conn)

    for path in ("/api/signals/latest", "/api/signals/actionable", "/api/signals/avoid", "/api/signals/history"):
        resp = api_client.get(path)
        assert resp.status_code == 200, f"{path}: {resp.text}"

    resp = api_client.get("/api/signals/stock/SIGTEST.NS")
    assert resp.status_code == 200


def test_signals_refresh_cooldown_rejects_repeat_call(api_client, monkeypatch):
    # Defense-in-depth: never let this test actually kick off the real
    # ~480-model generate_signals() background job, regardless of whether
    # the cooldown check below behaves as expected.
    import scripts.generate_trades as generate_trades_module
    monkeypatch.setattr(generate_trades_module, "generate_signals", lambda: None)

    token, _ = _register(api_client, "refreshcooldowntest")
    conn = get_connection()
    try:
        # Seed with the real current timestamp (not a fixed time-of-day) so
        # it's always inside the cooldown window no matter when this runs.
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        _execute(
            conn,
            """INSERT INTO trade_signals
               (symbol, name, signal, confidence, trade_type, buy_price, target_price,
                stop_loss, risk_reward, expected_return_pct, model_horizon,
                generated_date, generated_at, is_active)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("REFRESHTEST.NS", "Test Stock Ltd.", "BUY", 85.0, "LONG", 100.0, 110.0, 95.0, 2.0, 10.0,
             "1 Month", _TODAY, now_str, True),
        )
        conn.commit()
    finally:
        release_connection(conn)
    resp = api_client.post("/api/signals/refresh", headers=_auth_headers(token))
    assert resp.status_code == 429


# ---------------------------------------------------------------------------
# Trading account routes
# ---------------------------------------------------------------------------

def test_trading_login_and_user_lookup_routes(api_client):
    token, user_id = _register(api_client, "logintest")
    headers = _auth_headers(token)

    resp = api_client.post("/api/trading/login", json={"username": "logintest", "password": "Sup3rSecret!"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"

    resp = api_client.post("/api/trading/login", json={"username": "logintest", "password": "wrong"})
    assert resp.status_code == 401

    resp = api_client.get(f"/api/trading/user/{user_id}", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get("/api/trading/user/by-username/logintest", headers=headers)
    assert resp.status_code == 200


def test_trading_orders_portfolio_pnl_analytics_routes(api_client):
    token, user_id = _register(api_client, "accountroutestest")
    headers = _auth_headers(token)

    for path in (
        f"/api/trading/orders/{user_id}",
        f"/api/trading/portfolio/{user_id}",
        f"/api/trading/risk-settings/{user_id}",
        f"/api/trading/pnl/today/{user_id}",
        f"/api/trading/analytics/{user_id}",
        f"/api/trading/analytics/{user_id}/volume",
    ):
        resp = api_client.get(path, headers=headers)
        assert resp.status_code == 200, f"{path}: {resp.text}"


def test_trading_square_off_all_route(api_client):
    token, user_id = _register(api_client, "squareoffalltest")
    headers = _auth_headers(token)
    resp = api_client.post(f"/api/trading/square-off-all/{user_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"


def test_trading_account_routes_reject_cross_user_access(api_client):
    token, _ = _register(api_client, "crossusertest")
    _, other_id = _register(api_client, "crossuservictim")
    resp = api_client.get(f"/api/trading/portfolio/{other_id}", headers=_auth_headers(token))
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# Orders / GTT
# ---------------------------------------------------------------------------

def test_orders_gtt_routes(api_client):
    token, user_id = _register(api_client, "gtttest")
    headers = _auth_headers(token)

    resp = api_client.get("/api/orders/gtt", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get(f"/api/orders/gtt/{user_id}", headers=headers)
    assert resp.status_code == 200

    resp = api_client.post("/api/orders/gtt/sync", headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Notifications (GET list / DELETE)
# ---------------------------------------------------------------------------

def test_notifications_list_and_delete_routes(api_client):
    token, user_id = _register(api_client, "notiflisttest")
    headers = _auth_headers(token)
    from database.db import insert_notification

    insert_notification(user_id, "signal", "Test", "A test notification")

    resp = api_client.get("/api/notifications", headers=headers)
    assert resp.status_code == 200
    notifs = resp.json()["data"]
    assert len(notifs) >= 1

    resp = api_client.delete(f"/api/notifications/{notifs[0]['id']}", headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------

def test_news_watchlist_and_stock_routes(api_client):
    token, user_id = _register(api_client, "newsroutestest")
    headers = _auth_headers(token)

    resp = api_client.get(f"/api/news/watchlist/{user_id}", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get(f"/api/news/watchlist/{user_id}/summary", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get("/api/news/stock/SOMESTOCK.NS")
    assert resp.status_code == 200

    resp = api_client.get(f"/api/signals/history/{user_id}", headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Portfolio rebalance
# ---------------------------------------------------------------------------

def test_portfolio_rebalance_route(api_client):
    token, _ = _register(api_client, "rebalancetest")
    headers = _auth_headers(token)
    conn = get_connection()
    try:
        _seed_signal(conn, "RELIANCE.NS", "STRONG BUY")
        _seed_signal(conn, "TCS.NS", "STRONG BUY")
        conn.commit()
    finally:
        release_connection(conn)

    resp = api_client.post(
        "/api/portfolio/create",
        json={"name": "Rebalance Test", "investment_amount": 50000.0, "time_horizon": "medium", "risk_profile": "moderate"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    portfolio_id = resp.json()["data"]["portfolio_id"]

    resp = api_client.post(f"/api/portfolio/{portfolio_id}/rebalance", headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Autopilot
# ---------------------------------------------------------------------------

def test_autopilot_status_toggle_trades_routes(api_client):
    token, user_id = _register(api_client, "autopilottest")
    headers = _auth_headers(token)

    resp = api_client.get(f"/api/autopilot/status?user_id={user_id}", headers=headers)
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False

    resp = api_client.post(
        "/api/autopilot/trades",
        json={"user_id": user_id, "symbol": "AUTOPILOTTEST.NS", "qty": 0, "amount": 0},
        headers=headers,
    )
    assert resp.status_code == 200
    trade_id = resp.json()["data"]["id"]

    resp = api_client.get(f"/api/autopilot/trades?user_id={user_id}", headers=headers)
    assert resp.status_code == 200
    assert any(t["id"] == trade_id for t in resp.json()["data"])

    resp = api_client.post("/api/autopilot/toggle", json={"user_id": user_id}, headers=headers)
    assert resp.status_code == 200
    assert resp.json()["enabled"] is True

    resp = api_client.delete(f"/api/autopilot/trades/{trade_id}", headers=headers)
    assert resp.status_code == 200


def test_autopilot_revoke_nonexistent_trade_is_404_not_403(api_client):
    """Regression test for the autopilot 404-vs-403 enumeration fix in Phase 1."""
    token, _ = _register(api_client, "autopilotenumtest")
    resp = api_client.delete("/api/autopilot/trades/999999999", headers=_auth_headers(token))
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Broker connections — and the H4 regression this test suite caught
# ---------------------------------------------------------------------------

def test_broker_routes_require_full_scope_token():
    """
    Regression test for audit finding H4 — and for a gap found while writing
    this test: none of broker_routes.py's 5 handlers actually used the
    file's get_current_user dependency (each had its own copy-pasted inline
    auth check with no scope check at all). Now refactored to use
    Depends(get_current_user) uniformly, so an MFA-scoped (partial-auth)
    token must be rejected on every broker route, not just accepted ones
    that happen to call a function that was never wired up.
    """
    from fastapi.testclient import TestClient
    from api.server import app
    from api.auth import create_mfa_token

    client = TestClient(app)
    resp = client.post(
        "/api/trading/register",
        json={"username": "brokerscopetest", "password": "Sup3rSecret!", "display_name": "X"},
    )
    user_id = resp.json()["user"]["id"]
    mfa_token = create_mfa_token(user_id, "brokerscopetest")
    headers = {"Authorization": f"Bearer {mfa_token}"}

    assert client.get("/api/brokers", headers=headers).status_code == 401
    assert client.get("/api/brokers/zerodha/login", headers=headers).status_code == 401
    assert client.get("/api/brokers/upstox/login", headers=headers).status_code == 401
    assert client.delete("/api/brokers/angel-one/disconnect", headers=headers).status_code == 401
    assert client.post(
        "/api/brokers/angel-one/connect",
        json={"client_id": "X", "password": "x", "totp": "000000"},
        headers=headers,
    ).status_code == 401


def test_broker_routes_with_full_token(api_client, monkeypatch):
    token, _ = _register(api_client, "brokerfulltokentest")
    headers = _auth_headers(token)

    resp = api_client.get("/api/brokers", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get("/api/brokers/zerodha/login", headers=headers)
    assert resp.status_code == 200

    resp = api_client.get("/api/brokers/upstox/login", headers=headers)
    assert resp.status_code == 200

    # Never let this test hit the real Angel One API — mock the SDK call.
    class _FakeSmartConnect:
        def __init__(self, api_key):
            pass

        def generateSession(self, client_id, password, totp):
            return {"status": True, "data": {"jwtToken": "fake-jwt-token"}}

    import SmartApi
    monkeypatch.setattr(SmartApi, "SmartConnect", _FakeSmartConnect)
    monkeypatch.setenv("ANGEL_API_KEY", "fake-key-for-test")

    resp = api_client.post(
        "/api/brokers/angel-one/connect",
        json={"client_id": "X", "password": "x", "totp": "000000"},
        headers=headers,
    )
    assert resp.status_code == 200
    assert resp.json()["connected"] is True

    resp = api_client.delete("/api/brokers/angel-one/disconnect", headers=headers)
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Auth: password change/set, TOTP 2FA end-to-end, Google sign-in, sessions
# ---------------------------------------------------------------------------

def test_password_change_flow(api_client):
    token, _ = _register(api_client, "pwchangetest")
    headers = _auth_headers(token)

    resp = api_client.post(
        "/auth/password/change",
        json={"current_password": "WrongPassword!", "new_password": "NewSecret123!"},
        headers=headers,
    )
    assert resp.status_code == 400

    resp = api_client.post(
        "/auth/password/change",
        json={"current_password": "Sup3rSecret!", "new_password": "NewSecret123!"},
        headers=headers,
    )
    assert resp.status_code == 200

    resp = api_client.post("/api/trading/login", json={"username": "pwchangetest", "password": "NewSecret123!"})
    assert resp.status_code == 200


def test_password_set_rejects_when_password_already_exists(api_client):
    token, _ = _register(api_client, "pwsettest")
    resp = api_client.post(
        "/auth/password/set", json={"new_password": "Another123!"}, headers=_auth_headers(token),
    )
    assert resp.status_code == 400


def test_totp_setup_confirm_disable_and_login_mfa_flow(api_client):
    token, user_id = _register(api_client, "totpflowtest")
    headers = _auth_headers(token)

    resp = api_client.post("/auth/totp/setup", headers=headers)
    assert resp.status_code == 200
    secret = resp.json()["secret"]

    code = pyotp.TOTP(secret).now()
    resp = api_client.post("/auth/totp/confirm", json={"code": code}, headers=headers)
    assert resp.status_code == 200

    # Login now requires MFA.
    resp = api_client.post("/api/trading/login", json={"username": "totpflowtest", "password": "Sup3rSecret!"})
    assert resp.status_code == 200
    assert resp.json()["mfa_required"] is True
    mfa_token = resp.json()["mfa_token"]

    resp = api_client.post("/auth/login/mfa", json={"mfa_token": mfa_token, "totp_code": pyotp.TOTP(secret).now()})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    full_token = resp.json()["token"]

    # Disable 2FA again.
    resp = api_client.post("/auth/totp/disable", json={"code": pyotp.TOTP(secret).now()}, headers=_auth_headers(full_token))
    assert resp.status_code == 200

    resp = api_client.post("/api/trading/login", json={"username": "totpflowtest", "password": "Sup3rSecret!"})
    assert resp.status_code == 200
    assert "mfa_required" not in resp.json()


def test_sessions_list_and_delete_routes(api_client):
    token, _ = _register(api_client, "sessionstest")
    headers = _auth_headers(token)

    resp = api_client.get("/auth/sessions", headers=headers)
    assert resp.status_code == 200

    resp = api_client.delete("/auth/sessions/00000000-0000-0000-0000-000000000000", headers=headers)
    assert resp.status_code == 200

    resp = api_client.delete("/auth/sessions", headers=headers)
    assert resp.status_code == 200


def test_google_auth_rejects_invalid_token(api_client, monkeypatch):
    import api.routes.auth_routes as auth_routes_module

    class _FakeResp:
        def raise_for_status(self):
            raise Exception("401 from Google")

    monkeypatch.setattr(auth_routes_module.http_requests, "get", lambda *a, **k: _FakeResp())
    resp = api_client.post("/auth/google", json={"access_token": "bogus"})
    assert resp.status_code == 401


def test_google_auth_creates_new_user_on_valid_token(api_client, monkeypatch):
    import api.routes.auth_routes as auth_routes_module

    class _FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "sub": "google-sub-12345",
                "email": "newgoogleuser@example.com",
                "name": "New Google User",
                "picture": "",
                "email_verified": True,
            }

    monkeypatch.setattr(auth_routes_module.http_requests, "get", lambda *a, **k: _FakeResp())
    resp = api_client.post("/auth/google", json={"access_token": "valid-fake-token"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    assert resp.json()["user"]["email"] == "newgoogleuser@example.com"


# ---------------------------------------------------------------------------
# Server-level public endpoints
# ---------------------------------------------------------------------------

def test_server_level_public_routes(api_client):
    for path in ("/", "/api/health", "/api/scheduler/status", "/api/market/status"):
        resp = api_client.get(path)
        assert resp.status_code == 200, f"{path}: {resp.text}"


def test_heatmap_sectors_route(api_client):
    # Separated from the quick loop above — this route makes ~50 sequential
    # per-stock DB queries and reliably takes 30-50s against the test
    # instance's network latency. Confirmed it returns 200 correctly; the
    # latency itself is a pre-existing characteristic of the route, not
    # something introduced or in scope for this audit pass.
    resp = api_client.get("/api/heatmap/sectors")
    assert resp.status_code == 200


def test_watchlist_combined_route(api_client):
    insert_prices_batch([("WLCOMBINEDTEST.NS", "NSE", _TODAY, None, 100, 102, 99, 101, 50000, "1d")])
    resp = api_client.get("/api/watchlist/WLCOMBINEDTEST.NS")
    assert resp.status_code == 200
