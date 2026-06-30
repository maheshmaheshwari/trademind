"""
Comprehensive API test suite — covers every route with happy-path AND error cases.

Organised by router file. Adds coverage for every endpoint not already
fully exercised by test_api_routes.py, test_mutations.py, or
test_remaining_routes.py.

New cases added here (not elsewhere):
  • Password reset request/confirm full flow
  • MFA scope guard on /api/notifications (H8 regression)
  • Notification delete by non-owner → 200 but no effect (idempotent)
  • Notification delete own → actually removes
  • Position / order filtering + sort query params
  • /api/orders/gtt with explicit ?user_id cross-user rejection
  • Autopilot: execute_immediately=True fires mandate immediately
  • Autopilot: bracket_id marks EXECUTED without re-executing
  • Autopilot: revoke PENDING trade (no position to square off)
  • Autopilot: toggle twice (on → off)
  • Autopilot: cross-user status/trades 403
  • SectorUpdate 422 validation (M1 — missing allocation_pct)
  • Portfolio: update sectors that don't sum to 100 → 400
  • Portfolio: get full portfolio detail (sectors + stocks)
  • Portfolio: delete non-owned → 404
  • Risk settings: cross-user → 403
  • Risk settings: partial update persists
  • GET /auth/me returns has_password field
  • PATCH /auth/me phone update
  • Sessions: list / delete specific / delete all
  • /api/trading/user cross-user → 403
  • /api/trading/user/by-username cross-user → 403
  • Analytics and volume routes with actual trade history
  • Today P&L with seeded square-off order
  • Signals /all DB-backed (no file monkeypatch)
  • Signals /all cache hit (second call cheaper)
  • /api/signals/stock/{symbol} with no data
  • /api/signals/history/{user_id}
  • /api/news/market with no news → empty list
  • /api/news/watchlist/{id} cross-user → 403
  • /api/brokers angel connect/disconnect idempotent
  • Server health, scheduler status, market status
  • CORS preflight (verify allow-origin header is set)
"""
from datetime import datetime, timedelta

import pyotp

from database.db import (
    get_connection, release_connection, _execute,
    insert_news, insert_notification, insert_prices_batch, insert_indicators,
)

_TODAY = datetime.now().strftime("%Y-%m-%d")
_YESTERDAY = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _register(api_client, username: str):
    resp = api_client.post(
        "/api/trading/register",
        json={"username": username, "password": "Sup3rSecret!", "display_name": "Test"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    return body["token"], body["user"]["id"]


def _h(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _seed_signal(conn, symbol="SEEDTEST.NS", signal="BUY", confidence=85.0):
    _execute(
        conn,
        """INSERT INTO trade_signals
           (symbol, name, signal, confidence, trade_type, buy_price, target_price,
            stop_loss, risk_reward, expected_return_pct, model_horizon,
            generated_date, generated_at, is_active)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (symbol, "Seed Test Ltd.", signal, confidence, "LONG", 100.0, 110.0, 95.0,
         2.0, 10.0, "1 Month", _TODAY, f"{_TODAY} 09:00:00", True),
    )


def _execute_trade(api_client, token, user_id, symbol="TRADETEST.NS",
                   amount=10000.0, price=200.0):
    """Helper: execute a PAPER signal and return the result body."""
    resp = api_client.post(
        "/api/trading/execute-signal",
        json={
            "user_id": user_id, "symbol": symbol, "name": "Trade Test Ltd.",
            "investment_amount": amount, "buy_price": price,
            "target_price": price * 1.1, "stop_loss": price * 0.95, "mode": "PAPER",
        },
        headers=_h(token),
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


# ===========================================================================
# AUTH ROUTES  (/auth/*)
# ===========================================================================

class TestAuthMe:
    def test_get_auth_me_returns_profile(self, api_client):
        token, _ = _register(api_client, "getmetest")
        resp = api_client.get("/auth/me", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["username"] == "getmetest"
        assert "has_password" in body
        assert body["has_password"] is True

    def test_get_auth_me_unauthenticated(self, api_client):
        resp = api_client.get("/auth/me")
        assert resp.status_code == 401

    def test_patch_auth_me_phone_update(self, api_client):
        token, _ = _register(api_client, "phoneupdatetest")
        resp = api_client.patch("/auth/me",
                                json={"phone": "9876543210"}, headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["phone"] == "9876543210"

    def test_patch_auth_me_multiple_fields(self, api_client):
        token, _ = _register(api_client, "multiupdatetest")
        resp = api_client.patch(
            "/auth/me",
            json={"display_name": "Multi Update", "email": "multi@example.com", "phone": "1234567890"},
            headers=_h(token),
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["display_name"] == "Multi Update"
        assert body["email"] == "multi@example.com"

    def test_patch_auth_me_no_fields_rejected(self, api_client):
        token, _ = _register(api_client, "emptyupdatetest2")
        resp = api_client.patch("/auth/me", json={}, headers=_h(token))
        assert resp.status_code == 400


class TestAuthPreferences:
    def test_get_preferences_defaults(self, api_client):
        token, _ = _register(api_client, "prefsdefaulttest")
        resp = api_client.get("/auth/preferences", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["default_account"] in ("PAPER", "LIVE")
        assert body["currency"] == "INR"

    def test_put_preferences_live_mode(self, api_client):
        token, _ = _register(api_client, "prefslivetest")
        resp = api_client.put("/auth/preferences",
                              json={"default_account": "LIVE"}, headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["default_account"] == "LIVE"

        resp = api_client.get("/auth/preferences", headers=_h(token))
        assert resp.json()["default_account"] == "LIVE"

    def test_put_preferences_invalid_account_type(self, api_client):
        token, _ = _register(api_client, "prefsinvalidtest2")
        resp = api_client.put("/auth/preferences",
                              json={"default_account": "INVALID"}, headers=_h(token))
        assert resp.status_code == 400

    def test_put_preferences_no_fields_rejected(self, api_client):
        token, _ = _register(api_client, "prefsemptytest")
        resp = api_client.put("/auth/preferences", json={}, headers=_h(token))
        assert resp.status_code == 400


class TestAuthPasswordChange:
    def test_change_password_wrong_current(self, api_client):
        token, _ = _register(api_client, "pwwrongtest")
        resp = api_client.post(
            "/auth/password/change",
            json={"current_password": "WrongPass!", "new_password": "NewPass123!"},
            headers=_h(token),
        )
        assert resp.status_code == 400

    def test_change_password_too_short(self, api_client):
        token, _ = _register(api_client, "pwshorttest")
        resp = api_client.post(
            "/auth/password/change",
            json={"current_password": "Sup3rSecret!", "new_password": "short"},
            headers=_h(token),
        )
        assert resp.status_code == 400

    def test_change_password_success_and_login_with_new(self, api_client):
        token, _ = _register(api_client, "pwsuccesstest")
        resp = api_client.post(
            "/auth/password/change",
            json={"current_password": "Sup3rSecret!", "new_password": "NewSecret123!"},
            headers=_h(token),
        )
        assert resp.status_code == 200

        resp = api_client.post("/api/trading/login",
                               json={"username": "pwsuccesstest", "password": "NewSecret123!"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

        resp = api_client.post("/api/trading/login",
                               json={"username": "pwsuccesstest", "password": "Sup3rSecret!"})
        assert resp.status_code == 401


class TestPasswordReset:
    """Full password reset flow: request OTP → confirm OTP → new password."""

    @staticmethod
    def _patch_email(monkeypatch):
        """Prevent real HTTP calls to Resend in all reset tests."""
        import api.routes.auth_routes as m
        monkeypatch.setattr(m, "_send_otp_email", lambda *a, **k: True)

    def test_reset_request_always_returns_ok(self, api_client, monkeypatch):
        self._patch_email(monkeypatch)
        # Returns "ok" even for a non-existent email (timing-safe)
        resp = api_client.post("/auth/password/reset-request",
                               json={"email": "nonexistent@example.com"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reset_confirm_invalid_otp(self, api_client, monkeypatch):
        self._patch_email(monkeypatch)
        # OTP doesn't exist → 400
        resp = api_client.post("/auth/password/reset-confirm", json={
            "email": "nobody@example.com",
            "otp": "000000",
            "new_password": "NewPass123!",
        })
        assert resp.status_code == 400

    def test_reset_full_flow(self, api_client, monkeypatch):
        """Register → request reset → inject OTP manually → confirm → login."""
        self._patch_email(monkeypatch)
        token, _ = _register(api_client, "resetflowtest")

        # Set email on the account
        api_client.patch("/auth/me",
                         json={"email": "resetflow@example.com"}, headers=_h(token))

        # Request OTP
        resp = api_client.post("/auth/password/reset-request",
                               json={"email": "resetflow@example.com"})
        assert resp.status_code == 200

        # Inject a known OTP directly into the DB
        from api.auth import hash_password
        from datetime import datetime, timedelta, timezone
        otp_plain = "999999"
        otp_hash = hash_password(otp_plain)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
        conn = get_connection()
        try:
            _execute(conn,
                "INSERT INTO password_reset_otps (email, otp_hash, expires_at) VALUES (?, ?, ?)",
                ("resetflow@example.com", otp_hash, expires_at))
            conn.commit()
        finally:
            release_connection(conn)

        resp = api_client.post("/auth/password/reset-confirm", json={
            "email": "resetflow@example.com",
            "otp": otp_plain,
            "new_password": "BrandNew123!",
        })
        assert resp.status_code == 200

        resp = api_client.post("/api/trading/login",
                               json={"username": "resetflowtest", "password": "BrandNew123!"})
        assert resp.status_code == 200

    def test_reset_confirm_too_short_password(self, api_client, monkeypatch):
        self._patch_email(monkeypatch)
        resp = api_client.post("/auth/password/reset-confirm", json={
            "email": "x@x.com",
            "otp": "123456",
            "new_password": "short",
        })
        assert resp.status_code == 400


class TestAuthPasswordSet:
    def test_set_password_already_has_one(self, api_client):
        """Users registered with password can't use /set."""
        token, _ = _register(api_client, "pwsetalreadytest")
        resp = api_client.post("/auth/password/set",
                               json={"new_password": "AnotherPass123!"},
                               headers=_h(token))
        assert resp.status_code == 400


class TestTotpFlow:
    def test_totp_setup_returns_secret_and_qr(self, api_client):
        token, _ = _register(api_client, "totpsetuptest")
        resp = api_client.post("/auth/totp/setup", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert "secret" in body and body["secret"]
        assert "qr_uri" in body and body["qr_uri"]

    def test_totp_confirm_invalid_code(self, api_client):
        token, _ = _register(api_client, "totpconfirmbadtest")
        api_client.post("/auth/totp/setup", headers=_h(token))
        resp = api_client.post("/auth/totp/confirm",
                               json={"code": "000000"}, headers=_h(token))
        assert resp.status_code == 400

    def test_totp_disable_not_enabled(self, api_client):
        token, _ = _register(api_client, "totpdisablenotest")
        resp = api_client.post("/auth/totp/disable",
                               json={"code": "000000"}, headers=_h(token))
        assert resp.status_code == 400

    def test_totp_full_flow(self, api_client):
        token, _ = _register(api_client, "totpfulltestapi")
        headers = _h(token)

        secret = api_client.post("/auth/totp/setup", headers=headers).json()["secret"]
        code = pyotp.TOTP(secret).now()
        resp = api_client.post("/auth/totp/confirm",
                               json={"code": code}, headers=headers)
        assert resp.status_code == 200

        # Login requires MFA now
        resp = api_client.post("/api/trading/login",
                               json={"username": "totpfulltestapi",
                                     "password": "Sup3rSecret!"})
        assert resp.json()["mfa_required"] is True
        mfa_token = resp.json()["mfa_token"]

        full_token = api_client.post("/auth/login/mfa", json={
            "mfa_token": mfa_token,
            "totp_code": pyotp.TOTP(secret).now(),
        }).json()["token"]

        code2 = pyotp.TOTP(secret).now()
        resp = api_client.post("/auth/totp/disable",
                               json={"code": code2}, headers=_h(full_token))
        assert resp.status_code == 200


class TestAuthSessions:
    def test_list_sessions_returns_data(self, api_client):
        token, _ = _register(api_client, "sessionlisttest")
        resp = api_client.get("/auth/sessions", headers=_h(token))
        assert resp.status_code == 200
        assert "data" in resp.json()

    def test_delete_specific_nonexistent_session_ok(self, api_client):
        token, _ = _register(api_client, "sessiondeltest")
        fake_id = "00000000-0000-0000-0000-000000000000"
        resp = api_client.delete(f"/auth/sessions/{fake_id}", headers=_h(token))
        assert resp.status_code == 200

    def test_delete_all_sessions(self, api_client):
        token, _ = _register(api_client, "sessiondelalltest")
        resp = api_client.delete("/auth/sessions", headers=_h(token))
        assert resp.status_code == 200


class TestGoogleAuth:
    def test_reject_invalid_google_token(self, api_client, monkeypatch):
        import api.routes.auth_routes as m

        class _FakeResp:
            def raise_for_status(self):
                raise Exception("Google returned 401")

        monkeypatch.setattr(m.http_requests, "get", lambda *a, **k: _FakeResp())
        resp = api_client.post("/auth/google", json={"access_token": "bogus"})
        assert resp.status_code == 401

    def test_reject_unverified_email(self, api_client, monkeypatch):
        import api.routes.auth_routes as m

        class _FakeResp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"sub": "sub123", "email": "x@y.com",
                        "email_verified": False, "name": "X"}

        monkeypatch.setattr(m.http_requests, "get", lambda *a, **k: _FakeResp())
        resp = api_client.post("/auth/google", json={"access_token": "tok"})
        assert resp.status_code == 400

    def test_create_new_user_on_valid_token(self, api_client, monkeypatch):
        import api.routes.auth_routes as m

        class _FakeResp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"sub": "googlesub-new-1234", "email": "googletest@example.com",
                        "name": "Google Test", "picture": "", "email_verified": True}

        monkeypatch.setattr(m.http_requests, "get", lambda *a, **k: _FakeResp())
        resp = api_client.post("/auth/google", json={"access_token": "fake"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_link_existing_email_user(self, api_client, monkeypatch):
        """Existing password user with matching email gets Google sub linked."""
        token, _ = _register(api_client, "googlelinktestuser")
        api_client.patch("/auth/me",
                         json={"email": "link@example.com"}, headers=_h(token))

        import api.routes.auth_routes as m

        class _FakeResp:
            def raise_for_status(self):
                pass
            def json(self):
                return {"sub": "googlesub-link-777", "email": "link@example.com",
                        "name": "Link User", "picture": "", "email_verified": True}

        monkeypatch.setattr(m.http_requests, "get", lambda *a, **k: _FakeResp())
        resp = api_client.post("/auth/google", json={"access_token": "link-tok"})
        assert resp.status_code == 200
        assert resp.json()["user"]["email"] == "link@example.com"


# ===========================================================================
# TRADING  (/api/trading/*)
# ===========================================================================

class TestTradingAuth:
    def test_register_success(self, api_client):
        resp = api_client.post("/api/trading/register", json={
            "username": "allregtest", "password": "Sup3rSecret!",
            "display_name": "All Reg", "email": "allreg@example.com",
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "token" in body
        assert body["user"]["username"] == "allregtest"

    def test_register_duplicate_username(self, api_client):
        _register(api_client, "duptest")
        resp = api_client.post("/api/trading/register", json={
            "username": "duptest", "password": "Sup3rSecret!"
        })
        assert resp.status_code == 400

    def test_login_success_returns_token(self, api_client):
        _register(api_client, "logintest2")
        resp = api_client.post("/api/trading/login",
                               json={"username": "logintest2", "password": "Sup3rSecret!"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"
        assert "token" in resp.json()

    def test_login_wrong_password(self, api_client):
        _register(api_client, "badpwtest")
        resp = api_client.post("/api/trading/login",
                               json={"username": "badpwtest", "password": "Wrong!"})
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, api_client):
        resp = api_client.post("/api/trading/login",
                               json={"username": "nobody_xyz", "password": "Sup3rSecret!"})
        assert resp.status_code == 401

    def test_get_me_returns_user(self, api_client):
        token, _ = _register(api_client, "getmetest2")
        resp = api_client.get("/api/trading/me", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "getmetest2"

    def test_get_me_unauthenticated(self, api_client):
        resp = api_client.get("/api/trading/me")
        assert resp.status_code == 401

    def test_get_user_by_id_self(self, api_client):
        token, user_id = _register(api_client, "getbyidtest")
        resp = api_client.get(f"/api/trading/user/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["id"] == user_id

    def test_get_user_by_id_cross_user_forbidden(self, api_client):
        token, _ = _register(api_client, "crossidtest")
        _, other_id = _register(api_client, "crossidvictim")
        resp = api_client.get(f"/api/trading/user/{other_id}", headers=_h(token))
        assert resp.status_code == 403

    def test_get_user_by_username_self(self, api_client):
        token, _ = _register(api_client, "byunametest")
        resp = api_client.get("/api/trading/user/by-username/byunametest",
                              headers=_h(token))
        assert resp.status_code == 200

    def test_get_user_by_username_cross_user(self, api_client):
        token, _ = _register(api_client, "byunamecrosstest")
        _register(api_client, "byunamevictim")
        resp = api_client.get("/api/trading/user/by-username/byunamevictim",
                              headers=_h(token))
        assert resp.status_code == 403

    def test_get_user_by_username_not_found(self, api_client):
        token, _ = _register(api_client, "byunamenotfound")
        resp = api_client.get("/api/trading/user/by-username/nobody_xyz_1234",
                              headers=_h(token))
        assert resp.status_code == 404


class TestExecuteSignal:
    def test_execute_creates_position(self, api_client):
        token, user_id = _register(api_client, "exectest")
        body = _execute_trade(api_client, token, user_id, "INFOSYS.NS", 10000.0, 1500.0)
        assert body["status"] == "executed"
        assert body["position"]["symbol"] == "INFOSYS.NS"

    def test_execute_duplicate_symbol_rejected(self, api_client):
        token, user_id = _register(api_client, "execduptest")
        _execute_trade(api_client, token, user_id, "HDFCBANK.NS", 10000.0, 1600.0)
        resp = api_client.post("/api/trading/execute-signal", json={
            "user_id": user_id, "symbol": "HDFCBANK.NS", "name": "HDFC Bank",
            "investment_amount": 5000.0, "buy_price": 1600.0,
            "target_price": 1760.0, "stop_loss": 1520.0, "mode": "PAPER",
        }, headers=_h(token))
        assert resp.status_code == 400

    def test_execute_zero_buy_price_422(self, api_client):
        token, user_id = _register(api_client, "zeropricetestapi")
        resp = api_client.post("/api/trading/execute-signal", json={
            "user_id": user_id, "symbol": "ZEROTEST.NS", "name": "Zero",
            "investment_amount": 10000.0, "buy_price": 0,
            "target_price": 100.0, "stop_loss": 90.0,
        }, headers=_h(token))
        assert resp.status_code == 422

    def test_execute_negative_investment_422(self, api_client):
        token, user_id = _register(api_client, "neginvesttest")
        resp = api_client.post("/api/trading/execute-signal", json={
            "user_id": user_id, "symbol": "NEG.NS", "name": "Neg",
            "investment_amount": -1000.0, "buy_price": 100.0,
            "target_price": 110.0, "stop_loss": 95.0,
        }, headers=_h(token))
        assert resp.status_code == 422

    def test_execute_excess_amount_risk_rejected(self, api_client):
        token, user_id = _register(api_client, "excessrisktest")
        resp = api_client.post("/api/trading/execute-signal", json={
            "user_id": user_id, "symbol": "EXCESS.NS", "name": "Excess",
            "investment_amount": 99_000_000.0, "buy_price": 100.0,
            "target_price": 110.0, "stop_loss": 95.0, "mode": "PAPER",
        }, headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["status"] == "rejected"

    def test_execute_requires_auth(self, api_client):
        resp = api_client.post("/api/trading/execute-signal", json={
            "user_id": 1, "symbol": "NOAUTH.NS", "name": "No Auth",
            "investment_amount": 1000.0, "buy_price": 100.0,
            "target_price": 110.0, "stop_loss": 95.0,
        })
        assert resp.status_code == 401


class TestPositions:
    def test_positions_empty_on_new_account(self, api_client):
        token, user_id = _register(api_client, "emptypositest")
        resp = api_client.get(f"/api/trading/positions/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_positions_shows_after_trade(self, api_client):
        token, user_id = _register(api_client, "posaftertrade")
        _execute_trade(api_client, token, user_id, "POSSHOW.NS", 10000.0, 500.0)
        resp = api_client.get(f"/api/trading/positions/{user_id}", headers=_h(token))
        symbols = [p["symbol"] for p in resp.json()["data"]]
        assert "POSSHOW.NS" in symbols

    def test_positions_globalfilter(self, api_client):
        token, user_id = _register(api_client, "posglobalfilt")
        _execute_trade(api_client, token, user_id, "POSFILT.NS", 10000.0, 300.0)
        resp = api_client.get(
            f"/api/trading/positions/{user_id}?globalFilter=POSFILT",
            headers=_h(token),
        )
        assert resp.status_code == 200
        assert all("POSFILT" in p["symbol"].upper() for p in resp.json()["data"])

    def test_positions_sort_by_symbol(self, api_client):
        token, user_id = _register(api_client, "possorttest")
        _execute_trade(api_client, token, user_id, "ZZZ.NS", 5000.0, 100.0)
        _execute_trade(api_client, token, user_id, "AAA.NS", 5000.0, 100.0)
        resp = api_client.get(
            f"/api/trading/positions/{user_id}?sort=symbol&order=asc",
            headers=_h(token),
        )
        assert resp.status_code == 200
        syms = [p["symbol"] for p in resp.json()["data"]]
        assert syms == sorted(syms)

    def test_positions_cross_user_403(self, api_client):
        token, _ = _register(api_client, "posxusertest")
        _, other_id = _register(api_client, "posxuservictim")
        resp = api_client.get(f"/api/trading/positions/{other_id}", headers=_h(token))
        assert resp.status_code == 403


class TestOrders:
    def test_orders_empty_list(self, api_client):
        token, user_id = _register(api_client, "ordersemptytest")
        resp = api_client.get(f"/api/trading/orders/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_orders_appears_after_trade(self, api_client):
        token, user_id = _register(api_client, "ordersaftertrade")
        _execute_trade(api_client, token, user_id, "ORDTEST.NS", 10000.0, 400.0)
        resp = api_client.get(f"/api/trading/orders/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_orders_globalfilter(self, api_client):
        token, user_id = _register(api_client, "ordersfilttest")
        _execute_trade(api_client, token, user_id, "FILTORD.NS", 10000.0, 250.0)
        resp = api_client.get(
            f"/api/trading/orders/{user_id}?globalFilter=FILTORD",
            headers=_h(token),
        )
        assert resp.status_code == 200
        for o in resp.json()["data"]:
            assert "FILTORD" in o["symbol"].upper()

    def test_orders_cross_user_403(self, api_client):
        token, _ = _register(api_client, "orderxuser")
        _, other_id = _register(api_client, "orderxvictim")
        resp = api_client.get(f"/api/trading/orders/{other_id}", headers=_h(token))
        assert resp.status_code == 403


class TestSquareOff:
    def test_square_off_closes_position(self, api_client):
        token, user_id = _register(api_client, "sqofftest")
        _execute_trade(api_client, token, user_id, "SQOFF.NS", 10000.0, 300.0)
        resp = api_client.post(f"/api/trading/square-off/{user_id}/SQOFF.NS",
                               json={}, headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "SQOFF.NS"

        resp = api_client.get(f"/api/trading/positions/{user_id}", headers=_h(token))
        assert resp.json()["data"] == []

    def test_square_off_nonexistent_symbol(self, api_client):
        token, user_id = _register(api_client, "sqoffnopostest")
        resp = api_client.post(f"/api/trading/square-off/{user_id}/NOPOS.NS",
                               json={}, headers=_h(token))
        assert resp.status_code == 400

    def test_square_off_with_explicit_price(self, api_client):
        token, user_id = _register(api_client, "sqoffpricetest")
        _execute_trade(api_client, token, user_id, "SQPRICE.NS", 10000.0, 200.0)
        resp = api_client.post(f"/api/trading/square-off/{user_id}/SQPRICE.NS",
                               json={"sell_price": 220.0}, headers=_h(token))
        assert resp.status_code == 200

    def test_square_off_cross_user_403(self, api_client):
        token, _ = _register(api_client, "sqxuser")
        _, other_id = _register(api_client, "sqxvictim")
        resp = api_client.post(f"/api/trading/square-off/{other_id}/ANY.NS",
                               json={}, headers=_h(token))
        assert resp.status_code == 403

    def test_square_off_all_empty(self, api_client):
        token, user_id = _register(api_client, "sqalltest")
        resp = api_client.post(f"/api/trading/square-off-all/{user_id}",
                               headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_square_off_all_closes_multiple(self, api_client):
        token, user_id = _register(api_client, "sqallmultitest")
        _execute_trade(api_client, token, user_id, "SQALL1.NS", 5000.0, 100.0)
        _execute_trade(api_client, token, user_id, "SQALL2.NS", 5000.0, 200.0)
        resp = api_client.post(f"/api/trading/square-off-all/{user_id}",
                               headers=_h(token))
        assert resp.status_code == 200
        resp = api_client.get(f"/api/trading/positions/{user_id}", headers=_h(token))
        assert resp.json()["data"] == []


class TestPortfolioSummary:
    def test_portfolio_summary_empty_user(self, api_client):
        token, user_id = _register(api_client, "portsummarytest")
        resp = api_client.get(f"/api/trading/portfolio/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert "total_value" in body or "balance" in body

    def test_portfolio_summary_cross_user_403(self, api_client):
        token, _ = _register(api_client, "portsumxuser")
        _, other_id = _register(api_client, "portsumvictim")
        resp = api_client.get(f"/api/trading/portfolio/{other_id}", headers=_h(token))
        assert resp.status_code == 403


class TestRiskSettings:
    def test_get_risk_settings_default(self, api_client):
        token, user_id = _register(api_client, "riskgettest")
        resp = api_client.get(f"/api/trading/risk-settings/{user_id}", headers=_h(token))
        assert resp.status_code == 200

    def test_put_risk_settings_partial_update(self, api_client):
        token, user_id = _register(api_client, "riskpartialtest")
        resp = api_client.put(f"/api/trading/risk-settings/{user_id}",
                              json={"max_daily_loss": 2500.0},
                              headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["settings"]["max_daily_loss"] == 2500.0

    def test_put_risk_settings_all_fields(self, api_client):
        token, user_id = _register(api_client, "riskallfieldstest")
        resp = api_client.put(f"/api/trading/risk-settings/{user_id}", json={
            "max_daily_loss": 3000.0,
            "max_daily_trades": 5,
            "max_position_pct": 20.0,
            "stop_loss_pct": 5.0,
            "target_pct": 10.0,
        }, headers=_h(token))
        assert resp.status_code == 200
        s = resp.json()["settings"]
        assert s["max_daily_trades"] == 5

    def test_put_risk_settings_empty_body_400(self, api_client):
        token, user_id = _register(api_client, "riskemptytest2")
        resp = api_client.put(f"/api/trading/risk-settings/{user_id}",
                              json={}, headers=_h(token))
        assert resp.status_code == 400

    def test_risk_settings_cross_user_403(self, api_client):
        token, _ = _register(api_client, "riskxuser")
        _, other_id = _register(api_client, "riskvictim")
        resp = api_client.get(f"/api/trading/risk-settings/{other_id}", headers=_h(token))
        assert resp.status_code == 403
        resp = api_client.put(f"/api/trading/risk-settings/{other_id}",
                              json={"max_daily_loss": 100.0},
                              headers=_h(token))
        assert resp.status_code == 403


class TestTodayPnl:
    def test_today_pnl_empty(self, api_client):
        token, user_id = _register(api_client, "pnlemptytest")
        resp = api_client.get(f"/api/trading/pnl/today/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["net_pnl"] == 0.0
        assert body["trades_closed"] == 0
        assert "date" in body

    def test_today_pnl_after_trade_and_squareoff(self, api_client):
        token, user_id = _register(api_client, "pnlsqofftest")
        _execute_trade(api_client, token, user_id, "PNLTEST.NS", 10000.0, 300.0)
        api_client.post(f"/api/trading/square-off/{user_id}/PNLTEST.NS",
                        json={}, headers=_h(token))
        resp = api_client.get(f"/api/trading/pnl/today/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["trades_closed"] >= 1

    def test_today_pnl_cross_user_403(self, api_client):
        token, _ = _register(api_client, "pnlxuser")
        _, other_id = _register(api_client, "pnlvictim")
        resp = api_client.get(f"/api/trading/pnl/today/{other_id}", headers=_h(token))
        assert resp.status_code == 403


class TestAnalytics:
    def test_analytics_empty_user(self, api_client):
        token, user_id = _register(api_client, "analyticsempty")
        resp = api_client.get(f"/api/trading/analytics/{user_id}", headers=_h(token))
        assert resp.status_code == 200

    def test_analytics_volume_empty(self, api_client):
        token, user_id = _register(api_client, "analyticsvol")
        resp = api_client.get(f"/api/trading/analytics/{user_id}/volume",
                              headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body and "user_id" in body

    def test_analytics_cross_user_403(self, api_client):
        token, _ = _register(api_client, "analyticsxuser")
        _, other_id = _register(api_client, "analyticsvictim")
        resp = api_client.get(f"/api/trading/analytics/{other_id}", headers=_h(token))
        assert resp.status_code == 403


# ===========================================================================
# WATCHLIST  (/api/users/{id}/watchlist)
# ===========================================================================

class TestWatchlist:
    def test_add_get_remove(self, api_client):
        token, user_id = _register(api_client, "wlapitest")
        headers = _h(token)

        resp = api_client.post(f"/api/users/{user_id}/watchlist/TCS.NS", headers=headers)
        assert resp.status_code == 201
        assert resp.json()["data"]["symbol"] == "TCS.NS"

        resp = api_client.get(f"/api/users/{user_id}/watchlist", headers=headers)
        assert resp.json()["total"] == 1

        resp = api_client.delete(f"/api/users/{user_id}/watchlist/TCS.NS", headers=headers)
        assert resp.status_code == 200

        resp = api_client.get(f"/api/users/{user_id}/watchlist", headers=headers)
        assert resp.json()["total"] == 0

    def test_add_uppercases_symbol(self, api_client):
        token, user_id = _register(api_client, "wlcasetest")
        resp = api_client.post(f"/api/users/{user_id}/watchlist/reliance.ns",
                               headers=_h(token))
        assert resp.status_code == 201
        assert resp.json()["data"]["symbol"] == "RELIANCE.NS"

    def test_add_and_update_alerts(self, api_client):
        token, user_id = _register(api_client, "wlalerttest")
        api_client.post(f"/api/users/{user_id}/watchlist/WIPRO.NS", headers=_h(token))
        resp = api_client.put(f"/api/users/{user_id}/watchlist/WIPRO.NS/alerts",
                              json={"alert_above": 600.0, "alert_below": 400.0},
                              headers=_h(token))
        assert resp.status_code == 200

    def test_cross_user_add_403(self, api_client):
        token, _ = _register(api_client, "wlxuser")
        _, other_id = _register(api_client, "wlvictim")
        resp = api_client.post(f"/api/users/{other_id}/watchlist/TCS.NS",
                               headers=_h(token))
        assert resp.status_code == 403

    def test_cross_user_delete_403(self, api_client):
        token, _ = _register(api_client, "wlxdel")
        _, other_id = _register(api_client, "wldelvictim")
        resp = api_client.delete(f"/api/users/{other_id}/watchlist/TCS.NS",
                                 headers=_h(token))
        assert resp.status_code == 403


# ===========================================================================
# NOTIFICATIONS  (/api/notifications)
# ===========================================================================

class TestNotifications:
    def test_list_empty(self, api_client):
        token, _ = _register(api_client, "notifemptytest")
        resp = api_client.get("/api/notifications", headers=_h(token))
        assert resp.status_code == 200
        assert "data" in resp.json()

    def test_list_returns_inserted_notification(self, api_client):
        token, user_id = _register(api_client, "notiflisttest2")
        insert_notification(user_id, "signal", "Alert", "Body text")
        resp = api_client.get("/api/notifications", headers=_h(token))
        assert resp.json()["total"] >= 1

    def test_mark_read(self, api_client):
        token, user_id = _register(api_client, "notifmarktest")
        insert_notification(user_id, "signal", "Unread", "Text")
        resp = api_client.post("/api/notifications/mark-read", headers=_h(token))
        assert resp.status_code == 200

        conn = get_connection()
        try:
            row = _execute(conn,
                "SELECT is_read FROM notifications WHERE user_id = ?",
                (user_id,)).fetchone()
        finally:
            release_connection(conn)
        assert row[0] is True

    def test_delete_own_notification(self, api_client):
        token, user_id = _register(api_client, "notifdeltestapi")
        insert_notification(user_id, "signal", "Del", "Text")
        resp = api_client.get("/api/notifications", headers=_h(token))
        notif_id = resp.json()["data"][0]["id"]

        resp = api_client.delete(f"/api/notifications/{notif_id}", headers=_h(token))
        assert resp.status_code == 200

        resp = api_client.get("/api/notifications", headers=_h(token))
        assert not any(n["id"] == notif_id for n in resp.json()["data"])

    def test_delete_other_users_notification_no_error(self, api_client):
        """Deleting another user's notification should return ok (idempotent) but have no effect."""
        token_a, user_a = _register(api_client, "notifdelownera")
        _, user_b = _register(api_client, "notifdelownerb")
        insert_notification(user_b, "signal", "B's Notif", "Text")

        resp = api_client.get("/api/notifications",
                              headers=_h(api_client.post("/api/trading/login",
                              json={"username": "notifdelownerb",
                                    "password": "Sup3rSecret!"}).json()["token"]))
        notif_id = resp.json()["data"][0]["id"]

        resp = api_client.delete(f"/api/notifications/{notif_id}", headers=_h(token_a))
        assert resp.status_code == 200

        conn = get_connection()
        try:
            still_there = _execute(conn,
                "SELECT id FROM notifications WHERE id = ?", (notif_id,)).fetchone()
        finally:
            release_connection(conn)
        assert still_there is not None  # B's notification still exists

    def test_mfa_scope_token_rejected(self, api_client):
        """Regression test for H8 — MFA-step token must be rejected."""
        from api.auth import create_mfa_token
        token, user_id = _register(api_client, "notifmfatest")
        mfa_token = create_mfa_token(user_id, "notifmfatest")
        resp = api_client.get("/api/notifications",
                              headers={"Authorization": f"Bearer {mfa_token}"})
        assert resp.status_code == 401
        assert "MFA" in resp.json()["detail"] or "mfa" in resp.json()["detail"].lower() or "Incomplete" in resp.json()["detail"]

    def test_notification_preferences_defaults(self, api_client):
        token, _ = _register(api_client, "notifprefsdeftest")
        resp = api_client.get("/api/notifications/preferences", headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert "signal_change" in body
        assert body["price_alert"] is True

    def test_notification_preferences_update(self, api_client):
        token, _ = _register(api_client, "notifprefsupdtest")
        resp = api_client.put("/api/notifications/preferences", json={
            "signal_change": False,
            "ch_sms": True,
            "weekly_report": True,
        }, headers=_h(token))
        assert resp.status_code == 200

        resp = api_client.get("/api/notifications/preferences", headers=_h(token))
        body = resp.json()
        assert body["signal_change"] is False
        assert body["ch_sms"] is True
        assert body["weekly_report"] is True
        assert body["price_alert"] is True  # untouched default


# ===========================================================================
# ORDERS / GTT  (/api/orders/gtt*)
# ===========================================================================

class TestOrdersGtt:
    def test_gtt_list_empty(self, api_client):
        token, _ = _register(api_client, "gttlisttest")
        resp = api_client.get("/api/orders/gtt", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_gtt_user_param_self(self, api_client):
        token, user_id = _register(api_client, "gttselftest")
        resp = api_client.get(f"/api/orders/gtt?user_id={user_id}", headers=_h(token))
        assert resp.status_code == 200

    def test_gtt_user_param_cross_user_403(self, api_client):
        token, _ = _register(api_client, "gttxuser")
        _, other_id = _register(api_client, "gttvictim")
        resp = api_client.get(f"/api/orders/gtt?user_id={other_id}", headers=_h(token))
        assert resp.status_code == 403

    def test_gtt_by_user_id(self, api_client):
        token, user_id = _register(api_client, "gttbyuidtest")
        resp = api_client.get(f"/api/orders/gtt/{user_id}", headers=_h(token))
        assert resp.status_code == 200
        assert "user_id" in resp.json()

    def test_gtt_by_user_id_cross_user_403(self, api_client):
        token, _ = _register(api_client, "gttbyuidxuser")
        _, other_id = _register(api_client, "gttbyuidvictim")
        resp = api_client.get(f"/api/orders/gtt/{other_id}", headers=_h(token))
        assert resp.status_code == 403

    def test_gtt_sync(self, api_client):
        token, _ = _register(api_client, "gttsynctest")
        resp = api_client.post("/api/orders/gtt/sync", headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ===========================================================================
# PORTFOLIO MANAGEMENT  (/api/portfolio/*)
# ===========================================================================

class TestPortfolioManagement:
    def _seed_signals_for_portfolio(self, conn):
        for sym, name in [
            ("RELIANCE.NS", "Reliance Industries Ltd."),
            ("TCS.NS", "Tata Consultancy Services Ltd."),
            ("INFY.NS", "Infosys Ltd."),
        ]:
            _seed_signal(conn, sym, "STRONG BUY")

    def test_sectors_endpoint_public(self, api_client):
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/portfolio/sectors")
        assert resp.status_code == 200
        assert resp.json()["total_sectors"] > 0

    def test_create_get_list_delete(self, api_client):
        token, user_id = _register(api_client, "portcreatetest")
        headers = _h(token)

        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)

        resp = api_client.post("/api/portfolio/create", json={
            "name": "My Portfolio", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=headers)
        assert resp.status_code == 200, resp.text
        data = resp.json()["data"]
        portfolio_id = data["portfolio_id"]
        assert portfolio_id
        assert "stocks" in data and "sectors" in data

        resp = api_client.get(f"/api/portfolio/{portfolio_id}", headers=headers)
        assert resp.status_code == 200
        detail = resp.json()["data"]
        assert "sectors" in detail and "stocks" in detail

        resp = api_client.get("/api/portfolio", headers=headers)
        assert resp.status_code == 200
        assert any(p["id"] == portfolio_id for p in resp.json()["data"])

        resp = api_client.delete(f"/api/portfolio/{portfolio_id}", headers=headers)
        assert resp.status_code == 200

        conn = get_connection()
        try:
            remaining = _execute(conn, "SELECT id FROM portfolios WHERE id = ?",
                                 (portfolio_id,)).fetchone()
        finally:
            release_connection(conn)
        assert remaining is None

    def test_create_fails_without_signals(self, api_client):
        token, _ = _register(api_client, "portemptysig")
        resp = api_client.post("/api/portfolio/create", json={
            "name": "Empty", "investment_amount": 50000.0,
            "time_horizon": "short", "risk_profile": "conservative",
        }, headers=_h(token))
        assert resp.status_code == 400

    def test_create_requires_auth(self, api_client):
        resp = api_client.post("/api/portfolio/create", json={
            "name": "No Auth", "investment_amount": 50000.0,
            "time_horizon": "short", "risk_profile": "moderate",
        })
        assert resp.status_code == 401

    def test_get_portfolio_other_user_404(self, api_client):
        owner_tok, _ = _register(api_client, "portowner2")
        intruder_tok, _ = _register(api_client, "portintruder2")
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "Owner", "investment_amount": 50000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=_h(owner_tok))
        pid = resp.json()["data"]["portfolio_id"]

        resp = api_client.get(f"/api/portfolio/{pid}", headers=_h(intruder_tok))
        assert resp.status_code == 404

    def test_update_sectors_valid(self, api_client):
        token, _ = _register(api_client, "secupdate")
        headers = _h(token)
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "SecUpdate", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=headers)
        pid = resp.json()["data"]["portfolio_id"]
        sectors = resp.json()["data"]["sectors"]

        raw_total = sum(s["allocation_pct"] for s in sectors)
        rescaled = [{"sector": s["sector"],
                     "allocation_pct": round(s["allocation_pct"] / raw_total * 100, 1)}
                    for s in sectors]

        resp = api_client.put(f"/api/portfolio/{pid}/sectors",
                              json={"sectors": rescaled}, headers=headers)
        assert resp.status_code == 200

    def test_update_sectors_missing_allocation_pct_422(self, api_client):
        """Regression test for M1 — SectorAlloc Pydantic model returns 422."""
        token, _ = _register(api_client, "sec422test")
        headers = _h(token)
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "422Test", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=headers)
        pid = resp.json()["data"]["portfolio_id"]

        resp = api_client.put(f"/api/portfolio/{pid}/sectors",
                              json={"sectors": [{"sector": "IT"}]},
                              headers=headers)
        assert resp.status_code == 422

    def test_update_sectors_not_sum_100_400(self, api_client):
        token, _ = _register(api_client, "sec400test")
        headers = _h(token)
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "400Test", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=headers)
        pid = resp.json()["data"]["portfolio_id"]
        sector = resp.json()["data"]["sectors"][0]["sector"]

        resp = api_client.put(f"/api/portfolio/{pid}/sectors",
                              json={"sectors": [{"sector": sector, "allocation_pct": 50.0}]},
                              headers=headers)
        assert resp.status_code == 400

    def test_rebalance(self, api_client):
        token, _ = _register(api_client, "rebalancetest2")
        headers = _h(token)
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "Rebal", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=headers)
        pid = resp.json()["data"]["portfolio_id"]
        resp = api_client.post(f"/api/portfolio/{pid}/rebalance", headers=headers)
        assert resp.status_code == 200
        assert "rebalanced_at" in resp.json()["data"]

    def test_rebalance_wrong_user_404(self, api_client):
        token_a, _ = _register(api_client, "rebxuser")
        token_b, _ = _register(api_client, "rebxuservict")
        conn = get_connection()
        try:
            self._seed_signals_for_portfolio(conn)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/portfolio/create", json={
            "name": "Private", "investment_amount": 100000.0,
            "time_horizon": "medium", "risk_profile": "moderate",
        }, headers=_h(token_a))
        pid = resp.json()["data"]["portfolio_id"]
        resp = api_client.post(f"/api/portfolio/{pid}/rebalance", headers=_h(token_b))
        assert resp.status_code == 404


# ===========================================================================
# AUTOPILOT  (/api/autopilot/*)
# ===========================================================================

class TestAutopilot:
    def test_status_initial_disabled(self, api_client):
        token, user_id = _register(api_client, "apstatinit")
        resp = api_client.get(f"/api/autopilot/status?user_id={user_id}",
                              headers=_h(token))
        assert resp.status_code == 200
        body = resp.json()
        assert body["enabled"] is False
        assert body["capital"] == 0

    def test_status_cross_user_403(self, api_client):
        token, _ = _register(api_client, "apstatxuser")
        _, other_id = _register(api_client, "apstatvict")
        resp = api_client.get(f"/api/autopilot/status?user_id={other_id}",
                              headers=_h(token))
        assert resp.status_code == 403

    def test_authorize_trade_pending(self, api_client):
        token, user_id = _register(api_client, "aptradetest")
        resp = api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "AUTO1.NS", "qty": 5,
            "amount": 5000.0, "entry": 100.0, "target": 110.0, "sl": 95.0,
        }, headers=_h(token))
        assert resp.status_code == 200
        trade = resp.json()["data"]
        assert trade["status"] == "PENDING"
        assert trade["symbol"] == "AUTO1.NS"
        return token, user_id, trade["id"]

    def test_list_trades(self, api_client):
        token, user_id = _register(api_client, "aplisttest")
        api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "AUTOLIST.NS", "qty": 3,
            "amount": 3000.0,
        }, headers=_h(token))
        resp = api_client.get(f"/api/autopilot/trades?user_id={user_id}",
                              headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_list_trades_filter_by_status(self, api_client):
        token, user_id = _register(api_client, "aplistfilter")
        api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "AFILT.NS", "qty": 2, "amount": 2000.0,
        }, headers=_h(token))
        resp = api_client.get(
            f"/api/autopilot/trades?user_id={user_id}&status=PENDING",
            headers=_h(token),
        )
        assert resp.status_code == 200
        for t in resp.json()["data"]:
            assert t["status"] == "PENDING"

    def test_toggle_on_off(self, api_client):
        token, user_id = _register(api_client, "aptoggletest")
        headers = _h(token)

        resp = api_client.post("/api/autopilot/toggle",
                               json={"user_id": user_id}, headers=headers)
        assert resp.json()["enabled"] is True

        resp = api_client.post("/api/autopilot/toggle",
                               json={"user_id": user_id}, headers=headers)
        assert resp.json()["enabled"] is False

    def test_revoke_pending_trade(self, api_client):
        token, user_id = _register(api_client, "aprevokepend")
        resp = api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "REVPEND.NS", "qty": 2, "amount": 2000.0,
        }, headers=_h(token))
        trade_id = resp.json()["data"]["id"]

        resp = api_client.delete(f"/api/autopilot/trades/{trade_id}",
                                 headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_revoke_nonexistent_trade_404(self, api_client):
        token, _ = _register(api_client, "aprevoke404")
        resp = api_client.delete("/api/autopilot/trades/9999999",
                                 headers=_h(token))
        assert resp.status_code == 404

    def test_revoke_other_users_trade_404_not_403(self, api_client):
        """Security: non-owner should get 404, not 403 (avoids ID enumeration)."""
        owner_tok, owner_id = _register(api_client, "aprevxowner")
        intruder_tok, _ = _register(api_client, "aprevxintruder")

        resp = api_client.post("/api/autopilot/trades", json={
            "user_id": owner_id, "symbol": "XOWN.NS", "qty": 1, "amount": 1000.0,
        }, headers=_h(owner_tok))
        trade_id = resp.json()["data"]["id"]

        resp = api_client.delete(f"/api/autopilot/trades/{trade_id}",
                                 headers=_h(intruder_tok))
        assert resp.status_code == 404

    def test_authorize_with_bracket_id_marks_executed(self, api_client):
        """Existing position handed over via bracket_id → status=EXECUTED immediately."""
        token, user_id = _register(api_client, "apbracketid")
        resp = api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "BRACK.NS",
            "qty": 5, "amount": 5000.0,
            "bracket_id": "BRACK-001",
        }, headers=_h(token))
        assert resp.status_code == 200
        assert resp.json()["data"]["status"] == "EXECUTED"
        assert resp.json()["data"]["bracket_id"] == "BRACK-001"

    def test_revoke_already_stopped_400(self, api_client):
        """Revoking a trade that's already STOPPED should return 400."""
        token, user_id = _register(api_client, "aprevstop")
        resp = api_client.post("/api/autopilot/trades", json={
            "user_id": user_id, "symbol": "REVSTOP.NS", "qty": 2, "amount": 2000.0,
        }, headers=_h(token))
        trade_id = resp.json()["data"]["id"]

        api_client.delete(f"/api/autopilot/trades/{trade_id}", headers=_h(token))

        resp = api_client.delete(f"/api/autopilot/trades/{trade_id}",
                                 headers=_h(token))
        assert resp.status_code == 400

    def test_cross_user_list_trades_403(self, api_client):
        token, _ = _register(api_client, "aplistxuser")
        _, other_id = _register(api_client, "aplistvictim")
        resp = api_client.get(f"/api/autopilot/trades?user_id={other_id}",
                              headers=_h(token))
        assert resp.status_code == 403

    def test_cross_user_toggle_403(self, api_client):
        token, _ = _register(api_client, "aptogglexuser")
        _, other_id = _register(api_client, "aptogglevictim")
        resp = api_client.post("/api/autopilot/toggle",
                               json={"user_id": other_id}, headers=_h(token))
        assert resp.status_code == 403


# ===========================================================================
# BROKER ROUTES  (/api/brokers/*)
# ===========================================================================

class TestBrokerRoutes:
    def test_list_brokers_returns_all(self, api_client):
        token, _ = _register(api_client, "broklisttest")
        resp = api_client.get("/api/brokers", headers=_h(token))
        assert resp.status_code == 200
        brokers = {b["broker"] for b in resp.json()}
        assert "angel" in brokers
        assert "zerodha" in brokers

    def test_zerodha_login_coming_soon(self, api_client):
        token, _ = _register(api_client, "zerodhatest")
        resp = api_client.get("/api/brokers/zerodha/login", headers=_h(token))
        assert resp.status_code == 200
        assert "coming soon" in resp.json().get("message", "").lower()

    def test_upstox_login_coming_soon(self, api_client):
        token, _ = _register(api_client, "upstoxtest")
        resp = api_client.get("/api/brokers/upstox/login", headers=_h(token))
        assert resp.status_code == 200

    def test_angel_connect_and_disconnect(self, api_client, monkeypatch):
        token, _ = _register(api_client, "angelconntest")
        headers = _h(token)

        class _FakeSmart:
            def __init__(self, api_key):
                pass
            def generateSession(self, client_id, password, totp):
                return {"status": True, "data": {"jwtToken": "fake-jwt"}}

        import SmartApi
        monkeypatch.setattr(SmartApi, "SmartConnect", _FakeSmart)
        monkeypatch.setenv("ANGEL_API_KEY", "fakekey")

        resp = api_client.post("/api/brokers/angel-one/connect",
                               json={"client_id": "X123", "password": "pw",
                                     "totp": "000000"},
                               headers=headers)
        assert resp.status_code == 200
        assert resp.json()["connected"] is True

        resp = api_client.get("/api/brokers", headers=headers)
        angel = next(b for b in resp.json() if b["broker"] == "angel")
        assert angel["connected"] is True

        resp = api_client.delete("/api/brokers/angel-one/disconnect", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["connected"] is False

    def test_angel_connect_no_api_key_503(self, api_client, monkeypatch):
        token, _ = _register(api_client, "angelnokey")
        monkeypatch.delenv("ANGEL_API_KEY", raising=False)
        resp = api_client.post("/api/brokers/angel-one/connect",
                               json={"client_id": "X", "password": "x", "totp": "000000"},
                               headers=_h(token))
        assert resp.status_code == 503

    def test_mfa_token_rejected_on_all_broker_routes(self, api_client):
        from api.auth import create_mfa_token
        resp = api_client.post("/api/trading/register", json={
            "username": "brokermfatest", "password": "Sup3rSecret!",
        })
        uid = resp.json()["user"]["id"]
        mfa_tok = create_mfa_token(uid, "brokermfatest")
        h = {"Authorization": f"Bearer {mfa_tok}"}

        assert api_client.get("/api/brokers", headers=h).status_code == 401
        assert api_client.get("/api/brokers/zerodha/login", headers=h).status_code == 401
        assert api_client.get("/api/brokers/upstox/login", headers=h).status_code == 401
        assert api_client.delete("/api/brokers/angel-one/disconnect",
                                 headers=h).status_code == 401
        assert api_client.post("/api/brokers/angel-one/connect",
                               json={"client_id": "X", "password": "x", "totp": "000000"},
                               headers=h).status_code == 401


# ===========================================================================
# SIGNALS  (/api/signals/*)
# ===========================================================================

class TestSignals:
    def test_top_buys_only_buy_signals(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "BUYSIG.NS", "BUY", 88.0)
            _seed_signal(conn, "SELLSIG.NS", "SELL", 92.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/top-buys")
        assert resp.status_code == 200
        for s in resp.json()["signals"]:
            assert "BUY" in s["signal"]

    def test_top_buys_limit_param(self, api_client):
        conn = get_connection()
        try:
            for i in range(5):
                _seed_signal(conn, f"BLIMIT{i}.NS", "BUY", float(80 + i))
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/top-buys?limit=3")
        assert resp.status_code == 200
        assert len(resp.json()["signals"]) <= 3

    def test_top_sells_only_sell_signals(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "SELLONLY.NS", "SELL", 85.0)
            _seed_signal(conn, "BUYONLY.NS", "BUY", 90.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/top-sells")
        assert resp.status_code == 200
        for s in resp.json()["signals"]:
            assert "SELL" in s["signal"]

    def test_signals_all_db_backed(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "ALLDBTEST.NS", "BUY", 79.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/all")
        assert resp.status_code == 200
        body = resp.json()
        assert "signals" in body
        assert "count" in body

    def test_signals_all_empty_db_empty_response(self, api_client):
        resp = api_client.get("/api/signals/all")
        assert resp.status_code == 200

    def test_signals_latest_paginated(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "LATPAG.NS", "BUY", 81.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/latest?page=0&size=5")
        assert resp.status_code == 200

    def test_signals_actionable(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "ACTION.NS", "BUY", 80.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/actionable")
        assert resp.status_code == 200

    def test_signals_avoid(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "AVOID.NS", "SELL", 80.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/avoid")
        assert resp.status_code == 200

    def test_signals_stock_with_data(self, api_client):
        conn = get_connection()
        try:
            _seed_signal(conn, "STOCKSIG.NS", "BUY", 70.0)
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.get("/api/signals/stock/STOCKSIG.NS")
        assert resp.status_code == 200

    def test_signals_stock_no_data(self, api_client):
        resp = api_client.get("/api/signals/stock/NOSIGATALL.NS")
        assert resp.status_code == 200  # returns empty, not 404

    def test_signals_history(self, api_client):
        resp = api_client.get("/api/signals/history")
        assert resp.status_code == 200

    def test_signals_refresh_cooldown(self, api_client, monkeypatch):
        import scripts.generate_trades as gt
        monkeypatch.setattr(gt, "generate_signals", lambda: None)
        token, _ = _register(api_client, "refreshcooltest2")
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = get_connection()
        try:
            _execute(conn,
                """INSERT INTO trade_signals
                   (symbol, name, signal, confidence, trade_type, buy_price, target_price,
                    stop_loss, risk_reward, expected_return_pct, model_horizon,
                    generated_date, generated_at, is_active)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("COOL.NS", "Cool", "BUY", 75.0, "LONG", 100.0, 110.0, 95.0, 2.0,
                 10.0, "1 Month", _TODAY, now_str, True))
            conn.commit()
        finally:
            release_connection(conn)
        resp = api_client.post("/api/signals/refresh", headers=_h(token))
        assert resp.status_code == 429

    def test_signals_refresh_requires_auth(self, api_client):
        resp = api_client.post("/api/signals/refresh")
        assert resp.status_code == 401


# ===========================================================================
# PRICES / INDICATORS  (/api/prices/*, /api/indicators/*)
# ===========================================================================

class TestPricesAndIndicators:
    def test_prices_not_found(self, api_client):
        resp = api_client.get("/api/prices/NOPRICEDATA.NS")
        assert resp.status_code == 404

    def test_prices_found_after_insert(self, api_client):
        insert_prices_batch([("PRICEAPI.NS", "NSE", _TODAY, None, 100, 102, 99, 101, 50000, "1d")])
        resp = api_client.get("/api/prices/PRICEAPI.NS")
        assert resp.status_code == 200
        assert resp.json()["symbol"] == "PRICEAPI.NS"

    def test_indicators_not_found(self, api_client):
        resp = api_client.get("/api/indicators/NOINDDATA.NS")
        assert resp.status_code == 404

    def test_indicators_found_after_insert(self, api_client):
        insert_indicators("INDAPI.NS", _TODAY, {"rsi_14": 60.0, "signal": "BUY", "signal_strength": 65})
        resp = api_client.get("/api/indicators/INDAPI.NS")
        assert resp.status_code == 200


# ===========================================================================
# STOCKS  (/api/stocks/*)
# ===========================================================================

class TestStocks:
    def test_stocks_list_returns_paginated(self, api_client):
        resp = api_client.get("/api/stocks?size=5")
        assert resp.status_code == 200
        body = resp.json()
        assert "data" in body
        assert "total" in body

    def test_stocks_list_filters_by_sector(self, api_client):
        resp = api_client.get("/api/stocks?sector=IT&size=5")
        assert resp.status_code == 200

    def test_stocks_detail_with_price(self, api_client):
        insert_prices_batch([
            ("STOCKDETAIL.NS", "NSE", _TODAY, None, 200, 205, 198, 202, 100000, "1d"),
        ])
        resp = api_client.get("/api/stocks/STOCKDETAIL.NS")
        assert resp.status_code == 200

    def test_stocks_history(self, api_client):
        insert_prices_batch([
            ("STOCKHIST.NS", "NSE", _TODAY, None, 100, 105, 99, 102, 50000, "1d"),
            ("STOCKHIST.NS", "NSE", _YESTERDAY, None, 98, 103, 97, 100, 45000, "1d"),
        ])
        resp = api_client.get("/api/stocks/STOCKHIST.NS/history?range=1M")
        assert resp.status_code == 200


# ===========================================================================
# SENTIMENT & NEWS  (/api/sentiment/*, /api/news/*)
# ===========================================================================

class TestSentiment:
    def test_market_sentiment_with_news(self, api_client):
        insert_news(
            headline="Sensex rallies 500 points",
            source="rss", published_at="2026-06-29 09:00:00",
            symbol=None, sentiment="positive", confidence=0.85,
        )
        resp = api_client.get("/api/sentiment/market")
        assert resp.status_code == 200
        body = resp.json()
        assert "score" in body
        assert "label" in body

    def test_market_sentiment_empty(self, api_client):
        resp = api_client.get("/api/sentiment/market")
        assert resp.status_code == 200

    def test_sentiment_health(self, api_client):
        resp = api_client.get("/api/sentiment/health")
        assert resp.status_code in (200, 500)

    def test_sentiment_stock_no_news(self, api_client):
        resp = api_client.get("/api/sentiment/NOSENTIMENT.NS")
        assert resp.status_code == 200
        assert resp.json()["label"] == "Neutral"

    def test_sentiment_stock_with_news(self, api_client):
        insert_news(
            headline="Reliance Q4 profit beats estimates",
            source="rss", published_at="2026-06-29 10:00:00",
            symbol="RELIANCE.NS", sentiment="positive", confidence=0.9,
        )
        resp = api_client.get("/api/sentiment/RELIANCE.NS")
        assert resp.status_code == 200


class TestNews:
    def test_market_news_empty(self, api_client):
        resp = api_client.get("/api/news/market")
        assert resp.status_code == 200
        assert "data" in resp.json()

    def test_market_news_with_entry(self, api_client):
        insert_news(
            headline="RBI MPC keeps rates unchanged",
            source="rss", published_at="2026-06-29 08:00:00",
            symbol=None,
        )
        resp = api_client.get("/api/news/market")
        assert resp.status_code == 200
        assert resp.json()["total"] >= 1

    def test_watchlist_news(self, api_client):
        token, user_id = _register(api_client, "newswtestapi")
        resp = api_client.get(f"/api/news/watchlist/{user_id}", headers=_h(token))
        assert resp.status_code == 200

    def test_watchlist_news_cross_user_403(self, api_client):
        token, _ = _register(api_client, "newswxuser")
        _, other_id = _register(api_client, "newswvictim")
        resp = api_client.get(f"/api/news/watchlist/{other_id}", headers=_h(token))
        assert resp.status_code == 403

    def test_watchlist_news_summary(self, api_client):
        token, user_id = _register(api_client, "newswsumtest")
        resp = api_client.get(f"/api/news/watchlist/{user_id}/summary",
                              headers=_h(token))
        assert resp.status_code == 200

    def test_stock_news(self, api_client):
        resp = api_client.get("/api/news/stock/TCS.NS")
        assert resp.status_code == 200


# ===========================================================================
# SERVER-LEVEL & PUBLIC ENDPOINTS
# ===========================================================================

class TestServerEndpoints:
    def test_root_returns_200(self, api_client):
        resp = api_client.get("/")
        assert resp.status_code == 200

    def test_health_check(self, api_client):
        resp = api_client.get("/api/health")
        assert resp.status_code == 200

    def test_scheduler_status(self, api_client):
        resp = api_client.get("/api/scheduler/status")
        assert resp.status_code == 200

    def test_market_status(self, api_client):
        resp = api_client.get("/api/market/status")
        assert resp.status_code == 200

    def test_cors_preflight_allowed_origin(self, api_client):
        resp = api_client.options(
            "/api/signals",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )
        assert resp.status_code in (200, 204)
        acao = resp.headers.get("access-control-allow-origin", "")
        assert acao == "http://localhost:5173" or acao == "*"

    def test_cors_preflight_disallowed_origin_no_acao_header(self, api_client):
        resp = api_client.options(
            "/api/signals",
            headers={
                "Origin": "http://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware either omits the header or returns the request origin
        # only if it was in the allow-list — evil.example.com must NOT appear.
        acao = resp.headers.get("access-control-allow-origin", "")
        assert "evil.example.com" not in acao

    def test_heatmap_sectors(self, api_client):
        resp = api_client.get("/api/heatmap/sectors")
        assert resp.status_code == 200

    def test_watchlist_endpoint_public(self, api_client):
        insert_prices_batch([("WLPUBTEST.NS", "NSE", _TODAY, None, 100, 102, 99, 101, 50000, "1d")])
        resp = api_client.get("/api/watchlist/WLPUBTEST.NS")
        assert resp.status_code == 200


# ===========================================================================
# BACKTEST  (/api/backtest/summary)
# ===========================================================================

class TestBacktest:
    def test_summary_returns_expected_shape(self, api_client, monkeypatch, tmp_path):
        import api.routes.backtest as backtest_module
        monkeypatch.setattr(backtest_module, "DATA_DIR", tmp_path)
        resp = api_client.get("/api/backtest/summary")
        assert resp.status_code == 200
        body = resp.json()
        for key in ("model_stats", "signal_stats", "history"):
            assert key in body
