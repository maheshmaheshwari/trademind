"""
TradeMind AI — Extended Auth Routes

GET    /auth/me
PATCH  /auth/me
POST   /auth/google
POST   /auth/password/change
POST   /auth/password/reset-request
POST   /auth/password/reset-confirm
GET    /auth/preferences
PUT    /auth/preferences
POST   /auth/totp/setup
POST   /auth/totp/confirm
POST   /auth/totp/disable
GET    /auth/sessions
DELETE /auth/sessions/{session_id}
DELETE /auth/sessions
"""

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional

import pyotp
import requests as http_requests
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

from api.auth import decode_token, create_token, hash_password, verify_password
from api.rate_limit import limiter
from database.db import _execute, _row_to_dict, _rows_to_dicts, get_connection, release_connection
from trading.trading_engine import get_user

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Auth"])


# ---------------------------------------------------------------------------
# JWT dependency (shared pattern)
# ---------------------------------------------------------------------------

async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("scope") != "full":
        raise HTTPException(status_code=401, detail="Incomplete authentication — please complete MFA")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


class ResetRequestBody(BaseModel):
    email: str


class ResetConfirmBody(BaseModel):
    email: str
    otp: str
    new_password: str


class PreferencesRequest(BaseModel):
    default_account: Optional[str] = None
    currency: Optional[str] = None


class TotpConfirmRequest(BaseModel):
    code: str


class TotpDisableRequest(BaseModel):
    code: str


class GoogleAuthRequest(BaseModel):
    access_token: str


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fetch_user_full(user_id: int) -> Optional[dict]:
    """Fetch extended user row including new columns."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT id, username, display_name, email, phone, avatar_url,
                   totp_enabled, default_account, currency, virtual_balance,
                   password_hash
            FROM users
            WHERE id = ?
        """, (user_id,))
        row = _row_to_dict(cur)
        if row:
            # Expose whether the user has a password without leaking the hash
            row["has_password"] = bool(row.pop("password_hash", ""))
        return row
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

@router.get("/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return current user's profile."""
    profile = _fetch_user_full(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile


# ---------------------------------------------------------------------------
# POST /auth/google
# ---------------------------------------------------------------------------

@router.post("/auth/google")
async def google_auth(req: GoogleAuthRequest):
    """
    Sign in or register via Google OAuth.
    Accepts the access_token from Google Identity Services (implicit flow),
    fetches the user profile from Google's userinfo endpoint, then:
      - existing google_sub  → login
      - matching email       → link accounts + login
      - new user             → auto-register + login
    Returns the same {status, user, token} shape as POST /login.
    """
    # Fetch user info from Google
    try:
        resp = http_requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {req.access_token}"},
            timeout=10,
        )
        resp.raise_for_status()
        idinfo = resp.json()
    except Exception as e:
        logger.warning(f"Google userinfo fetch failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid Google access token")

    google_sub     = idinfo.get("sub", "")
    email          = idinfo.get("email", "")
    name           = idinfo.get("name", "") or idinfo.get("given_name", "")
    avatar_url     = idinfo.get("picture", "")
    email_verified = idinfo.get("email_verified", False)

    if not google_sub:
        raise HTTPException(status_code=401, detail="Invalid Google token: missing sub")
    if not email_verified:
        raise HTTPException(status_code=400, detail="Google account email is not verified")

    conn = get_connection()
    try:
        # 1. Returning Google user
        cur = _execute(conn, "SELECT * FROM users WHERE google_sub = ?", (google_sub,))
        user = _row_to_dict(cur)

        if not user and email:
            # 2. Existing password user with same email — link accounts
            cur = _execute(conn, "SELECT * FROM users WHERE email = ?", (email,))
            user = _row_to_dict(cur)
            if user:
                _execute(conn,
                    "UPDATE users SET google_sub = ?, avatar_url = ? WHERE id = ?",
                    (google_sub, avatar_url, user["id"]),
                )
                conn.commit()

        if not user:
            # 3. New user — auto-register
            base_username = email.split("@")[0][:28].lower()
            username = base_username
            counter = 1
            while True:
                cur = _execute(conn, "SELECT id FROM users WHERE username = ?", (username,))
                if not _row_to_dict(cur):
                    break
                username = f"{base_username}_{counter}"
                counter += 1

            _execute(conn, """
                INSERT INTO users
                  (username, display_name, email, password_hash, google_sub, avatar_url, virtual_balance)
                VALUES (?, ?, ?, '', ?, ?, 1000000)
            """, (username, name or base_username, email, google_sub, avatar_url))
            conn.commit()

            cur = _execute(conn, "SELECT * FROM users WHERE google_sub = ?", (google_sub,))
            user = _row_to_dict(cur)

        if not user:
            raise HTTPException(status_code=500, detail="Failed to create or find user")

        from trading.trading_engine import _safe_user
        token = create_token(user["id"], user["username"])
        return {"status": "success", "user": _safe_user(user), "token": token}

    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# PATCH /auth/me
# ---------------------------------------------------------------------------

@router.patch("/auth/me")
async def update_me(req: UpdateProfileRequest, user: dict = Depends(get_current_user)):
    """Update display_name, email, or phone."""
    _ALLOWED_PROFILE_FIELDS = {"display_name", "email", "phone"}
    updates = {k: v for k, v in req.dict().items() if v is not None and k in _ALLOWED_PROFILE_FIELDS}
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user["id"]]

    conn = get_connection()
    try:
        _execute(conn, f"UPDATE users SET {set_clause} WHERE id = ?", tuple(values))
        conn.commit()
    finally:
        release_connection(conn)

    return _fetch_user_full(user["id"])


# ---------------------------------------------------------------------------
# POST /auth/password/change
# ---------------------------------------------------------------------------

@router.post("/auth/password/change")
async def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    """Change password — requires current password verification."""
    if not verify_password(req.current_password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    new_hash = hash_password(req.new_password)
    conn = get_connection()
    try:
        _execute(conn, "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user["id"]))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "message": "Password changed successfully"}


# ---------------------------------------------------------------------------
# POST /auth/password/set  (Google-only users setting a password for the first time)
# ---------------------------------------------------------------------------

class SetPasswordRequest(BaseModel):
    new_password: str


@router.post("/auth/password/set")
async def set_password(req: SetPasswordRequest, user: dict = Depends(get_current_user)):
    """Set a password for the first time — only allowed when password_hash is empty (Google-only account)."""
    if user.get("password_hash", ""):
        raise HTTPException(status_code=400, detail="Account already has a password. Use /auth/password/change instead.")

    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    new_hash = hash_password(req.new_password)
    conn = get_connection()
    try:
        _execute(conn, "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user["id"]))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "message": "Password set successfully"}


# ---------------------------------------------------------------------------
# POST /auth/password/reset-request
# ---------------------------------------------------------------------------

@router.post("/auth/password/reset-request")
@limiter.limit("5/hour")
async def password_reset_request(req: ResetRequestBody, request: Request):
    """Generate a 6-digit OTP and store its hash for password reset.

    Rate-limited (audit M2) — was previously unthrottled (row-growth/spam vector)."""
    otp = "".join(secrets.choice("0123456789") for _ in range(6))
    otp_hash = hash_password(otp)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)

    conn = get_connection()
    try:
        _execute(conn, """
            INSERT INTO password_reset_otps (email, otp_hash, expires_at)
            VALUES (?, ?, ?)
        """, (req.email, otp_hash, expires_at))
        conn.commit()
    finally:
        release_connection(conn)

    # SMTP not configured — OTP is not logged to prevent credential exposure in logs

    return {"status": "ok", "message": "If that email exists, an OTP has been sent"}


# ---------------------------------------------------------------------------
# POST /auth/password/reset-confirm
# ---------------------------------------------------------------------------

@router.post("/auth/password/reset-confirm")
@limiter.limit("10/hour")
async def password_reset_confirm(req: ResetConfirmBody, request: Request):
    """Verify the OTP and set a new password.

    Rate-limited (audit H2) — a 6-digit OTP (1M space) was brute-forceable
    within its 15-minute validity window with no limit on confirm attempts."""
    if len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT id, otp_hash, expires_at, used
            FROM password_reset_otps
            WHERE email = ?
              AND used = FALSE
            ORDER BY created_at DESC
            LIMIT 1
        """, (req.email,))
        row = _row_to_dict(cur)

        if not row:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")

        # Check expiry
        expires_at = row["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) > expires_at:
            raise HTTPException(status_code=400, detail="OTP has expired")

        # Verify OTP
        if not verify_password(req.otp, row["otp_hash"]):
            raise HTTPException(status_code=400, detail="Invalid OTP")

        # Look up user
        user_cur = _execute(conn, "SELECT id FROM users WHERE email = ?", (req.email,))
        user_row = _row_to_dict(user_cur)
        if not user_row:
            raise HTTPException(status_code=400, detail="No account found for that email")

        new_hash = hash_password(req.new_password)

        # Update password
        _execute(conn, "UPDATE users SET password_hash = ? WHERE id = ?",
                 (new_hash, user_row["id"]))

        # Mark OTP as used
        _execute(conn, "UPDATE password_reset_otps SET used = TRUE WHERE id = ?", (row["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "message": "Password reset successfully"}


# ---------------------------------------------------------------------------
# GET /auth/preferences
# ---------------------------------------------------------------------------

@router.get("/auth/preferences")
async def get_preferences(user: dict = Depends(get_current_user)):
    """Return user preferences: default_account, currency."""
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT default_account, currency FROM users WHERE id = ?",
                       (user["id"],))
        row = _row_to_dict(cur)
    finally:
        release_connection(conn)

    return {
        "default_account": (row or {}).get("default_account", "PAPER"),
        "currency": (row or {}).get("currency", "INR"),
    }


# ---------------------------------------------------------------------------
# PUT /auth/preferences
# ---------------------------------------------------------------------------

@router.put("/auth/preferences")
async def update_preferences(req: PreferencesRequest, user: dict = Depends(get_current_user)):
    """Update default_account and/or currency."""
    _ALLOWED_PREF_FIELDS = {"default_account", "currency"}
    updates = {k: v for k, v in req.dict().items() if v is not None and k in _ALLOWED_PREF_FIELDS}
    if not updates:
        raise HTTPException(status_code=400, detail="No preferences to update")

    if "default_account" in updates and updates["default_account"] not in ("PAPER", "LIVE"):
        raise HTTPException(status_code=400, detail="default_account must be PAPER or LIVE")

    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [user["id"]]

    conn = get_connection()
    try:
        _execute(conn, f"UPDATE users SET {set_clause} WHERE id = ?", tuple(values))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", **updates}


# ---------------------------------------------------------------------------
# POST /auth/totp/setup
# ---------------------------------------------------------------------------

@router.post("/auth/totp/setup")
async def totp_setup(user: dict = Depends(get_current_user)):
    """Generate a new TOTP secret and return provisioning URI."""
    secret = pyotp.random_base32()
    qr_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user["username"], issuer_name="TradeMind"
    )

    # Generate a base64-encoded PNG QR code so the frontend can use it as <img src=...>
    try:
        import qrcode
        import io
        import base64
        qr = qrcode.QRCode(box_size=6, border=4)
        qr.add_data(qr_uri)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        qr_image = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        qr_image = None

    conn = get_connection()
    try:
        _execute(conn, "UPDATE users SET totp_secret = ?, totp_enabled = FALSE WHERE id = ?",
                 (secret, user["id"]))
        conn.commit()
    finally:
        release_connection(conn)

    return {"qr_uri": qr_uri, "qr_image": qr_image, "secret": secret}


# ---------------------------------------------------------------------------
# POST /auth/totp/confirm
# ---------------------------------------------------------------------------

@router.post("/auth/totp/confirm")
async def totp_confirm(req: TotpConfirmRequest, user: dict = Depends(get_current_user)):
    """Verify TOTP code and enable 2FA."""
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT totp_secret FROM users WHERE id = ?", (user["id"],))
        row = _row_to_dict(cur)
    finally:
        release_connection(conn)

    secret = (row or {}).get("totp_secret")
    if not secret:
        raise HTTPException(status_code=400, detail="TOTP setup not initiated. Call /auth/totp/setup first")

    totp = pyotp.TOTP(secret)
    if not totp.verify(req.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    conn = get_connection()
    try:
        _execute(conn, "UPDATE users SET totp_enabled = TRUE WHERE id = ?", (user["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "message": "Two-factor authentication enabled"}


# ---------------------------------------------------------------------------
# POST /auth/totp/disable
# ---------------------------------------------------------------------------

@router.post("/auth/totp/disable")
async def totp_disable(req: TotpDisableRequest, user: dict = Depends(get_current_user)):
    """Verify TOTP code and disable 2FA."""
    conn = get_connection()
    try:
        cur = _execute(conn, "SELECT totp_secret, totp_enabled FROM users WHERE id = ?", (user["id"],))
        row = _row_to_dict(cur)
    finally:
        release_connection(conn)

    secret = (row or {}).get("totp_secret")
    enabled = (row or {}).get("totp_enabled", False)

    if not secret or not enabled:
        raise HTTPException(status_code=400, detail="Two-factor authentication is not enabled")

    totp = pyotp.TOTP(secret)
    if not totp.verify(req.code, valid_window=1):
        raise HTTPException(status_code=400, detail="Invalid TOTP code")

    conn = get_connection()
    try:
        _execute(conn, "UPDATE users SET totp_secret = NULL, totp_enabled = FALSE WHERE id = ?",
                 (user["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "message": "Two-factor authentication disabled"}


# ---------------------------------------------------------------------------
# GET /auth/sessions
# ---------------------------------------------------------------------------

@router.get("/auth/sessions")
async def list_sessions(user: dict = Depends(get_current_user)):
    """Return all active sessions for the current user."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT id, device, ip_address, location, created_at, last_seen
            FROM user_sessions
            WHERE user_id = ?
            ORDER BY last_seen DESC
        """, (user["id"],))
        sessions = _rows_to_dicts(cur)
    finally:
        release_connection(conn)

    # Convert UUID to string for JSON serialisation
    for s in sessions:
        if s.get("id") is not None:
            s["id"] = str(s["id"])

    return {"data": sessions, "total": len(sessions)}


# ---------------------------------------------------------------------------
# DELETE /auth/sessions/{session_id}
# ---------------------------------------------------------------------------

@router.delete("/auth/sessions/{session_id}")
async def delete_session(session_id: str, user: dict = Depends(get_current_user)):
    """Delete a specific session by ID."""
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM user_sessions WHERE id = ? AND user_id = ?",
                 (session_id, user["id"]))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# DELETE /auth/sessions  (delete all)
# ---------------------------------------------------------------------------

@router.delete("/auth/sessions")
async def delete_all_sessions(user: dict = Depends(get_current_user)):
    """Delete all sessions for the current user."""
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM user_sessions WHERE user_id = ?", (user["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# POST /auth/login/mfa — complete login after TOTP verification
# ---------------------------------------------------------------------------

class MfaLoginRequest(BaseModel):
    mfa_token: str
    totp_code: str


@router.post("/auth/login/mfa")
async def login_mfa(req: MfaLoginRequest):
    """
    Complete a login that requires TOTP.

    Accepts the short-lived mfa_token issued by POST /login when totp_enabled=TRUE,
    verifies the 6-digit TOTP code, and returns the full JWT on success.
    """
    payload = decode_token(req.mfa_token)
    if not payload:
        raise HTTPException(status_code=401, detail="MFA token expired or invalid. Please log in again.")
    if payload.get("scope") != "mfa":
        raise HTTPException(status_code=401, detail="Invalid token scope.")

    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found.")

    conn = get_connection()
    try:
        row = _execute(conn,
            "SELECT totp_secret, totp_enabled FROM users WHERE id = ?",
            (user["id"],)
        ).fetchone()
    finally:
        release_connection(conn)

    if not row or not row[1] or not row[0]:
        raise HTTPException(status_code=400, detail="2FA is not enabled on this account.")

    totp_secret = row[0]
    totp = pyotp.TOTP(totp_secret)

    # Allow 1 window of drift (30s before/after) to handle clock skew
    if not totp.verify(req.totp_code.strip(), valid_window=1):
        raise HTTPException(status_code=401, detail="Invalid authenticator code. Please try again.")

    from trading.trading_engine import _safe_user
    token = create_token(user["id"], user["username"])
    return {"status": "success", "user": _safe_user(user), "token": token}
