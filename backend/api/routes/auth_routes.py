"""
TradeMind AI — Extended Auth Routes

GET    /auth/me
PATCH  /auth/me
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
import random
import string
from datetime import datetime, timedelta, timezone
from typing import Optional

import pyotp
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from api.auth import decode_token, hash_password, verify_password
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


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fetch_user_full(user_id: int) -> Optional[dict]:
    """Fetch extended user row including new columns."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT id, username, display_name, email, phone, avatar_url,
                   totp_enabled, default_account, currency, virtual_balance
            FROM users
            WHERE id = ?
        """, (user_id,))
        return _row_to_dict(cur)
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# GET /auth/me
# ---------------------------------------------------------------------------

@router.get("/auth/me")
async def get_me(authorization: Optional[str] = Header(None)):
    """Return current user's profile."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    profile = _fetch_user_full(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile


# ---------------------------------------------------------------------------
# PATCH /auth/me
# ---------------------------------------------------------------------------

@router.patch("/auth/me")
async def update_me(req: UpdateProfileRequest, authorization: Optional[str] = Header(None)):
    """Update display_name, email, or phone."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    updates = {k: v for k, v in req.dict().items() if v is not None}
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
async def change_password(req: ChangePasswordRequest, authorization: Optional[str] = Header(None)):
    """Change password — requires current password verification."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
# POST /auth/password/reset-request
# ---------------------------------------------------------------------------

@router.post("/auth/password/reset-request")
async def password_reset_request(req: ResetRequestBody):
    """Generate a 6-digit OTP and store its hash for password reset."""
    otp = "".join(random.choices(string.digits, k=6))
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

    # SMTP not configured — log OTP for dev/testing only
    logger.info("[DEV] Password reset OTP for %s: %s", req.email, otp)

    return {"status": "ok", "message": "If that email exists, an OTP has been sent"}


# ---------------------------------------------------------------------------
# POST /auth/password/reset-confirm
# ---------------------------------------------------------------------------

@router.post("/auth/password/reset-confirm")
async def password_reset_confirm(req: ResetConfirmBody):
    """Verify the OTP and set a new password."""
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
async def get_preferences(authorization: Optional[str] = Header(None)):
    """Return user preferences: default_account, currency."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
async def update_preferences(req: PreferencesRequest, authorization: Optional[str] = Header(None)):
    """Update default_account and/or currency."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    updates = {k: v for k, v in req.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No preferences to update")

    # Validate values
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
async def totp_setup(authorization: Optional[str] = Header(None)):
    """Generate a new TOTP secret and return provisioning URI."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    secret = pyotp.random_base32()
    qr_uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user["username"], issuer_name="TradeMind"
    )

    conn = get_connection()
    try:
        # Store secret but don't enable yet (requires confirmation)
        _execute(conn, "UPDATE users SET totp_secret = ?, totp_enabled = FALSE WHERE id = ?",
                 (secret, user["id"]))
        conn.commit()
    finally:
        release_connection(conn)

    return {"qr_uri": qr_uri, "secret": secret}


# ---------------------------------------------------------------------------
# POST /auth/totp/confirm
# ---------------------------------------------------------------------------

@router.post("/auth/totp/confirm")
async def totp_confirm(req: TotpConfirmRequest, authorization: Optional[str] = Header(None)):
    """Verify TOTP code and enable 2FA."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
async def totp_disable(req: TotpDisableRequest, authorization: Optional[str] = Header(None)):
    """Verify TOTP code and disable 2FA."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
async def list_sessions(authorization: Optional[str] = Header(None)):
    """Return all active sessions for the current user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
async def delete_session(session_id: str, authorization: Optional[str] = Header(None)):
    """Delete a specific session by ID."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

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
async def delete_all_sessions(authorization: Optional[str] = Header(None)):
    """Delete all sessions for the current user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM user_sessions WHERE user_id = ?", (user["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok"}
