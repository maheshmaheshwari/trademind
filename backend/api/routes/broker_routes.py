"""
TradeMind AI — Broker Connection Routes

GET    /api/brokers
POST   /api/brokers/angel-one/connect
DELETE /api/brokers/angel-one/disconnect
GET    /api/brokers/zerodha/login
GET    /api/brokers/upstox/login
"""

import base64
import hashlib
import logging
import os
from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from api.auth import decode_token
from database.db import _execute, _row_to_dict, _rows_to_dicts, get_connection, release_connection
from trading.trading_engine import get_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/brokers", tags=["Brokers"])


# ---------------------------------------------------------------------------
# Encryption helpers (Fernet, key derived from JWT_SECRET)
# ---------------------------------------------------------------------------

def _get_fernet():
    """Return a Fernet instance keyed by BROKER_ENCRYPTION_KEY (separate from JWT_SECRET)."""
    from cryptography.fernet import Fernet
    broker_key = os.getenv("BROKER_ENCRYPTION_KEY") or os.getenv("JWT_SECRET", "")
    key = base64.urlsafe_b64encode(hashlib.sha256(broker_key.encode()).digest())
    return Fernet(key)


def _encrypt(plaintext: str) -> str:
    return _get_fernet().encrypt(plaintext.encode()).decode()


def _decrypt(ciphertext: str) -> str:
    try:
        return _get_fernet().decrypt(ciphertext.encode()).decode()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# JWT dependency
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

class AngelConnectRequest(BaseModel):
    client_id: str
    password: str
    totp: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BROKER_DEFAULTS = [
    {"broker": "angel",   "name": "Angel One", "connected": False, "client_id": None},
    {"broker": "zerodha", "name": "Zerodha",   "connected": False, "client_id": None},
    {"broker": "upstox",  "name": "Upstox",    "connected": False, "client_id": None},
    {"broker": "groww",   "name": "Groww",     "connected": False, "client_id": None, "coming_soon": True},
]


def _get_broker_rows(user_id: int) -> dict:
    """Return a dict of broker -> row from broker_connections."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT broker, connected, client_id
            FROM broker_connections
            WHERE user_id = ?
        """, (user_id,))
        rows = _rows_to_dicts(cur)
    finally:
        release_connection(conn)
    return {r["broker"]: r for r in rows}


# ---------------------------------------------------------------------------
# GET /api/brokers
# ---------------------------------------------------------------------------

@router.get("")
async def list_brokers(authorization: Optional[str] = Header(None)):
    """List all broker connections with their current status."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    db_rows = _get_broker_rows(user["id"])

    result = []
    for b in _BROKER_DEFAULTS:
        entry = dict(b)  # copy defaults
        if b["broker"] in db_rows:
            row = db_rows[b["broker"]]
            entry["connected"] = bool(row.get("connected"))
            entry["client_id"] = row.get("client_id")
        result.append(entry)

    return result


# ---------------------------------------------------------------------------
# POST /api/brokers/angel-one/connect
# ---------------------------------------------------------------------------

@router.post("/angel-one/connect")
async def angel_connect(req: AngelConnectRequest, authorization: Optional[str] = Header(None)):
    """Connect Angel One SmartAPI — stores encrypted access token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    api_key = os.getenv("ANGEL_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=503, detail="Angel One API key not configured on server")

    # Attempt SmartAPI login
    try:
        from SmartApi import SmartConnect
        smart = SmartConnect(api_key=api_key)
        data = smart.generateSession(req.client_id, req.password, req.totp)
    except ImportError:
        raise HTTPException(status_code=503, detail="SmartApi SDK not installed on server")
    except Exception as e:
        logger.error("Angel One SmartAPI error: %s", e)
        raise HTTPException(status_code=502, detail=f"Angel One API error: {e}")

    if not data.get("status"):
        msg = data.get("message", "Login failed")
        raise HTTPException(status_code=401, detail=f"Angel One login failed: {msg}")

    jwt_token = data["data"]["jwtToken"]
    encrypted_token = _encrypt(jwt_token)

    conn = get_connection()
    try:
        _execute(conn, """
            INSERT INTO broker_connections (user_id, broker, access_token, client_id, connected)
            VALUES (?, 'angel', ?, ?, TRUE)
            ON CONFLICT (user_id, broker) DO UPDATE SET
                access_token = EXCLUDED.access_token,
                client_id    = EXCLUDED.client_id,
                connected    = TRUE
        """, (user["id"], encrypted_token, req.client_id))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "broker": "angel", "connected": True, "client_id": req.client_id}


# ---------------------------------------------------------------------------
# DELETE /api/brokers/angel-one/disconnect
# ---------------------------------------------------------------------------

@router.delete("/angel-one/disconnect")
async def angel_disconnect(authorization: Optional[str] = Header(None)):
    """Disconnect Angel One — clears stored credentials."""
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
        _execute(conn, """
            UPDATE broker_connections
            SET access_token = NULL, connected = FALSE
            WHERE user_id = ? AND broker = 'angel'
        """, (user["id"],))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", "broker": "angel", "connected": False}


# ---------------------------------------------------------------------------
# GET /api/brokers/zerodha/login
# ---------------------------------------------------------------------------

@router.get("/zerodha/login")
async def zerodha_login(authorization: Optional[str] = Header(None)):
    """Zerodha OAuth — coming soon."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"redirect_url": None, "message": "Zerodha OAuth coming soon"}


# ---------------------------------------------------------------------------
# GET /api/brokers/upstox/login
# ---------------------------------------------------------------------------

@router.get("/upstox/login")
async def upstox_login(authorization: Optional[str] = Header(None)):
    """Upstox OAuth — coming soon."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return {"redirect_url": None, "message": "Upstox OAuth coming soon"}
