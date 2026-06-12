"""
TradeMind AI — Authentication Utilities

Password hashing (bcrypt) and JWT token management.
"""
import os
import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict

# JWT secret — must be set via environment variable
_jwt_secret = os.getenv("JWT_SECRET")
if not _jwt_secret:
    raise RuntimeError(
        "JWT_SECRET environment variable is not set. "
        "Set it to a strong random string before starting the server."
    )
JWT_SECRET = _jwt_secret
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24 * 7  # 7 days


def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user_id: int, username: str) -> str:
    """Create a full JWT access token (7-day lifetime)."""
    payload = {
        "user_id": user_id,
        "username": username,
        "scope": "full",
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def create_mfa_token(user_id: int, username: str) -> str:
    """Create a short-lived MFA-step token (5-minute lifetime, scope=mfa only)."""
    payload = {
        "user_id": user_id,
        "username": username,
        "scope": "mfa",
        "exp": datetime.utcnow() + timedelta(minutes=5),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Optional[Dict]:
    """Decode and validate a JWT token. Returns payload or None."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
