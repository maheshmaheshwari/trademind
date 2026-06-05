"""
TradeMind AI — Notifications Routes

GET    /api/notifications
POST   /api/notifications/mark-read
DELETE /api/notifications/{id}
"""
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException

from api.auth import decode_token
from database.db import delete_notification, get_notifications, mark_notifications_read
from trading.trading_engine import get_user

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


async def _current_user_id(authorization: Optional[str] = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    payload = decode_token(authorization.split(" ", 1)[1])
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user["id"]


@router.get("")
async def list_notifications(user_id: int = Depends(_current_user_id)):
    return get_notifications(user_id)


@router.post("/mark-read")
async def mark_read(user_id: int = Depends(_current_user_id)):
    mark_notifications_read(user_id)
    return {"status": "ok"}


@router.delete("/{notif_id}")
async def remove_notification(notif_id: int, user_id: int = Depends(_current_user_id)):
    delete_notification(notif_id, user_id)
    return {"status": "ok"}
