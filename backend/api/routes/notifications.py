"""
TradeMind AI — Notifications Routes

GET    /api/notifications
POST   /api/notifications/mark-read
DELETE /api/notifications/{id}
GET    /api/notifications/preferences
PUT    /api/notifications/preferences
"""
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel

from api.auth import decode_token
from database.db import (
    delete_notification, get_notifications, mark_notifications_read,
    _execute, _row_to_dict, get_connection, release_connection,
)
from trading.trading_engine import get_user
from api.schemas import NotificationsListOut, StatusOut

router = APIRouter(prefix="/api/notifications", tags=["Notifications"])


async def _current_user_id(authorization: Optional[str] = Header(None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    payload = decode_token(authorization.split(" ", 1)[1])
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    if payload.get("scope") != "full":
        raise HTTPException(status_code=401, detail="Incomplete authentication — please complete MFA")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user["id"]


# ---------------------------------------------------------------------------
# Preference default values
# ---------------------------------------------------------------------------

_PREF_DEFAULTS = {
    "signal_change": True,
    "price_alert": True,
    "trade_executed": True,
    "news_sentiment": False,
    "eod_summary": True,
    "weekly_report": False,
    "ch_email": True,
    "ch_push": True,
    "ch_sms": False,
}


class NotificationPreferencesRequest(BaseModel):
    signal_change: Optional[bool] = None
    price_alert: Optional[bool] = None
    trade_executed: Optional[bool] = None
    news_sentiment: Optional[bool] = None
    eod_summary: Optional[bool] = None
    weekly_report: Optional[bool] = None
    ch_email: Optional[bool] = None
    ch_push: Optional[bool] = None
    ch_sms: Optional[bool] = None


@router.get("", response_model=NotificationsListOut)
async def list_notifications(user_id: int = Depends(_current_user_id)):
    return get_notifications(user_id)


@router.post("/mark-read", response_model=StatusOut)
async def mark_read(user_id: int = Depends(_current_user_id)):
    mark_notifications_read(user_id)
    return {"status": "ok"}


@router.delete("/{notif_id}", response_model=StatusOut)
async def remove_notification(notif_id: int, user_id: int = Depends(_current_user_id)):
    delete_notification(notif_id, user_id)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /api/notifications/preferences
# ---------------------------------------------------------------------------

@router.get("/preferences")
async def get_notification_preferences(user_id: int = Depends(_current_user_id)):
    """Return notification preference settings for the current user."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT signal_change, price_alert, trade_executed, news_sentiment,
                   eod_summary, weekly_report, ch_email, ch_push, ch_sms
            FROM notification_preferences
            WHERE user_id = ?
        """, (user_id,))
        row = _row_to_dict(cur)
    finally:
        release_connection(conn)

    if not row:
        return dict(_PREF_DEFAULTS)

    # Merge with defaults for any NULL columns
    result = dict(_PREF_DEFAULTS)
    for k, v in row.items():
        if v is not None:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# PUT /api/notifications/preferences
# ---------------------------------------------------------------------------

@router.put("/preferences")
async def update_notification_preferences(
    req: NotificationPreferencesRequest,
    user_id: int = Depends(_current_user_id),
):
    """Upsert notification preference settings for the current user."""
    # Build full preferences dict: start from defaults, overlay request values
    prefs = dict(_PREF_DEFAULTS)
    for field, val in req.dict().items():
        if val is not None:
            prefs[field] = val

    conn = get_connection()
    try:
        _execute(conn, """
            INSERT INTO notification_preferences
                (user_id, signal_change, price_alert, trade_executed, news_sentiment,
                 eod_summary, weekly_report, ch_email, ch_push, ch_sms, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
            ON CONFLICT (user_id) DO UPDATE SET
                signal_change  = EXCLUDED.signal_change,
                price_alert    = EXCLUDED.price_alert,
                trade_executed = EXCLUDED.trade_executed,
                news_sentiment = EXCLUDED.news_sentiment,
                eod_summary    = EXCLUDED.eod_summary,
                weekly_report  = EXCLUDED.weekly_report,
                ch_email       = EXCLUDED.ch_email,
                ch_push        = EXCLUDED.ch_push,
                ch_sms         = EXCLUDED.ch_sms,
                updated_at     = NOW()
        """, (
            user_id,
            prefs["signal_change"],
            prefs["price_alert"],
            prefs["trade_executed"],
            prefs["news_sentiment"],
            prefs["eod_summary"],
            prefs["weekly_report"],
            prefs["ch_email"],
            prefs["ch_push"],
            prefs["ch_sms"],
        ))
        conn.commit()
    finally:
        release_connection(conn)

    return {"status": "ok", **prefs}
