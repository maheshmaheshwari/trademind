"""
TradeMind AI — User Watchlist Routes

GET    /api/users/{user_id}/watchlist
POST   /api/users/{user_id}/watchlist/{symbol}
DELETE /api/users/{user_id}/watchlist/{symbol}
PUT    /api/users/{user_id}/watchlist/{symbol}/alerts
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from database.db import (
    add_to_watchlist,
    get_watchlist,
    remove_from_watchlist,
    update_watchlist_alerts,
)
from api.routes.trading import get_current_user

router = APIRouter(prefix="/api/users", tags=["Watchlist"])


class AlertRequest(BaseModel):
    alert_above: Optional[float] = None
    alert_below: Optional[float] = None


@router.get("/{user_id}/watchlist")
async def list_watchlist(user_id: int, user=Depends(get_current_user)):
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    items = get_watchlist(user_id)
    return {"data": items, "total": len(items)}


@router.post("/{user_id}/watchlist/{symbol}", status_code=201)
async def add_to_watchlist_route(user_id: int, symbol: str, user=Depends(get_current_user)):
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    add_to_watchlist(user_id, symbol.upper())
    return {"status": "ok", "data": {"user_id": user_id, "symbol": symbol.upper()}}


@router.delete("/{user_id}/watchlist/{symbol}")
async def remove_from_watchlist_route(user_id: int, symbol: str, user=Depends(get_current_user)):
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    remove_from_watchlist(user_id, symbol.upper())
    return {"status": "ok"}


@router.put("/{user_id}/watchlist/{symbol}/alerts")
async def update_alerts(user_id: int, symbol: str, req: AlertRequest, user=Depends(get_current_user)):
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    update_watchlist_alerts(user_id, symbol.upper(), req.alert_above, req.alert_below)
    return {"status": "ok"}
