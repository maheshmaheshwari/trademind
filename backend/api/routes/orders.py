"""
TradeMind AI — Orders Routes

GET  /api/orders/gtt              — all GTT orders (optionally ?user_id=N)
GET  /api/orders/gtt/{user_id}    — GTT orders for a specific user
POST /api/orders/gtt/sync         — trigger Angel One GTT status sync
"""
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from database.db import _execute, _rows_to_dicts, get_connection
from api.routes.trading import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/orders", tags=["Orders"])


@router.get("/gtt")
async def get_gtt_orders(user_id: Optional[int] = Query(default=None), user=Depends(get_current_user)):
    """
    Return GTT orders.
    Pass ?user_id=N to filter by user (recommended).
    Without user_id returns only the authenticated user's orders.
    """
    conn = get_connection()
    try:
        # Always restrict to the authenticated user unless they match explicitly
        effective_user_id = user["id"]
        if user_id and user_id != effective_user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        cur = _execute(conn,
            "SELECT * FROM orders WHERE gtt_rule_id IS NOT NULL AND user_id = ? ORDER BY created_at DESC",
            (effective_user_id,))
        rows = _rows_to_dicts(cur)
        return {"data": rows, "total": len(rows)}
    finally:
        conn.close()


@router.get("/gtt/{user_id}")
async def get_user_gtt_orders(user_id: int, user=Depends(get_current_user)):
    """Return GTT orders for a specific user."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT * FROM orders WHERE gtt_rule_id IS NOT NULL AND user_id = ? ORDER BY created_at DESC",
            (user_id,))
        rows = _rows_to_dicts(cur)
        return {"data": rows, "total": len(rows), "user_id": user_id}
    finally:
        conn.close()


@router.post("/gtt/sync")
async def sync_gtt(background_tasks: BackgroundTasks, user=Depends(get_current_user)):
    """Manually trigger an Angel One GTT status sync in the background."""
    def _run():
        try:
            from trading.gtt_manager import sync_gtt_statuses
            sync_gtt_statuses()
        except Exception as e:
            logger.error(f"Manual GTT sync failed: {e}")
    background_tasks.add_task(_run)
    return {"status": "ok", "message": "GTT sync started"}
