"""
TradeMind AI — Autopilot / AI-Authorized Trades Routes

Full wiring:
  • authorize_trade  — saves the mandate; if autopilot is ON immediately calls
                       execute_signal() (PAPER or LIVE), stores bracket_id + GTT IDs,
                       sets status → EXECUTED.
  • toggle           — flip enabled flag; when turning ON, fires all PENDING mandates.
  • revoke_trade     — cancels GTTs on Angel One, squares off open position, sets STOPPED.
  • list_trades      — filtered list for the UI.
  • get_status       — summary stats (capital, active, realized, projected).
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from pydantic import BaseModel

from api.auth import decode_token
from database.db import _execute, _rows_to_dicts, get_connection, release_connection, insert_notification

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/autopilot", tags=["Autopilot"])


async def _get_current_user(authorization: Optional[str] = Header(None)):
    from trading.trading_engine import get_user
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
# Pydantic models
# ---------------------------------------------------------------------------

class ToggleBody(BaseModel):
    user_id: int


class AuthorizeTradeBody(BaseModel):
    user_id: int
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    signal: str = "BUY"
    mode: str = "PAPER"
    qty: int = 0
    amount: float = 0
    entry: Optional[float] = None
    target: Optional[float] = None
    sl: Optional[float] = None
    exp_profit: float = 0
    max_loss: float = 0
    cmp: Optional[float] = None
    bracket_id: Optional[str] = None  # pass when adding an already-open position
    execute_immediately: bool = False  # True = always execute now (buy button), False = respect autopilot toggle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_settings(conn, user_id: int) -> dict:
    """Return autopilot_settings row, creating it if absent."""
    cur = _execute(conn, "SELECT * FROM autopilot_settings WHERE user_id = ?", (user_id,))
    row = _rows_to_dicts(cur)
    if not row:
        _execute(conn,
            "INSERT INTO autopilot_settings (user_id, enabled) VALUES (?, ?)", (user_id, False))
        conn.commit()
        return {"user_id": user_id, "enabled": False}
    return row[0]


def _execute_mandate(trade: dict) -> dict:
    """
    Call trading_engine.execute_signal() for one authorized trade, then
    update the authorized_trades row with bracket_id, GTT IDs, and
    status = EXECUTED.

    Returns a result dict with keys: success, bracket_id, sl_gtt_id, target_gtt_id, error
    """
    from trading.trading_engine import execute_signal

    result = {"success": False, "bracket_id": None, "sl_gtt_id": None, "target_gtt_id": None, "error": None}

    if not trade.get("entry") or not trade.get("target") or not trade.get("sl"):
        result["error"] = "Missing entry/target/sl — cannot execute"
        return result

    if not trade.get("qty") or trade["qty"] < 1:
        result["error"] = "qty < 1 — cannot execute"
        return result

    # Atomically claim the trade — prevents double-execution if both
    # authorize_trade and _fire_pending_mandates race to the same record.
    claim_conn = get_connection()
    try:
        cur = _execute(claim_conn,
            "UPDATE authorized_trades SET status='EXECUTING' WHERE id=? AND status='PENDING' RETURNING id",
            (trade["id"],))
        claimed = cur.fetchone()
        claim_conn.commit()
    finally:
        release_connection(claim_conn)
    if not claimed:
        return {"success": False, "error": "Trade already claimed by another executor"}

    try:
        exec_result = execute_signal(
            user_id    = trade["user_id"],
            symbol     = trade["symbol"],
            name       = trade.get("name") or trade["symbol"],
            investment_amount = trade["amount"],
            buy_price  = trade["entry"],
            target_price = trade["target"],
            stop_loss  = trade["sl"],
            signal     = trade.get("signal", "BUY"),
            confidence = 0,
            horizon    = "autopilot",
            mode       = trade.get("mode", "PAPER"),
        )

        bracket_id    = exec_result.get("bracket_id")
        gtt_info      = exec_result.get("gtt") or {}
        sl_gtt_id     = str(gtt_info.get("sl_rule_id"))    if gtt_info.get("sl_rule_id")     else None
        target_gtt_id = str(gtt_info.get("target_rule_id")) if gtt_info.get("target_rule_id") else None

        # Update the authorized_trades row
        conn = get_connection()
        try:
            _execute(conn,
                """UPDATE authorized_trades
                   SET status = 'EXECUTED', bracket_id = ?, sl_gtt_id = ?, target_gtt_id = ?,
                       updated_at = NOW()
                   WHERE id = ?""",
                (bracket_id, sl_gtt_id, target_gtt_id, trade["id"]))
            conn.commit()
        finally:
            release_connection(conn)

        result.update(success=True, bracket_id=bracket_id,
                      sl_gtt_id=sl_gtt_id, target_gtt_id=target_gtt_id)

        logger.info(
            f"✅ Autopilot executed: {trade['symbol']} user={trade['user_id']} "
            f"bracket={bracket_id} mode={trade.get('mode','PAPER')}"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"❌ Autopilot execute_mandate failed for {trade['symbol']}: {e}")

    return result


def _fire_pending_mandates(user_id: int):
    """Execute all PENDING mandates for a user (called when autopilot is turned ON)."""
    conn = get_connection()
    try:
        # Atomic claim (audit M12): a single UPDATE...WHERE status='PENDING'
        # RETURNING * means a rapid toggle-off/on that spawns a second
        # concurrent _fire_pending_mandates can't also claim the same rows —
        # by the time it runs its own claim, status is no longer 'PENDING'
        # for anything the first call already grabbed. status is set
        # straight to 'EXECUTED' as the claim marker since the
        # authorized_trades CHECK constraint has no separate "claimed" state;
        # failed executions are reverted back to 'PENDING' below so they
        # remain retryable, matching the pre-existing failure semantics.
        cur = _execute(conn,
            "UPDATE authorized_trades SET status = 'EXECUTED', updated_at = NOW() "
            "WHERE user_id = ? AND status = 'PENDING' RETURNING *",
            (user_id,))
        pending = _rows_to_dicts(cur)
        conn.commit()
    finally:
        release_connection(conn)

    if not pending:
        logger.info(f"Autopilot ON: no pending mandates for user {user_id}")
        return

    logger.info(f"Autopilot ON: firing {len(pending)} pending mandates for user {user_id}")
    fired, failed = 0, 0
    for trade in pending:
        res = _execute_mandate(trade)
        if res["success"]:
            fired += 1
            # execute_signal already inserts the "Order placed" notification
        else:
            failed += 1
            logger.warning(f"Mandate {trade['symbol']} failed: {res['error']}")
            conn = get_connection()
            try:
                _execute(conn, "UPDATE authorized_trades SET status = 'PENDING', updated_at = NOW() WHERE id = ?", (trade["id"],))
                conn.commit()
            finally:
                release_connection(conn)

    logger.info(f"Autopilot fired: {fired} executed, {failed} failed")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/status")
async def get_status(user_id: int, user=Depends(_get_current_user)):
    """Autopilot enabled flag + summary stats."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    conn = get_connection()
    try:
        settings = _ensure_settings(conn, user_id)
        cur = _execute(conn,
            "SELECT * FROM authorized_trades WHERE user_id = ?", (user_id,))
        trades = _rows_to_dicts(cur)

        capital   = sum(t["amount"] for t in trades if t["status"] == "EXECUTED")
        active    = sum(1 for t in trades if t["status"] in ("EXECUTED", "PENDING"))
        realized  = sum(
            t["actual_pnl"] for t in trades
            if t["actual_pnl"] is not None and t["status"] in ("COMPLETED", "STOPPED")
        )
        projected = sum(
            t["exp_profit"] for t in trades
            if t["status"] in ("EXECUTED", "PENDING")
        )
        return {
            "enabled": settings["enabled"],
            "capital": capital,
            "active": active,
            "realized_pnl": realized,
            "projected_profit": projected,
        }
    finally:
        release_connection(conn)


@router.post("/toggle")
async def toggle_autopilot(body: ToggleBody, background_tasks: BackgroundTasks, user=Depends(_get_current_user)):
    """Flip autopilot on/off. Turning ON fires all pending mandates in the background."""
    if user["id"] != body.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    conn = get_connection()
    try:
        settings = _ensure_settings(conn, body.user_id)
        new_state = not settings["enabled"]
        _execute(conn,
            "UPDATE autopilot_settings SET enabled = ?, updated_at = NOW() WHERE user_id = ?",
            (new_state, body.user_id))
        conn.commit()
    finally:
        release_connection(conn)

    # Fire pending mandates in the background when turning ON
    if new_state:
        background_tasks.add_task(_fire_pending_mandates, body.user_id)

    return {"enabled": new_state}


@router.get("/trades")
async def list_trades(user_id: int, status: Optional[str] = None, user=Depends(_get_current_user)):
    """List authorized trades, optionally filtered by status."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    conn = get_connection()
    try:
        if status and status != "All":
            cur = _execute(conn,
                "SELECT * FROM authorized_trades WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
                (user_id, status))
        else:
            cur = _execute(conn,
                "SELECT * FROM authorized_trades WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,))
        rows = _rows_to_dicts(cur)
        return {"data": rows, "total": len(rows)}
    finally:
        release_connection(conn)


@router.post("/trades")
async def authorize_trade(body: AuthorizeTradeBody, background_tasks: BackgroundTasks, user=Depends(_get_current_user)):
    """
    Authorize a new AI-managed trade.

    If autopilot is currently ON, executes the trade immediately (in the background)
    and sets status = EXECUTED.
    If OFF, saves with status = PENDING — will fire when autopilot is next turned ON.
    """
    if user["id"] != body.user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    conn = get_connection()
    try:
        settings = _ensure_settings(conn, body.user_id)
        autopilot_on = settings["enabled"]

        # If a bracket_id is supplied the position is already open — mark EXECUTED immediately
        # so the autopilot page reflects the real state and square_off() can settle it.
        initial_status = "EXECUTED" if body.bracket_id else "PENDING"

        cur = _execute(conn,
            """INSERT INTO authorized_trades
               (user_id, symbol, name, sector, signal, mode, qty, amount,
                entry, target, sl, exp_profit, max_loss, cmp, bracket_id, status)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               RETURNING *""",
            (body.user_id, body.symbol, body.name, body.sector,
             body.signal, body.mode, body.qty, body.amount,
             body.entry, body.target, body.sl,
             body.exp_profit, body.max_loss, body.cmp,
             body.bracket_id, initial_status))
        conn.commit()
        rows = _rows_to_dicts(cur)
        trade = rows[0] if rows else {}
    except Exception as e:
        conn.rollback()
        logger.error(f"authorize_trade insert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        release_connection(conn)

    # Already-open position handed to autopilot — no need to re-execute
    if body.bracket_id:
        return {"status": "ok", "message": "Existing position handed to autopilot — AI will monitor SL/Target", "data": trade}

    # execute_immediately=True (buy button): fire now regardless of autopilot toggle
    # autopilot_on=True: also fire now
    if (body.execute_immediately or autopilot_on) and trade:
        background_tasks.add_task(_execute_mandate, trade)
        return {"status": "ok", "message": "Order placed — AI is managing SL/Target", "data": trade}

    return {"status": "ok", "message": "Mandate saved — will execute when autopilot is turned ON", "data": trade}


@router.delete("/trades/{trade_id}")
async def revoke_trade(trade_id: int, user=Depends(_get_current_user)):
    """
    Revoke an authorized trade:
      1. Cancel SL + Target GTTs on Angel One (if LIVE mode and GTT IDs present)
      2. Square off the open position (if status = EXECUTED)
      3. Set status = STOPPED
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT * FROM authorized_trades WHERE id = ?", (trade_id,))
        rows = _rows_to_dicts(cur)
        if not rows:
            raise HTTPException(status_code=404, detail="Trade not found")
        trade = rows[0]
    finally:
        release_connection(conn)

    # 404 for both "doesn't exist" and "not yours" — a distinct 403 here
    # would let a caller enumerate valid trade ids by status-code alone.
    if user["id"] != trade["user_id"]:
        raise HTTPException(status_code=404, detail="Trade not found")

    if trade["status"] not in ("PENDING", "EXECUTED"):
        raise HTTPException(status_code=400,
            detail=f"Cannot revoke a trade with status '{trade['status']}'")

    actual_pnl = None

    # ── Step 1: cancel GTTs ──────────────────────────────────────────────────
    if trade.get("mode") == "LIVE":
        from trading.gtt_manager import cancel_gtt
        for gtt_field in ("sl_gtt_id", "target_gtt_id"):
            gtt_id = trade.get(gtt_field)
            if gtt_id:
                try:
                    cancel_gtt(int(gtt_id))
                    logger.info(f"Cancelled GTT {gtt_id} for trade {trade_id}")
                except Exception as e:
                    logger.warning(f"Could not cancel GTT {gtt_id}: {e}")

    # ── Step 2: square off the open position (if executed) ───────────────────
    if trade["status"] == "EXECUTED" and trade.get("bracket_id"):
        try:
            from trading.trading_engine import square_off
            sq = square_off(trade["user_id"], trade["symbol"])
            actual_pnl = sq.get("pnl")
            logger.info(
                f"Squared off {trade['symbol']} for user {trade['user_id']} — P&L: ₹{actual_pnl}"
            )
        except Exception as e:
            logger.warning(f"Square-off failed for {trade['symbol']}: {e} — marking STOPPED anyway")

    # ── Step 3: mark STOPPED ─────────────────────────────────────────────────
    conn = get_connection()
    try:
        if actual_pnl is not None:
            _execute(conn,
                "UPDATE authorized_trades SET status = 'STOPPED', actual_pnl = ?, updated_at = NOW() WHERE id = ?",
                (actual_pnl, trade_id))
        else:
            _execute(conn,
                "UPDATE authorized_trades SET status = 'STOPPED', updated_at = NOW() WHERE id = ?",
                (trade_id,))
        conn.commit()
    finally:
        release_connection(conn)

    # Notify
    try:
        insert_notification(
            user_id=trade["user_id"], type="trade",
            title=f"Autopilot revoked: {trade['symbol']}",
            message=f"Authorization removed — position {'closed' if actual_pnl is not None else 'pending cancellation'}",
            icon="AlertCircle", color="#EF4444",
        )
    except Exception:
        pass

    return {
        "status": "ok",
        "message": f"Trade {trade_id} revoked",
        "actual_pnl": actual_pnl,
    }
