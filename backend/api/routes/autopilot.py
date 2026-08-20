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
from database.db import (_execute, _rows_to_dicts, get_active_signal_id, get_connection,
                         get_latest_close_map, insert_notification, release_connection)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/autopilot", tags=["Autopilot"])

# Minimum upside a BUY mandate must offer, as a percent of entry. Round-trip
# costs (brokerage 0.05% + STT 0.1% + SEBI + stamp duty) are roughly 0.165%, so
# below this a trade cannot profit even when it reaches its target.
MIN_UPSIDE_PCT = 0.5


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

def _pnl_bounds(entry, target, sl, qty, client_exp_profit=None, client_max_loss=None):
    """(exp_profit, max_loss) for a mandate — derived here, and SIGNED.

    Two things were wrong with taking these from the client. The values were
    whatever the browser posted, so nothing server-side could vouch for them;
    and `max_loss` arrived as a positive magnitude — StockPage sent
    (entry - sl) * qty — leaving the frontend to bolt a "−" onto the number at
    render time. A loss that is only negative because a template prepends a
    glyph is not negative data: sort it, sum it, or export it and the sign is
    simply gone.

    Derived from the levels instead, so the sign falls out of the arithmetic:
    profit is measured up to the target, loss down to the stop. For a long,
    max_loss is therefore negative on its own.

    Falls back to the client's figure only when a level is missing, normalising
    its sign so a legacy positive magnitude still reads as a loss.
    """
    def _num(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    e, t, s_, q = _num(entry), _num(target), _num(sl), _num(qty)

    if None not in (e, t, q):
        exp_profit = round((t - e) * q, 2)
    else:
        exp_profit = abs(_num(client_exp_profit) or 0)

    if None not in (e, s_, q):
        max_loss = round((s_ - e) * q, 2)
    else:
        max_loss = -abs(_num(client_max_loss) or 0)

    return exp_profit, max_loss


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
        # What the entry actually filled at — below `entry` whenever the market
        # was cheaper than the authorised price. Recorded so the mandate's P&L
        # is computed off the real cost basis and agrees with the position's.
        fill_price    = (exec_result.get("position") or {}).get("buy_price")

        # Update the authorized_trades row
        conn = get_connection()
        try:
            _execute(conn,
                """UPDATE authorized_trades
                   SET status = 'OPEN', bracket_id = ?, sl_gtt_id = ?, target_gtt_id = ?,
                       fill_price = ?, updated_at = NOW()
                   WHERE id = ?""",
                (bracket_id, sl_gtt_id, target_gtt_id, fill_price, trade["id"]))
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
        # Read only — do NOT claim here. _execute_mandate does the atomic claim
        # (SET status='EXECUTING' WHERE status='PENDING'), which is the single
        # place that decides who owns a mandate.
        #
        # This used to pre-claim by setting 'EXECUTED', because the CHECK
        # constraint had no 'EXECUTING' state. That deadlocked the two claims
        # against each other: this call consumed the PENDING state, then
        # _execute_mandate looked for PENDING, matched nothing, returned
        # "Trade already claimed by another executor", and the error handler
        # below reverted the row to PENDING. Every mandate cycled
        # PENDING -> EXECUTED -> PENDING and no order was ever placed, so
        # turning autopilot ON silently did nothing. 'EXECUTING' is now a
        # permitted status, so the real claim works and this must not race it.
        cur = _execute(conn,
            "SELECT * FROM authorized_trades WHERE user_id = ? AND status = 'PENDING'",
            (user_id,))
        pending = _rows_to_dicts(cur)
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
                # Only release a claim we actually hold. An unconditional revert
                # would reset a row another executor had just claimed, handing
                # the same mandate to two executors — the precise race the
                # EXECUTING claim exists to prevent.
                _execute(conn,
                    "UPDATE authorized_trades SET status = 'PENDING', updated_at = NOW() "
                    "WHERE id = ? AND status = 'EXECUTING'", (trade["id"],))
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

        capital   = sum(t["amount"] for t in trades if t["status"] == "OPEN")
        active    = sum(1 for t in trades if t["status"] in ("OPEN", "PENDING"))
        realized  = sum(
            t["actual_pnl"] for t in trades
            if t["actual_pnl"] is not None and t["status"] in ("COMPLETED", "STOPPED")
        )
        projected = sum(
            t["exp_profit"] for t in trades
            if t["status"] in ("OPEN", "PENDING")
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


@router.get("/recommendations")
async def get_recommendations(user_id: int, max_concurrent: int = 8,
                              user=Depends(_get_current_user)):
    """Conviction-ranked STRONG BUY trades to take now (recommend-only, PAPER).

    Applies the backtest-validated portfolio construction — rank by conviction,
    cap concurrent positions, risk-size, respect cash — and returns the ordered
    set. Places NO trades; the user reviews and authorizes via POST /trades."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    from trading.signal_selector import recommend_trades
    return recommend_trades(user_id, max_concurrent=max_concurrent)


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

        # Existing rows were stored with a positive max_loss (the browser sent a
        # magnitude), so recompute from the levels on the way out. The API is the
        # authority on the sign — a client should never have to add one.
        for r in rows:
            r["exp_profit"], r["max_loss"] = _pnl_bounds(
                r.get("entry"), r.get("target"), r.get("sl"), r.get("qty"),
                r.get("exp_profit"), r.get("max_loss"))

        # `cmp` is written at authorisation and then only refreshed by
        # price_monitor, which walks open *positions*. That leaves it stale for
        # every row that is not an open position:
        #   PENDING  — no position yet, so it still holds the entry price
        #   COMPLETED / STOPPED / CANCELLED — the position is gone, so it holds
        #     whatever the price was at the moment the trade closed, frozen
        # Both render as a "current price" that is not current. Overlay the
        # latest close on all of them; only EXECUTED rows keep their own value,
        # because price_monitor's LTP is fresher than a close.
        stale = [r for r in rows
                 if r.get("status") != "OPEN" or r.get("cmp") is None]
        if stale:
            close_map = get_latest_close_map(
                sorted({r["symbol"] for r in stale}), conn=conn)
            for r in stale:
                if close_map.get(r["symbol"]) is not None:
                    r["cmp"] = close_map[r["symbol"]]

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

        # Refuse a trade with no upside. Independent of where the numbers came
        # from — a target at or below the entry is not a trade, whatever the
        # model or the client believed.
        #
        # This is the second line of defence; generate_trades now rejects the
        # malformed target_pct that caused it (see MIN_TARGET_PCT). It stays
        # because that bug reached a real mandate: authorized_trades id 3,
        # AADHARHFC, entry 518.50 / target 518.53 — three paise of upside a
        # share, less than the round-trip fees — and nothing between the model
        # and the user's capital questioned it.
        if body.signal in ("BUY", "STRONG BUY") and body.entry and body.target:
            # A minimum upside, not merely target > entry. AADHARHFC's target of
            # 518.53 against an entry of 518.50 IS greater — by three paise — so
            # a strict inequality would have waved it through. Round-trip costs
            # (brokerage 0.05% + STT 0.1% + SEBI + stamp) are about 0.165%, so
            # anything under MIN_UPSIDE_PCT cannot profit even if it works.
            upside_pct = (body.target / body.entry - 1) * 100
            if upside_pct < MIN_UPSIDE_PCT:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Target ₹{body.target:,.2f} is only {upside_pct:.3f}% above "
                            f"entry ₹{body.entry:,.2f} — below the {MIN_UPSIDE_PCT}% "
                            f"minimum, which round-trip fees would consume."))
            if body.sl and body.sl >= body.entry:
                raise HTTPException(
                    status_code=400,
                    detail=(f"Stop-loss ₹{body.sl:,.2f} is not below entry "
                            f"₹{body.entry:,.2f} — it would trigger immediately."))

        # If a bracket_id is supplied the position is already open — mark EXECUTED immediately
        # so the autopilot page reflects the real state and square_off() can settle it.
        initial_status = "OPEN" if body.bracket_id else "PENDING"

        cur = _execute(conn,
            # trade_signal_id is captured HERE, at authorisation — the signal the
            # user actually saw and agreed to. trading_engine resolves its own
            # link at execution time via get_active_signal_id(), which records
            # whatever happens to be active when the order fires; for a mandate
            # authorised on 2026-06-26 that was the 2026-06-23 signal. For
            # "which recommendation did the user act on", only this one is true.
            """INSERT INTO authorized_trades
               (user_id, symbol, name, sector, signal, mode, qty, amount,
                entry, target, sl, exp_profit, max_loss, cmp, bracket_id, status,
                trade_signal_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               RETURNING *""",
            (body.user_id, body.symbol, body.name, body.sector,
             body.signal, body.mode, body.qty, body.amount,
             body.entry, body.target, body.sl,
             *_pnl_bounds(body.entry, body.target, body.sl, body.qty,
                          body.exp_profit, body.max_loss),
             body.cmp,
             body.bracket_id, initial_status,
             get_active_signal_id(body.symbol)))
        conn.commit()
        rows = _rows_to_dicts(cur)
        trade = rows[0] if rows else {}
    except HTTPException:
        # Deliberate 4xx (e.g. the no-upside validation above) must reach the
        # client as itself. The bare `except Exception` below would otherwise
        # re-wrap it as a 500, turning "target is not above entry" into an
        # opaque server error.
        conn.rollback()
        raise
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

    if trade["status"] not in ("PENDING", "OPEN"):
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
    if trade["status"] == "OPEN" and trade.get("bracket_id"):
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
