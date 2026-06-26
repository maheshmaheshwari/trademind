"""
Nifty 500 AI — Signals API Routes

GET /api/signals/top-buys?limit=10
GET /api/signals/top-sells?limit=10
GET /api/signals/all              — all signals across all horizons (flat list)
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request

from database.db import get_top_signals, get_connection, release_connection, _execute, _rows_to_dicts
from api.rate_limit import limiter
from api.routes.trading import get_current_user as _get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

_HORIZON_SHORT = {
    "1 Week": "1W", "2 Weeks": "2W", "1 Month": "1M",
    "2 Months": "2M", "3 Months": "3M", "6 Months": "6M",
}

# Load sector map once at startup from the static tokens file
_SECTOR_MAP: dict = {}
_TOKENS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "angel_tokens.json")
try:
    with open(_TOKENS_FILE) as _f:
        _tokens = json.load(_f)
    _SECTOR_MAP = {f"{sym}.NS": info.get("sector", "Unknown") for sym, info in _tokens.items()}
except Exception:
    pass

# In-process cache: rebuilt only when the DB has a newer generated_date or > 5 min old
_cache: dict = {"date": None, "payload": None, "ts": 0.0}
_CACHE_TTL = 300  # seconds


def _build_signals_payload() -> dict:
    import time
    conn = get_connection()
    try:
        cur = _execute(conn,
            """SELECT symbol, name, signal, confidence, model_horizon,
                      expected_return_pct, buy_price, target_price, stop_loss,
                      risk_reward, sentiment, model_accuracy, model_precision,
                      model_name, generated_at, generated_date
               FROM trade_signals
               WHERE generated_date = (SELECT MAX(generated_date) FROM trade_signals)
                 AND is_active = TRUE
               ORDER BY confidence DESC""", ())
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
    finally:
        release_connection(conn)

    now_utc = datetime.now(timezone.utc)
    out = []
    latest_gen_at = None
    latest_date   = None
    for raw in rows:
        r = dict(zip(cols, raw))
        raw_signal = r.get("signal", "HOLD")
        signal     = "BUY" if "BUY" in raw_signal else "SELL" if "SELL" in raw_signal else "HOLD"
        horizon_l  = r.get("model_horizon", "")
        horizon    = _HORIZON_SHORT.get(horizon_l, horizon_l)

        gen_at = r.get("generated_at")
        if gen_at and latest_gen_at is None:
            latest_gen_at = str(gen_at)
        if r.get("generated_date") and latest_date is None:
            latest_date = r["generated_date"]

        updated_min = 0
        if gen_at:
            try:
                dt = gen_at if hasattr(gen_at, "tzinfo") else datetime.fromisoformat(str(gen_at).replace(" ", "T").replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                updated_min = max(0, int((now_utc - dt).total_seconds() / 60))
            except Exception:
                pass

        sent_val = 0.0
        try:
            s = json.loads(r.get("sentiment") or "{}")
            sent_val = float(s.get("sent_stock") or s.get("mkt_sentiment") or 0)
        except Exception:
            pass

        out.append({
            "symbol":       r.get("symbol", ""),
            "name":         r.get("name", ""),
            "sector":       _SECTOR_MAP.get(r.get("symbol", ""), "Unknown"),
            "signal":       signal,
            "raw_signal":   raw_signal,
            "confidence":   round(float(r.get("confidence") or 0)),
            "horizon":      horizon,
            "horizon_long": horizon_l,
            "expReturn":    r.get("expected_return_pct"),
            "buy_price":    r.get("buy_price"),
            "target_price": r.get("target_price"),
            "stop_loss":    r.get("stop_loss"),
            "risk_reward":  r.get("risk_reward"),
            "sentiment":    round(sent_val, 4),
            "accuracy":     r.get("model_accuracy"),
            "precision":    r.get("model_precision"),
            "model_name":   r.get("model_name"),
            "updatedMin":   updated_min,
            "generated_at": str(gen_at or ""),
        })

    payload = {
        "count":        len(out),
        "generated_at": latest_gen_at,
        "total_stocks": len({r["symbol"] for r in out}),
        "signals":      out,
    }
    _cache["date"]    = str(latest_date)
    _cache["payload"] = payload
    _cache["ts"]      = time.monotonic()
    return payload


@router.get("/signals/all")
async def get_all_signals():
    """Return every signal across all horizons as a flat list — powers the AI Signals page."""
    import time
    # Serve from cache if fresh (< 5 min) and date hasn't changed
    if _cache["payload"] and (time.monotonic() - _cache["ts"]) < _CACHE_TTL:
        return _cache["payload"]

    # Check if DB has a newer date before doing a full fetch
    try:
        conn = get_connection()
        try:
            cur = _execute(conn, "SELECT MAX(generated_date) FROM trade_signals", ())
            row = cur.fetchone()
            db_date = str(row[0]) if row and row[0] else None
        finally:
            release_connection(conn)
    except Exception as e:
        logger.error(f"signals/all DB check failed: {e}")
        if _cache["payload"]:
            return _cache["payload"]
        raise HTTPException(status_code=500, detail="Failed to load signals")

    if _cache["payload"] and db_date == _cache["date"] and (time.monotonic() - _cache["ts"]) < _CACHE_TTL:
        return _cache["payload"]

    try:
        return _build_signals_payload()
    except Exception as e:
        logger.error(f"signals/all build failed: {e}")
        if _cache["payload"]:
            return _cache["payload"]
        raise HTTPException(status_code=500, detail="Failed to load signals")


@router.get("/signals/top-buys")
async def top_buys(
    limit: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Get top stocks with the strongest BUY signal today.

    Returns stocks sorted by signal confidence, including both
    "BUY" and "STRONG BUY" signals.

    Args:
        limit: Maximum number of results (default 10, max 50)
    """
    try:
        signals = get_top_signals(signal_type="BUY", limit=limit)

        if not signals:
            return {
                "count": 0,
                "signals": [],
                "message": "No BUY signals found. Run: python main.py collect",
            }

        return {
            "count": len(signals),
            "signals": signals,
        }

    except Exception as e:
        logger.error(f"Error fetching top buys: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@router.get("/signals/top-sells")
async def top_sells(
    limit: int = Query(default=10, ge=1, le=50, description="Number of results"),
):
    """
    Get top stocks with the strongest SELL signal today.

    Returns stocks sorted by signal confidence, including both
    "SELL" and "STRONG SELL" signals.

    Args:
        limit: Maximum number of results (default 10, max 50)
    """
    try:
        signals = get_top_signals(signal_type="SELL", limit=limit)

        if not signals:
            return {
                "count": 0,
                "signals": [],
                "message": "No SELL signals found. Run: python main.py collect",
            }

        return {
            "count": len(signals),
            "signals": signals,
        }

    except Exception as e:
        logger.error(f"Error fetching top sells: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


_SIGNAL_REFRESH_COOLDOWN_MINUTES = 30


@router.post("/signals/refresh")
@limiter.limit("3/hour")
async def refresh_signals(request: Request, background_tasks: BackgroundTasks, user=Depends(_get_current_user)):
    """Trigger async regeneration of all trade signals from stored ML models.

    Rate-limited as defense-in-depth on top of the cooldown check below (audit M13/M15)."""
    # Audit M13: any authenticated user could previously trigger this
    # repeatedly, queuing concurrent expensive (~480-model) background jobs
    # with no limit. A cooldown based on the most recent generated_at makes
    # repeated calls within the window a no-op rather than piling up work.
    from database.db import get_connection, release_connection, _execute
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT MAX(generated_at) FROM trade_signals").fetchone()
    finally:
        release_connection(conn)

    last_generated_at = row[0] if row else None
    if last_generated_at:
        age_minutes = (datetime.now(timezone.utc) - last_generated_at.replace(tzinfo=timezone.utc)).total_seconds() / 60
        if age_minutes < _SIGNAL_REFRESH_COOLDOWN_MINUTES:
            raise HTTPException(
                status_code=429,
                detail=f"Signals were refreshed {age_minutes:.0f} min ago — please wait "
                       f"{_SIGNAL_REFRESH_COOLDOWN_MINUTES - age_minutes:.0f} more minutes.",
            )

    def _run():
        try:
            from scripts.generate_trades import generate_signals
            generate_signals()
        except Exception as e:
            logger.error(f"Background signal refresh failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "ok", "message": "Signal refresh started in background"}
