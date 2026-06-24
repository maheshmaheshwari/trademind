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

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from database.db import get_top_signals
from api.routes.trading import get_current_user as _get_current_user

logger = logging.getLogger(__name__)

router = APIRouter()

_SIGNALS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "trade_signals_latest.json",
)

_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "angel_tokens.json",
)

_HORIZON_SHORT = {
    "1 Week": "1W", "2 Weeks": "2W", "1 Month": "1M",
    "2 Months": "2M", "3 Months": "3M", "6 Months": "6M",
}

def _load_sector_map() -> dict:
    try:
        with open(_TOKENS_FILE) as f:
            tokens = json.load(f)
        return {f"{sym}.NS": info.get("sector", "Unknown") for sym, info in tokens.items()}
    except Exception:
        return {}


@router.get("/signals/all")
async def get_all_signals():
    """Return every signal across all horizons as a flat list — powers the AI Signals page."""
    try:
        with open(_SIGNALS_FILE) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {"count": 0, "signals": [], "generated_at": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load signals: {e}")

    sector_map = _load_sector_map()
    now_utc    = datetime.now(timezone.utc)

    all_cats = (
        raw.get("actionable_trades", []) +
        raw.get("avoid_list", []) +
        raw.get("hold_list", [])
    )

    out = []
    for t in all_cats:
        sym        = t.get("symbol", "")
        raw_signal = t.get("signal", "HOLD")
        signal     = "BUY" if "BUY" in raw_signal else "SELL" if "SELL" in raw_signal else "HOLD"

        model      = t.get("model") or {}
        trade      = t.get("trade") or {}
        sentiment  = t.get("sentiment") or {}
        horizon_l  = model.get("horizon", "")
        horizon    = _HORIZON_SHORT.get(horizon_l, horizon_l)

        updated_min = 0
        gen_at = t.get("generated_at")
        if gen_at:
            try:
                dt = datetime.fromisoformat(str(gen_at).replace(" ", "T").replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                updated_min = max(0, int((now_utc - dt).total_seconds() / 60))
            except Exception:
                pass

        out.append({
            "symbol":       sym,
            "name":         t.get("name", sym.replace(".NS", "")),
            "sector":       sector_map.get(sym, "Unknown"),
            "signal":       signal,
            "raw_signal":   raw_signal,
            "confidence":   round(float(t.get("confidence") or 0)),
            "horizon":      horizon,
            "horizon_long": horizon_l,
            "expReturn":    trade.get("expected_return_pct"),
            "buy_price":    trade.get("buy_price"),
            "target_price": trade.get("target_price"),
            "stop_loss":    trade.get("stop_loss"),
            "risk_reward":  trade.get("risk_reward"),
            "sentiment":    round(float(sentiment.get("sent_stock") or sentiment.get("mkt_sentiment") or 0), 4),
            "accuracy":     model.get("accuracy"),
            "precision":    model.get("precision"),
            "model_name":   model.get("name"),
            "updatedMin":   updated_min,
            "generated_at": t.get("generated_at"),
        })

    out.sort(key=lambda x: x["confidence"], reverse=True)

    return {
        "count":        len(out),
        "generated_at": raw.get("generated_at"),
        "total_stocks": raw.get("total_stocks"),
        "signals":      out,
    }


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


@router.post("/signals/refresh")
async def refresh_signals(background_tasks: BackgroundTasks, user=Depends(_get_current_user)):
    """Trigger async regeneration of all trade signals from stored ML models."""
    def _run():
        try:
            from generate_trades import generate_signals
            generate_signals()
        except Exception as e:
            logger.error(f"Background signal refresh failed: {e}")

    background_tasks.add_task(_run)
    return {"status": "ok", "message": "Signal refresh started in background"}
