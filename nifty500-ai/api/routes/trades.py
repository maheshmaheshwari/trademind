"""
Nifty 500 AI — Trade Signals API Routes

Serves the latest trade signals and trade history via REST API.
"""
import json
import os
from fastapi import APIRouter

router = APIRouter(prefix="/api/signals", tags=["Signals"])

SIGNALS_LATEST = "data/trade_signals_latest.json"
SIGNALS_HISTORY = "data/trade_history.json"


@router.get("/latest")
async def get_latest_signals():
    """Get the most recent trade signals with buy/target/stop-loss."""
    if os.path.exists(SIGNALS_LATEST):
        with open(SIGNALS_LATEST) as f:
            return {"data": json.load(f)}
    return {"data": {"trades": [], "summary": {}}}


@router.get("/history")
async def get_signal_history():
    """Get all historical signal runs."""
    if os.path.exists(SIGNALS_HISTORY):
        with open(SIGNALS_HISTORY) as f:
            history = json.load(f)
        return {"data": history, "total": len(history)}
    return {"data": [], "total": 0}


@router.get("/actionable")
async def get_actionable_signals():
    """Get only STRONG BUY and BUY signals with trade details."""
    if os.path.exists(SIGNALS_LATEST):
        with open(SIGNALS_LATEST) as f:
            data = json.load(f)
        return {
            "data": data.get("actionable_trades", []),
            "total": len(data.get("actionable_trades", [])),
        }
    return {"data": [], "total": 0}


@router.get("/avoid")
async def get_avoid_signals():
    """Get SELL and STRONG SELL signals."""
    if os.path.exists(SIGNALS_LATEST):
        with open(SIGNALS_LATEST) as f:
            data = json.load(f)
        return {
            "data": data.get("avoid_list", []),
            "total": len(data.get("avoid_list", [])),
        }
    return {"data": [], "total": 0}
