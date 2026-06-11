"""
Nifty 500 AI — Trade Signals API Routes (Paginated)

GET /api/signals/latest?page=0&size=25&sort=confidence&order=desc&globalFilter=TCS&filters=[...]
"""
import json
from fastapi import APIRouter, Query
from typing import Optional

from database.db import get_trade_signals_formatted, get_signal_history
from api.schemas import PaginatedSignalsOut, SignalForStockOut, SignalHistoryOut

router = APIRouter(prefix="/api/signals", tags=["Signals"])


def _paginate_signals(data: list, page: int, size: int, sort: str, order_dir: str,
                      global_filter: str, filters: str) -> dict:
    """Apply server-side filtering, sorting, and pagination to signal list."""
    result = list(data)

    # Global filter
    if global_filter:
        q = global_filter.lower()
        result = [s for s in result if (
            q in str(s.get("symbol", "")).lower() or
            q in str(s.get("name", "")).lower() or
            q in str(s.get("signal", "")).lower()
        )]

    # Column filters
    if filters:
        try:
            col_filters = json.loads(filters)
            for cf in col_filters:
                col_id = cf.get("id", "")
                val = str(cf.get("value", "")).lower()
                if val and col_id:
                    # Handle nested fields like trade.buy_price
                    if "." in col_id:
                        parts = col_id.split(".")
                        result = [s for s in result if val in str(_nested_get(s, parts)).lower()]
                    else:
                        result = [s for s in result if val in str(s.get(col_id, "")).lower()]
        except json.JSONDecodeError:
            pass

    # Sorting
    sort_fields = ("symbol", "name", "signal", "confidence", "buy_price",
                   "target_price", "stop_loss", "risk_reward", "expected_return_pct")
    if sort and sort in sort_fields:
        reverse = order_dir == "desc"
        result.sort(key=lambda s: (s.get(sort) is None, s.get(sort, 0)), reverse=reverse)

    total = len(result)
    start = page * size
    end = start + size

    return {
        "data": result[start:end],
        "total": total,
        "page": page,
        "size": size,
    }


def _nested_get(d: dict, keys: list):
    """Get nested dict value from key path like ['trade', 'buy_price']."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, "")
        else:
            return ""
    return d


@router.get("/latest", response_model=PaginatedSignalsOut)
async def get_latest_signals(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=500),
    sort: Optional[str] = Query(default="confidence"),
    order: Optional[str] = Query(default="desc"),
    globalFilter: Optional[str] = Query(default=None),
    filters: Optional[str] = Query(default=None),
):
    """Get the most recent trade signals with pagination."""
    raw = get_trade_signals_formatted()
    all_trades = raw.get("trades", raw.get("data", []))
    if isinstance(raw, dict) and "trades" not in raw and "data" not in raw:
        # If it returns the full dict, get all signal items
        all_trades = []
        for key in ("actionable_trades", "avoid_list", "hold_list"):
            all_trades.extend(raw.get(key, []))

    result = _paginate_signals(all_trades, page, size, sort, order, globalFilter, filters)
    result["summary"] = raw.get("summary", {})
    return result


@router.get("/stock/{symbol}", response_model=SignalForStockOut)
async def get_signal_for_stock(symbol: str):
    """Get the latest trade signal for a specific stock symbol."""
    raw = get_trade_signals_formatted()
    all_trades = raw.get("trades", raw.get("data", []))

    sym_clean = symbol.upper()
    match = next(
        (s for s in all_trades if
         s.get("symbol", "").upper() == sym_clean or
         s.get("symbol", "").upper().replace(".NS", "") == sym_clean.replace(".NS", "")),
        None
    )
    if not match:
        return {"signal": None}
    return {"signal": match}


@router.get("/history", response_model=SignalHistoryOut)
async def get_signal_history_endpoint():
    """Get all historical signal runs grouped by date."""
    history = get_signal_history(limit=30)
    return {"data": history, "total": len(history)}


@router.get("/actionable", response_model=PaginatedSignalsOut)
async def get_actionable_signals(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=500),
    sort: Optional[str] = Query(default="confidence"),
    order: Optional[str] = Query(default="desc"),
    globalFilter: Optional[str] = Query(default=None),
    filters: Optional[str] = Query(default=None),
):
    """Get only STRONG BUY and BUY signals with pagination."""
    raw = get_trade_signals_formatted(signal_filter=["STRONG BUY", "BUY"])
    all_trades = raw.get("actionable_trades", [])
    return _paginate_signals(all_trades, page, size, sort, order, globalFilter, filters)


@router.get("/avoid", response_model=PaginatedSignalsOut)
async def get_avoid_signals(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=25, ge=1, le=500),
    sort: Optional[str] = Query(default="confidence"),
    order: Optional[str] = Query(default="desc"),
    globalFilter: Optional[str] = Query(default=None),
    filters: Optional[str] = Query(default=None),
):
    """Get SELL and STRONG SELL signals with pagination."""
    raw = get_trade_signals_formatted(signal_filter=["SELL", "STRONG SELL"])
    all_trades = raw.get("avoid_list", [])
    return _paginate_signals(all_trades, page, size, sort, order, globalFilter, filters)
