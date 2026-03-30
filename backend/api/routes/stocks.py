"""
Nifty 500 AI — Stocks List API Route (Paginated)

GET /api/stocks?page=0&size=25&sort=symbol&order=asc&globalFilter=TCS&filters=[...]
"""

import json
import logging
import os

from fastapi import APIRouter, Query
from typing import Optional

from database.db import get_connection

logger = logging.getLogger(__name__)
router = APIRouter()

# Load sector data from angel_tokens.json
_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "angel_tokens.json"
)
_SECTOR_MAP = {}
try:
    with open(_TOKENS_FILE) as f:
        _tokens = json.load(f)
    for sym, info in _tokens.items():
        _SECTOR_MAP[f"{sym}.NS"] = {
            "name": info.get("name", sym),
            "sector": info.get("sector", "Unknown"),
        }
except FileNotFoundError:
    logger.warning("angel_tokens.json not found for sector mapping")


@router.get("/stocks")
async def get_all_stocks(
    search: Optional[str] = Query(default=None, description="Search by symbol or name (legacy)"),
    sector: Optional[str] = Query(default=None, description="Filter by sector (legacy)"),
    # MRT pagination params
    page: int = Query(default=0, ge=0, description="Page index (0-based)"),
    size: int = Query(default=25, ge=1, le=500, description="Page size"),
    sort: Optional[str] = Query(default=None, description="Sort column"),
    order: Optional[str] = Query(default="asc", description="Sort order: asc/desc"),
    globalFilter: Optional[str] = Query(default=None, description="Global search"),
    filters: Optional[str] = Query(default=None, description="Column filters JSON"),
):
    conn = get_connection()
    try:
        # Get latest price for each stock
        rows = conn.execute("""
            SELECT p.symbol, p.open, p.high, p.low, p.close, p.volume, p.date
            FROM prices p
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date
                FROM prices
                WHERE interval = '1d'
                GROUP BY symbol
            ) latest ON p.symbol = latest.symbol AND p.date = latest.max_date AND p.interval = '1d'
            ORDER BY p.symbol
        """).fetchall()

        # Get previous day prices for change calculation
        prev_rows = conn.execute("""
            SELECT p.symbol, p.close
            FROM prices p
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date
                FROM prices
                WHERE interval = '1d' AND date < (SELECT MAX(date) FROM prices WHERE interval = '1d')
                GROUP BY symbol
            ) prev ON p.symbol = prev.symbol AND p.date = prev.max_date AND p.interval = '1d'
        """).fetchall()
        prev_close_map = {r[0]: r[1] for r in prev_rows}

        stocks = []
        for row in rows:
            symbol = row[0]
            close = row[4]
            prev_close = prev_close_map.get(symbol, close)
            change = close - prev_close if prev_close else 0
            change_pct = (change / prev_close * 100) if prev_close and prev_close > 0 else 0

            info = _SECTOR_MAP.get(symbol, {})
            name = info.get("name", symbol.replace(".NS", ""))
            sector_name = info.get("sector", "Unknown")

            stocks.append({
                "symbol": symbol,
                "name": name,
                "sector": sector_name,
                "open": row[1],
                "high": row[2],
                "low": row[3],
                "close": close,
                "volume": row[5],
                "date": row[6],
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": prev_close,
            })

        # ---- Filtering ----

        # Legacy search param (backwards compatible)
        if search:
            q = search.lower()
            stocks = [s for s in stocks if q in s["symbol"].lower() or q in s["name"].lower()]

        # Legacy sector param
        if sector:
            stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]

        # MRT Global filter
        if globalFilter:
            q = globalFilter.lower()
            stocks = [s for s in stocks if (
                q in s["symbol"].lower() or
                q in s["name"].lower() or
                q in s["sector"].lower()
            )]

        # MRT Column filters
        if filters:
            try:
                col_filters = json.loads(filters)
                for cf in col_filters:
                    col_id = cf.get("id", "")
                    val = str(cf.get("value", "")).lower()
                    if val:
                        stocks = [s for s in stocks if val in str(s.get(col_id, "")).lower()]
            except json.JSONDecodeError:
                pass

        # Get unique sectors (from filtered set)
        all_sectors = sorted(set(s["sector"] for s in stocks if s["sector"] != "Unknown"))

        # ---- Sorting ----
        if sort and sort in ("symbol", "name", "sector", "close", "change_pct", "volume", "open", "high", "low"):
            reverse = order == "desc"
            stocks.sort(key=lambda s: (s.get(sort) is None, s.get(sort, "")), reverse=reverse)

        # ---- Pagination ----
        total = len(stocks)
        start = page * size
        end = start + size
        paginated = stocks[start:end]

        return {
            "data": paginated,
            "total": total,
            "page": page,
            "size": size,
            "sectors": all_sectors,
            # Legacy compat
            "count": total,
            "stocks": paginated,
        }

    except Exception as e:
        logger.error(f"Error fetching stocks: {e}")
        return {"data": [], "total": 0, "page": 0, "size": size, "sectors": [], "error": str(e)}
    finally:
        conn.close()
