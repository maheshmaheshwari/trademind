"""
Nifty 500 AI — Stocks List API Route (Paginated)

GET /api/stocks?page=0&size=25&sort=symbol&order=asc&globalFilter=TCS&filters=[...]
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Query
from typing import Optional

from database.db import get_connection, release_connection, _execute, get_trade_signals_formatted
from api.schemas import PaginatedStocksOut

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Static maps loaded once at startup ────────────────────────────────────────

_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "..", "data", "angel_tokens.json"
)
_SECTOR_MAP: dict = {}
try:
    with open(_TOKENS_FILE) as f:
        _tokens = json.load(f)
    for sym, info in _tokens.items():
        _SECTOR_MAP[f"{sym}.NS"] = {
            "name":   info.get("name", sym),
            "sector": info.get("sector", "Unknown"),
        }
except FileNotFoundError:
    logger.warning("angel_tokens.json not found for sector mapping")

_HORIZON_SHORT = {
    "1 Week":    "1W",
    "2 Weeks":   "2W",
    "1 Month":   "1M",
    "2 Months":  "2M",
    "3 Months":  "3M",
    "6 Months":  "6M",
}

# ── Signal data helper ─────────────────────────────────────────────────────────

def _build_signal_map() -> dict:
    """
    Load trade_signals_latest.json and return a symbol-keyed dict with
    normalised signal fields ready to be merged into the stocks response.
    """
    try:
        raw = get_trade_signals_formatted()
        sig_map: dict = {}
        now_utc = datetime.now(timezone.utc)

        for category in ("actionable_trades", "avoid_list", "hold_list"):
            for t in raw.get(category, []):
                sym = t.get("symbol", "")
                if not sym:
                    continue

                raw_signal = t.get("signal", "HOLD")
                signal = "BUY" if "BUY" in raw_signal else "SELL" if "SELL" in raw_signal else "HOLD"

                horizon_long = (t.get("model") or {}).get("horizon", "")
                horizon = _HORIZON_SHORT.get(horizon_long, horizon_long)

                trade      = t.get("trade")      or {}
                sentiment  = t.get("sentiment")  or {}
                exp_ret    = trade.get("expected_return_pct")

                # minutes since signals were generated
                updated_min = 0
                gen_at = t.get("generated_at")
                if gen_at:
                    try:
                        if isinstance(gen_at, str):
                            gen_at = gen_at.replace(" ", "T")
                            if gen_at.endswith("+00:00"):
                                pass
                            elif "+" not in gen_at and not gen_at.endswith("Z"):
                                gen_at += "+00:00"
                        dt = datetime.fromisoformat(str(gen_at).replace("Z", "+00:00"))
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        updated_min = max(0, int((now_utc - dt).total_seconds() / 60))
                    except Exception:
                        updated_min = 0

                sent_val = (sentiment.get("sent_stock") or sentiment.get("mkt_sentiment") or 0)

                sig_map[sym] = {
                    "signal":               signal,
                    "confidence":           round(float(t.get("confidence") or 0)),
                    "horizon":              horizon,
                    "expected_return_pct":  exp_ret,
                    "expReturn":            exp_ret,      # alias for frontend Stock type
                    "sentiment":            round(float(sent_val), 4),
                    "updatedMin":           updated_min,
                    "target_price":         trade.get("target_price"),
                    "stop_loss":            trade.get("stop_loss"),
                }
        return sig_map
    except Exception as exc:
        logger.warning(f"Could not merge signal data into stocks: {exc}")
        return {}


# ── Route ──────────────────────────────────────────────────────────────────────

@router.get("/stocks", response_model=PaginatedStocksOut)
async def get_all_stocks(
    search:       Optional[str] = Query(default=None),
    sector:       Optional[str] = Query(default=None),
    page:         int           = Query(default=0,    ge=0),
    size:         int           = Query(default=25,   ge=1, le=500),
    sort:         Optional[str] = Query(default=None),
    order:        Optional[str] = Query(default="asc"),
    globalFilter: Optional[str] = Query(default=None),
    filters:      Optional[str] = Query(default=None),
):
    conn = get_connection()
    try:
        # ── Price data ────────────────────────────────────────────────────────
        rows = _execute(conn, """
            SELECT p.symbol, p.open, p.high, p.low, p.close, p.volume, p.date
            FROM prices p
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date
                FROM prices WHERE interval = '1d'
                GROUP BY symbol
            ) latest
              ON p.symbol = latest.symbol
             AND p.date   = latest.max_date
             AND p.interval = '1d'
            ORDER BY p.symbol
        """).fetchall()

        prev_rows = _execute(conn, """
            SELECT p.symbol, p.close
            FROM prices p
            INNER JOIN (
                SELECT symbol, MAX(date) as max_date
                FROM prices
                WHERE interval = '1d'
                  AND date < (SELECT MAX(date) FROM prices WHERE interval = '1d')
                GROUP BY symbol
            ) prev
              ON p.symbol = prev.symbol
             AND p.date   = prev.max_date
             AND p.interval = '1d'
        """).fetchall()
        prev_close_map = {r[0]: r[1] for r in prev_rows}

        # ── Signal data (merged per symbol) ───────────────────────────────────
        sig_map = _build_signal_map()

        stocks = []
        for row in rows:
            symbol      = row[0]
            close       = row[4]
            prev_close  = prev_close_map.get(symbol, close)
            change      = close - prev_close if prev_close else 0
            change_pct  = (change / prev_close * 100) if prev_close and prev_close > 0 else 0

            info        = _SECTOR_MAP.get(symbol, {})
            sig         = sig_map.get(symbol, {})

            stocks.append({
                "symbol":               symbol,
                "name":                 info.get("name", symbol.replace(".NS", "")),
                "sector":               info.get("sector", "Unknown"),
                "open":                 row[1],
                "high":                 row[2],
                "low":                  row[3],
                "close":                close,
                "price":                close,          # alias: Stock.price
                "volume":               row[5],
                "date":                 row[6],
                "change":               round(change, 2),
                "change_pct":           round(change_pct, 2),
                "prev_close":           prev_close,
                # Signal fields — null when no signal exists for this stock
                "signal":               sig.get("signal"),
                "confidence":           sig.get("confidence"),
                "horizon":              sig.get("horizon"),
                "expected_return_pct":  sig.get("expected_return_pct"),
                "expReturn":            sig.get("expReturn"),
                "sentiment":            sig.get("sentiment", 0),
                "updatedMin":           sig.get("updatedMin", 0),
                "target_price":         sig.get("target_price"),
                "stop_loss":            sig.get("stop_loss"),
            })

        # ── Filtering ─────────────────────────────────────────────────────────
        if search:
            q = search.lower()
            stocks = [s for s in stocks if q in s["symbol"].lower() or q in s["name"].lower()]

        if sector and sector != "All":
            stocks = [s for s in stocks if s["sector"].lower() == sector.lower()]

        if globalFilter:
            q = globalFilter.lower()
            stocks = [s for s in stocks if (
                q in s["symbol"].lower() or
                q in s["name"].lower() or
                q in s["sector"].lower()
            )]

        if filters:
            try:
                for cf in json.loads(filters):
                    col_id = cf.get("id", "")
                    val    = str(cf.get("value", "")).lower()
                    if val and col_id:
                        stocks = [s for s in stocks if val in str(s.get(col_id, "")).lower()]
            except json.JSONDecodeError:
                pass

        all_sectors = sorted(set(s["sector"] for s in stocks if s["sector"] != "Unknown"))

        # ── Sorting ───────────────────────────────────────────────────────────
        sortable = {
            "symbol", "name", "sector", "close", "price", "change_pct",
            "volume", "signal", "confidence", "horizon", "expected_return_pct",
            "expReturn", "sentiment",
        }
        if sort and sort in sortable:
            reverse = order == "desc"
            # Nulls always last regardless of sort direction
            stocks.sort(
                key=lambda s: (s.get(sort) is None, -(s.get(sort) or 0) if reverse else (s.get(sort) or 0))
            )

        # ── Pagination ────────────────────────────────────────────────────────
        total     = len(stocks)
        paginated = stocks[page * size: page * size + size]

        return {
            "data":    paginated,
            "total":   total,
            "page":    page,
            "size":    size,
            "sectors": all_sectors,
            "count":   total,
            "stocks":  paginated,
        }

    except Exception as e:
        logger.error(f"Error fetching stocks: {e}", exc_info=True)
        return {"data": [], "total": 0, "page": 0, "size": size, "sectors": [], "error": str(e)}
    finally:
        release_connection(conn)


# ── Stock Detail ───────────────────────────────────────────────────────────────

_HORIZON_SHORT_MAP = {
    "1 Week": "1W", "2 Weeks": "2W", "1 Month": "1M",
    "2 Months": "2M", "3 Months": "3M", "6 Months": "6M",
}

# In-memory cache for yfinance fundamentals {symbol: {"mcap": N, "pe": N, "ts": epoch}}
_FUNDAMENTALS_CACHE: dict = {}
_FUNDAMENTALS_TTL  = 3600   # 1 hour


def _get_fundamentals(sym: str) -> dict:
    """
    Fetch market cap (Cr) and trailing P/E from yfinance.
    Results are cached for 1 hour to avoid hammering the API.
    """
    import time
    cached = _FUNDAMENTALS_CACHE.get(sym)
    if cached and time.time() - cached["ts"] < _FUNDAMENTALS_TTL:
        return cached

    result = {"mcap": 0, "pe": 0, "ts": time.time()}
    try:
        import yfinance as yf
        info = yf.Ticker(sym).fast_info
        mcap_raw = getattr(info, "market_cap", None)
        if mcap_raw:
            result["mcap"] = round(mcap_raw / 1e7, 2)   # convert ₹ → Cr
        # fast_info doesn't have PE; fall back to info dict (slower)
        try:
            full = yf.Ticker(sym).info
            pe_raw = full.get("trailingPE") or full.get("forwardPE")
            if pe_raw:
                result["pe"] = round(float(pe_raw), 2)
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"yfinance fundamentals fetch failed for {sym}: {e}")

    _FUNDAMENTALS_CACHE[sym] = result
    return result


@router.get("/stocks/{symbol}")
async def get_stock_detail(symbol: str):
    """
    Full detail for a single stock — used by StockDrawer.

    Returns price, 52w hi/lo, spark, signal, all trade_signal fields,
    technical indicators, news, and yfinance fundamentals (mcap, pe).
    """
    sym = symbol.upper()
    conn = get_connection()
    try:
        now_utc = datetime.now(timezone.utc)

        # ── 1. Price history (last 365 days) ─────────────────────────────────
        price_rows = _execute(conn, """
            SELECT close, volume, date FROM prices
            WHERE symbol = ? AND interval = '1d'
              AND date >= CURRENT_DATE - INTERVAL '365 days'
            ORDER BY date ASC
        """, (sym,)).fetchall()

        if not price_rows:
            return {"data": None, "error": "No price data found"}

        closes     = [float(r[0]) for r in price_rows]
        high52     = round(max(closes), 2)
        low52      = round(min(closes), 2)
        spark      = closes[-60:]
        cur_price  = closes[-1]
        prev_price = closes[-2] if len(closes) >= 2 else closes[-1]
        change_pct = round((cur_price - prev_price) / prev_price * 100, 2) if prev_price else 0
        volume_m   = round(float(price_rows[-1][1] or 0) / 1_000_000, 2)

        # ── 2. Signal + all trade fields ──────────────────────────────────────
        sig_rows = _execute(conn, """
            SELECT signal, confidence, model_horizon, expected_return_pct,
                   sentiment, generated_at,
                   stop_loss, target_price, buy_price, risk_reward,
                   atr_14, atr_pct, avg_daily_volume, daily_turnover_cr,
                   liquidity, max_safe_qty, max_qty_per_user,
                   max_investment_per_user, min_qty,
                   model_name, model_accuracy, model_precision,
                   consumed_volume, recommended_volume
            FROM trade_signals
            WHERE symbol = ? AND is_active = TRUE
            ORDER BY generated_date DESC, confidence DESC
        """, (sym,)).fetchall()

        best_signal    = "HOLD"
        best_conf      = 0
        best_horizon   = ""
        exp_ret        = 0.0
        sentiment_val  = 0.0
        updated_min    = 0
        stop_loss      = None
        target_price   = None
        buy_price      = None
        risk_reward    = None
        atr_14         = None
        atr_pct        = None
        avg_daily_vol  = None
        daily_turnover = None
        liquidity      = None
        max_safe_qty   = None
        max_qty_user   = None
        consumed_vol   = 0
        recommended_vol = 0
        max_invest     = None
        min_qty        = None
        model_name     = None
        model_accuracy = None
        model_precision= None
        horizons: list = []
        seen_h: set    = set()

        for row in sig_rows:
            raw_sig = row[0] or "HOLD"
            sig     = "BUY" if "BUY" in raw_sig else "SELL" if "SELL" in raw_sig else "HOLD"
            conf    = round(float(row[1] or 0))
            h_long  = row[2] or ""
            h_short = _HORIZON_SHORT_MAP.get(h_long, h_long)

            if h_short not in seen_h:
                seen_h.add(h_short)
                horizons.append({"h": h_short, "sig": sig, "conf": conf})

            if conf > best_conf:
                best_signal    = sig
                best_conf      = conf
                best_horizon   = h_short
                exp_ret        = float(row[3] or 0)

                raw_sent = row[4] or 0
                try:
                    if isinstance(raw_sent, str):
                        d = json.loads(raw_sent)
                        raw_sent = d.get("sent_stock") or d.get("mkt_sentiment") or 0
                    sentiment_val = float(raw_sent)
                except Exception:
                    sentiment_val = 0.0

                gen_at = row[5]
                if gen_at:
                    try:
                        if hasattr(gen_at, "tzinfo") and gen_at.tzinfo:
                            updated_min = max(0, int((now_utc - gen_at).total_seconds() / 60))
                        else:
                            updated_min = max(0, int((now_utc - gen_at.replace(tzinfo=timezone.utc)).total_seconds() / 60))
                    except Exception:
                        updated_min = 0

                stop_loss       = float(row[6])  if row[6]  is not None else None
                target_price    = float(row[7])  if row[7]  is not None else None
                buy_price       = float(row[8])  if row[8]  is not None else None
                risk_reward     = float(row[9])  if row[9]  is not None else None
                atr_14          = float(row[10]) if row[10] is not None else None
                atr_pct         = float(row[11]) if row[11] is not None else None
                avg_daily_vol   = int(row[12])   if row[12] is not None else None
                daily_turnover  = float(row[13]) if row[13] is not None else None
                liquidity       = row[14]
                max_safe_qty    = int(row[15])   if row[15] is not None else None
                max_qty_user    = int(row[16])   if row[16] is not None else None
                max_invest      = float(row[17]) if row[17] is not None else None
                min_qty         = int(row[18])   if row[18] is not None else None
                model_name      = row[19]
                model_accuracy  = float(row[20]) if row[20] is not None else None
                model_precision = float(row[21]) if row[21] is not None else None
                consumed_vol    = int(row[22])   if row[22] is not None else 0
                recommended_vol = int(row[23])   if row[23] is not None else 0

        # ── 3. Technical indicators (latest row) ──────────────────────────────
        ti_row = _execute(conn, """
            SELECT rsi_14, macd, macd_signal, macd_hist,
                   bb_upper, bb_middle, bb_lower,
                   sma_20, sma_50, sma_200,
                   atr_14, adx_14,
                   support_1, support_2, resistance_1, resistance_2
            FROM technical_indicators
            WHERE symbol = ?
            ORDER BY date DESC LIMIT 1
        """, (sym,)).fetchone()

        tech = {}
        if ti_row:
            keys = ["rsi_14", "macd", "macd_signal", "macd_hist",
                    "bb_upper", "bb_middle", "bb_lower",
                    "sma_20", "sma_50", "sma_200",
                    "atr_14", "adx_14",
                    "support_1", "support_2", "resistance_1", "resistance_2"]
            tech = {k: (round(float(v), 2) if v is not None else None)
                    for k, v in zip(keys, ti_row)}

        # ── 4. Recent news ────────────────────────────────────────────────────
        news_rows = _execute(conn, """
            SELECT headline, source, published_at, sentiment
            FROM news_sentiment
            WHERE symbol = ? AND headline IS NOT NULL
            ORDER BY published_at DESC LIMIT 6
        """, (sym,)).fetchall()

        news = []
        for nr in news_rows:
            title    = nr[0] or ""
            src      = nr[1] or ""
            pub      = nr[2]
            sent_v   = float(nr[3] or 0)
            sent_lbl = "pos" if sent_v > 0.05 else "neg" if sent_v < -0.05 else "neu"
            time_str = pub.strftime("%b %d") if pub else ""
            news.append({"src": src, "time": time_str, "sent": sent_lbl, "title": title})

        # ── 5. Name + sector ──────────────────────────────────────────────────
        info   = _SECTOR_MAP.get(sym, {})
        name   = info.get("name",   sym.replace(".NS", ""))
        sector = info.get("sector", "Unknown")

        # ── 6. Fundamentals (mcap, pe) via yfinance with 1-hour cache ─────────
        fundamentals = _get_fundamentals(sym)

        return {
            "data": {
                # Core identity
                "symbol":    sym,
                "name":      name,
                "sector":    sector,
                # Price
                "price":     cur_price,
                "change":    change_pct,
                "high52":    high52,
                "low52":     low52,
                "volume":    volume_m,
                "spark":     spark,
                # Fundamentals
                "mcap":      fundamentals["mcap"],
                "pe":        fundamentals["pe"],
                # Signal
                "signal":    best_signal,
                "confidence":best_conf,
                "horizon":   best_horizon,
                "expReturn": exp_ret,
                "expected_return_pct": exp_ret,
                "sentiment": round(sentiment_val, 4),
                "updatedMin":updated_min,
                # Trade levels
                "stop_loss":    stop_loss,
                "target_price": target_price,
                "buy_price":    buy_price,
                "risk_reward":  risk_reward,
                # Volatility
                "atr_14":       atr_14,
                "atr_pct":      atr_pct,
                # Liquidity
                "avg_daily_volume":       avg_daily_vol,
                "daily_turnover_cr":      daily_turnover,
                "liquidity":              liquidity,
                "max_safe_qty":           max_safe_qty,
                "suggested_qty_per_user":        max_qty_user,
                "suggested_investment_per_user": max_invest,
                "min_qty":                min_qty,
                "consumed_volume":        consumed_vol,
                "recommended_volume":     recommended_vol,
                "remaining_volume":       max(0, (recommended_vol or 0) - (consumed_vol or 0)),
                # Model
                "model_name":      model_name,
                "model_accuracy":  model_accuracy,
                "model_precision": model_precision,
                # Technical indicators
                "technical": tech,
                # News & horizons
                "news":     news,
                "horizons": horizons,
            }
        }

    except Exception as e:
        logger.error(f"Error fetching stock detail for {sym}: {e}", exc_info=True)
        return {"data": None, "error": str(e)}
    finally:
        release_connection(conn)
