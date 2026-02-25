"""
Nifty 500 AI — Portfolio Management API Routes

AI-driven portfolio creation, sector allocation, and stock picking.
"""
import json
import os
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import libsql_experimental as libsql

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

DB_PATH = "nifty500.db"
TOKENS_PATH = "data/angel_tokens.json"
SIGNALS_PATH = "data/trade_signals_latest.json"


# ==========================================
# Pydantic Models
# ==========================================

class PortfolioCreate(BaseModel):
    name: str
    investment_amount: float
    time_horizon: str  # "short", "medium", "long"
    risk_profile: str = "moderate"  # "conservative", "moderate", "aggressive"


class SectorUpdate(BaseModel):
    sectors: List[dict]  # [{"sector": "IT", "allocation_pct": 20}, ...]


# ==========================================
# Helper: Load signals + token data
# ==========================================

def load_signals():
    if os.path.exists(SIGNALS_PATH):
        with open(SIGNALS_PATH) as f:
            return json.load(f)
    return {"trades": [], "actionable_trades": [], "avoid_list": [], "hold_list": []}


def load_tokens():
    with open(TOKENS_PATH) as f:
        return json.load(f)


def get_sector_map():
    """Map symbol → sector from angel_tokens.json"""
    tokens = load_tokens()
    return {f"{sym}.NS": info.get("sector", "Unknown") for sym, info in tokens.items()}


def get_conn():
    return libsql.connect(DB_PATH)


# ==========================================
# AI Allocation Engine
# ==========================================

def ai_allocate_sectors(signals_data, time_horizon, risk_profile):
    """
    Score sectors based on ML signal confidence and allocate percentages.
    
    - Aggressive: heavier on sectors with STRONG BUY signals
    - Conservative: more diversified, caps single sector at 15%
    - Moderate: balanced approach
    """
    sector_map = get_sector_map()
    
    # All trades from signals
    all_trades = signals_data.get("actionable_trades", []) + signals_data.get("hold_list", [])
    
    # Score each sector
    sector_scores = {}
    sector_stocks = {}
    
    for trade in all_trades:
        sym = trade["symbol"]
        sector = sector_map.get(sym, "Unknown")
        conf = trade.get("confidence", 50) / 100.0
        signal = trade.get("signal", "HOLD")
        
        # Weighted score: STRONG BUY = 2x, BUY = 1.5x, HOLD = 0.5x
        weight = {"STRONG BUY": 2.0, "BUY": 1.5, "HOLD": 0.5}.get(signal, 0.3)
        score = conf * weight
        
        if sector not in sector_scores:
            sector_scores[sector] = 0
            sector_stocks[sector] = []
        sector_scores[sector] += score
        sector_stocks[sector].append(trade)
    
    if not sector_scores:
        return [], {}
    
    # Normalize to percentages
    total_score = sum(sector_scores.values())
    allocations = {}
    
    for sector, score in sector_scores.items():
        pct = (score / total_score) * 100
        allocations[sector] = round(pct, 1)
    
    # Apply risk profile caps
    max_sector_pct = {"conservative": 15, "moderate": 25, "aggressive": 40}.get(risk_profile, 25)
    min_sectors = {"conservative": 6, "moderate": 4, "aggressive": 3}.get(risk_profile, 4)
    
    # Cap and redistribute
    capped = {}
    overflow = 0
    for sector, pct in sorted(allocations.items(), key=lambda x: -x[1]):
        if pct > max_sector_pct:
            overflow += pct - max_sector_pct
            capped[sector] = max_sector_pct
        else:
            capped[sector] = pct
    
    # Redistribute overflow to smaller sectors
    if overflow > 0:
        small_sectors = [s for s, p in capped.items() if p < max_sector_pct]
        if small_sectors:
            extra_each = overflow / len(small_sectors)
            for s in small_sectors:
                capped[s] = round(capped[s] + extra_each, 1)
    
    # Sort by allocation
    sorted_allocs = sorted(capped.items(), key=lambda x: -x[1])
    
    return sorted_allocs, sector_stocks


def pick_stocks_for_portfolio(sector_stocks, allocations, investment_amount, time_horizon):
    """Pick best stocks per sector based on confidence and risk:reward."""
    picks = []
    
    # Horizon filter: prefer models matching the time horizon
    horizon_pref = {
        "short": ["1 Week", "2 Weeks", "1 Month"],
        "medium": ["1 Month", "2 Months", "3 Months"],
        "long": ["3 Months", "6 Months"],
    }.get(time_horizon, ["2 Months", "3 Months"])
    
    for sector, alloc_pct in allocations:
        sector_amount = investment_amount * (alloc_pct / 100)
        stocks = sector_stocks.get(sector, [])
        
        # Sort by confidence (prioritize matching horizons)
        def sort_key(t):
            horizon_match = 1 if t.get("model", {}).get("horizon", "") in horizon_pref else 0
            return (horizon_match, t.get("confidence", 0))
        
        stocks.sort(key=sort_key, reverse=True)
        
        # Pick top 2-3 stocks per sector
        max_picks = min(3, len(stocks))
        for stock in stocks[:max_picks]:
            price = stock.get("price", {}).get("current") or stock.get("trade", {}).get("buy_price", 0)
            if price and price > 0:
                per_stock_amount = sector_amount / max_picks
                qty = int(per_stock_amount / price)
                if qty < 1:
                    qty = 1
                
                picks.append({
                    "symbol": stock["symbol"],
                    "sector": sector,
                    "signal": stock.get("signal", "HOLD"),
                    "confidence": stock.get("confidence", 50),
                    "buy_price": stock.get("trade", {}).get("buy_price") or price,
                    "target_price": stock.get("trade", {}).get("target_price"),
                    "stop_loss": stock.get("trade", {}).get("stop_loss"),
                    "allocated_amount": round(per_stock_amount, 2),
                    "quantity": qty,
                })
    
    return picks


# ==========================================
# API Endpoints
# ==========================================

@router.get("/sectors")
async def get_all_sectors():
    """List all 21 available sectors with stock counts and signal summary."""
    sector_map = get_sector_map()
    signals = load_signals()
    
    # Count stocks per sector
    sector_counts = {}
    for sym, sector in sector_map.items():
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
    
    # Count signals per sector
    sector_signals = {}
    all_trades = (signals.get("actionable_trades", []) + 
                  signals.get("hold_list", []) + 
                  signals.get("avoid_list", []))
    
    for trade in all_trades:
        sector = sector_map.get(trade["symbol"], "Unknown")
        if sector not in sector_signals:
            sector_signals[sector] = {"STRONG BUY": 0, "BUY": 0, "HOLD": 0, "SELL": 0, "STRONG SELL": 0}
        sig = trade.get("signal", "HOLD")
        if sig in sector_signals[sector]:
            sector_signals[sector][sig] += 1
    
    result = []
    for sector, count in sorted(sector_counts.items(), key=lambda x: -x[1]):
        result.append({
            "sector": sector,
            "total_stocks": count,
            "signals": sector_signals.get(sector, {}),
        })
    
    return {"data": result, "total_sectors": len(result)}


@router.post("/create")
async def create_portfolio(body: PortfolioCreate):
    """Create a portfolio with AI-suggested sector allocation."""
    signals = load_signals()
    
    # AI allocation
    allocations, sector_stocks = ai_allocate_sectors(
        signals, body.time_horizon, body.risk_profile
    )
    
    if not allocations:
        raise HTTPException(400, "No signals available to build portfolio")
    
    # Pick stocks
    picks = pick_stocks_for_portfolio(
        sector_stocks, allocations, body.investment_amount, body.time_horizon
    )
    
    # Save to DB
    conn = get_conn()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO portfolios (name, investment_amount, time_horizon, risk_profile) VALUES (?, ?, ?, ?)",
            (body.name, body.investment_amount, body.time_horizon, body.risk_profile)
        )
        conn.commit()
        portfolio_id = cursor.execute("SELECT last_insert_rowid()").fetchone()[0]
        
        # Save sector allocations
        for sector, pct in allocations:
            num = len([p for p in picks if p["sector"] == sector])
            cursor.execute(
                "INSERT INTO portfolio_sectors (portfolio_id, sector, allocation_pct, ai_suggested_pct, num_stocks) VALUES (?, ?, ?, ?, ?)",
                (portfolio_id, sector, pct, pct, num)
            )
        
        # Save stock picks
        for pick in picks:
            cursor.execute(
                """INSERT INTO portfolio_stocks 
                (portfolio_id, symbol, sector, signal, confidence, buy_price, target_price, stop_loss, allocated_amount, quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (portfolio_id, pick["symbol"], pick["sector"], pick["signal"],
                 pick["confidence"], pick["buy_price"], pick["target_price"],
                 pick["stop_loss"], pick["allocated_amount"], pick["quantity"])
            )
        
        conn.commit()
    finally:
        conn.close()
    
    return {
        "data": {
            "portfolio_id": portfolio_id,
            "name": body.name,
            "investment_amount": body.investment_amount,
            "time_horizon": body.time_horizon,
            "risk_profile": body.risk_profile,
            "sectors": [{"sector": s, "allocation_pct": p, "num_stocks": len([x for x in picks if x["sector"] == s])} for s, p in allocations],
            "stocks": picks,
            "total_stocks": len(picks),
        }
    }


@router.get("/{portfolio_id}")
async def get_portfolio(portfolio_id: int):
    """Get portfolio details with sectors and stocks."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Portfolio not found")
        
        cols = ["id", "name", "investment_amount", "time_horizon", "risk_profile", "created_at", "updated_at"]
        portfolio = dict(zip(cols, row))
        
        # Sectors
        sector_rows = conn.execute(
            "SELECT sector, allocation_pct, ai_suggested_pct, num_stocks FROM portfolio_sectors WHERE portfolio_id = ?",
            (portfolio_id,)
        ).fetchall()
        portfolio["sectors"] = [
            {"sector": r[0], "allocation_pct": r[1], "ai_suggested_pct": r[2], "num_stocks": r[3]}
            for r in sector_rows
        ]
        
        # Stocks
        stock_rows = conn.execute(
            "SELECT symbol, sector, signal, confidence, buy_price, target_price, stop_loss, allocated_amount, quantity, status FROM portfolio_stocks WHERE portfolio_id = ?",
            (portfolio_id,)
        ).fetchall()
        stock_cols = ["symbol", "sector", "signal", "confidence", "buy_price", "target_price", "stop_loss", "allocated_amount", "quantity", "status"]
        portfolio["stocks"] = [dict(zip(stock_cols, r)) for r in stock_rows]
        portfolio["total_stocks"] = len(stock_rows)
        
        return {"data": portfolio}
    finally:
        conn.close()


@router.get("")
async def list_portfolios():
    """List all portfolios."""
    conn = get_conn()
    try:
        rows = conn.execute("SELECT id, name, investment_amount, time_horizon, risk_profile, created_at FROM portfolios ORDER BY created_at DESC").fetchall()
        cols = ["id", "name", "investment_amount", "time_horizon", "risk_profile", "created_at"]
        portfolios = [dict(zip(cols, r)) for r in rows]
        return {"data": portfolios, "total": len(portfolios)}
    finally:
        conn.close()


@router.put("/{portfolio_id}/sectors")
async def update_sectors(portfolio_id: int, body: SectorUpdate):
    """Update sector allocations (user customization)."""
    conn = get_conn()
    try:
        # Verify portfolio exists
        row = conn.execute("SELECT id FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Portfolio not found")
        
        # Validate total = 100%
        total = sum(s["allocation_pct"] for s in body.sectors)
        if abs(total - 100) > 1:
            raise HTTPException(400, f"Sector allocations must sum to 100% (got {total}%)")
        
        # Update each sector
        for sector_data in body.sectors:
            conn.execute(
                "UPDATE portfolio_sectors SET allocation_pct = ? WHERE portfolio_id = ? AND sector = ?",
                (sector_data["allocation_pct"], portfolio_id, sector_data["sector"])
            )
        
        conn.execute("UPDATE portfolios SET updated_at = datetime('now') WHERE id = ?", (portfolio_id,))
        conn.commit()
        
        return {"data": {"message": "Sectors updated", "portfolio_id": portfolio_id}}
    finally:
        conn.close()


@router.post("/{portfolio_id}/rebalance")
async def rebalance_portfolio(portfolio_id: int):
    """Re-run AI allocation with current signals."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT * FROM portfolios WHERE id = ?", (portfolio_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Portfolio not found")
        
        cols = ["id", "name", "investment_amount", "time_horizon", "risk_profile", "created_at", "updated_at"]
        portfolio = dict(zip(cols, row))
        
        signals = load_signals()
        allocations, sector_stocks = ai_allocate_sectors(
            signals, portfolio["time_horizon"], portfolio["risk_profile"]
        )
        picks = pick_stocks_for_portfolio(
            sector_stocks, allocations, portfolio["investment_amount"], portfolio["time_horizon"]
        )
        
        # Clear old and insert new
        conn.execute("DELETE FROM portfolio_sectors WHERE portfolio_id = ?", (portfolio_id,))
        conn.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = ?", (portfolio_id,))
        
        for sector, pct in allocations:
            num = len([p for p in picks if p["sector"] == sector])
            conn.execute(
                "INSERT INTO portfolio_sectors (portfolio_id, sector, allocation_pct, ai_suggested_pct, num_stocks) VALUES (?, ?, ?, ?, ?)",
                (portfolio_id, sector, pct, pct, num)
            )
        
        for pick in picks:
            conn.execute(
                """INSERT INTO portfolio_stocks 
                (portfolio_id, symbol, sector, signal, confidence, buy_price, target_price, stop_loss, allocated_amount, quantity)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (portfolio_id, pick["symbol"], pick["sector"], pick["signal"],
                 pick["confidence"], pick["buy_price"], pick["target_price"],
                 pick["stop_loss"], pick["allocated_amount"], pick["quantity"])
            )
        
        conn.execute("UPDATE portfolios SET updated_at = datetime('now') WHERE id = ?", (portfolio_id,))
        conn.commit()
        
        return {"data": {
            "portfolio_id": portfolio_id,
            "sectors": [{"sector": s, "allocation_pct": p} for s, p in allocations],
            "stocks": picks,
            "rebalanced_at": datetime.now().isoformat(),
        }}
    finally:
        conn.close()


@router.delete("/{portfolio_id}")
async def delete_portfolio(portfolio_id: int):
    """Delete a portfolio."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = ?", (portfolio_id,))
        conn.execute("DELETE FROM portfolio_sectors WHERE portfolio_id = ?", (portfolio_id,))
        conn.execute("DELETE FROM portfolios WHERE id = ?", (portfolio_id,))
        conn.commit()
        return {"data": {"message": "Portfolio deleted", "portfolio_id": portfolio_id}}
    finally:
        conn.close()
