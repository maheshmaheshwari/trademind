"""
TradeMind AI — Trading API Routes

Virtual paper trading + live trading endpoints.
Every user gets ₹10L virtual money. One-click signal execution
creates BUY + SL + Target bracket orders automatically.
"""
import json
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional
from trading.trading_engine import (
    create_user, get_user, get_user_by_username, _safe_user,
    execute_signal, get_positions, get_orders,
    square_off, square_off_all, get_portfolio_summary,
)
from trading.risk_manager import (
    check_order, get_risk_settings, update_risk_settings,
)
from trading.price_monitor import update_position_prices
from api.auth import hash_password, verify_password, create_token, decode_token

router = APIRouter(prefix="/api/trading", tags=["Trading"])


# ==========================================
# Request Models
# ==========================================

class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: Optional[str] = None
    email: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    display_name: Optional[str] = None


class ExecuteSignalRequest(BaseModel):
    user_id: int
    symbol: str
    name: Optional[str] = None
    investment_amount: float
    buy_price: float
    target_price: float
    stop_loss: float
    signal: Optional[str] = "BUY"
    confidence: Optional[float] = 0
    horizon: Optional[str] = "Unknown"
    max_safe_qty: Optional[int] = None
    mode: Optional[str] = "PAPER"  # "PAPER" or "LIVE" (Angel One GTT)


class RiskSettingsRequest(BaseModel):
    max_daily_loss: Optional[float] = None
    max_daily_trades: Optional[int] = None
    max_position_pct: Optional[float] = None
    auto_stop_loss: Optional[int] = None
    auto_target: Optional[int] = None


class SquareOffRequest(BaseModel):
    sell_price: Optional[float] = None


# ==========================================
# JWT Auth Dependency
# ==========================================

async def get_current_user(authorization: Optional[str] = Header(None)):
    """Extract user from JWT Bearer token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")
    token = authorization.split(" ", 1)[1]
    payload = decode_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user = get_user(payload["user_id"])
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# ==========================================
# Auth Endpoints
# ==========================================

@router.post("/register")
async def api_register(req: RegisterRequest):
    """Create a new account with hashed password. Returns JWT token."""
    if len(req.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")
    try:
        pw_hash = hash_password(req.password)
        user = create_user(req.username, pw_hash, req.display_name, req.email)
        safe = _safe_user(user)
        token = create_token(user["id"], user["username"])
        return {"status": "success", "user": safe, "token": token}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/login")
async def api_login(req: LoginRequest):
    """Login with username + password. Returns JWT token."""
    user = get_user_by_username(req.username)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    safe = _safe_user(user)
    token = create_token(user["id"], user["username"])
    return {"status": "success", "user": safe, "token": token}


@router.get("/me")
async def api_get_me(user=Depends(get_current_user)):
    """Get current user from JWT token."""
    return _safe_user(user)


# ==========================================
# Legacy User Endpoints (kept for compat)
# ==========================================

@router.post("/user")
async def api_create_user(req: CreateUserRequest):
    """Create a virtual trading account with ₹10,00,000 (legacy, use /register)."""
    try:
        pw_hash = hash_password(req.username)  # default password = username
        user = create_user(req.username, pw_hash, req.display_name)
        return {"status": "success", "user": _safe_user(user)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/user/{user_id}")
async def api_get_user(user_id: int):
    """Get user account details."""
    user = get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _safe_user(user)


@router.get("/user/by-username/{username}")
async def api_get_user_by_username(username: str):
    """Get user by username."""
    user = get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _safe_user(user)


# ==========================================
# Trade Execution
# ==========================================

@router.post("/execute-signal")
async def api_execute_signal(req: ExecuteSignalRequest, user=Depends(get_current_user)):
    """
    One-click trade: AI signal → auto bracket order (BUY + SL + TARGET).

    Runs 6 risk checks, then creates:
    1. BUY order (executed immediately)
    2. STOP_LOSS order (pending, auto-triggers)
    3. TARGET order (pending, auto-triggers)
    """
    from api.server import run_in_thread

    # Always use the authenticated user's ID — ignore any user_id in the request body
    auth_user_id = user["id"]

    # Calculate quantity for risk check
    quantity = int(req.investment_amount / req.buy_price) if req.buy_price > 0 else 0

    # B4: Run blocking risk check + order execution in thread pool — not blocking event loop
    approved, reason, checks = await run_in_thread(
        check_order,
        user_id=auth_user_id,
        symbol=req.symbol,
        investment_amount=req.investment_amount,
        quantity=quantity,
        max_safe_qty=req.max_safe_qty,
    )

    if not approved:
        return {
            "status": "rejected",
            "reason": reason,
            "risk_checks": checks,
        }

    # Execute the bracket order (blocking DB write — offloaded to thread pool)
    try:
        result = await run_in_thread(
            execute_signal,
            user_id=auth_user_id,
            symbol=req.symbol,
            name=req.name or req.symbol.replace(".NS", ""),
            investment_amount=req.investment_amount,
            buy_price=req.buy_price,
            target_price=req.target_price,
            stop_loss=req.stop_loss,
            signal=req.signal,
            confidence=req.confidence,
            horizon=req.horizon,
            mode=req.mode or "PAPER",
        )
        result["status"] = "executed"
        result["risk_checks"] = checks
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ==========================================
# Positions & Orders
# ==========================================

@router.get("/positions/{user_id}")
async def api_get_positions(
    user_id: int,
    page: int = 0, size: int = 25,
    sort: Optional[str] = None, order: Optional[str] = "asc",
    globalFilter: Optional[str] = None, filters: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get all open positions with unrealized P&L (paginated)."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    update_position_prices(user_id)
    positions = get_positions(user_id)

    # Global filter
    if globalFilter:
        q = globalFilter.lower()
        positions = [p for p in positions if (
            q in str(p.get("symbol", "")).lower() or
            q in str(p.get("name", "")).lower()
        )]

    # Column filters
    if filters:
        try:
            for cf in json.loads(filters):
                col_id = cf.get("id", "")
                val = str(cf.get("value", "")).lower()
                if val and col_id:
                    positions = [p for p in positions if val in str(p.get(col_id, "")).lower()]
        except (json.JSONDecodeError, TypeError):
            pass

    # Sort
    if sort:
        reverse = order == "desc"
        positions.sort(key=lambda p: (p.get(sort) is None, p.get(sort, 0)), reverse=reverse)

    total = len(positions)
    start = page * size
    paginated = positions[start:start + size]

    return {"user_id": user_id, "data": paginated, "positions": paginated,
            "total": total, "page": page, "size": size, "count": total}


@router.get("/orders/{user_id}")
async def api_get_orders(
    user_id: int, limit: int = 200,
    page: int = 0, size: int = 25,
    sort: Optional[str] = None, order: Optional[str] = "desc",
    globalFilter: Optional[str] = None, filters: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Get order history (paginated)."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    orders = get_orders(user_id, limit)

    # Global filter
    if globalFilter:
        q = globalFilter.lower()
        orders = [o for o in orders if (
            q in str(o.get("symbol", "")).lower() or
            q in str(o.get("purpose", "")).lower() or
            q in str(o.get("status", "")).lower()
        )]

    # Column filters
    if filters:
        try:
            for cf in json.loads(filters):
                col_id = cf.get("id", "")
                val = str(cf.get("value", "")).lower()
                if val and col_id:
                    orders = [o for o in orders if val in str(o.get(col_id, "")).lower()]
        except (json.JSONDecodeError, TypeError):
            pass

    # Sort
    if sort:
        reverse = order == "desc"
        orders.sort(key=lambda o: (o.get(sort) is None, o.get(sort, "")), reverse=reverse)

    total = len(orders)
    start = page * size
    paginated = orders[start:start + size]

    return {"user_id": user_id, "data": paginated, "orders": paginated,
            "total": total, "page": page, "size": size, "count": total}


@router.post("/square-off/{user_id}/{symbol}")
async def api_square_off(user_id: int, symbol: str, req: SquareOffRequest = None, user=Depends(get_current_user)):
    """Sell an entire position."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        sell_price = req.sell_price if req else None
        result = square_off(user_id, symbol, sell_price)
        return {"status": "success", **result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/square-off-all/{user_id}")
async def api_square_off_all(user_id: int, user=Depends(get_current_user)):
    """Emergency kill switch: sell ALL positions."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    result = square_off_all(user_id)
    return {"status": "success", **result}


# ==========================================
# Portfolio & Risk
# ==========================================

@router.get("/portfolio/{user_id}")
async def api_portfolio_summary(user_id: int, user=Depends(get_current_user)):
    """Full portfolio summary: balance, invested, P&L, win rate, positions."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        # Update prices before summary
        update_position_prices(user_id)
        summary = get_portfolio_summary(user_id)
        return summary
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/risk-settings/{user_id}")
async def api_get_risk_settings(user_id: int, user=Depends(get_current_user)):
    """Get risk management settings."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    settings = get_risk_settings(user_id)
    return settings


@router.put("/risk-settings/{user_id}")
async def api_update_risk_settings(user_id: int, req: RiskSettingsRequest, user=Depends(get_current_user)):
    """Update risk management settings."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    updates = {k: v for k, v in req.dict().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")
    settings = update_risk_settings(user_id, updates)
    return {"status": "updated", "settings": settings}


@router.get("/pnl/today/{user_id}")
async def api_today_pnl(user_id: int, user=Depends(get_current_user)):
    """Get today's realized P&L."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    from database.db import get_connection, _execute
    from datetime import datetime

    conn = get_connection()
    today = datetime.now().strftime("%Y-%m-%d")

    row = _execute(conn, """
        SELECT
            COALESCE(SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END), 0) as profit,
            COALESCE(SUM(CASE WHEN pnl < 0 THEN pnl ELSE 0 END), 0) as loss,
            COALESCE(SUM(pnl), 0) as net,
            COUNT(CASE WHEN pnl IS NOT NULL THEN 1 END) as trades
        FROM orders
        WHERE user_id = ? AND DATE(created_at) = ? AND order_purpose = 'SQUARE_OFF'
    """, (user_id, today)).fetchone()
    
    conn.close()
    
    return {
        "date": today,
        "profit": round(row[0], 2),
        "loss": round(row[1], 2),
        "net_pnl": round(row[2], 2),
        "today_pnl": round(row[2], 2),
        "today_pnl_pct": 0.0,
        "trades_closed": row[3],
    }


@router.get("/analytics/{user_id}")
async def api_user_analytics(user_id: int, user=Depends(get_current_user)):
    """
    Comprehensive user trading analytics:
    - Overall win/loss summary
    - P&L breakdown by signal type, horizon, confidence band
    - AI signal accuracy (did signals the user acted on turn out correct?)
    - Per-signal volume consumed
    - Best and worst trades
    """
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    from database.db import get_user_analytics
    return get_user_analytics(user_id)


@router.get("/analytics/{user_id}/volume")
async def api_user_signal_volume(user_id: int, user=Depends(get_current_user)):
    """Per-user breakdown of AI signal volume consumed."""
    if user["id"] != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    from database.db import get_user_signal_volume
    data = get_user_signal_volume(user_id)
    return {"data": data, "total": len(data), "user_id": user_id}
