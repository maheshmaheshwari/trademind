"""
Signals API Routes

Endpoints for retrieving trading signals.
"""

from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from app.api.deps import ApiKey, DbSession
from app.models import Signal, Stock
from app.schemas.signal import (
    SignalFilter,
    SignalResponse,
    SignalsListResponse,
    SignalHistoryResponse,
    TopSignalsResponse,
)
from app.services.cache import get_cached_signals, cache_signals

router = APIRouter(prefix="/signals", tags=["Signals"])


def create_signal_response(signal: Signal, stock: Stock) -> SignalResponse:
    """Convert Signal model to API response."""
    return SignalResponse(
        id=signal.id,
        symbol=stock.symbol,
        stock_name=stock.name,
        sector=stock.sector,
        date=signal.date,
        signal_type=signal.signal_type,
        confidence=signal.confidence,
        probability_buy=signal.probability_buy,
        probability_hold=signal.probability_hold,
        probability_avoid=signal.probability_avoid,
        risk_reward_ratio=signal.risk_reward_ratio,
        suggested_timeframe_days=signal.suggested_timeframe_days,
        suggested_entry=signal.suggested_entry,
        suggested_stop_loss=signal.suggested_stop_loss,
        suggested_target=signal.suggested_target,
        model_version=signal.model_version,
        reasoning=signal.reasoning,
        generated_at=signal.generated_at,
    )


@router.get("/today", response_model=SignalsListResponse)
async def get_today_signals(
    db: DbSession,
    api_key: ApiKey,
    signal_type: Optional[str] = Query(None, description="Filter by signal type: BUY, HOLD, AVOID"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence score"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    market_cap_category: Optional[str] = Query(None, description="Filter by market cap: Large, Mid, Small"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("confidence", description="Sort by: confidence, symbol, sector"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
) -> SignalsListResponse:
    """
    Get today's trading signals with optional filters.
    
    Returns paginated list of signals generated for the current trading day.
    """
    today = date.today()
    
    # Try cache first
    cached = await get_cached_signals(today, signal_type, min_confidence)
    if cached:
        return cached
    
    # Build query
    query = (
        select(Signal, Stock)
        .join(Stock, Signal.stock_id == Stock.id)
        .where(Signal.date == today)
        .where(Stock.is_active == True)
    )
    
    # Apply filters
    if signal_type:
        query = query.where(Signal.signal_type == signal_type.upper())
    if min_confidence is not None:
        query = query.where(Signal.confidence >= min_confidence)
    if sector:
        query = query.where(Stock.sector == sector)
    if market_cap_category:
        query = query.where(Stock.market_cap_category == market_cap_category)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply sorting
    if sort_by == "confidence":
        order_col = Signal.confidence
    elif sort_by == "symbol":
        order_col = Stock.symbol
    elif sort_by == "sector":
        order_col = Stock.sector
    else:
        order_col = Signal.confidence
    
    if sort_order == "desc":
        query = query.order_by(order_col.desc())
    else:
        query = query.order_by(order_col.asc())
    
    # Apply pagination
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    
    # Execute query
    result = await db.execute(query)
    rows = result.all()
    
    # Build response
    items = [create_signal_response(signal, stock) for signal, stock in rows]
    
    # Count by type
    buy_count_result = await db.execute(
        select(func.count()).where(Signal.date == today, Signal.signal_type == "BUY")
    )
    hold_count_result = await db.execute(
        select(func.count()).where(Signal.date == today, Signal.signal_type == "HOLD")
    )
    avoid_count_result = await db.execute(
        select(func.count()).where(Signal.date == today, Signal.signal_type == "AVOID")
    )
    
    response = SignalsListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 0,
        generated_date=today,
        model_version=items[0].model_version if items else "unknown",
        buy_count=buy_count_result.scalar() or 0,
        hold_count=hold_count_result.scalar() or 0,
        avoid_count=avoid_count_result.scalar() or 0,
    )
    
    # Cache response
    await cache_signals(today, signal_type, min_confidence, response)
    
    return response


@router.get("/top", response_model=TopSignalsResponse)
async def get_top_signals(
    db: DbSession,
    api_key: ApiKey,
    limit: int = Query(10, ge=1, le=50, description="Number of top signals"),
    target_date: Optional[date] = Query(None, description="Date (defaults to today)"),
) -> TopSignalsResponse:
    """
    Get top BUY signals by confidence.
    """
    target = target_date or date.today()
    
    query = (
        select(Signal, Stock)
        .join(Stock, Signal.stock_id == Stock.id)
        .where(Signal.date == target)
        .where(Signal.signal_type == "BUY")
        .where(Stock.is_active == True)
        .order_by(Signal.confidence.desc())
        .limit(limit)
    )
    
    result = await db.execute(query)
    rows = result.all()
    
    top_buys = [create_signal_response(signal, stock) for signal, stock in rows]
    
    # High confidence count
    high_conf_result = await db.execute(
        select(func.count())
        .where(Signal.date == target)
        .where(Signal.signal_type == "BUY")
        .where(Signal.confidence >= 0.7)
    )
    
    return TopSignalsResponse(
        date=target,
        top_buys=top_buys,
        high_confidence_count=high_conf_result.scalar() or 0,
        model_version=top_buys[0].model_version if top_buys else "unknown",
    )


@router.get("/{symbol}", response_model=SignalHistoryResponse)
async def get_signal_history(
    symbol: str,
    db: DbSession,
    api_key: ApiKey,
    days: int = Query(30, ge=1, le=365, description="Number of days of history"),
) -> SignalHistoryResponse:
    """
    Get historical signals for a specific stock.
    """
    symbol = symbol.upper()
    
    # Get stock
    stock_result = await db.execute(
        select(Stock).where(Stock.symbol == symbol)
    )
    stock = stock_result.scalar_one_or_none()
    
    if not stock:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Stock with symbol '{symbol}' not found"
        )
    
    # Get signals
    start_date = date.today() - timedelta(days=days)
    
    query = (
        select(Signal)
        .where(Signal.stock_id == stock.id)
        .where(Signal.date >= start_date)
        .order_by(Signal.date.desc())
    )
    
    result = await db.execute(query)
    signals = result.scalars().all()
    
    return SignalHistoryResponse(
        symbol=stock.symbol,
        stock_name=stock.name,
        signals=[create_signal_response(s, stock) for s in signals],
        total=len(signals),
    )
