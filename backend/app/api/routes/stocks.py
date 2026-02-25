"""
Stocks API Routes

Endpoints for stock search and information.
"""

from typing import Optional

from fastapi import APIRouter, Query
from sqlalchemy import func, or_, select

from app.api.deps import ApiKey, DbSession
from app.models import Stock, OHLCData, TechnicalIndicator
from app.schemas.stock import (
    StockListResponse,
    StockResponse,
    StockSearchResponse,
    StockWithLatestPrice,
)

router = APIRouter(prefix="/stocks", tags=["Stocks"])


@router.get("/search", response_model=StockListResponse)
async def search_stocks(
    db: DbSession,
    api_key: ApiKey,
    q: Optional[str] = Query(None, min_length=1, description="Search query (symbol or name)"),
    sector: Optional[str] = Query(None, description="Filter by sector"),
    market_cap_category: Optional[str] = Query(None, description="Filter by market cap: Large, Mid, Small"),
    is_nifty50: Optional[bool] = Query(None, description="Filter NIFTY 50 stocks"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> StockListResponse:
    """
    Search stocks by symbol or name with filters.
    """
    query = select(Stock).where(Stock.is_active == True)
    
    # Apply search query
    if q:
        search_pattern = f"%{q.upper()}%"
        query = query.where(
            or_(
                Stock.symbol.ilike(search_pattern),
                Stock.name.ilike(search_pattern),
            )
        )
    
    # Apply filters
    if sector:
        query = query.where(Stock.sector == sector)
    if market_cap_category:
        query = query.where(Stock.market_cap_category == market_cap_category)
    if is_nifty50 is not None:
        query = query.where(Stock.is_nifty50 == is_nifty50)
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0
    
    # Apply pagination and ordering
    offset = (page - 1) * page_size
    query = query.order_by(Stock.symbol).offset(offset).limit(page_size)
    
    # Execute query
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    items = [
        StockSearchResponse(
            id=s.id,
            symbol=s.symbol,
            name=s.name,
            sector=s.sector,
            market_cap_category=s.market_cap_category,
            is_nifty50=s.is_nifty50,
        )
        for s in stocks
    ]
    
    return StockListResponse(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size if total > 0 else 0,
    )


@router.get("/sectors", response_model=list[str])
async def get_sectors(
    db: DbSession,
    api_key: ApiKey,
) -> list[str]:
    """
    Get list of all available sectors.
    """
    query = (
        select(Stock.sector)
        .where(Stock.is_active == True)
        .where(Stock.sector.isnot(None))
        .group_by(Stock.sector)
        .order_by(Stock.sector)
    )
    
    result = await db.execute(query)
    return [row[0] for row in result.all()]


@router.get("/nifty50", response_model=list[StockSearchResponse])
async def get_nifty50_stocks(
    db: DbSession,
    api_key: ApiKey,
) -> list[StockSearchResponse]:
    """
    Get all NIFTY 50 constituent stocks.
    """
    query = (
        select(Stock)
        .where(Stock.is_active == True)
        .where(Stock.is_nifty50 == True)
        .order_by(Stock.symbol)
    )
    
    result = await db.execute(query)
    stocks = result.scalars().all()
    
    return [
        StockSearchResponse(
            id=s.id,
            symbol=s.symbol,
            name=s.name,
            sector=s.sector,
            market_cap_category=s.market_cap_category,
            is_nifty50=s.is_nifty50,
        )
        for s in stocks
    ]


@router.get("/{symbol}", response_model=StockWithLatestPrice)
async def get_stock_details(
    symbol: str,
    db: DbSession,
    api_key: ApiKey,
) -> StockWithLatestPrice:
    """
    Get detailed information about a stock.
    """
    from datetime import date
    from fastapi import HTTPException, status
    
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
    
    # Get latest OHLC data
    ohlc_result = await db.execute(
        select(OHLCData)
        .where(OHLCData.stock_id == stock.id)
        .order_by(OHLCData.date.desc())
        .limit(2)
    )
    ohlc_data = ohlc_result.scalars().all()
    
    latest_close = None
    price_change = None
    price_change_percent = None
    
    if ohlc_data:
        latest_close = ohlc_data[0].close
        if len(ohlc_data) > 1:
            prev_close = ohlc_data[1].close
            price_change = latest_close - prev_close
            price_change_percent = (price_change / prev_close) * 100
    
    return StockWithLatestPrice(
        id=stock.id,
        symbol=stock.symbol,
        name=stock.name,
        sector=stock.sector,
        market_cap_category=stock.market_cap_category,
        is_nifty50=stock.is_nifty50,
        latest_close=latest_close,
        price_change_1d=price_change,
        price_change_percent_1d=price_change_percent,
    )
