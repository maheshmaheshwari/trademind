"""
Market API Routes

Endpoints for market overview and data.
"""

from datetime import date, datetime

from fastapi import APIRouter, Query
from sqlalchemy import func, select

from app.api.deps import ApiKey, DbSession
from app.models import Signal, Stock, TechnicalIndicator
from app.schemas.market import (
    IndexData,
    MarketBreadth,
    MarketOverview,
    MarketRegime,
    MarketSummaryResponse,
    SectorPerformance,
    VIXData,
)
from app.services.data_ingestion import get_index_data
from app.services.cache import get_cached_market_overview, cache_market_overview

router = APIRouter(prefix="/market", tags=["Market"])


def get_vix_level(vix_value: float) -> tuple[str, str]:
    """Classify VIX level and provide description."""
    if vix_value < 13:
        return "LOW", "Market is calm with low volatility expectations"
    elif vix_value < 20:
        return "MODERATE", "Normal market volatility"
    elif vix_value < 30:
        return "HIGH", "Elevated fear and uncertainty in the market"
    else:
        return "EXTREME", "Extreme fear - potential market stress"


def classify_market_regime(
    nifty_change: float,
    vix_value: float,
    advance_decline: float
) -> MarketRegime:
    """Classify current market regime based on indicators."""
    indicators = {
        "nifty_change": nifty_change,
        "vix": vix_value,
        "advance_decline_ratio": advance_decline,
    }
    
    if nifty_change > 0.5 and advance_decline > 1.2 and vix_value < 20:
        return MarketRegime(
            regime="BULL",
            confidence=min(0.9, 0.5 + (advance_decline - 1) * 0.2),
            description="Strong bullish sentiment with broad market participation",
            indicators=indicators,
        )
    elif nifty_change < -0.5 and advance_decline < 0.8 and vix_value > 18:
        return MarketRegime(
            regime="BEAR",
            confidence=min(0.9, 0.5 + (1 - advance_decline) * 0.3),
            description="Bearish conditions with elevated volatility",
            indicators=indicators,
        )
    else:
        return MarketRegime(
            regime="SIDEWAYS",
            confidence=0.6,
            description="Range-bound market with mixed signals",
            indicators=indicators,
        )


@router.get("/overview", response_model=MarketOverview)
async def get_market_overview(
    db: DbSession,
    api_key: ApiKey,
) -> MarketOverview:
    """
    Get comprehensive market overview.
    
    Includes index data, VIX, market regime, breadth, and sector performance.
    """
    today = date.today()
    now = datetime.utcnow()
    
    # Try cache first
    cached = await get_cached_market_overview(today)
    if cached:
        return cached
    
    # Fetch index data (from cache or external API)
    nifty50_data = await get_index_data("^NSEI")
    india_vix_data = await get_index_data("^INDIAVIX")
    
    # Create index response
    nifty50 = IndexData(
        name="NIFTY 50",
        symbol="^NSEI",
        value=nifty50_data.get("close", 0),
        change=nifty50_data.get("change", 0),
        change_percent=nifty50_data.get("change_percent", 0),
        last_updated=now,
    )
    
    # VIX data
    vix_value = india_vix_data.get("close", 15)
    vix_level, vix_desc = get_vix_level(vix_value)
    india_vix = VIXData(
        value=vix_value,
        change=india_vix_data.get("change", 0),
        change_percent=india_vix_data.get("change_percent", 0),
        level=vix_level,
        description=vix_desc,
    )
    
    # Market breadth from indicators
    breadth_query = select(
        func.count().filter(TechnicalIndicator.returns_1d > 0).label("advances"),
        func.count().filter(TechnicalIndicator.returns_1d < 0).label("declines"),
        func.count().filter(TechnicalIndicator.returns_1d == 0).label("unchanged"),
    ).where(TechnicalIndicator.date == today)
    
    breadth_result = await db.execute(breadth_query)
    breadth_row = breadth_result.one_or_none()
    
    advances = breadth_row.advances if breadth_row else 250
    declines = breadth_row.declines if breadth_row else 200
    unchanged = breadth_row.unchanged if breadth_row else 50
    
    advance_decline_ratio = advances / max(declines, 1)
    
    market_breadth = MarketBreadth(
        advances=advances,
        declines=declines,
        unchanged=unchanged,
        advance_decline_ratio=round(advance_decline_ratio, 2),
        percent_above_20ema=0.55,  # TODO: Calculate from data
        percent_above_50ema=0.50,
        percent_above_200ema=0.60,
    )
    
    # Market regime
    market_regime = classify_market_regime(
        nifty50.change_percent,
        vix_value,
        advance_decline_ratio,
    )
    
    # Sector performance
    sector_query = (
        select(Stock.sector)
        .where(Stock.is_active == True)
        .where(Stock.sector.isnot(None))
        .group_by(Stock.sector)
    )
    sector_result = await db.execute(sector_query)
    sectors = [row[0] for row in sector_result.all()]
    
    sector_performance = []
    for sector in sectors[:10]:  # Top 10 sectors
        sector_performance.append(
            SectorPerformance(
                sector=sector,
                change_percent_1d=0.0,  # TODO: Calculate
                change_percent_5d=0.0,
                change_percent_20d=0.0,
                buy_signals_count=0,
            )
        )
    
    # Signal counts
    signal_counts = await db.execute(
        select(
            Signal.signal_type,
            func.count().label("count")
        )
        .where(Signal.date == today)
        .group_by(Signal.signal_type)
    )
    
    counts_dict = {row[0]: row[1] for row in signal_counts.all()}
    
    high_conf_result = await db.execute(
        select(func.count())
        .where(Signal.date == today)
        .where(Signal.signal_type == "BUY")
        .where(Signal.confidence >= 0.7)
    )
    
    response = MarketOverview(
        date=today,
        last_updated=now,
        nifty50=nifty50,
        india_vix=india_vix,
        market_regime=market_regime,
        market_breadth=market_breadth,
        sector_performance=sector_performance,
        total_buy_signals=counts_dict.get("BUY", 0),
        total_hold_signals=counts_dict.get("HOLD", 0),
        total_avoid_signals=counts_dict.get("AVOID", 0),
        high_confidence_buys=high_conf_result.scalar() or 0,
    )
    
    # Cache response
    await cache_market_overview(today, response)
    
    return response


@router.get("/summary", response_model=MarketSummaryResponse)
async def get_market_summary(
    db: DbSession,
    api_key: ApiKey,
) -> MarketSummaryResponse:
    """
    Get simplified market summary.
    """
    today = date.today()
    
    # Fetch index data
    nifty50_data = await get_index_data("^NSEI")
    india_vix_data = await get_index_data("^INDIAVIX")
    
    # Market regime
    regime = classify_market_regime(
        nifty50_data.get("change_percent", 0),
        india_vix_data.get("close", 15),
        1.0,  # Default A/D ratio
    )
    
    # Signal counts
    buy_count_result = await db.execute(
        select(func.count())
        .where(Signal.date == today)
        .where(Signal.signal_type == "BUY")
    )
    
    high_conf_result = await db.execute(
        select(func.count())
        .where(Signal.date == today)
        .where(Signal.signal_type == "BUY")
        .where(Signal.confidence >= 0.7)
    )
    
    return MarketSummaryResponse(
        date=today,
        nifty50_value=nifty50_data.get("close", 0),
        nifty50_change_percent=nifty50_data.get("change_percent", 0),
        india_vix=india_vix_data.get("close", 15),
        market_regime=regime.regime,
        advance_decline_ratio=1.0,
        buy_signals_count=buy_count_result.scalar() or 0,
        high_confidence_buys=high_conf_result.scalar() or 0,
    )
