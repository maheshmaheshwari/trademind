"""
Market Schemas

Pydantic models for market-related API operations.
"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class IndexData(BaseModel):
    """Index data point."""
    
    name: str
    symbol: str
    value: float
    change: float
    change_percent: float
    last_updated: datetime


class MarketRegime(BaseModel):
    """Market regime classification."""
    
    regime: Literal["BULL", "BEAR", "SIDEWAYS"]
    confidence: float = Field(..., ge=0, le=1)
    description: str
    indicators: dict = Field(default_factory=dict)


class SectorPerformance(BaseModel):
    """Sector performance summary."""
    
    sector: str
    change_percent_1d: float
    change_percent_5d: float
    change_percent_20d: float
    top_gainer: Optional[str] = None
    top_loser: Optional[str] = None
    buy_signals_count: int = 0


class MarketBreadth(BaseModel):
    """Market breadth indicators."""
    
    advances: int
    declines: int
    unchanged: int
    advance_decline_ratio: float
    percent_above_20ema: float
    percent_above_50ema: float
    percent_above_200ema: float


class VIXData(BaseModel):
    """India VIX data."""
    
    value: float
    change: float
    change_percent: float
    level: Literal["LOW", "MODERATE", "HIGH", "EXTREME"]
    description: str


class MarketOverview(BaseModel):
    """Complete market overview response."""
    
    date: date
    last_updated: datetime
    
    # Index data
    nifty50: IndexData
    nifty_bank: Optional[IndexData] = None
    nifty_it: Optional[IndexData] = None
    
    # Volatility
    india_vix: VIXData
    
    # Market regime
    market_regime: MarketRegime
    
    # Breadth
    market_breadth: MarketBreadth
    
    # Sector performance
    sector_performance: list[SectorPerformance] = []
    
    # Signal summary
    total_buy_signals: int = 0
    total_hold_signals: int = 0
    total_avoid_signals: int = 0
    high_confidence_buys: int = 0
    
    # Disclaimer
    disclaimer: str = "Market data may be delayed. This is not financial advice."


class MarketSummaryResponse(BaseModel):
    """Simplified market summary."""
    
    date: date
    nifty50_value: float
    nifty50_change_percent: float
    india_vix: float
    market_regime: str
    advance_decline_ratio: float
    buy_signals_count: int
    high_confidence_buys: int
