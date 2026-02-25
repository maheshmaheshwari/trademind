"""
Signal Schemas

Pydantic models for signal-related API operations.
"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class SignalResponse(BaseModel):
    """Schema for individual signal response."""
    
    id: int
    symbol: str
    stock_name: str
    sector: Optional[str] = None
    
    # Signal details
    date: date
    signal_type: Literal["BUY", "HOLD", "AVOID"]
    confidence: float = Field(..., ge=0, le=1, description="Confidence score (0-1)")
    
    # Probabilities
    probability_buy: float
    probability_hold: float
    probability_avoid: float
    
    # Trading suggestions
    risk_reward_ratio: Optional[float] = None
    suggested_timeframe_days: int
    suggested_entry: Optional[float] = None
    suggested_stop_loss: Optional[float] = None
    suggested_target: Optional[float] = None
    
    # Meta
    model_version: str
    reasoning: Optional[str] = None
    generated_at: datetime
    
    # Disclaimer
    disclaimer: str = Field(
        default="This is not financial advice. Do your own research before trading.",
        description="Legal disclaimer"
    )
    
    class Config:
        from_attributes = True


class SignalsListResponse(BaseModel):
    """Paginated list of signals."""
    
    items: list[SignalResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
    generated_date: date
    model_version: str
    
    # Summary stats
    buy_count: int = 0
    hold_count: int = 0
    avoid_count: int = 0


class SignalFilter(BaseModel):
    """Filters for signal queries."""
    
    signal_type: Optional[Literal["BUY", "HOLD", "AVOID"]] = None
    min_confidence: Optional[float] = Field(None, ge=0, le=1)
    sector: Optional[str] = None
    market_cap_category: Optional[Literal["Large", "Mid", "Small"]] = None
    
    # Pagination
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    
    # Sorting
    sort_by: Literal["confidence", "symbol", "sector"] = "confidence"
    sort_order: Literal["asc", "desc"] = "desc"


class SignalHistoryResponse(BaseModel):
    """Historical signals for a stock."""
    
    symbol: str
    stock_name: str
    signals: list[SignalResponse]
    total: int


class TopSignalsResponse(BaseModel):
    """Top signals summary."""
    
    date: date
    top_buys: list[SignalResponse]
    high_confidence_count: int
    model_version: str
