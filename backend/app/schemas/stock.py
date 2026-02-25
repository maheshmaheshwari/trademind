"""
Stock Schemas

Pydantic models for stock-related API operations.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class StockBase(BaseModel):
    """Base schema for stock data."""
    
    symbol: str = Field(..., min_length=1, max_length=20, description="Stock ticker symbol")
    name: str = Field(..., min_length=1, max_length=200, description="Company name")
    sector: Optional[str] = Field(None, max_length=100, description="Industry sector")
    industry: Optional[str] = Field(None, max_length=100, description="Specific industry")


class StockCreate(StockBase):
    """Schema for creating a new stock entry."""
    
    isin: Optional[str] = Field(None, max_length=20, description="ISIN code")
    market_cap: Optional[float] = Field(None, ge=0, description="Market capitalization")
    market_cap_category: Optional[str] = Field(None, description="Large/Mid/Small cap")
    is_nifty50: bool = False
    is_nifty100: bool = False
    is_nifty500: bool = True


class StockResponse(StockBase):
    """Schema for stock API responses."""
    
    id: int
    isin: Optional[str] = None
    market_cap: Optional[float] = None
    market_cap_category: Optional[str] = None
    is_active: bool
    is_nifty50: bool
    is_nifty100: bool
    is_nifty500: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class StockSearchResponse(BaseModel):
    """Schema for stock search results."""
    
    id: int
    symbol: str
    name: str
    sector: Optional[str] = None
    market_cap_category: Optional[str] = None
    is_nifty50: bool = False
    
    class Config:
        from_attributes = True


class StockWithLatestPrice(StockSearchResponse):
    """Stock with latest price information."""
    
    latest_close: Optional[float] = None
    price_change_1d: Optional[float] = None
    price_change_percent_1d: Optional[float] = None


class StockListResponse(BaseModel):
    """Paginated list of stocks."""
    
    items: list[StockSearchResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
