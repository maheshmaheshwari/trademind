"""
Stock Model

Represents a stock in NIFTY 500 index.
"""

from typing import Optional

from sqlalchemy import Boolean, Float, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Stock(Base):
    """Stock entity representing an NSE-listed stock."""
    
    __tablename__ = "stocks"
    
    # Stock identifiers
    symbol: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    isin: Mapped[Optional[str]] = mapped_column(String(20), unique=True, nullable=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Classification
    sector: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    industry: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Market data
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    market_cap_category: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # Large/Mid/Small
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_nifty50: Mapped[bool] = mapped_column(Boolean, default=False)
    is_nifty100: Mapped[bool] = mapped_column(Boolean, default=False)
    is_nifty500: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    ohlc_data = relationship("OHLCData", back_populates="stock", lazy="dynamic")
    indicators = relationship("TechnicalIndicator", back_populates="stock", lazy="dynamic")
    signals = relationship("Signal", back_populates="stock", lazy="dynamic")
    financial_data = relationship("FinancialData", back_populates="stock", lazy="dynamic")
    growth_metrics = relationship("GrowthMetrics", back_populates="stock", lazy="dynamic")
    
    # Indexes
    __table_args__ = (
        Index("ix_stocks_sector", "sector"),
        Index("ix_stocks_industry", "industry"),
        Index("ix_stocks_market_cap_category", "market_cap_category"),
    )
    
    def __repr__(self) -> str:
        return f"<Stock(symbol={self.symbol}, name={self.name})>"
