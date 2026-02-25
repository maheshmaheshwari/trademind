"""
OHLC Data Model

Stores daily Open-High-Low-Close-Volume data for stocks.
"""

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, Index, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class OHLCData(Base):
    """Daily OHLC data for a stock."""
    
    __tablename__ = "ohlc_data"
    
    # Foreign key
    stock_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("stocks.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # Date
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # Price data
    open: Mapped[float] = mapped_column(Float, nullable=False)
    high: Mapped[float] = mapped_column(Float, nullable=False)
    low: Mapped[float] = mapped_column(Float, nullable=False)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Adjusted close (for splits/dividends)
    adj_close: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Volume
    volume: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Relationship
    stock = relationship("Stock", back_populates="ohlc_data")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_ohlc_stock_date"),
        Index("ix_ohlc_stock_date", "stock_id", "date"),
    )
    
    def __repr__(self) -> str:
        return f"<OHLCData(stock_id={self.stock_id}, date={self.date}, close={self.close})>"
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3
    
    @property
    def price_range(self) -> float:
        """Calculate daily price range."""
        return self.high - self.low
    
    @property
    def body_size(self) -> float:
        """Calculate candlestick body size."""
        return abs(self.close - self.open)
    
    @property
    def is_bullish(self) -> bool:
        """Check if candlestick is bullish."""
        return self.close > self.open
