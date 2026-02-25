"""
Technical Indicator Model

Stores computed technical indicators for each stock.
"""

from datetime import date
from typing import Optional

from sqlalchemy import Date, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TechnicalIndicator(Base):
    """Computed technical indicators for a stock on a given date."""
    
    __tablename__ = "technical_indicators"
    
    # Foreign key
    stock_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("stocks.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Date
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    
    # Returns
    returns_1d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    returns_5d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    returns_20d: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # RSI
    rsi_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # EMAs
    ema_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ema_200: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Volatility
    volatility_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    atr_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Relative strength
    relative_strength_nifty: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Market regime (Bull/Bear/Sideways)
    market_regime: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Volume indicators
    volume_sma_20: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    volume_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Trend indicators
    adx_14: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_signal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    macd_histogram: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Bollinger Bands
    bb_upper: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_middle: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_lower: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    bb_width: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Relationship
    stock = relationship("Stock", back_populates="indicators")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("stock_id", "date", name="uq_indicator_stock_date"),
        Index("ix_indicator_stock_date", "stock_id", "date"),
        Index("ix_indicator_rsi", "rsi_14"),
        Index("ix_indicator_regime", "market_regime"),
    )
    
    def __repr__(self) -> str:
        return f"<TechnicalIndicator(stock_id={self.stock_id}, date={self.date})>"
    
    @property
    def is_oversold(self) -> bool:
        """Check if RSI indicates oversold condition."""
        return self.rsi_14 is not None and self.rsi_14 < 30
    
    @property
    def is_overbought(self) -> bool:
        """Check if RSI indicates overbought condition."""
        return self.rsi_14 is not None and self.rsi_14 > 70
    
    @property
    def is_trending(self) -> bool:
        """Check if ADX indicates a trending market."""
        return self.adx_14 is not None and self.adx_14 > 25
