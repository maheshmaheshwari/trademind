"""
Financial Data Model

Stores yearly financial metrics scraped from Multibagg.ai for stocks.
Includes profit & loss, balance sheet, cash flow, and growth metrics.
"""

from datetime import date
from typing import Optional

from sqlalchemy import Date, Float, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class FinancialData(Base):
    """Yearly financial data for a stock scraped from Multibagg.ai."""
    
    __tablename__ = "financial_data"
    
    # Foreign key
    stock_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("stocks.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # Fiscal year (e.g., 2024 for FY ending March 2024)
    fiscal_year: Mapped[int] = mapped_column(Integer, nullable=False)
    fiscal_year_end: Mapped[Optional[date]] = mapped_column(Date, nullable=True)
    
    # ===== Profit & Loss Statement =====
    # Revenue
    revenue: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    revenue_growth_yoy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Operating metrics
    operating_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    operating_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Net profit
    net_profit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    net_profit_margin: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Other P&L items
    ebitda: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    eps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Balance Sheet =====
    total_assets: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_liabilities: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Debt
    total_debt: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    debt_to_equity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Working capital
    current_assets: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_liabilities: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Cash Flow =====
    operating_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    investing_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    financing_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    free_cash_flow: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Return Ratios (Yearly) =====
    roe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Return on Equity
    roce: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Return on Capital Employed
    roa: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Return on Assets
    
    # ===== Valuation Ratios =====
    pe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pb_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Dividend =====
    dividend_payout: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Data source =====
    source: Mapped[str] = mapped_column(String(50), default="multibagg.ai")
    
    # Relationship
    stock = relationship("Stock", back_populates="financial_data")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("stock_id", "fiscal_year", name="uq_financial_stock_year"),
        Index("ix_financial_stock_year", "stock_id", "fiscal_year"),
        Index("ix_financial_fiscal_year", "fiscal_year"),
    )
    
    def __repr__(self) -> str:
        return f"<FinancialData(stock_id={self.stock_id}, fiscal_year={self.fiscal_year})>"


class GrowthMetrics(Base):
    """CAGR and growth metrics for stocks (point-in-time snapshot)."""
    
    __tablename__ = "growth_metrics"
    
    # Foreign key
    stock_id: Mapped[int] = mapped_column(
        Integer, 
        ForeignKey("stocks.id", ondelete="CASCADE"), 
        nullable=False
    )
    
    # Snapshot date (when the data was scraped)
    snapshot_date: Mapped[date] = mapped_column(Date, nullable=False)
    
    # ===== Sales CAGR =====
    sales_cagr_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sales_cagr_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sales_cagr_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sales_cagr_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Profit CAGR =====
    profit_cagr_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_cagr_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_cagr_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    profit_cagr_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Return Metrics (Multi-year averages) =====
    roe_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roe_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    roce_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roce_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roce_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roce_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    roa_1y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roa_3y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roa_5y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roa_10y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Current Valuation =====
    market_cap: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pe_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pb_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dividend_yield: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== 52-Week Range =====
    high_52w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    low_52w: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # ===== Data source =====
    source: Mapped[str] = mapped_column(String(50), default="multibagg.ai")
    
    # Relationship
    stock = relationship("Stock", back_populates="growth_metrics")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("stock_id", "snapshot_date", name="uq_growth_stock_date"),
        Index("ix_growth_stock_date", "stock_id", "snapshot_date"),
        Index("ix_growth_snapshot_date", "snapshot_date"),
    )
    
    def __repr__(self) -> str:
        return f"<GrowthMetrics(stock_id={self.stock_id}, snapshot_date={self.snapshot_date})>"
