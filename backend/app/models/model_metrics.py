"""
Model Metrics

Stores ML model training metrics and performance data.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class ModelMetrics(Base):
    """ML model training and performance metrics."""
    
    __tablename__ = "model_metrics"
    
    # Model identification
    version: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), default="xgboost")
    
    # Training info
    trained_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    training_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    validation_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Feature info
    feature_count: Mapped[int] = mapped_column(Integer, nullable=False)
    feature_names: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Classification metrics
    accuracy: Mapped[float] = mapped_column(Float, nullable=False)
    precision_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_hold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision_avoid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_buy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_hold: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall_avoid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Backtesting metrics
    backtest_return: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_sharpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_max_drawdown: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_win_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    backtest_profit_factor: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Model hyperparameters
    hyperparameters: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Model path
    model_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Status
    is_active: Mapped[bool] = mapped_column(default=False)
    
    # Notes
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    def __repr__(self) -> str:
        return f"<ModelMetrics(version={self.version}, accuracy={self.accuracy:.2%})>"
    
    @property
    def is_performant(self) -> bool:
        """Check if model meets minimum performance thresholds."""
        return (
            self.accuracy >= 0.55 and
            self.backtest_sharpe is not None and self.backtest_sharpe >= 0.5 and
            self.backtest_max_drawdown is not None and self.backtest_max_drawdown <= 0.20
        )
