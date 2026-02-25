"""
Health API Routes

Endpoints for service and model health checks.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import func, select

from app.api.deps import DbSession
from app.config import settings
from app.models import ModelMetrics, Signal, Stock, OHLCData

router = APIRouter(prefix="/model", tags=["Health"])


class ServiceHealth(BaseModel):
    """Service health status."""
    status: str
    version: str
    environment: str
    timestamp: datetime


class DatabaseHealth(BaseModel):
    """Database health status."""
    status: str
    stocks_count: int
    ohlc_records_count: int
    signals_count: int


class ModelHealth(BaseModel):
    """ML model health status."""
    status: str
    current_version: str
    last_trained: Optional[datetime] = None
    accuracy: Optional[float] = None
    backtest_sharpe: Optional[float] = None
    backtest_max_drawdown: Optional[float] = None
    is_active: bool


class HealthResponse(BaseModel):
    """Complete health check response."""
    service: ServiceHealth
    database: DatabaseHealth
    model: ModelHealth


@router.get("/health", response_model=HealthResponse)
async def get_model_health(
    db: DbSession,
) -> HealthResponse:
    """
    Get complete system health status.
    
    This endpoint is public (no API key required) for monitoring purposes.
    """
    now = datetime.utcnow()
    
    # Service health
    service = ServiceHealth(
        status="healthy",
        version="1.0.0",
        environment=settings.app_env,
        timestamp=now,
    )
    
    # Database health
    try:
        stocks_count_result = await db.execute(select(func.count()).select_from(Stock))
        ohlc_count_result = await db.execute(select(func.count()).select_from(OHLCData))
        signals_count_result = await db.execute(select(func.count()).select_from(Signal))
        
        database = DatabaseHealth(
            status="connected",
            stocks_count=stocks_count_result.scalar() or 0,
            ohlc_records_count=ohlc_count_result.scalar() or 0,
            signals_count=signals_count_result.scalar() or 0,
        )
    except Exception as e:
        database = DatabaseHealth(
            status=f"error: {str(e)}",
            stocks_count=0,
            ohlc_records_count=0,
            signals_count=0,
        )
    
    # Model health
    try:
        model_result = await db.execute(
            select(ModelMetrics)
            .where(ModelMetrics.is_active == True)
            .order_by(ModelMetrics.trained_at.desc())
            .limit(1)
        )
        active_model = model_result.scalar_one_or_none()
        
        if active_model:
            model = ModelHealth(
                status="active",
                current_version=active_model.version,
                last_trained=active_model.trained_at,
                accuracy=active_model.accuracy,
                backtest_sharpe=active_model.backtest_sharpe,
                backtest_max_drawdown=active_model.backtest_max_drawdown,
                is_active=True,
            )
        else:
            model = ModelHealth(
                status="no_active_model",
                current_version=settings.current_model_version,
                is_active=False,
            )
    except Exception as e:
        model = ModelHealth(
            status=f"error: {str(e)}",
            current_version="unknown",
            is_active=False,
        )
    
    return HealthResponse(
        service=service,
        database=database,
        model=model,
    )


@router.get("/versions", response_model=list[ModelHealth])
async def get_model_versions(
    db: DbSession,
) -> list[ModelHealth]:
    """
    Get all model versions and their metrics.
    """
    result = await db.execute(
        select(ModelMetrics).order_by(ModelMetrics.trained_at.desc()).limit(10)
    )
    models = result.scalars().all()
    
    return [
        ModelHealth(
            status="active" if m.is_active else "inactive",
            current_version=m.version,
            last_trained=m.trained_at,
            accuracy=m.accuracy,
            backtest_sharpe=m.backtest_sharpe,
            backtest_max_drawdown=m.backtest_max_drawdown,
            is_active=m.is_active,
        )
        for m in models
    ]


@router.get("/ping")
async def ping() -> dict:
    """Simple ping endpoint for load balancer health checks."""
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}
