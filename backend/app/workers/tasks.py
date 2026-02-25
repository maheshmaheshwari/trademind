"""
Celery Tasks

Background tasks for data pipeline.
"""

import logging
from datetime import date, timedelta

from celery import shared_task
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import sync_engine

logger = logging.getLogger(__name__)

# Sync session for Celery tasks
SessionLocal = sessionmaker(bind=sync_engine, autocommit=False, autoflush=False)


def get_sync_session():
    """Get synchronous database session for Celery tasks."""
    return SessionLocal()


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def ingest_daily_data(self, target_date: str = None):
    """
    Ingest daily OHLC data for all stocks.
    
    Args:
        target_date: Date string (YYYY-MM-DD) or None for yesterday
    """
    import asyncio
    from app.database import async_session_maker
    from app.services.data_ingestion import StockDataFetcher
    
    if target_date:
        ingest_date = date.fromisoformat(target_date)
    else:
        ingest_date = date.today() - timedelta(days=1)
    
    logger.info(f"Starting data ingestion for {ingest_date}")
    
    async def run_ingestion():
        async with async_session_maker() as session:
            fetcher = StockDataFetcher(session)
            summary = await fetcher.ingest_all_stocks(start_date=ingest_date)
            return summary
    
    try:
        summary = asyncio.run(run_ingestion())
        logger.info(f"Data ingestion complete: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def compute_daily_indicators(self, target_date: str = None):
    """
    Compute technical indicators for all stocks.
    
    Args:
        target_date: Date string (YYYY-MM-DD) or None for today
    """
    import asyncio
    from app.database import async_session_maker
    from app.services.feature_engineering import FeatureEngineer
    
    if target_date:
        compute_date = date.fromisoformat(target_date)
    else:
        compute_date = date.today()
    
    logger.info(f"Starting indicator computation for {compute_date}")
    
    async def run_computation():
        async with async_session_maker() as session:
            engineer = FeatureEngineer(session)
            summary = await engineer.compute_all_indicators(end_date=compute_date)
            return summary
    
    try:
        summary = asyncio.run(run_computation())
        logger.info(f"Indicator computation complete: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Indicator computation failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def run_daily_inference(self, target_date: str = None):
    """
    Run model inference for all stocks.
    
    Args:
        target_date: Date string (YYYY-MM-DD) or None for today
    """
    import asyncio
    from app.database import async_session_maker
    from app.ml.inference import ModelInference
    
    if target_date:
        inference_date = date.fromisoformat(target_date)
    else:
        inference_date = date.today()
    
    logger.info(f"Starting model inference for {inference_date}")
    
    async def run_inference():
        async with async_session_maker() as session:
            inference = ModelInference(session)
            predictions = await inference.predict_all_stocks(prediction_date=inference_date)
            return predictions
    
    try:
        predictions = asyncio.run(run_inference())
        logger.info(f"Inference complete: {len(predictions)} predictions generated")
        
        # Store predictions in cache for signal generation
        return {"predictions_count": len(predictions), "predictions": predictions}
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=3, default_retry_delay=300)
def generate_daily_signals(self, target_date: str = None, predictions: dict = None):
    """
    Generate trading signals from model predictions.
    
    Args:
        target_date: Date string (YYYY-MM-DD) or None for today
        predictions: Dict of stock_id -> probabilities (optional)
    """
    import asyncio
    from app.database import async_session_maker
    from app.ml.inference import ModelInference
    from app.services.signal_generator import SignalGenerator
    from app.services.cache import invalidate_cache
    
    if target_date:
        signal_date = date.fromisoformat(target_date)
    else:
        signal_date = date.today()
    
    logger.info(f"Starting signal generation for {signal_date}")
    
    async def run_signal_generation():
        async with async_session_maker() as session:
            # Get predictions if not provided
            if predictions is None:
                inference = ModelInference(session)
                preds = await inference.predict_all_stocks(prediction_date=signal_date)
            else:
                preds = predictions
            
            # Generate signals
            generator = SignalGenerator(session)
            summary = await generator.generate_all_signals(preds, signal_date)
            
            # Invalidate cache
            await invalidate_cache(f"signals:{signal_date.isoformat()}*")
            await invalidate_cache(f"market:{signal_date.isoformat()}")
            
            return summary
    
    try:
        summary = asyncio.run(run_signal_generation())
        logger.info(f"Signal generation complete: {summary}")
        return summary
    except Exception as e:
        logger.error(f"Signal generation failed: {e}")
        raise self.retry(exc=e)


@shared_task(bind=True, max_retries=1, default_retry_delay=600)
def train_model(self, version: str = None):
    """
    Train a new model version.
    
    Args:
        version: Model version string (auto-generated if None)
    """
    import asyncio
    from app.database import async_session_maker
    from app.ml.training import ModelTrainer
    
    logger.info(f"Starting model training (version: {version or 'auto'})")
    
    async def run_training():
        async with async_session_maker() as session:
            trainer = ModelTrainer(session)
            model_path, metrics = await trainer.train_and_save(version=version)
            return {"model_path": model_path, "metrics": metrics}
    
    try:
        result = asyncio.run(run_training())
        logger.info(f"Model training complete: {result}")
        return result
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise self.retry(exc=e)


@shared_task
def run_complete_pipeline(target_date: str = None):
    """
    Run the complete daily pipeline.
    
    This chains: ingestion -> indicators -> inference -> signals
    """
    from celery import chain
    
    logger.info(f"Starting complete pipeline for {target_date or 'today'}")
    
    # Chain tasks
    pipeline = chain(
        ingest_daily_data.s(target_date),
        compute_daily_indicators.s(target_date),
        run_daily_inference.s(target_date),
        generate_daily_signals.s(target_date),
    )
    
    result = pipeline.apply_async()
    
    return {"task_id": result.id, "status": "pipeline_started"}


@shared_task
def health_check():
    """Simple health check task."""
    return {"status": "healthy", "worker": "celery"}
