"""
Run Complete Pipeline Script

Manually trigger the full data pipeline.
"""

import asyncio
import logging
from datetime import date, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import async_session_maker, init_db
from app.services.data_ingestion import StockDataFetcher
from app.services.feature_engineering import FeatureEngineer
from app.ml.inference import ModelInference
from app.services.signal_generator import SignalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def run_pipeline(target_date: date = None):
    """Run the complete daily pipeline."""
    
    if target_date is None:
        target_date = date.today()
    
    logger.info(f"Running pipeline for {target_date}")
    
    await init_db()
    
    async with async_session_maker() as session:
        # Step 1: Data Ingestion
        logger.info("Step 1: Data Ingestion")
        fetcher = StockDataFetcher(session)
        ingestion_result = await fetcher.ingest_all_stocks(start_date=target_date - timedelta(days=1))
        logger.info(f"Ingestion: {ingestion_result}")
        
        # Step 2: Feature Engineering
        logger.info("Step 2: Feature Engineering")
        engineer = FeatureEngineer(session)
        feature_result = await engineer.compute_all_indicators(end_date=target_date)
        logger.info(f"Features: {feature_result}")
        
        # Step 3: Model Inference
        logger.info("Step 3: Model Inference")
        try:
            inference = ModelInference(session)
            predictions = await inference.predict_all_stocks(prediction_date=target_date)
            logger.info(f"Predictions: {len(predictions)} generated")
        except FileNotFoundError:
            logger.warning("No trained model found. Skipping inference.")
            predictions = {}
        
        # Step 4: Signal Generation
        if predictions:
            logger.info("Step 4: Signal Generation")
            generator = SignalGenerator(session)
            signal_result = await generator.generate_all_signals(predictions, target_date)
            logger.info(f"Signals: {signal_result}")
        else:
            logger.warning("No predictions available. Skipping signal generation.")
    
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TradeMind pipeline")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    args = parser.parse_args()
    
    target = date.fromisoformat(args.date) if args.date else None
    asyncio.run(run_pipeline(target))
