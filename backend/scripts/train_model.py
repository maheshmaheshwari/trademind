"""
Train Model Script

Train a new model version manually.
"""

import asyncio
import logging
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.database import async_session_maker, init_db
from app.ml.training import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def train_model(version: str = None):
    """Train a new model version."""
    
    if version is None:
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Training model version: {version}")
    
    await init_db()
    
    async with async_session_maker() as session:
        trainer = ModelTrainer(session)
        
        try:
            model_path, metrics = await trainer.train_and_save(version=version)
            
            logger.info(f"Model saved to: {model_path}")
            logger.info(f"Metrics:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
            logger.info(f"  Backtest Return: {metrics.get('total_return', 'N/A'):.2f}%")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.4f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2f}%")
            logger.info(f"  Win Rate: {metrics.get('win_rate', 'N/A'):.2%}")
            
        except ValueError as e:
            logger.error(f"Training failed: {e}")
            logger.info("Make sure you have sufficient historical data before training.")
            logger.info("Run: python scripts/seed_database.py --days 730")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TradeMind model")
    parser.add_argument("--version", type=str, help="Model version name")
    args = parser.parse_args()
    
    asyncio.run(train_model(args.version))
