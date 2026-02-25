"""
Model Inference

Run batch predictions on stock data.
"""

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Stock, OHLCData, TechnicalIndicator
from app.ml.model import StockClassifier, LABEL_MAP_INV

logger = logging.getLogger(__name__)


class ModelInference:
    """Run inference on stock data."""
    
    def __init__(self, db: AsyncSession, model_path: Optional[str] = None):
        self.db = db
        self.model_path = model_path or self._get_latest_model_path()
        self.model: Optional[StockClassifier] = None
    
    def _get_latest_model_path(self) -> str:
        """Get path to latest model file."""
        model_dir = Path(settings.model_path)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        model_files = list(model_dir.glob("model_*.pkl"))
        
        if not model_files:
            raise FileNotFoundError("No model files found")
        
        # Sort by modification time
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        
        return str(latest)
    
    def load_model(self) -> None:
        """Load the model from disk."""
        self.model = StockClassifier.load(self.model_path)
        logger.info(f"Loaded model from {self.model_path}")
    
    async def predict_for_stock(
        self,
        stock: Stock,
        prediction_date: Optional[date] = None,
    ) -> Optional[dict]:
        """
        Generate prediction for a single stock.
        
        Args:
            stock: Stock model instance
            prediction_date: Date for prediction
            
        Returns:
            Dict with probabilities or None if insufficient data
        """
        if self.model is None:
            self.load_model()
        
        if prediction_date is None:
            prediction_date = date.today()
        
        # Get latest indicators
        indicator_result = await self.db.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.stock_id == stock.id)
            .where(TechnicalIndicator.date <= prediction_date)
            .order_by(TechnicalIndicator.date.desc())
            .limit(1)
        )
        indicator = indicator_result.scalar_one_or_none()
        
        if not indicator:
            logger.warning(f"No indicators found for {stock.symbol}")
            return None
        
        # Get latest price
        ohlc_result = await self.db.execute(
            select(OHLCData)
            .where(OHLCData.stock_id == stock.id)
            .where(OHLCData.date <= prediction_date)
            .order_by(OHLCData.date.desc())
            .limit(1)
        )
        ohlc = ohlc_result.scalar_one_or_none()
        
        if not ohlc:
            logger.warning(f"No OHLC data found for {stock.symbol}")
            return None
        
        # Create indicator DataFrame
        indicator_df = pd.DataFrame([{
            "returns_1d": indicator.returns_1d,
            "returns_5d": indicator.returns_5d,
            "returns_20d": indicator.returns_20d,
            "rsi_14": indicator.rsi_14,
            "ema_20": indicator.ema_20,
            "ema_50": indicator.ema_50,
            "volatility_20": indicator.volatility_20,
            "atr_14": indicator.atr_14,
            "volume_ratio": indicator.volume_ratio,
            "adx_14": indicator.adx_14,
            "macd_histogram": indicator.macd_histogram,
            "bb_upper": indicator.bb_upper,
            "bb_lower": indicator.bb_lower,
            "relative_strength_nifty": indicator.relative_strength_nifty or 1.0,
        }])
        
        close_price = pd.Series([ohlc.close])
        
        # Prepare features
        features = StockClassifier.prepare_features(indicator_df, close_price)
        
        # Predict
        probabilities = self.model.predict_proba(features)[0]
        
        return {
            "buy": float(probabilities[2]),  # BUY is class 2
            "hold": float(probabilities[1]),  # HOLD is class 1
            "avoid": float(probabilities[0]),  # AVOID is class 0
        }
    
    async def predict_all_stocks(
        self,
        prediction_date: Optional[date] = None,
    ) -> dict[int, dict]:
        """
        Generate predictions for all active stocks.
        
        Args:
            prediction_date: Date for predictions
            
        Returns:
            Dict mapping stock_id to probabilities
        """
        if self.model is None:
            self.load_model()
        
        # Get all active stocks
        stocks_result = await self.db.execute(
            select(Stock).where(Stock.is_active == True)
        )
        stocks = stocks_result.scalars().all()
        
        predictions = {}
        
        for stock in stocks:
            try:
                pred = await self.predict_for_stock(stock, prediction_date)
                if pred:
                    predictions[stock.id] = pred
            except Exception as e:
                logger.error(f"Prediction failed for {stock.symbol}: {e}")
        
        logger.info(f"Generated predictions for {len(predictions)}/{len(stocks)} stocks")
        
        return predictions
    
    async def get_top_predictions(
        self,
        n: int = 20,
        prediction_date: Optional[date] = None,
    ) -> list[dict]:
        """
        Get top N BUY predictions.
        
        Args:
            n: Number of top predictions
            prediction_date: Date for predictions
            
        Returns:
            List of stock predictions sorted by BUY probability
        """
        predictions = await self.predict_all_stocks(prediction_date)
        
        # Get stock info
        stock_ids = list(predictions.keys())
        stocks_result = await self.db.execute(
            select(Stock).where(Stock.id.in_(stock_ids))
        )
        stocks = {s.id: s for s in stocks_result.scalars().all()}
        
        # Combine and sort
        results = []
        for stock_id, probs in predictions.items():
            stock = stocks.get(stock_id)
            if stock:
                results.append({
                    "stock_id": stock_id,
                    "symbol": stock.symbol,
                    "name": stock.name,
                    "sector": stock.sector,
                    "probabilities": probs,
                    "buy_probability": probs["buy"],
                })
        
        # Sort by BUY probability
        results.sort(key=lambda x: x["buy_probability"], reverse=True)
        
        return results[:n]
