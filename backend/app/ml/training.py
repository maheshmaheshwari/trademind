"""
Model Training Pipeline

Train and validate XGBoost models.
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Stock, OHLCData, TechnicalIndicator, ModelMetrics
from app.ml.model import StockClassifier, FEATURE_COLUMNS
from app.ml.metrics import calculate_metrics, backtest_strategy

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and validate stock classification models."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.lookahead_days = 10  # Days to look ahead for labeling
        self.buy_threshold = 5.0  # %return for BUY
        self.avoid_threshold = -3.0  # %return for AVOID
    
    async def prepare_training_data(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training dataset from database.
        
        Args:
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Tuple of (features DataFrame, labels Series)
        """
        if end_date is None:
            end_date = date.today() - timedelta(days=self.lookahead_days + 1)
        if start_date is None:
            start_date = end_date - timedelta(days=365 * 2)  # 2 years of data
        
        logger.info(f"Preparing training data from {start_date} to {end_date}")
        
        # Get all active stocks
        stocks_result = await self.db.execute(
            select(Stock).where(Stock.is_active == True)
        )
        stocks = stocks_result.scalars().all()
        
        all_features = []
        all_labels = []
        
        for stock in stocks:
            try:
                features, labels = await self._prepare_stock_data(
                    stock, start_date, end_date
                )
                if features is not None and len(features) > 0:
                    all_features.append(features)
                    all_labels.append(labels)
            except Exception as e:
                logger.warning(f"Failed to prepare data for {stock.symbol}: {e}")
        
        if not all_features:
            raise ValueError("No training data available")
        
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_labels, ignore_index=True)
        
        logger.info(f"Prepared {len(X)} training samples from {len(stocks)} stocks")
        
        return X, y
    
    async def _prepare_stock_data(
        self,
        stock: Stock,
        start_date: date,
        end_date: date,
    ) -> tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepare features and labels for a single stock."""
        
        # Get OHLC data
        ohlc_result = await self.db.execute(
            select(OHLCData)
            .where(OHLCData.stock_id == stock.id)
            .where(OHLCData.date >= start_date)
            .order_by(OHLCData.date)
        )
        ohlc_records = ohlc_result.scalars().all()
        
        if len(ohlc_records) < 50:
            return None, None
        
        # Get indicators
        indicator_result = await self.db.execute(
            select(TechnicalIndicator)
            .where(TechnicalIndicator.stock_id == stock.id)
            .where(TechnicalIndicator.date >= start_date)
            .where(TechnicalIndicator.date <= end_date)
            .order_by(TechnicalIndicator.date)
        )
        indicators = indicator_result.scalars().all()
        
        if len(indicators) < 30:
            return None, None
        
        # Create DataFrames
        ohlc_df = pd.DataFrame([{
            "date": r.date,
            "close": r.close,
        } for r in ohlc_records])
        ohlc_df.set_index("date", inplace=True)
        
        indicator_df = pd.DataFrame([{
            "date": r.date,
            "returns_1d": r.returns_1d,
            "returns_5d": r.returns_5d,
            "returns_20d": r.returns_20d,
            "rsi_14": r.rsi_14,
            "ema_20": r.ema_20,
            "ema_50": r.ema_50,
            "volatility_20": r.volatility_20,
            "atr_14": r.atr_14,
            "volume_ratio": r.volume_ratio,
            "adx_14": r.adx_14,
            "macd_histogram": r.macd_histogram,
            "bb_upper": r.bb_upper,
            "bb_lower": r.bb_lower,
            "relative_strength_nifty": r.relative_strength_nifty,
        } for r in indicators])
        indicator_df.set_index("date", inplace=True)
        
        # Merge with close prices
        merged = indicator_df.join(ohlc_df, how="inner")
        
        if len(merged) < 30:
            return None, None
        
        # Prepare features
        features = StockClassifier.prepare_features(merged, merged["close"])
        
        # Calculate future returns for labels
        future_returns = merged["close"].pct_change(self.lookahead_days).shift(-self.lookahead_days) * 100
        labels = StockClassifier.create_labels(
            future_returns, self.buy_threshold, self.avoid_threshold
        )
        
        # Remove rows without labels (end of data)
        valid_mask = labels.notna()
        features = features[valid_mask]
        labels = labels[valid_mask]
        
        return features, labels
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> tuple[StockClassifier, dict]:
        """
        Train model with time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Labels
            n_splits: Number of CV splits
            
        Returns:
            Tuple of (trained model, validation metrics)
        """
        logger.info("Starting model training with time-series CV")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train fold model
            fold_model = StockClassifier()
            fold_model.fit(X_train, y_train)
            
            # Validate
            y_pred = fold_model.predict(X_val)
            y_proba = fold_model.predict_proba(X_val)
            
            fold_metrics = calculate_metrics(y_val, y_pred, y_proba)
            cv_metrics.append(fold_metrics)
            
            logger.info(f"Fold {fold + 1}: Accuracy = {fold_metrics['accuracy']:.4f}")
        
        # Average CV metrics
        avg_metrics = {}
        for key in cv_metrics[0].keys():
            if isinstance(cv_metrics[0][key], (int, float)):
                avg_metrics[key] = np.mean([m[key] for m in cv_metrics])
        
        # Train final model on all data
        final_model = StockClassifier()
        final_model.fit(X, y)
        
        logger.info(f"Training complete. Avg CV Accuracy: {avg_metrics['accuracy']:.4f}")
        
        return final_model, avg_metrics
    
    async def train_and_save(
        self,
        version: Optional[str] = None,
    ) -> tuple[str, dict]:
        """
        Complete training pipeline: prepare data, train, save.
        
        Args:
            version: Model version string (auto-generated if None)
            
        Returns:
            Tuple of (model_path, metrics)
        """
        if version is None:
            version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Prepare data
        X, y = await self.prepare_training_data()
        
        # Train model
        model, metrics = self.train_model(X, y)
        
        # Save model
        model_path = Path(settings.model_path) / f"model_{version}.pkl"
        model.save(str(model_path))
        
        # Run backtest
        backtest_metrics = await self._run_backtest(model, X, y)
        metrics.update(backtest_metrics)
        
        # Save metrics to database
        model_metrics = ModelMetrics(
            version=version,
            model_type="xgboost",
            trained_at=datetime.utcnow(),
            training_samples=len(X),
            validation_samples=int(len(X) * 0.2),
            feature_count=len(FEATURE_COLUMNS),
            feature_names={"columns": FEATURE_COLUMNS},
            accuracy=metrics["accuracy"],
            precision_buy=metrics.get("precision_buy"),
            precision_hold=metrics.get("precision_hold"),
            precision_avoid=metrics.get("precision_avoid"),
            recall_buy=metrics.get("recall_buy"),
            recall_hold=metrics.get("recall_hold"),
            recall_avoid=metrics.get("recall_avoid"),
            f1_score=metrics.get("f1_score"),
            backtest_return=metrics.get("total_return"),
            backtest_sharpe=metrics.get("sharpe_ratio"),
            backtest_max_drawdown=metrics.get("max_drawdown"),
            backtest_win_rate=metrics.get("win_rate"),
            hyperparameters={"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1},
            model_path=str(model_path),
            is_active=True,
        )
        
        # Deactivate previous active models
        from sqlalchemy import update
        await self.db.execute(
            update(ModelMetrics).where(ModelMetrics.is_active == True).values(is_active=False)
        )
        
        self.db.add(model_metrics)
        await self.db.commit()
        
        logger.info(f"Model {version} saved and registered")
        
        return str(model_path), metrics
    
    async def _run_backtest(
        self,
        model: StockClassifier,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict:
        """Run simple backtest on training data."""
        
        # Use last 20% for backtest
        split_idx = int(len(X) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        return backtest_strategy(y_test.values, y_pred)
