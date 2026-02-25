"""
XGBoost Stock Classifier Model

Classification model for BUY/HOLD/AVOID signals.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from app.config import settings

logger = logging.getLogger(__name__)

# Feature columns used for training
FEATURE_COLUMNS = [
    "returns_1d",
    "returns_5d",
    "returns_20d",
    "rsi_14",
    "ema_20_ratio",  # price / ema_20
    "ema_50_ratio",  # price / ema_50
    "volatility_20",
    "atr_14_pct",  # atr / price
    "volume_ratio",
    "adx_14",
    "macd_histogram",
    "bb_position",  # (price - bb_lower) / (bb_upper - bb_lower)
    "relative_strength",
]

# Label encoding
LABEL_MAP = {"AVOID": 0, "HOLD": 1, "BUY": 2}
LABEL_MAP_INV = {v: k for k, v in LABEL_MAP.items()}


class StockClassifier:
    """XGBoost classifier for stock signals."""
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 3,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize classifier with hyperparameters.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            min_child_weight: Minimum child weight
            subsample: Subsample ratio
            colsample_bytree: Column sample ratio
            random_state: Random seed
        """
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            objective="multi:softprob",
            num_class=3,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
        self.feature_columns = FEATURE_COLUMNS
        self.is_fitted = False
    
    @staticmethod
    def prepare_features(df: pd.DataFrame, close_price: pd.Series) -> pd.DataFrame:
        """
        Prepare feature matrix from raw indicator data.
        
        Args:
            df: DataFrame with technical indicators
            close_price: Close price series
            
        Returns:
            DataFrame with computed features
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features["returns_1d"] = df["returns_1d"]
        features["returns_5d"] = df["returns_5d"]
        features["returns_20d"] = df["returns_20d"]
        
        # RSI
        features["rsi_14"] = df["rsi_14"]
        
        # EMA ratios
        features["ema_20_ratio"] = close_price / df["ema_20"]
        features["ema_50_ratio"] = close_price / df["ema_50"]
        
        # Volatility
        features["volatility_20"] = df["volatility_20"]
        features["atr_14_pct"] = df["atr_14"] / close_price * 100
        
        # Volume
        features["volume_ratio"] = df["volume_ratio"]
        
        # Trend
        features["adx_14"] = df["adx_14"]
        features["macd_histogram"] = df["macd_histogram"]
        
        # Bollinger position
        bb_range = df["bb_upper"] - df["bb_lower"]
        features["bb_position"] = (close_price - df["bb_lower"]) / bb_range
        
        # Relative strength (placeholder - would be vs NIFTY)
        features["relative_strength"] = df.get("relative_strength_nifty", 1.0)
        
        return features
    
    @staticmethod
    def create_labels(
        future_returns: pd.Series,
        buy_threshold: float = 5.0,
        avoid_threshold: float = -3.0,
    ) -> pd.Series:
        """
        Create labels from future returns.
        
        Args:
            future_returns: Returns over lookahead period
            buy_threshold: %return threshold for BUY
            avoid_threshold: %return threshold for AVOID
            
        Returns:
            Series with labels (0=AVOID, 1=HOLD, 2=BUY)
        """
        labels = pd.Series(index=future_returns.index, dtype=int)
        labels[:] = LABEL_MAP["HOLD"]
        labels[future_returns >= buy_threshold] = LABEL_MAP["BUY"]
        labels[future_returns <= avoid_threshold] = LABEL_MAP["AVOID"]
        
        return labels
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StockClassifier":
        """
        Train the classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            
        Returns:
            Self
        """
        # Drop rows with missing values
        valid_mask = X.notna().all(axis=1) & y.notna()
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        logger.info(f"Training on {len(X_clean)} samples")
        
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, 3) with probabilities
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Handle missing values
        X_filled = X.fillna(X.mean())
        
        return self.model.predict_proba(X_filled)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array with predicted labels
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores."""
        if not self.is_fitted:
            return {}
        
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))
    
    def save(self, filepath: str) -> None:
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "wb") as f:
            pickle.dump({
                "model": self.model,
                "feature_columns": self.feature_columns,
                "is_fitted": self.is_fitted,
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> "StockClassifier":
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded classifier
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        classifier = cls()
        classifier.model = data["model"]
        classifier.feature_columns = data["feature_columns"]
        classifier.is_fitted = data["is_fitted"]
        
        logger.info(f"Model loaded from {filepath}")
        
        return classifier
