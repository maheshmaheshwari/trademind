"""
Service Tests

Unit tests for service layer.
"""

import pytest
import pandas as pd
import numpy as np

from app.services.feature_engineering import FeatureEngineer
from app.services.signal_generator import SignalGenerator
from app.ml.model import StockClassifier


class TestFeatureEngineer:
    """Tests for FeatureEngineer service."""
    
    def test_calculate_returns(self):
        """Test returns calculation."""
        prices = pd.Series([100, 105, 103, 108, 110])
        returns = FeatureEngineer.calculate_returns(prices, [1, 2])
        
        assert "returns_1d" in returns
        assert "returns_2d" in returns
        assert len(returns["returns_1d"]) == len(prices)
    
    def test_calculate_rsi(self):
        """Test RSI calculation."""
        prices = pd.Series(np.random.randn(100).cumsum() + 100)
        rsi = FeatureEngineer.calculate_rsi(prices, 14)
        
        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        assert all(valid_rsi >= 0)
        assert all(valid_rsi <= 100)
    
    def test_calculate_ema(self):
        """Test EMA calculation."""
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        emas = FeatureEngineer.calculate_ema(prices, [3, 5])
        
        assert "ema_3" in emas
        assert "ema_5" in emas
        assert len(emas["ema_3"]) == len(prices)
    
    def test_calculate_volatility(self):
        """Test volatility calculation."""
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        volatility = FeatureEngineer.calculate_volatility(prices, 20)
        
        # Volatility should be positive
        valid_vol = volatility.dropna()
        assert all(valid_vol >= 0)
    
    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = pd.Series(np.random.randn(50).cumsum() + 100)
        upper, middle, lower, width = FeatureEngineer.calculate_bollinger_bands(prices)
        
        valid_idx = middle.dropna().index
        # Upper should be above middle, lower below
        assert all(upper.loc[valid_idx] >= middle.loc[valid_idx])
        assert all(lower.loc[valid_idx] <= middle.loc[valid_idx])
    
    def test_classify_market_regime(self):
        """Test market regime classification."""
        # Bull market
        regime = FeatureEngineer.classify_market_regime(10.0, 0.15, 30)
        assert regime == "BULL"
        
        # Bear market
        regime = FeatureEngineer.classify_market_regime(-8.0, 0.30, 25)
        assert regime == "BEAR"
        
        # Sideways
        regime = FeatureEngineer.classify_market_regime(2.0, 0.18, 20)
        assert regime == "SIDEWAYS"


class TestSignalGenerator:
    """Tests for SignalGenerator (without database)."""
    
    def test_determine_signal_type(self):
        """Test signal type determination."""
        # BUY dominant
        signal = SignalGenerator.determine_signal_type(
            SignalGenerator(None), 0.7, 0.2, 0.1
        )
        assert signal == "BUY"
        
        # AVOID dominant
        signal = SignalGenerator.determine_signal_type(
            SignalGenerator(None), 0.1, 0.2, 0.7
        )
        assert signal == "AVOID"
        
        # HOLD dominant
        signal = SignalGenerator.determine_signal_type(
            SignalGenerator(None), 0.2, 0.6, 0.2
        )
        assert signal == "HOLD"
    
    def test_calculate_confidence(self):
        """Test confidence calculation."""
        gen = SignalGenerator(None)
        
        # High confidence (clear winner)
        conf = gen.calculate_confidence(0.8, 0.1, 0.1, "BUY")
        assert conf > 0.7
        
        # Lower confidence (close probabilities)
        conf = gen.calculate_confidence(0.4, 0.35, 0.25, "BUY")
        assert conf < 0.5
    
    def test_determine_timeframe(self):
        """Test timeframe determination."""
        gen = SignalGenerator(None)
        
        # High confidence, low volatility = longer timeframe
        tf = gen.determine_timeframe(0.85, 0.12)
        assert tf >= 15
        
        # Lower confidence, high volatility = shorter timeframe
        tf = gen.determine_timeframe(0.55, 0.35)
        assert tf <= 10


class TestStockClassifier:
    """Tests for StockClassifier model."""
    
    def test_create_labels(self):
        """Test label creation from returns."""
        returns = pd.Series([10, -5, 2, -1, 7])
        labels = StockClassifier.create_labels(returns)
        
        assert labels[0] == 2  # BUY (10% > 5%)
        assert labels[1] == 0  # AVOID (-5% < -3%)
        assert labels[2] == 1  # HOLD (2%)
        assert labels[4] == 2  # BUY (7% > 5%)
    
    def test_fit_predict(self):
        """Test model training and prediction."""
        # Create synthetic data
        np.random.seed(42)
        n_samples = 200
        
        X = pd.DataFrame({
            "returns_1d": np.random.randn(n_samples),
            "returns_5d": np.random.randn(n_samples),
            "returns_20d": np.random.randn(n_samples),
            "rsi_14": np.random.uniform(20, 80, n_samples),
            "ema_20_ratio": np.random.uniform(0.95, 1.05, n_samples),
            "ema_50_ratio": np.random.uniform(0.90, 1.10, n_samples),
            "volatility_20": np.random.uniform(0.1, 0.3, n_samples),
            "atr_14_pct": np.random.uniform(1, 3, n_samples),
            "volume_ratio": np.random.uniform(0.5, 2, n_samples),
            "adx_14": np.random.uniform(15, 40, n_samples),
            "macd_histogram": np.random.randn(n_samples),
            "bb_position": np.random.uniform(0, 1, n_samples),
            "relative_strength": np.random.uniform(0.8, 1.2, n_samples),
        })
        
        y = pd.Series(np.random.choice([0, 1, 2], n_samples))
        
        # Train model
        model = StockClassifier(n_estimators=10, max_depth=3)
        model.fit(X, y)
        
        assert model.is_fitted
        
        # Predict
        probas = model.predict_proba(X)
        assert probas.shape == (n_samples, 3)
        assert np.allclose(probas.sum(axis=1), 1.0)
        
        preds = model.predict(X)
        assert len(preds) == n_samples
        assert all(p in [0, 1, 2] for p in preds)
    
    def test_feature_importance(self):
        """Test feature importance extraction."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            f"feature_{i}": np.random.randn(n_samples) for i in range(5)
        })
        y = pd.Series(np.random.choice([0, 1, 2], n_samples))
        
        model = StockClassifier(n_estimators=10, max_depth=3)
        model.feature_columns = list(X.columns)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        assert len(importance) == 5
        assert all(isinstance(v, float) for v in importance.values())
