"""
Feature Engineering Service

Compute technical indicators from OHLC data.
"""

import logging
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Stock, OHLCData, TechnicalIndicator

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Compute and store technical indicators."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @staticmethod
    def calculate_returns(close_prices: pd.Series, periods: list[int]) -> dict:
        """
        Calculate returns for different periods.
        
        Args:
            close_prices: Series of closing prices
            periods: List of periods (e.g., [1, 5, 20])
            
        Returns:
            Dict with returns for each period
        """
        returns = {}
        for period in periods:
            returns[f"returns_{period}d"] = close_prices.pct_change(period) * 100
        return returns
    
    @staticmethod
    def calculate_rsi(close_prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            close_prices: Series of closing prices
            period: RSI period (default 14)
            
        Returns:
            RSI values
        """
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_ema(close_prices: pd.Series, periods: list[int]) -> dict:
        """
        Calculate Exponential Moving Averages.
        
        Args:
            close_prices: Series of closing prices
            periods: List of EMA periods
            
        Returns:
            Dict with EMA values
        """
        emas = {}
        for period in periods:
            emas[f"ema_{period}"] = close_prices.ewm(span=period, adjust=False).mean()
        return emas
    
    @staticmethod
    def calculate_volatility(close_prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate rolling volatility (standard deviation of returns).
        
        Args:
            close_prices: Series of closing prices
            period: Rolling window period
            
        Returns:
            Volatility values (annualized)
        """
        returns = close_prices.pct_change()
        volatility = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
        return volatility
    
    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period
            
        Returns:
            ATR values
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def calculate_macd(close_prices: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD, Signal Line, and Histogram.
        
        Args:
            close_prices: Series of closing prices
            
        Returns:
            Tuple of (MACD, Signal, Histogram)
        """
        ema_12 = close_prices.ewm(span=12, adjust=False).mean()
        ema_26 = close_prices.ewm(span=26, adjust=False).mean()
        
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return macd, signal, histogram
    
    @staticmethod
    def calculate_bollinger_bands(close_prices: pd.Series, period: int = 20, std_dev: float = 2) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Args:
            close_prices: Series of closing prices
            period: SMA period
            std_dev: Number of standard deviations
            
        Returns:
            Tuple of (upper, middle, lower, width)
        """
        middle = close_prices.rolling(window=period).mean()
        std = close_prices.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        width = (upper - lower) / middle * 100
        
        return upper, middle, lower, width
    
    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADX period
            
        Returns:
            ADX values
        """
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # When +DM > -DM, -DM = 0 and vice versa
        plus_dm[(plus_dm <= minus_dm)] = 0
        minus_dm[(minus_dm <= plus_dm)] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def classify_market_regime(
        returns_20d: float,
        volatility: float,
        adx: float,
    ) -> str:
        """
        Classify market regime based on indicators.
        
        Args:
            returns_20d: 20-day returns
            volatility: Current volatility
            adx: ADX value
            
        Returns:
            Market regime: BULL, BEAR, or SIDEWAYS
        """
        if returns_20d > 5 and volatility < 0.25 and adx > 25:
            return "BULL"
        elif returns_20d < -5 and volatility > 0.20:
            return "BEAR"
        else:
            return "SIDEWAYS"
    
    async def compute_indicators_for_stock(
        self,
        stock: Stock,
        end_date: Optional[date] = None,
    ) -> int:
        """
        Compute and store indicators for a stock.
        
        Args:
            stock: Stock model instance
            end_date: End date for computation
            
        Returns:
            Number of indicator records created
        """
        if end_date is None:
            end_date = date.today()
        
        start_date = end_date - timedelta(days=300)  # Need history for indicators
        
        # Fetch OHLC data
        result = await self.db.execute(
            select(OHLCData)
            .where(OHLCData.stock_id == stock.id)
            .where(OHLCData.date >= start_date)
            .order_by(OHLCData.date)
        )
        ohlc_records = result.scalars().all()
        
        if len(ohlc_records) < 50:
            logger.warning(f"Insufficient data for {stock.symbol}: {len(ohlc_records)} records")
            return 0
        
        # Convert to DataFrame
        df = pd.DataFrame([{
            "date": r.date,
            "open": r.open,
            "high": r.high,
            "low": r.low,
            "close": r.close,
            "volume": r.volume,
        } for r in ohlc_records])
        df.set_index("date", inplace=True)
        
        # Calculate all indicators
        returns = self.calculate_returns(df["close"], [1, 5, 20])
        rsi = self.calculate_rsi(df["close"])
        emas = self.calculate_ema(df["close"], [20, 50, 200])
        volatility = self.calculate_volatility(df["close"])
        atr = self.calculate_atr(df["high"], df["low"], df["close"])
        macd, macd_signal, macd_hist = self.calculate_macd(df["close"])
        bb_upper, bb_middle, bb_lower, bb_width = self.calculate_bollinger_bands(df["close"])
        adx = self.calculate_adx(df["high"], df["low"], df["close"])
        
        # Volume indicators
        volume_sma = df["volume"].rolling(window=20).mean()
        volume_ratio = df["volume"] / volume_sma
        
        records_created = 0
        
        # Only store for recent dates
        recent_dates = df.index[-30:]
        
        for idx in recent_dates:
            # Check if already exists
            existing = await self.db.execute(
                select(TechnicalIndicator).where(
                    TechnicalIndicator.stock_id == stock.id,
                    TechnicalIndicator.date == idx,
                )
            )
            
            if existing.scalar_one_or_none():
                continue
            
            # Get regime
            regime = self.classify_market_regime(
                returns["returns_20d"].get(idx, 0) or 0,
                volatility.get(idx, 0) or 0,
                adx.get(idx, 0) or 0,
            )
            
            indicator = TechnicalIndicator(
                stock_id=stock.id,
                date=idx,
                returns_1d=self._safe_value(returns["returns_1d"].get(idx)),
                returns_5d=self._safe_value(returns["returns_5d"].get(idx)),
                returns_20d=self._safe_value(returns["returns_20d"].get(idx)),
                rsi_14=self._safe_value(rsi.get(idx)),
                ema_20=self._safe_value(emas["ema_20"].get(idx)),
                ema_50=self._safe_value(emas["ema_50"].get(idx)),
                ema_200=self._safe_value(emas["ema_200"].get(idx)),
                volatility_20=self._safe_value(volatility.get(idx)),
                atr_14=self._safe_value(atr.get(idx)),
                market_regime=regime,
                volume_sma_20=self._safe_value(volume_sma.get(idx)),
                volume_ratio=self._safe_value(volume_ratio.get(idx)),
                adx_14=self._safe_value(adx.get(idx)),
                macd=self._safe_value(macd.get(idx)),
                macd_signal=self._safe_value(macd_signal.get(idx)),
                macd_histogram=self._safe_value(macd_hist.get(idx)),
                bb_upper=self._safe_value(bb_upper.get(idx)),
                bb_middle=self._safe_value(bb_middle.get(idx)),
                bb_lower=self._safe_value(bb_lower.get(idx)),
                bb_width=self._safe_value(bb_width.get(idx)),
            )
            self.db.add(indicator)
            records_created += 1
        
        await self.db.commit()
        logger.info(f"Created {records_created} indicator records for {stock.symbol}")
        
        return records_created
    
    @staticmethod
    def _safe_value(val) -> Optional[float]:
        """Convert value to float, handling NaN."""
        if val is None or pd.isna(val):
            return None
        return float(val)
    
    async def compute_all_indicators(self, end_date: Optional[date] = None) -> dict:
        """
        Compute indicators for all active stocks.
        
        Returns:
            Summary of computation results
        """
        result = await self.db.execute(
            select(Stock).where(Stock.is_active == True)
        )
        stocks = result.scalars().all()
        
        summary = {
            "total_stocks": len(stocks),
            "successful": 0,
            "failed": 0,
            "records_created": 0,
            "errors": [],
        }
        
        for stock in stocks:
            try:
                records = await self.compute_indicators_for_stock(stock, end_date)
                summary["records_created"] += records
                summary["successful"] += 1
            except Exception as e:
                logger.error(f"Failed to compute indicators for {stock.symbol}: {e}")
                summary["failed"] += 1
                summary["errors"].append({"symbol": stock.symbol, "error": str(e)})
        
        return summary
