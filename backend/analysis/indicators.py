"""
Nifty 500 AI — Technical Indicators Calculator

Calculates all key technical indicators using the 'ta' library.
Includes RSI, MACD, Bollinger Bands, SMA, EMA, ATR, ADX, Stochastic,
OBV, and pivot point support/resistance levels.

Usage:
    from analysis.indicators import calculate_all, calculate_support_resistance
    import pandas as pd

    # df must have columns: open, high, low, close, volume
    df = pd.DataFrame(price_data)
    df = calculate_all(df)
    print(df[['close', 'rsi_14', 'macd', 'sma_200']].tail())
"""

import logging
from typing import Optional

import pandas as pd
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.trend import (
    ADXIndicator,
    EMAIndicator,
    MACD,
    SMAIndicator,
)
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

logger = logging.getLogger(__name__)


def calculate_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate ALL technical indicators and add them as new columns to the DataFrame.

    Requires at least 200 rows for SMA-200 to be meaningful.
    NaN values at the beginning are normal (indicators need warmup data).

    Args:
        df: DataFrame with columns — date, open, high, low, close, volume.
            The 'close' column is required; others enhance accuracy.

    Returns:
        Same DataFrame with additional indicator columns added.

    Example:
        df = pd.DataFrame(get_prices("TCS.NS", days=365))
        df = calculate_all(df)
        print(df.columns.tolist())  # Shows all new indicator columns
    """
    if df.empty or len(df) < 14:
        logger.warning("Not enough data to calculate indicators (need at least 14 rows)")
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    volume = df["volume"] if "volume" in df.columns else pd.Series([0] * len(df))

    # ==========================================
    # RSI (Relative Strength Index)
    # RSI < 30 = oversold (potential buy)
    # RSI > 70 = overbought (potential sell)
    # ==========================================
    try:
        rsi = RSIIndicator(close=close, window=14)
        df["rsi_14"] = rsi.rsi()
    except Exception as e:
        logger.warning(f"RSI calculation failed: {e}")
        df["rsi_14"] = None

    # ==========================================
    # MACD (Moving Average Convergence Divergence)
    # MACD crossing above signal = bullish
    # MACD crossing below signal = bearish
    # ==========================================
    try:
        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()
    except Exception as e:
        logger.warning(f"MACD calculation failed: {e}")
        df["macd"] = df["macd_signal"] = df["macd_hist"] = None

    # ==========================================
    # Bollinger Bands
    # Price near lower band = potential support
    # Price near upper band = potential resistance
    # ==========================================
    try:
        bb = BollingerBands(close=close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
    except Exception as e:
        logger.warning(f"Bollinger Bands calculation failed: {e}")
        df["bb_upper"] = df["bb_middle"] = df["bb_lower"] = None

    # ==========================================
    # Simple Moving Averages (SMA)
    # Price > SMA-200 = long-term bullish trend
    # SMA-50 crossing above SMA-200 = "Golden Cross" (strong buy)
    # ==========================================
    try:
        df["sma_20"] = SMAIndicator(close=close, window=20).sma_indicator()
        df["sma_50"] = SMAIndicator(close=close, window=50).sma_indicator()
        df["sma_200"] = SMAIndicator(close=close, window=200).sma_indicator()
    except Exception as e:
        logger.warning(f"SMA calculation failed: {e}")
        df["sma_20"] = df["sma_50"] = df["sma_200"] = None

    # ==========================================
    # Exponential Moving Averages (EMA)
    # Faster response to recent price changes than SMA
    # ==========================================
    try:
        df["ema_9"] = EMAIndicator(close=close, window=9).ema_indicator()
        df["ema_21"] = EMAIndicator(close=close, window=21).ema_indicator()
    except Exception as e:
        logger.warning(f"EMA calculation failed: {e}")
        df["ema_9"] = df["ema_21"] = None

    # ==========================================
    # ATR (Average True Range) — measures volatility
    # Higher ATR = more volatile stock
    # ==========================================
    try:
        atr = AverageTrueRange(high=high, low=low, close=close, window=14)
        df["atr_14"] = atr.average_true_range()
    except Exception as e:
        logger.warning(f"ATR calculation failed: {e}")
        df["atr_14"] = None

    # ==========================================
    # ADX (Average Directional Index) — trend strength
    # ADX > 25 = strong trend, ADX < 20 = weak/no trend
    # ==========================================
    try:
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        df["adx_14"] = adx.adx()
    except Exception as e:
        logger.warning(f"ADX calculation failed: {e}")
        df["adx_14"] = None

    # ==========================================
    # Stochastic Oscillator
    # %K > 80 = overbought, %K < 20 = oversold
    # ==========================================
    try:
        stoch = StochRSIIndicator(close=close, window=14, smooth1=3, smooth2=3)
        df["stoch_k"] = stoch.stochrsi_k() * 100  # Convert to 0-100 scale
        df["stoch_d"] = stoch.stochrsi_d() * 100
    except Exception as e:
        logger.warning(f"Stochastic calculation failed: {e}")
        df["stoch_k"] = df["stoch_d"] = None

    # ==========================================
    # OBV (On-Balance Volume) — volume-based trend confirmation
    # Rising OBV + rising price = strong uptrend
    # ==========================================
    try:
        obv = OnBalanceVolumeIndicator(close=close, volume=volume)
        df["obv"] = obv.on_balance_volume()
    except Exception as e:
        logger.warning(f"OBV calculation failed: {e}")
        df["obv"] = None

    # ==========================================
    # Support & Resistance (Pivot Points)
    # ==========================================
    df = calculate_support_resistance(df)

    logger.info(f"Calculated all indicators — {len(df)} rows, {len(df.columns)} columns")
    return df


def calculate_support_resistance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate support and resistance levels using the Classic Pivot Point formula.

    Formula:
        Pivot = (High + Low + Close) / 3
        R1 = 2 * Pivot - Low
        R2 = Pivot + (High - Low)
        R3 = High + 2 * (Pivot - Low)
        S1 = 2 * Pivot - High
        S2 = Pivot - (High - Low)
        S3 = Low - 2 * (High - Low)

    Uses the previous day's data to calculate today's levels.

    Args:
        df: DataFrame with 'high', 'low', 'close' columns.

    Returns:
        DataFrame with support_1/2/3 and resistance_1/2/3 columns added.
    """
    if df.empty or len(df) < 2:
        return df

    try:
        # Use previous day's high, low, close
        prev_high = df["high"].shift(1)
        prev_low = df["low"].shift(1)
        prev_close = df["close"].shift(1)

        # Pivot point
        pivot = (prev_high + prev_low + prev_close) / 3

        # Resistance levels
        df["resistance_1"] = (2 * pivot - prev_low).round(2)
        df["resistance_2"] = (pivot + (prev_high - prev_low)).round(2)
        df["resistance_3"] = (prev_high + 2 * (pivot - prev_low)).round(2)

        # Support levels
        df["support_1"] = (2 * pivot - prev_high).round(2)
        df["support_2"] = (pivot - (prev_high - prev_low)).round(2)
        df["support_3"] = (prev_low - 2 * (prev_high - prev_low)).round(2)

    except Exception as e:
        logger.warning(f"Support/Resistance calculation failed: {e}")
        for col in ["support_1", "support_2", "support_3",
                     "resistance_1", "resistance_2", "resistance_3"]:
            df[col] = None

    return df
