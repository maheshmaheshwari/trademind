"""
Nifty 500 AI — Trading Signal Generator

Generates BUY/SELL/HOLD signals based on technical indicators.
Uses a rules-based approach combining multiple indicators for confirmation.

Signal Levels:
    STRONG BUY  — multiple strong bullish confirmations
    BUY         — 2+ bullish indicators
    HOLD        — mixed or insufficient signals
    SELL        — 2+ bearish indicators
    STRONG SELL — multiple strong bearish confirmations

Usage:
    from analysis.signals import generate_signal, process_all_stocks
    from analysis.indicators import calculate_all
    import pandas as pd

    df = pd.DataFrame(price_data)
    df = calculate_all(df)
    signal, strength, reasons = generate_signal(df)
    print(f"Signal: {signal} (strength: {strength}%)")
    for r in reasons:
        print(f"  • {r}")
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.indicators import calculate_all
from database.db import (
    get_all_prices_df,
    get_all_symbols,
    get_connection,
    init_database,
    insert_ai_signal,
    insert_indicators,
    _execute,
)
import joblib
import os

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame, symbol: str) -> Tuple[str, float, List[str]]:
    """
    Analyze a DataFrame of OHLCV + indicators and generate a trading signal.
    Uses the trained XGBoost model if available, otherwise falls back to rules.
    """
    if df.empty or len(df) < 14:
        return "HOLD", 0.0, ["Insufficient data for analysis"]

    # 1. Attempt to load the trained model for the specific symbol
    model_path = f"models/best_{symbol}_v2.pkl"
    if not os.path.exists(model_path):
        model_path = f"models/xgb_{symbol}_v2.pkl" # fallback to older filename

    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            
            # 2. Engineer the exact features the model expects for the LATEST row
            # Note: We must replicate the feature engineering from model_training.py exactly
            df_feat = df.copy()
            
            # Fetch rolling sentiment from DB (since it's not in standard indicators)
            sentiment_rolling_3d = 0.0
            try:
                conn = get_connection()
                cur = _execute(conn, """
                    SELECT sentiment, confidence
                    FROM news_sentiment
                    WHERE symbol = ? OR symbol IS NULL
                    ORDER BY published_at DESC LIMIT 10
                """, (symbol,))
                news_rows = cur.fetchall()
                conn.close()
                if news_rows:
                    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
                    scores = []
                    for row in news_rows:
                        sent = row[0] or 'neutral'
                        conf = float(row[1] or 0)
                        scores.append(sentiment_map.get(sent, 0) * conf)
                    sentiment_rolling_3d = float(sum(scores) / len(scores)) if scores else 0.0
            except Exception:
                pass
                
            # Compute advanced technicals for the latest subset
            # Distance from MA
            df_feat['dist_sma_20'] = (df_feat['close'] / df_feat['sma_20'] - 1).fillna(0)
            df_feat['dist_sma_50'] = (df_feat['close'] / df_feat['sma_50'] - 1).fillna(0)
            df_feat['dist_sma_200'] = (df_feat['close'] / df_feat['sma_200'] - 1).fillna(0)
            
            # BB Position
            bb_range = df_feat['bb_upper'] - df_feat['bb_lower']
            df_feat['bb_position'] = np.where(bb_range > 0, (df_feat['close'] - df_feat['bb_lower']) / bb_range, 0.5)
            
            # Slopes
            df_feat['rsi_slope'] = df_feat['rsi_14'].diff(3).fillna(0)
            df_feat['macd_hist_slope'] = df_feat['macd_hist'].diff(2).fillna(0)
            
            # Returns
            df_feat['return_3d'] = df_feat['close'].pct_change(3).fillna(0)
            df_feat['return_5d'] = df_feat['close'].pct_change(5).fillna(0)
            
            # Add Sentiment
            df_feat['sentiment_rolling_3d'] = sentiment_rolling_3d
            
            # Extract the features expected by the model in the correct order
            # The model was trained on these columns:
            expected_features = [
                'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'atr_14', 'adx_14', 
                'stoch_k', 'stoch_d', 'india_vix', 'fii_net', 'dii_net', 
                'sentiment_rolling_3d', 'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 
                'bb_position', 'rsi_slope', 'macd_hist_slope', 'return_3d', 'return_5d'
            ]
            
            # Get the very last row for prediction
            latest_feat = df_feat.iloc[-1:].copy()
            
            # Fill missing market context if not fetched directly in this simple pass
            for col in ['india_vix', 'fii_net', 'dii_net']:
                if col not in latest_feat.columns:
                    latest_feat[col] = 0.0 # Will be populated natively in a full pipeline
                    
            X_live = latest_feat[expected_features]
            
            # 3. Model Inference
            prob_buy = float(model.predict_proba(X_live)[0][1])
            
            high_precision_threshold = 0.60
            
            reasons = [
                f"XGBoost ML Model Prediction Probability: {prob_buy*100:.1f}%",
                f"Current RSI: {float(latest_feat['rsi_14'].iloc[-1]):.1f}",
                f"MACD Histogram: {float(latest_feat['macd_hist'].iloc[-1]):.2f}",
                f"3-Day Rolling Sentiment: {sentiment_rolling_3d:.2f}"
            ]
            
            if prob_buy >= high_precision_threshold:
                return "STRONG BUY", prob_buy * 100.0, reasons
            elif prob_buy >= 0.50:
                return "BUY", prob_buy * 100.0, reasons
            elif prob_buy <= 0.30:
                return "STRONG SELL", (1 - prob_buy) * 100.0, reasons
            elif prob_buy <= 0.45:
                return "SELL", (1 - prob_buy) * 100.0, reasons
            else:
                return "HOLD", prob_buy * 100.0, reasons
                
        except Exception as e:
            logger.error(f"Failed to run ML model for {symbol}: {e}")
            # Fall back to rules below if it fails

    # ==========================================
    # FALLBACK RULES-BASED SYSTEM (Legacy)
    # ==========================================
    latest = df.iloc[-1]
    bullish_points = 0
    bearish_points = 0
    reasons = ["⚠️ ML Model not found; using legacy rules-based fallback."]

    rsi = latest.get("rsi_14")
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30: bullish_points += 2
        elif rsi > 70: bearish_points += 2

    macd_val = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    if macd_val is not None and macd_signal is not None and not pd.isna(macd_val):
        if macd_val > macd_signal: bullish_points += 1
        else: bearish_points += 1

    net_score = bullish_points - bearish_points
    strength = min(100.0, abs(net_score) / max((bullish_points + bearish_points), 1) * 100)

    if net_score >= 2: signal = "BUY"
    elif net_score <= -2: signal = "SELL"
    else: signal = "HOLD"

    return signal, round(strength, 1), reasons


def process_stock(symbol: str, days: int = 365, conn: Optional[Any] = None) -> Optional[Dict]:
    """
    Process a single stock: calculate indicators and generate signal.

    Fetches price data from the database, calculates all indicators,
    generates a signal, and stores both in the database.

    Args:
        symbol: Stock symbol (e.g. "TCS.NS")
        days: Number of days of price data to use

    Returns:
        Dict with signal details, or None if processing failed.
    """
    try:
        # Get price data from database
        prices = get_all_prices_df(symbol, days=days)
        if not prices or len(prices) < 14:
            logger.warning(f"Not enough price data for {symbol} ({len(prices) if prices else 0} rows)")
            return None

        # Create DataFrame
        df = pd.DataFrame(prices)

        # Calculate all technical indicators
        df = calculate_all(df)

        # Generate trading signal using ML Model
        signal, strength, reasons = generate_signal(df, symbol)

        # Get the latest row of indicators for storage
        latest = df.iloc[-1]
        latest_date = latest.get("date", datetime.now().strftime("%Y-%m-%d"))

        # Store indicators in database
        indicators_data = {
            "rsi_14": _safe_float(latest.get("rsi_14")),
            "macd": _safe_float(latest.get("macd")),
            "macd_signal": _safe_float(latest.get("macd_signal")),
            "macd_hist": _safe_float(latest.get("macd_hist")),
            "bb_upper": _safe_float(latest.get("bb_upper")),
            "bb_middle": _safe_float(latest.get("bb_middle")),
            "bb_lower": _safe_float(latest.get("bb_lower")),
            "sma_20": _safe_float(latest.get("sma_20")),
            "sma_50": _safe_float(latest.get("sma_50")),
            "sma_200": _safe_float(latest.get("sma_200")),
            "ema_9": _safe_float(latest.get("ema_9")),
            "ema_21": _safe_float(latest.get("ema_21")),
            "atr_14": _safe_float(latest.get("atr_14")),
            "adx_14": _safe_float(latest.get("adx_14")),
            "stoch_k": _safe_float(latest.get("stoch_k")),
            "stoch_d": _safe_float(latest.get("stoch_d")),
            "obv": _safe_float(latest.get("obv")),
            "support_1": _safe_float(latest.get("support_1")),
            "support_2": _safe_float(latest.get("support_2")),
            "support_3": _safe_float(latest.get("support_3")),
            "resistance_1": _safe_float(latest.get("resistance_1")),
            "resistance_2": _safe_float(latest.get("resistance_2")),
            "resistance_3": _safe_float(latest.get("resistance_3")),
            "signal": signal,
            "signal_strength": strength,
        }

        insert_indicators(symbol, str(latest_date), indicators_data, conn=conn)

        # Calculate target price and stop loss based on ATR
        close_price = float(latest.get("close", 0))
        atr = _safe_float(latest.get("atr_14"))

        target_price = None
        stop_loss = None

        if close_price and atr:
            if "BUY" in signal:
                target_price = round(close_price + (2 * atr), 2)
                stop_loss = round(close_price - (1.5 * atr), 2)
            elif "SELL" in signal:
                target_price = round(close_price - (2 * atr), 2)
                stop_loss = round(close_price + (1.5 * atr), 2)

        # Store AI signal
        model_ver = f"v2.0.0-ml" if "ML Model Prediction" in " ".join(reasons) else "v1.0.0-rules-fallback"
        insert_ai_signal(
            symbol=symbol,
            signal=signal,
            confidence=strength,
            model_version=model_ver,
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasons,
            features_used={"indicators": list(indicators_data.keys())},
            conn=conn,
        )

        return {
            "symbol": symbol,
            "signal": signal,
            "strength": strength,
            "reasons": reasons,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "close_price": close_price,
        }

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}")
        return None


def process_all_stocks() -> Dict:
    """
    Calculate indicators and generate signals for ALL stocks in the database.

    Returns:
        Summary dict with counts and top signals.

    Example:
        summary = process_all_stocks()
        print(f"Processed {summary['processed']} stocks")
        for s in summary['top_buys'][:5]:
            print(f"  {s['symbol']}: {s['signal']} ({s['strength']}%)")
    """
    try:
        init_database()

        # Get all symbols that have price data
        conn = get_connection()
        try:
            cur = _execute(conn,
                "SELECT DISTINCT symbol FROM prices WHERE interval = '1d' ORDER BY symbol"
            )
            symbols = [row[0] for row in cur.fetchall()]
        finally:
            conn.close()

        if not symbols:
            print("⚠️  No price data in database. Run price collector first!")
            return {"processed": 0, "failed": 0, "top_buys": [], "top_sells": []}

        processed = 0
        failed = 0
        all_signals = []

        print(f"\n🔬 Processing indicators for {len(symbols)} stocks...\n")

        for symbol in tqdm(symbols, desc="Analyzing", unit="stock"):
            result = process_stock(symbol)
            if result:
                processed += 1
                all_signals.append(result)
            else:
                failed += 1

        # Sort signals
        buy_signals = sorted(
            [s for s in all_signals if "BUY" in s["signal"]],
            key=lambda x: x["strength"],
            reverse=True,
        )
        sell_signals = sorted(
            [s for s in all_signals if "SELL" in s["signal"]],
            key=lambda x: x["strength"],
            reverse=True,
        )

        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 Signal Generation Summary")
        print(f"{'='*60}")
        print(f"✅ Processed: {processed}/{len(symbols)} stocks")
        print(f"❌ Failed: {failed}/{len(symbols)} stocks")
        print(f"\n🟢 Top BUY Signals:")
        for s in buy_signals[:10]:
            print(f"   {s['symbol']:20s} — {s['signal']:12s} (strength: {s['strength']:5.1f}%)")
        print(f"\n🔴 Top SELL Signals:")
        for s in sell_signals[:10]:
            print(f"   {s['symbol']:20s} — {s['signal']:12s} (strength: {s['strength']:5.1f}%)")
        print(f"{'='*60}\n")

        return {
            "processed": processed,
            "failed": failed,
            "top_buys": buy_signals[:10],
            "top_sells": sell_signals[:10],
            "all_signals": all_signals,
        }
    except Exception as e:
        logger.error(f"process_all_stocks failed: {e}")
        raise


def _safe_float(value) -> Optional[float]:
    """Convert a value to float safely, returning None for NaN/None."""
    if value is None:
        return None
    try:
        import math
        f = float(value)
        return round(f, 4) if not math.isnan(f) else None
    except (ValueError, TypeError):
        return None


# ==========================================
# Quick test when run directly
# ==========================================
if __name__ == "__main__":
    """Quick test: process signals for all stocks in the database."""
    logging.basicConfig(level=logging.INFO)
    summary = process_all_stocks()
    print(f"\nDone! Processed {summary['processed']} stocks.")
