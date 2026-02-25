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
from typing import Dict, List, Optional, Tuple

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
)

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame) -> Tuple[str, float, List[str]]:
    """
    Analyze a DataFrame of OHLCV + indicators and generate a trading signal.

    Args:
        df: DataFrame with calculated technical indicators
            (must have been processed by calculate_all() first)

    Returns:
        Tuple of (signal, strength, reasons):
            - signal: "STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"
            - strength: 0 to 100 (higher = more confident)
            - reasons: List of human-readable reason strings

    Example:
        df = calculate_all(price_dataframe)
        signal, strength, reasons = generate_signal(df)
        print(f"{signal} — confidence {strength}%")
    """
    if df.empty or len(df) < 14:
        return "HOLD", 0.0, ["Insufficient data for analysis"]

    # Get the latest row (most recent data point)
    latest = df.iloc[-1]

    # Scoring system: +1 for bullish, -1 for bearish
    bullish_points = 0
    bearish_points = 0
    reasons = []

    # ==========================================
    # 1. RSI Analysis
    # ==========================================
    rsi = latest.get("rsi_14")
    if rsi is not None and not pd.isna(rsi):
        if rsi < 30:
            bullish_points += 2
            reasons.append(f"RSI oversold at {rsi:.1f} (below 30 — strong buy zone)")
        elif rsi < 40:
            bullish_points += 1
            reasons.append(f"RSI near oversold at {rsi:.1f} (bullish)")
        elif rsi > 70:
            bearish_points += 2
            reasons.append(f"RSI overbought at {rsi:.1f} (above 70 — sell zone)")
        elif rsi > 60:
            bearish_points += 1
            reasons.append(f"RSI elevated at {rsi:.1f} (bearish caution)")
        else:
            reasons.append(f"RSI neutral at {rsi:.1f}")

    # ==========================================
    # 2. MACD Analysis
    # ==========================================
    macd_val = latest.get("macd")
    macd_signal = latest.get("macd_signal")
    macd_hist = latest.get("macd_hist")

    if macd_val is not None and macd_signal is not None and not pd.isna(macd_val):
        if macd_val > macd_signal:
            bullish_points += 1
            reasons.append("MACD above signal line (bullish crossover)")
        else:
            bearish_points += 1
            reasons.append("MACD below signal line (bearish crossover)")

        # Check histogram momentum
        if macd_hist is not None and not pd.isna(macd_hist):
            if len(df) >= 2:
                prev_hist = df.iloc[-2].get("macd_hist")
                if prev_hist is not None and not pd.isna(prev_hist):
                    if macd_hist > prev_hist and macd_hist > 0:
                        bullish_points += 1
                        reasons.append("MACD histogram increasing (growing bullish momentum)")
                    elif macd_hist < prev_hist and macd_hist < 0:
                        bearish_points += 1
                        reasons.append("MACD histogram decreasing (growing bearish momentum)")

    # ==========================================
    # 3. Moving Average Analysis (SMA)
    # ==========================================
    close = latest.get("close")
    sma_50 = latest.get("sma_50")
    sma_200 = latest.get("sma_200")

    if close is not None and sma_200 is not None and not pd.isna(sma_200):
        if close > sma_200:
            bullish_points += 1
            reasons.append(f"Price ({close:.2f}) above 200-day SMA ({sma_200:.2f}) — long-term uptrend")
        else:
            bearish_points += 1
            reasons.append(f"Price ({close:.2f}) below 200-day SMA ({sma_200:.2f}) — long-term downtrend")

    if sma_50 is not None and sma_200 is not None and not pd.isna(sma_50):
        if sma_50 > sma_200:
            bullish_points += 1
            reasons.append("Golden Cross: 50-day SMA above 200-day SMA (bullish)")
        else:
            bearish_points += 1
            reasons.append("Death Cross: 50-day SMA below 200-day SMA (bearish)")

    # ==========================================
    # 4. Bollinger Band Analysis
    # ==========================================
    bb_lower = latest.get("bb_lower")
    bb_upper = latest.get("bb_upper")

    if close is not None and bb_lower is not None and not pd.isna(bb_lower):
        if close <= bb_lower:
            bullish_points += 1
            reasons.append(f"Price touching lower Bollinger Band ({bb_lower:.2f}) — potential bounce")
        elif close >= bb_upper and bb_upper is not None and not pd.isna(bb_upper):
            bearish_points += 1
            reasons.append(f"Price touching upper Bollinger Band ({bb_upper:.2f}) — potential pullback")

    # ==========================================
    # 5. ADX Trend Strength
    # ==========================================
    adx = latest.get("adx_14")
    if adx is not None and not pd.isna(adx):
        if adx > 25:
            reasons.append(f"ADX at {adx:.1f} — strong trend in place")
        else:
            reasons.append(f"ADX at {adx:.1f} — weak/no trend (choppy market)")

    # ==========================================
    # 6. Stochastic Oscillator
    # ==========================================
    stoch_k = latest.get("stoch_k")
    if stoch_k is not None and not pd.isna(stoch_k):
        if stoch_k < 20:
            bullish_points += 1
            reasons.append(f"Stochastic %K oversold at {stoch_k:.1f}")
        elif stoch_k > 80:
            bearish_points += 1
            reasons.append(f"Stochastic %K overbought at {stoch_k:.1f}")

    # ==========================================
    # 7. EMA Cross
    # ==========================================
    ema_9 = latest.get("ema_9")
    ema_21 = latest.get("ema_21")
    if ema_9 is not None and ema_21 is not None and not pd.isna(ema_9):
        if ema_9 > ema_21:
            bullish_points += 1
            reasons.append("EMA-9 above EMA-21 (short-term bullish)")
        else:
            bearish_points += 1
            reasons.append("EMA-9 below EMA-21 (short-term bearish)")

    # ==========================================
    # DETERMINE SIGNAL
    # ==========================================
    net_score = bullish_points - bearish_points
    total_signals = bullish_points + bearish_points

    if total_signals == 0:
        return "HOLD", 0.0, ["No indicators available for analysis"]

    # Determine signal level
    if net_score >= 5 and rsi is not None and rsi < 40:
        signal = "STRONG BUY"
    elif net_score >= 3:
        signal = "STRONG BUY"
    elif net_score >= 2:
        signal = "BUY"
    elif net_score <= -5 and rsi is not None and rsi > 70:
        signal = "STRONG SELL"
    elif net_score <= -3:
        signal = "STRONG SELL"
    elif net_score <= -2:
        signal = "SELL"
    else:
        signal = "HOLD"

    # Calculate signal strength (0-100)
    strength = min(100.0, abs(net_score) / max(total_signals, 1) * 100)

    reasons.insert(0, f"Net Score: {net_score} (Bullish: {bullish_points}, Bearish: {bearish_points})")

    return signal, round(strength, 1), reasons


def process_stock(symbol: str, days: int = 365) -> Optional[Dict]:
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

        # Generate trading signal
        signal, strength, reasons = generate_signal(df)

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

        insert_indicators(symbol, str(latest_date), indicators_data)

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
        insert_ai_signal(
            symbol=symbol,
            signal=signal,
            confidence=strength,
            model_version="v1.0.0-rules",
            target_price=target_price,
            stop_loss=stop_loss,
            reasoning=reasons,
            features_used={"indicators": list(indicators_data.keys())},
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
    init_database()

    # Get all symbols that have price data
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM prices WHERE interval = '1d' ORDER BY symbol"
        ).fetchall()
        symbols = [row[0] for row in rows]
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
