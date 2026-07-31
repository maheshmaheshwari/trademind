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
import re
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
    release_connection,
    init_database,
    insert_indicators,
    _execute,
)
import joblib
import os

logger = logging.getLogger(__name__)


def generate_signal(df: pd.DataFrame, symbol: str) -> Tuple[str, float, List[str]]:
    """
    Generate a trading signal for `symbol`.

    Model-only: load the final production artifact from final_models/, run the
    full v4 feature pipeline, and predict. There is deliberately NO rules-based
    fallback — a symbol with no usable model returns HOLD.

    The old fallback scored RSI/MACD/Bollinger/delivery votes whenever the .pkl
    was missing, which meant a brand-new listing with a handful of bars produced
    a real-looking BUY/SELL indistinguishable in the UI from a model signal. Its
    strength figure was also misleading: strength = |bull-bear| / (bull+bear),
    so a single indicator firing alone scored 100 — unanimity among one voter,
    not confidence. Symbols without a model now stay HOLD until the weekly
    retrain gives them one.
    """
    if df.empty or len(df) < 14:
        return "HOLD", 0.0, ["Insufficient data for analysis"]

    # ── 1. Try final production model (final_models/{symbol}_final.pkl) ────────
    if not re.fullmatch(r'[A-Z0-9&\-]+(?:\.NS)?', symbol.upper()):
        logger.warning("Rejected unsafe symbol for model load: %s", symbol)
        return "HOLD", 0.0, ["Invalid symbol format"]

    _backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _models_dir = os.path.realpath(os.path.join(_backend_dir, "final_models"))

    final_path = os.path.join(_models_dir, f"{symbol}_final.pkl")
    if not os.path.exists(final_path):
        bare = symbol.replace(".NS", "")
        final_path = os.path.join(_models_dir, f"{bare}_final.pkl")

    # Prevent path traversal: resolved path must be inside final_models/
    if os.path.exists(final_path) and not os.path.realpath(final_path).startswith(_models_dir):
        logger.error("Path traversal attempt rejected for symbol: %s", symbol)
        return "HOLD", 0.0, ["Invalid symbol path"]

    if os.path.exists(final_path):
        try:
            from analysis.model_training import load_data_for_symbol, engineer_features_and_target
            artifact = joblib.load(final_path)
            features  = artifact["features"]
            threshold = artifact.get("threshold", 0.5)
            horizon   = artifact.get("horizon", "Unknown")
            fwd       = artifact.get("forward_days", 20)
            tgt_pct   = artifact.get("target_pct", 3.5)
            model_name = artifact.get("model_name", "Ensemble")
            metrics   = artifact.get("metrics", {})

            # Rebuild the full feature matrix from DB (market data + sentiment included)
            raw_df = load_data_for_symbol(symbol)
            if raw_df.empty or len(raw_df) < 60:
                raise ValueError("Insufficient data for full pipeline")

            X, _ = engineer_features_and_target(raw_df, forward_days=fwd, target_pct=tgt_pct)
            if X.empty:
                raise ValueError("Feature matrix empty after engineering")

            latest = X.iloc[-1:].copy()
            for f in features:
                if f not in latest.columns:
                    latest[f] = 0.0
            latest = latest[features].replace([np.inf, -np.inf], 0).fillna(0)
            # TabNet's DataLoader indexes rows by integer position, which fails on
            # named-column DataFrames. Also, MPS (Apple Silicon) rejects float64 —
            # use float32 numpy array for all models.
            latest_arr = latest.to_numpy(dtype=np.float32)

            # Ensemble or single model
            sub_models  = artifact.get("sub_models")
            sub_weights = artifact.get("sub_weights")
            if sub_models and sub_weights:
                total_w = sum(sub_weights.values())
                buy_prob = sum(
                    sub_models[mn].predict_proba(latest_arr)[0][1] * (sub_weights[mn] / total_w)
                    for mn in sub_models
                    if hasattr(sub_models[mn], "predict_proba")
                )
            else:
                model = artifact["model"]
                buy_prob = float(model.predict_proba(latest_arr)[0][1])

            acc  = metrics.get("accuracy", 0)
            prec = metrics.get("precision", 0)
            delivery_pct = float(raw_df["delivery_pct"].iloc[-1]) if "delivery_pct" in raw_df.columns else 50.0
            delivery_label = (
                f"High institutional ({delivery_pct:.1f}%)" if delivery_pct > 65
                else f"Speculative ({delivery_pct:.1f}%)" if delivery_pct < 30
                else f"Normal ({delivery_pct:.1f}%)"
            )
            reasons = [
                f"{model_name} model | horizon: {horizon} | buy_prob: {buy_prob*100:.1f}%",
                f"Model accuracy: {acc*100:.1f}%  precision: {prec*100:.1f}%",
                f"RSI-14: {float(df['rsi_14'].iloc[-1]):.1f}",
                f"MACD hist: {float(df['macd_hist'].iloc[-1]):.3f}",
                f"Delivery: {delivery_label}",
            ]

            if buy_prob >= 0.75 and acc >= 0.80:
                return "STRONG BUY", round(buy_prob * 100, 1), reasons
            elif buy_prob >= threshold:
                return "BUY", round(buy_prob * 100, 1), reasons
            elif buy_prob <= 0.25 and acc >= 0.80:
                return "STRONG SELL", round((1 - buy_prob) * 100, 1), reasons
            elif buy_prob <= (1 - threshold):
                return "SELL", round((1 - buy_prob) * 100, 1), reasons
            else:
                return "HOLD", round(buy_prob * 100, 1), reasons

        except Exception as e:
            logger.error(f"Final model inference failed for {symbol}: {e}")
            return "HOLD", 0.0, [f"Model inference failed for {symbol} — no signal"]

    # ── 2. No model → no signal ───────────────────────────────────────────────
    # Not an error: newly added constituents sit here until the weekly retrain
    # trains them. Strength is 0.0 so nothing downstream can read it as
    # conviction.
    logger.info("No final model for %s — returning HOLD (no rules fallback)", symbol)
    return "HOLD", 0.0, ["No trained model available — awaiting weekly retrain"]


def process_stock(symbol: str, days: int = 400, conn: Optional[Any] = None) -> Optional[Dict]:
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
            "signal_strength": _safe_float(strength),
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
            release_connection(conn)

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
