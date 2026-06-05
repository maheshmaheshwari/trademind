"""
Nifty 500 AI — Trade Signal Generator with Actionable Details

Loads all final models, runs inference on latest data, and produces
a readable JSON with:
  - Buy Price (entry)
  - Target Price (sell for profit)
  - Stop Loss (exit to limit loss)
  - Risk:Reward ratio
  - Signal + confidence

Each run appends a NEW timestamped record to data/trade_history.json
and also writes the latest to data/trade_signals_latest.json

Usage:
    PYTHONPATH=. python generate_trades.py
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from analysis.model_training import load_data_for_symbol, engineer_features_and_target

FINAL_DIR = "final_models"
OUTPUT_DIR = "data"


def calculate_trade_levels(df, signal, horizon, model_target_pct):
    """Calculate buy price, target price, and stop loss based on ATR and model horizon."""
    latest_close = float(df["close"].iloc[-1])
    latest_high = float(df["high"].iloc[-1])
    latest_low = float(df["low"].iloc[-1])
    
    # ATR-based stop loss (use 14-day ATR if available)
    if "atr_14" in df.columns and not pd.isna(df["atr_14"].iloc[-1]):
        atr = float(df["atr_14"].iloc[-1])
    else:
        # Calculate simple ATR from price data
        highs = df["high"].tail(14)
        lows = df["low"].tail(14)
        closes = df["close"].tail(14)
        tr = pd.concat([
            highs - lows,
            (highs - closes.shift()).abs(),
            (lows - closes.shift()).abs()
        ], axis=1).max(axis=1)
        atr = float(tr.mean())
    
    atr_pct = atr / latest_close * 100 if latest_close > 0 else 2.0
    
    # Target % based on model horizon
    target_pct = model_target_pct  # e.g., 0.5% for 1 week, 5% for 3 months
    
    # Buy/Sell logic
    if signal in ("STRONG BUY", "BUY"):
        buy_price = latest_close
        target_price = round(buy_price * (1 + target_pct / 100), 2)
        # Stop loss: 1.5x ATR below buy price (tighter for short horizon)
        sl_multiplier = 1.5 if "Week" in str(horizon) else 2.0
        stop_loss = round(buy_price - (atr * sl_multiplier), 2)
        trade_type = "LONG"
    elif signal in ("SELL", "STRONG SELL"):
        buy_price = None  # Don't buy
        target_price = round(latest_close * (1 - target_pct / 100), 2)  # Price expected to drop to
        stop_loss = round(latest_close + (atr * 1.5), 2)  # Stop loss above current for short
        trade_type = "SHORT"
    else:  # HOLD
        buy_price = None
        target_price = None
        stop_loss = None
        trade_type = "HOLD"
    
    # Risk/Reward ratio
    if trade_type == "LONG" and stop_loss and target_price:
        risk = buy_price - stop_loss
        reward = target_price - buy_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
    elif trade_type == "SHORT" and stop_loss and target_price:
        risk = stop_loss - latest_close
        reward = latest_close - target_price
        rr_ratio = round(reward / risk, 2) if risk > 0 else 0
    else:
        rr_ratio = None
    
    return {
        "trade_type": trade_type,
        "current_price": latest_close,
        "buy_price": buy_price,
        "target_price": target_price,
        "stop_loss": stop_loss,
        "atr_14": round(atr, 2),
        "atr_pct": round(atr_pct, 2),
        "risk_reward": rr_ratio,
        "expected_return_pct": target_pct,
    }


def _get_delivery_pct(symbol: str) -> float:
    """Fetch latest delivery % for a symbol from DB. Returns 50.0 if not available."""
    try:
        from database.db import get_connection, _execute
        conn = get_connection()
        cur = _execute(conn,
            "SELECT delivery_pct FROM delivery_data WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            (symbol,))
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] else 50.0
    except Exception:
        return 50.0


def _get_consumed_volume(symbol: str) -> int:
    """Fetch already-consumed volume for the active signal of this symbol."""
    try:
        from database.db import get_connection, _execute
        conn = get_connection()
        cur = _execute(conn,
            "SELECT consumed_volume, recommended_volume FROM trade_signals WHERE symbol = ? AND is_active = TRUE ORDER BY generated_date DESC LIMIT 1",
            (symbol,))
        row = cur.fetchone()
        conn.close()
        if row:
            consumed    = int(row[0] or 0)
            recommended = int(row[1] or 0)
            return consumed, recommended
        return 0, 0
    except Exception:
        return 0, 0


def calculate_market_impact(qty: int, adv: int, price: float, volatility_pct: float) -> float:
    """
    Square-root market impact model (industry standard).

    Impact = σ × sqrt(Q / ADV)
    where:
        σ   = daily volatility (as fraction)
        Q   = order quantity
        ADV = average daily volume

    Returns expected price impact as a percentage.
    For Indian markets, this is well-calibrated for orders < 5% ADV.
    """
    if adv <= 0 or price <= 0:
        return 0.0
    sigma = volatility_pct / 100.0
    participation = qty / adv
    impact_pct = sigma * (participation ** 0.5) * 100
    return round(impact_pct, 3)


def calculate_position_sizing(df, signal, buy_price, symbol: str = ""):
    """
    Calculate safe position size using market impact model.

    Uses the square-root market impact formula + delivery % adjustment:
    - High delivery % (>60%) → institutional stock, can absorb more volume
    - Low delivery  % (<30%) → speculative stock, reduce safe qty
    - Checks already-consumed volume from active signal
    - Ensures per-user order won't move the market price by > 0.5%
    """
    vol_20d = int(df["volume"].tail(20).mean()) if len(df) >= 20 else int(df["volume"].mean())
    vol_50d = int(df["volume"].tail(50).mean()) if len(df) >= 50 else vol_20d
    avg_daily_volume = min(vol_20d, vol_50d)

    price = buy_price if buy_price else float(df["close"].iloc[-1])

    # Daily volatility (annualised → daily)
    returns = df["close"].pct_change().dropna()
    volatility_pct = float(returns.tail(20).std() * 100) if len(returns) >= 5 else 2.0

    # ── Delivery % adjustment ──────────────────────────────────────────────
    delivery_pct = _get_delivery_pct(symbol) if symbol else 50.0
    if delivery_pct >= 60:
        delivery_factor = 1.3    # institutional stock — can absorb more
    elif delivery_pct >= 45:
        delivery_factor = 1.0    # normal
    elif delivery_pct >= 30:
        delivery_factor = 0.75   # moderate caution
    else:
        delivery_factor = 0.5    # speculative — reduce significantly

    # ── Base safe qty: 2% of ADV × delivery factor ────────────────────────
    SAFE_VOLUME_PCT       = 0.02
    ESTIMATED_USERS       = 100
    max_safe_qty_raw      = int(avg_daily_volume * SAFE_VOLUME_PCT * delivery_factor)

    # ── Market impact check: ensure per-platform order < 0.5% price impact ─
    # Solve: σ × sqrt(Q/ADV) = 0.005  →  Q = ADV × (0.005/σ)²
    target_impact_pct = 0.5
    sigma = volatility_pct / 100.0
    if sigma > 0:
        impact_limit_qty = int(avg_daily_volume * (target_impact_pct / 100.0 / sigma) ** 2)
    else:
        impact_limit_qty = max_safe_qty_raw

    max_safe_qty = min(max_safe_qty_raw, impact_limit_qty)
    max_qty_per_user = max(1, int(max_safe_qty / ESTIMATED_USERS))

    # ── Remaining capacity (subtract already-consumed volume) ──────────────
    consumed_volume, recommended_volume = _get_consumed_volume(symbol) if symbol else (0, max_safe_qty)
    remaining_platform = max(0, (recommended_volume or max_safe_qty) - consumed_volume)
    remaining_per_user = max(1, int(remaining_platform / ESTIMATED_USERS))
    # Use minimum of calculated safe qty and remaining capacity
    effective_qty_per_user = min(max_qty_per_user, remaining_per_user)

    # Actual price impact for the per-user order
    price_impact = calculate_market_impact(effective_qty_per_user, avg_daily_volume, price, volatility_pct)

    # ── Turnover and liquidity ─────────────────────────────────────────────
    daily_turnover = avg_daily_volume * price
    if daily_turnover >= 50_00_00_000:
        liquidity = "VERY_HIGH"
    elif daily_turnover >= 10_00_00_000:
        liquidity = "HIGH"
    elif daily_turnover >= 2_00_00_000:
        liquidity = "MEDIUM"
    elif daily_turnover >= 50_00_000:
        liquidity = "LOW"
    else:
        liquidity = "VERY_LOW"

    min_qty = max(1, int(5000 / price)) if price > 0 else 1

    return {
        "avg_daily_volume":         avg_daily_volume,
        "avg_daily_volume_20d":     vol_20d,
        "avg_daily_volume_50d":     vol_50d,
        "daily_turnover_cr":        round(daily_turnover / 1_00_00_000, 2),
        "max_safe_qty_total":       max_safe_qty,
        "max_qty_per_user":         effective_qty_per_user,
        "max_safe_investment":      round(max_safe_qty * price, 2),
        "max_investment_per_user":  round(effective_qty_per_user * price, 2),
        "min_qty":                  min_qty,
        "liquidity":                liquidity,
        "safe_volume_pct":          SAFE_VOLUME_PCT * 100,
        "estimated_concurrent_users": ESTIMATED_USERS,
        # Market impact analytics
        "volatility_pct":           round(volatility_pct, 2),
        "delivery_pct":             round(delivery_pct, 1),
        "delivery_factor":          delivery_factor,
        "price_impact_pct":         price_impact,
        "consumed_volume":          consumed_volume,
        "remaining_volume":         remaining_platform,
        "volume_utilisation_pct":   round(consumed_volume / max(recommended_volume or max_safe_qty, 1) * 100, 1),
    }


def generate_signals():
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    
    # Load stock names from angel_tokens.json
    name_map = {}
    tokens_path = os.path.join(OUTPUT_DIR, "angel_tokens.json")
    if os.path.exists(tokens_path):
        with open(tokens_path) as f:
            tokens = json.load(f)
        for sym, info in tokens.items():
            name_map[f"{sym}.NS"] = info.get("name", sym)
    
    model_files = sorted([f for f in os.listdir(FINAL_DIR) if f.endswith("_final.pkl")])
    print(f"📊 {len(model_files)} final models found\n")
    
    trades = []
    errors = []
    
    for mf in model_files:
        symbol = mf.replace("_final.pkl", "")
        
        try:
            artifact = joblib.load(os.path.join(FINAL_DIR, mf))
            model = artifact["model"]
            features = artifact["features"]
            metrics = artifact["metrics"]
            model_name = artifact.get("model_name", "Unknown")
            horizon = artifact.get("horizon", "Unknown")
            threshold = artifact.get("threshold", 0.5)
            target_pct = artifact.get("target_pct", 2.0)
            
            # Load latest data
            df = load_data_for_symbol(symbol)
            if df.empty:
                print(f"Skipping {symbol}: df is empty")
                continue
            if len(df) < 60:
                print(f"Skipping {symbol}: df length {len(df)} < 60")
                continue
            
            # Engineer features
            try:
                X, _ = engineer_features_and_target(df, forward_days=5, target_pct=0.5)
            except Exception as e:
                print(f"Skipping {symbol}: Error in engineer_features_and_target: {e}")
                continue
                
            if X.empty:
                print(f"Skipping {symbol}: X is empty")
                continue
            
            latest = X.iloc[-1:]
            missing = [f for f in features if f not in latest.columns]
            for f in missing:
                latest[f] = 0
            latest = latest[features]
            latest = latest.replace([np.inf, -np.inf], 0).fillna(0)
            
            # Predict
            prob = model.predict_proba(latest)[0]
            buy_prob = prob[1] if len(prob) > 1 else prob[0]
            
            acc = metrics["accuracy"]
            prec = metrics["precision"]
            
            # Determine signal
            if buy_prob >= 0.75 and acc >= 0.80:
                signal = "STRONG BUY"
            elif buy_prob >= 0.60 and acc >= 0.70:
                signal = "BUY"
            elif buy_prob >= 0.40:
                signal = "HOLD"
            elif buy_prob >= 0.25:
                signal = "SELL"
            else:
                signal = "STRONG SELL"
            
            # Calculate trade levels
            levels = calculate_trade_levels(df, signal, horizon, target_pct)
            
            # Calculate position sizing with volume caps
            position = calculate_position_sizing(df, signal, levels["buy_price"], symbol=symbol)
            
            # Feature importance — top 5
            top_features = []
            if hasattr(model, "feature_importances_"):
                imp = dict(zip(features, model.feature_importances_))
                top_features = [{"feature": f, "importance": round(float(v), 4)} 
                               for f, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:5]]
            
            # Sentiment info
            sent_info = {}
            for col in ["sent_stock", "mkt_sentiment"]:
                if col in df.columns:
                    val = df[col].iloc[-1]
                    sent_info[col] = round(float(val), 4) if not pd.isna(val) else 0
            
            trade = {
                "symbol": symbol,
                "name": name_map.get(symbol, symbol.replace(".NS", "")),
                "signal": signal,
                "confidence": round(float(buy_prob) * 100, 1),
                "trade": {
                    "type": levels["trade_type"],
                    "buy_price": levels["buy_price"],
                    "target_price": levels["target_price"],
                    "stop_loss": levels["stop_loss"],
                    "risk_reward": levels["risk_reward"],
                    "expected_return_pct": levels["expected_return_pct"],
                },
                "position": {
                    "avg_daily_volume":          position["avg_daily_volume"],
                    "daily_turnover_cr":         position["daily_turnover_cr"],
                    "liquidity":                 position["liquidity"],
                    "max_safe_qty":              position["max_safe_qty_total"],
                    "max_qty_per_user":          position["max_qty_per_user"],
                    "max_investment_per_user":   position["max_investment_per_user"],
                    "min_qty":                   position["min_qty"],
                    "recommended_volume":        position["max_safe_qty_total"],
                    # Market impact analytics
                    "volatility_pct":            position["volatility_pct"],
                    "delivery_pct":              position["delivery_pct"],
                    "price_impact_pct":          position["price_impact_pct"],
                    "consumed_volume":           position["consumed_volume"],
                    "remaining_volume":          position["remaining_volume"],
                    "volume_utilisation_pct":    position["volume_utilisation_pct"],
                },
                "price": {
                    "current": levels["current_price"],
                    "atr_14": levels["atr_14"],
                    "atr_pct": levels["atr_pct"],
                },
                "model": {
                    "name": model_name,
                    "horizon": horizon,
                    "accuracy": round(acc * 100, 1),
                    "precision": round(prec * 100, 1),
                },
                "sentiment": sent_info,
                "top_drivers": top_features,
                "generated_at": timestamp,
            }
            trades.append(trade)
            
            # Print actionable trades only
            if signal in ("STRONG BUY", "BUY"):
                icon = "🟢🟢" if signal == "STRONG BUY" else "🟢"
                print(f"   {icon} {symbol:<18} {signal:<12} conf:{buy_prob:.0%}  "
                      f"BUY:₹{levels['buy_price']:.2f}  TARGET:₹{levels['target_price']:.2f}  "
                      f"SL:₹{levels['stop_loss']:.2f}  R:R={levels['risk_reward']}")
            elif signal in ("SELL", "STRONG SELL"):
                icon = "🔴🔴" if signal == "STRONG SELL" else "🔴"
                print(f"   {icon} {symbol:<18} {signal:<12} conf:{buy_prob:.0%}  "
                      f"AVOID BUYING  price:₹{levels['current_price']:.2f}")
            
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
    
    # Sort by signal priority + confidence
    signal_order = {"STRONG BUY": 0, "BUY": 1, "HOLD": 2, "SELL": 3, "STRONG SELL": 4}
    trades.sort(key=lambda t: (signal_order.get(t["signal"], 5), -t["confidence"]))
    
    # Build output
    output = {
        "generated_at": timestamp,
        "total_models": len(model_files),
        "total_signals": len(trades),
        "summary": {
            "STRONG_BUY": len([t for t in trades if t["signal"] == "STRONG BUY"]),
            "BUY": len([t for t in trades if t["signal"] == "BUY"]),
            "HOLD": len([t for t in trades if t["signal"] == "HOLD"]),
            "SELL": len([t for t in trades if t["signal"] == "SELL"]),
            "STRONG_SELL": len([t for t in trades if t["signal"] == "STRONG SELL"]),
        },
        "actionable_trades": [t for t in trades if t["signal"] in ("STRONG BUY", "BUY")],
        "avoid_list": [t for t in trades if t["signal"] in ("SELL", "STRONG SELL")],
        "hold_list": [t for t in trades if t["signal"] == "HOLD"],
        "errors": errors[:5],
    }
    
    # ==========================================
    # Store trade signals in database
    # ==========================================
    try:
        from database.db import insert_trade_signals_batch

        # Store all trades (deduplicates on symbol + date)
        all_trades = trades  # trades list has all 493 signals
        stored = insert_trade_signals_batch(all_trades, today, timestamp, sync=False)
        print(f"\n   💾 Stored {stored} trade signals in database for {today}")
    except Exception as e:
        print(f"\n   ⚠️  DB storage failed: {e}")

    # Save latest snapshot
    latest_file = os.path.join(OUTPUT_DIR, "trade_signals_latest.json")
    with open(latest_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    
    # Append to history (one entry per run)
    history_file = os.path.join(OUTPUT_DIR, "trade_history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                history = json.load(f)
        except:
            history = []
    
    history_entry = {
        "run_id": now.strftime("%Y%m%d_%H%M%S"),
        "generated_at": timestamp,
        "summary": output["summary"],
        "actionable_trades": output["actionable_trades"],
        "avoid_list": [{"symbol": t["symbol"], "signal": t["signal"], "confidence": t["confidence"], 
                        "price": t["price"]["current"]} for t in output["avoid_list"]],
    }
    history.append(history_entry)
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 TRADE SIGNALS — {timestamp}")
    print(f"{'='*70}")
    print(f"   🟢🟢 STRONG BUY: {output['summary']['STRONG_BUY']}")
    print(f"   🟢   BUY:        {output['summary']['BUY']}")
    print(f"   ⚪   HOLD:       {output['summary']['HOLD']}")
    print(f"   🔴   SELL:       {output['summary']['SELL']}")
    print(f"   🔴🔴 STRONG SELL: {output['summary']['STRONG_SELL']}")
    print(f"\n   💾 Latest:  {latest_file}")
    print(f"   📜 History: {history_file} ({len(history)} runs)")
    
    if output["actionable_trades"]:
        print(f"\n{'='*70}")
        print(f"💰 TOP TRADES (sorted by confidence)")
        print(f"{'='*70}")
        print(f"{'Symbol':<18} {'Signal':<12} {'Conf':>5} {'Buy':>10} {'Target':>10} {'SL':>10} {'R:R':>5} {'Horizon':<10} {'Liquidity':<10} {'MaxQty':>8}")
        print(f"{'-'*105}")
        for t in output["actionable_trades"]:
            print(f"{t['symbol']:<18} {t['signal']:<12} {t['confidence']:>4.0f}% "
                  f"₹{t['trade']['buy_price']:>8.2f} ₹{t['trade']['target_price']:>8.2f} "
                  f"₹{t['trade']['stop_loss']:>8.2f} {t['trade']['risk_reward'] or 0:>4.1f}x {t['model']['horizon']:<10} "
                  f"{t['position']['liquidity']:<10} {t['position']['max_qty_per_user']:>7}")
    
    return latest_file


if __name__ == "__main__":
    generate_signals()
