"""
Nifty 500 AI — Full Pipeline: Fetch + Indicators + Train + Save Final Models

For all 499 Nifty stocks:
1) Fetch 5 years of price data from Angel One (skip if already in DB)
2) Calculate technical indicators
3) Train ML models with 81 features (including 12 news sentiment features)
4) Save the best model to final_models/<SYMBOL>_final.pkl

Usage:
    python train_all_final.py              # All 499 stocks
    python train_all_final.py --skip 100   # Resume from stock #100
    python train_all_final.py --batch 50   # Only process 50 stocks per run
"""
import os, sys, json, time, datetime, traceback
import pandas as pd
import numpy as np
import pyotp
import joblib
import libsql_experimental as libsql
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv(override=True)

FINAL_MODELS_DIR = "final_models"
os.makedirs(FINAL_MODELS_DIR, exist_ok=True)


def login_angel():
    """Login to Angel One SmartAPI."""
    smart_api = SmartConnect(api_key=os.getenv("ANGEL_API_KEY"))
    totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET")).now()
    smart_api.generateSession(os.getenv("ANGEL_CLIENT_ID"), os.getenv("ANGEL_PASSWORD"), totp)
    print("🔑 Logged in to Angel One")
    return smart_api


def fetch_and_store_prices(smart_api, conn, symbol, token_info):
    """Fetch 5 years of price data from Angel One and store in local DB."""
    ns_sym = f"{symbol}.NS"
    token = token_info["token"]
    
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=5 * 365)
    
    res = smart_api.getCandleData({
        "exchange": "NSE",
        "symboltoken": token,
        "interval": "ONE_DAY",
        "fromdate": start_date.strftime("%Y-%m-%d 09:15"),
        "todate": end_date.strftime("%Y-%m-%d 15:30"),
    })
    data = res.get("data")
    if not data:
        return 0
    
    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    for _, row in df.iterrows():
        conn.execute(
            "INSERT OR REPLACE INTO prices (symbol,exchange,date,time,open,high,low,close,volume,interval) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ns_sym, "NSE", row["date"], None, round(row["open"], 2), round(row["high"], 2),
             round(row["low"], 2), round(row["close"], 2), int(row["volume"]), "1d"))
    
    # Calculate and store indicators
    from analysis.indicators import calculate_all
    from analysis.signals import _safe_float
    
    df_calc = calculate_all(df)
    for j in range(50, len(df_calc)):
        r = df_calc.iloc[j]
        conn.execute(
            """INSERT OR REPLACE INTO technical_indicators 
            (symbol,date,rsi_14,macd,macd_signal,macd_hist,bb_upper,bb_middle,bb_lower,
             sma_20,sma_50,sma_200,ema_9,ema_21,atr_14,adx_14,stoch_k,stoch_d,obv,
             support_1,support_2,support_3,resistance_1,resistance_2,resistance_3,signal,signal_strength)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (ns_sym, r["date"],
             _safe_float(r.get("rsi_14")), _safe_float(r.get("macd")),
             _safe_float(r.get("macd_signal")), _safe_float(r.get("macd_hist")),
             _safe_float(r.get("bb_upper")), _safe_float(r.get("bb_middle")),
             _safe_float(r.get("bb_lower")), _safe_float(r.get("sma_20")),
             _safe_float(r.get("sma_50")), _safe_float(r.get("sma_200")),
             _safe_float(r.get("ema_9")), _safe_float(r.get("ema_21")),
             _safe_float(r.get("atr_14")), _safe_float(r.get("adx_14")),
             _safe_float(r.get("stoch_k")), _safe_float(r.get("stoch_d")),
             _safe_float(r.get("obv")),
             _safe_float(r.get("support_1")), _safe_float(r.get("support_2")),
             _safe_float(r.get("support_3")), _safe_float(r.get("resistance_1")),
             _safe_float(r.get("resistance_2")), _safe_float(r.get("resistance_3")),
             None, None))
    
    conn.commit()
    return len(data)


def train_stock(symbol_ns):
    """Train model for a stock and save to final_models/ if it passes quality bar."""
    from analysis.model_training import train_and_evaluate
    
    # train_and_evaluate saves to models/best_<symbol>_v3.pkl
    train_and_evaluate(symbol_ns)
    
    # Check if model was saved and is good enough
    src_path = f"models/best_{symbol_ns}_v3.pkl"
    if os.path.exists(src_path):
        artifact = joblib.load(src_path)
        acc = artifact["metrics"]["accuracy"]
        prec = artifact["metrics"]["precision"]
        
        # Save to final_models/ regardless of quality (user wants all)
        dst_path = os.path.join(FINAL_MODELS_DIR, f"{symbol_ns}_final.pkl")
        joblib.dump(artifact, dst_path)
        
        return {
            "symbol": symbol_ns,
            "model": artifact.get("model_name", "?"),
            "horizon": artifact.get("horizon", "?"),
            "accuracy": acc,
            "precision": prec,
            "f1": artifact["metrics"]["f1"],
            "saved": dst_path,
        }
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0, help="Skip first N stocks")
    parser.add_argument("--batch", type=int, default=0, help="Process only N stocks (0=all)")
    args = parser.parse_args()
    
    # Load all stocks
    with open("data/angel_tokens.json") as f:
        token_map = json.load(f)
    
    from database.models import ALL_TABLES, CREATE_INDEXES
    conn = libsql.connect("nifty500.db")
    for sql in ALL_TABLES:
        conn.execute(sql)
    for sql in CREATE_INDEXES:
        conn.execute(sql)
    conn.commit()
    
    # Check existing price data
    existing = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall())
    
    # Check existing final models
    existing_models = set()
    for f in os.listdir(FINAL_MODELS_DIR):
        if f.endswith("_final.pkl"):
            existing_models.add(f.replace("_final.pkl", ""))
    
    stocks = list(token_map.items())
    total = len(stocks)
    
    # Determine which stocks need price fetching
    to_fetch = [(sym, info) for sym, info in stocks if f"{sym}.NS" not in existing]
    
    print(f"{'='*60}")
    print(f"🚀 NIFTY 500 — FULL PIPELINE")
    print(f"{'='*60}")
    print(f"   Total stocks:        {total}")
    print(f"   Already have prices: {len(existing)}")
    print(f"   Need price fetch:    {len(to_fetch)}")
    print(f"   Existing final models: {len(existing_models)}")
    print(f"   Skip: {args.skip} | Batch: {args.batch or 'all'}")
    print(f"{'='*60}\n")
    
    # Step 1: Fetch missing price data
    if to_fetch:
        print(f"📥 Fetching price data for {len(to_fetch)} stocks from Angel One...\n")
        smart_api = login_angel()
        
        fetch_start = args.skip
        fetch_end = len(to_fetch) if not args.batch else min(args.skip + args.batch, len(to_fetch))
        
        for i, (sym, info) in enumerate(to_fetch[fetch_start:fetch_end], fetch_start + 1):
            ns_sym = f"{sym}.NS"
            print(f"   [{i}/{len(to_fetch)}] {ns_sym}...", end=" ", flush=True)
            try:
                count = fetch_and_store_prices(smart_api, conn, sym, info)
                if count > 0:
                    print(f"✅ {count} prices + indicators")
                    existing.add(ns_sym)
                else:
                    print("⚠️ no data")
            except Exception as e:
                err_msg = str(e)
                if "Too many" in err_msg or "rate" in err_msg.lower():
                    print(f"⏳ rate limited, waiting 10s...")
                    time.sleep(10)
                    try:
                        count = fetch_and_store_prices(smart_api, conn, sym, info)
                        if count > 0:
                            print(f"✅ {count} prices + indicators (retry)")
                            existing.add(ns_sym)
                        else:
                            print("⚠️ no data (retry)")
                    except Exception as e2:
                        print(f"❌ {e2}")
                else:
                    print(f"❌ {e}")
            time.sleep(1.0)
        
        conn.close()
        try:
            smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID"))
        except:
            pass
        print(f"\n✅ Price fetch complete. {len(existing)} stocks in DB.\n")
    
    # Step 2: Train models for all stocks with price data
    print(f"{'='*60}")
    print(f"🧠 MODEL TRAINING — {len(existing)} stocks")
    print(f"{'='*60}\n")
    
    results = []
    symbols = sorted(existing)
    
    for i, sym in enumerate(symbols, 1):
        # Skip if final model already exists
        if sym in existing_models:
            # Load existing model info
            try:
                artifact = joblib.load(os.path.join(FINAL_MODELS_DIR, f"{sym}_final.pkl"))
                results.append({
                    "symbol": sym,
                    "model": artifact.get("model_name", "?"),
                    "horizon": artifact.get("horizon", "?"),
                    "accuracy": artifact["metrics"]["accuracy"],
                    "precision": artifact["metrics"]["precision"],
                    "f1": artifact["metrics"]["f1"],
                    "saved": "(existing)",
                })
                print(f"   [{i}/{len(symbols)}] {sym}: final model exists (skip)")
                continue
            except:
                pass
        
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(symbols)}] {sym}")
        print(f"{'#'*60}")
        
        try:
            result = train_stock(sym)
            if result:
                results.append(result)
                status = "✅" if result["accuracy"] >= 0.70 and result["precision"] >= 0.70 else "⚠️"
                print(f"   {status} → {result['saved']}")
            else:
                print(f"   ⚠️ No model produced")
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            traceback.print_exc()
    
    # Final Summary
    print(f"\n\n{'='*70}")
    print(f"📊 FINAL MODELS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<17} {'Model':<12} {'Horizon':<10} {'Acc':>6} {'Prec':>6} {'F1':>6}")
    print(f"{'-'*65}")
    
    for r in sorted(results, key=lambda x: x["accuracy"], reverse=True):
        ai = "✅" if r["accuracy"] >= 0.70 else ""
        pi = "✅" if r["precision"] >= 0.70 else ""
        print(f"{r['symbol']:<17} {r['model']:<12} {str(r['horizon']):<10} {r['accuracy']:>5.1%}{ai} {r['precision']:>5.1%}{pi} {r['f1']:>5.1%}")
    
    hit_both = [r for r in results if r["accuracy"] >= 0.70 and r["precision"] >= 0.70]
    total_models = len([f for f in os.listdir(FINAL_MODELS_DIR) if f.endswith(".pkl")])
    
    print(f"\n{'='*70}")
    print(f"   Total final models saved: {total_models}")
    print(f"   ≥70% acc AND prec: {len(hit_both)}/{len(results)}")
    print(f"   Models directory: {FINAL_MODELS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
