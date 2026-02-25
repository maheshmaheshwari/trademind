"""
Batch fetch top Nifty stocks and train models for each.
1) Fetches price data from Angel One
2) Writes prices + indicators to local SQLite
3) Runs multi-horizon model training
"""
import os, sys, json, time, datetime
import pandas as pd
import pyotp
import libsql_experimental as libsql
from dotenv import load_dotenv
from SmartApi import SmartConnect

load_dotenv(override=True)

# Top 30 Nifty stocks by market cap (symbol -> token from angel_tokens.json)
TOP_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
    "BHARTIARTL", "SBIN", "LT", "ITC", "HINDUNILVR",
    "KOTAKBANK", "AXISBANK", "BAJFINANCE", "MARUTI", "TATAMOTORS",
    "SUNPHARMA", "TITAN", "WIPRO", "HCLTECH", "NTPC",
    "ONGC", "POWERGRID", "DRREDDY", "TATASTEEL", "COALINDIA",
    "M&M", "ADANIPORTS", "BAJAJFINSV", "INDUSINDBK", "TECHM",
]


def main():
    # Load token map
    with open("data/angel_tokens.json") as f:
        token_map = json.load(f)
    
    # Check what we already have
    conn = libsql.connect("nifty500.db")
    existing = set(r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices").fetchall())
    print(f"📦 {len(existing)} symbols already in DB")
    
    # Find which top stocks need fetching
    to_fetch = []
    for sym in TOP_STOCKS:
        ns_sym = f"{sym}.NS"
        if ns_sym in existing:
            cnt = conn.execute("SELECT COUNT(*) FROM prices WHERE symbol = ?", (ns_sym,)).fetchone()[0]
            print(f"  ✅ {ns_sym}: {cnt} rows (skip)")
            continue
        if sym not in token_map:
            print(f"  ⚠️ {sym}: not in angel_tokens.json (skip)")
            continue
        to_fetch.append((sym, token_map[sym]))
    conn.close()
    
    if not to_fetch:
        print("\n✅ All top stocks already in DB!")
    else:
        print(f"\n⚠️ {len(to_fetch)} stocks to fetch from Angel One")
        
        # Login to Angel One
        smart_api = SmartConnect(api_key=os.getenv("ANGEL_API_KEY"))
        totp = pyotp.TOTP(os.getenv("ANGEL_TOTP_SECRET")).now()
        smart_api.generateSession(os.getenv("ANGEL_CLIENT_ID"), os.getenv("ANGEL_PASSWORD"), totp)
        print("🔑 Logged in to Angel One\n")
        
        from analysis.indicators import calculate_all
        from analysis.signals import _safe_float
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5 * 365)
        
        conn = libsql.connect("nifty500.db")
        
        for i, (sym, info) in enumerate(to_fetch, 1):
            ns_sym = f"{sym}.NS"
            token = info["token"]
            print(f"  [{i}/{len(to_fetch)}] {ns_sym} (token {token})...", end=" ", flush=True)
            
            try:
                res = smart_api.getCandleData({
                    "exchange": "NSE",
                    "symboltoken": token,
                    "interval": "ONE_DAY",
                    "fromdate": start_date.strftime("%Y-%m-%d 09:15"),
                    "todate": end_date.strftime("%Y-%m-%d 15:30"),
                })
                data = res.get("data")
                if not data:
                    print("❌ no data")
                    continue
                
                df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume"])
                df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                
                # Write prices
                for _, row in df.iterrows():
                    conn.execute(
                        "INSERT OR REPLACE INTO prices (symbol,exchange,date,time,open,high,low,close,volume,interval) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (ns_sym, "NSE", row["date"], None, round(row["open"], 2), round(row["high"], 2),
                         round(row["low"], 2), round(row["close"], 2), int(row["volume"]), "1d"))
                
                # Write indicators
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
                print(f"✅ {len(data)}p | {len(df_calc)-50}i")
                time.sleep(0.3)  # Rate limit
                
            except Exception as e:
                print(f"❌ {e}")
                time.sleep(1)
        
        conn.close()
        try:
            smart_api.terminateSession(os.getenv("ANGEL_CLIENT_ID"))
        except:
            pass
    
    # Now run model training for all stocks in DB
    print(f"\n{'='*60}")
    print(f"🧠 BATCH MODEL TRAINING")
    print(f"{'='*60}\n")
    
    from analysis.model_training import train_and_evaluate
    
    conn = libsql.connect("nifty500.db")
    symbols = [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM prices ORDER BY symbol").fetchall()]
    conn.close()
    
    print(f"Training models for {len(symbols)} stocks...\n")
    
    results_summary = []
    for i, sym in enumerate(symbols, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(symbols)}] {sym}")
        print(f"{'#'*60}")
        try:
            train_and_evaluate(sym)
            
            # Check if model was saved
            model_path = f"models/best_{sym}_v3.pkl"
            if os.path.exists(model_path):
                import joblib
                artifact = joblib.load(model_path)
                results_summary.append({
                    'symbol': sym,
                    'model': artifact.get('model_name', '?'),
                    'horizon': artifact.get('horizon', '?'),
                    'accuracy': artifact['metrics']['accuracy'],
                    'precision': artifact['metrics']['precision'],
                    'f1': artifact['metrics']['f1'],
                })
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Final summary
    print(f"\n\n{'='*70}")
    print(f"📊 BATCH TRAINING FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Symbol':<15} {'Model':<12} {'Horizon':<10} {'Acc':>6} {'Prec':>6} {'F1':>6}")
    print(f"{'-'*60}")
    for r in sorted(results_summary, key=lambda x: x['accuracy'], reverse=True):
        ai = "✅" if r['accuracy'] >= 0.70 else ""
        pi = "✅" if r['precision'] >= 0.70 else ""
        print(f"{r['symbol']:<15} {r['model']:<12} {r['horizon']:<10} {r['accuracy']:>5.1%}{ai} {r['precision']:>5.1%}{pi} {r['f1']:>5.1%}")
    
    hit_both = [r for r in results_summary if r['accuracy'] >= 0.70 and r['precision'] >= 0.70]
    print(f"\n✅ {len(hit_both)}/{len(results_summary)} stocks hit both 70% accuracy AND precision")


if __name__ == "__main__":
    main()
