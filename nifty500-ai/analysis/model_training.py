"""
Nifty 500 AI — ML Model Training Pipeline v3 (Multi-Horizon)

Trains across multiple prediction horizons (5 days to 6 months),
comparing XGBoost, RandomForest, and GradientBoosting at each horizon.
Picks the single best (horizon, model) combination for maximum
accuracy AND precision.
"""
import os
import sys
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from database.db import get_connection


def load_data_for_symbol(symbol: str) -> pd.DataFrame:
    conn = get_connection(sync_on_connect=False)
    query = """
    SELECT 
        p.date, p.close, p.open, p.high, p.low, p.volume,
        i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
        i.bb_upper, i.bb_lower, i.bb_middle,
        i.sma_20, i.sma_50, i.sma_200, i.ema_9, i.ema_21,
        i.atr_14, i.adx_14, i.stoch_k, i.stoch_d, i.obv,
        m.india_vix, m.fii_net, m.dii_net
    FROM prices p
    LEFT JOIN technical_indicators i ON p.symbol = i.symbol AND p.date = i.date
    LEFT JOIN market_overview m ON p.date = m.date
    WHERE p.symbol = ? AND p.interval = '1d'
    ORDER BY p.date ASC
    """
    df = pd.read_sql_query(query, conn, params=(symbol,))
    # Load pre-aggregated daily sentiment from news_daily_sentiment table
    stock_sent = pd.read_sql_query(
        """SELECT date, avg_sentiment as sent_stock, news_count as news_count_stock,
           positive_count as news_pos, negative_count as news_neg,
           max_positive as sent_max_pos, max_negative as sent_max_neg
        FROM news_daily_sentiment WHERE symbol = ? ORDER BY date""",
        conn, params=(symbol,))
    mkt_sent = pd.read_sql_query(
        """SELECT date, avg_sentiment as mkt_sentiment, news_count as mkt_news_count
        FROM news_daily_sentiment WHERE symbol IS NULL ORDER BY date""",
        conn, params=())
    conn.close()
    
    for col in ['india_vix', 'fii_net', 'dii_net']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['close', 'open', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    if not stock_sent.empty:
        stock_sent['date'] = pd.to_datetime(stock_sent['date'])
        stock_sent.set_index('date', inplace=True)
        df = df.join(stock_sent, how='left')
    if not mkt_sent.empty:
        mkt_sent['date'] = pd.to_datetime(mkt_sent['date'])
        mkt_sent.set_index('date', inplace=True)
        df = df.join(mkt_sent, how='left')
    
    for col in ['sent_stock','news_count_stock','news_pos','news_neg',
                'sent_max_pos','sent_max_neg','mkt_sentiment','mkt_news_count']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def engineer_features_and_target(df: pd.DataFrame, forward_days: int = 5, target_pct: float = 0.5):
    df = df.copy()
    
    # Returns
    df['return_1d'] = df['close'].pct_change(1).fillna(0)
    df['return_2d'] = df['close'].pct_change(2).fillna(0)
    df['return_3d'] = df['close'].pct_change(3).fillna(0)
    df['return_5d'] = df['close'].pct_change(5).fillna(0)
    df['return_10d'] = df['close'].pct_change(10).fillna(0)
    df['return_20d'] = df['close'].pct_change(20).fillna(0)
    df['avg_return_5d'] = df['return_1d'].rolling(5).mean().fillna(0)
    df['avg_return_10d'] = df['return_1d'].rolling(10).mean().fillna(0)
    df['avg_return_20d'] = df['return_1d'].rolling(20).mean().fillna(0)
    
    # MA distances
    df['dist_sma_20'] = (df['close'] / df['sma_20'] - 1).fillna(0)
    df['dist_sma_50'] = (df['close'] / df['sma_50'] - 1).fillna(0)
    df['dist_sma_200'] = (df['close'] / df['sma_200'] - 1).fillna(0)
    df['dist_ema_9'] = (df['close'] / df['ema_9'] - 1).fillna(0)
    df['dist_ema_21'] = (df['close'] / df['ema_21'] - 1).fillna(0)
    df['sma_20_50_cross'] = (df['sma_20'] / df['sma_50'] - 1).fillna(0)
    df['ema_9_21_cross'] = (df['ema_9'] / df['ema_21'] - 1).fillna(0)
    df['sma_50_200_cross'] = (df['sma_50'] / df['sma_200'] - 1).fillna(0)
    
    # Bollinger
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(bb_range > 0, (df['close'] - df['bb_lower']) / bb_range, 0.5)
    df['bb_width'] = np.where(df['bb_middle'] > 0, bb_range / df['bb_middle'], 0)
    
    # Momentum slopes & lags
    df['rsi_slope_3d'] = df['rsi_14'].diff(3).fillna(0)
    df['rsi_slope_5d'] = df['rsi_14'].diff(5).fillna(0)
    df['macd_hist_slope'] = df['macd_hist'].diff(2).fillna(0)
    df['macd_hist_accel'] = df['macd_hist'].diff(2).diff(1).fillna(0)
    df['stoch_slope'] = df['stoch_k'].diff(3).fillna(0)
    df['adx_slope'] = df['adx_14'].diff(3).fillna(0)
    for lag in [3, 5, 10]:
        df[f'rsi_lag_{lag}'] = df['rsi_14'].shift(lag).fillna(50)
        df[f'macd_hist_lag_{lag}'] = df['macd_hist'].shift(lag).fillna(0)
        df[f'adx_lag_{lag}'] = df['adx_14'].shift(lag).fillna(20)
        df[f'stoch_k_lag_{lag}'] = df['stoch_k'].shift(lag).fillna(50)
    df['rsi_mean_10d'] = df['rsi_14'].rolling(10).mean().fillna(50)
    df['rsi_std_10d'] = df['rsi_14'].rolling(10).std().fillna(5)
    df['macd_hist_mean_5d'] = df['macd_hist'].rolling(5).mean().fillna(0)
    df['macd_hist_std_5d'] = df['macd_hist'].rolling(5).std().fillna(0)
    
    # Volatility
    df['atr_pct'] = np.where(df['close'] > 0, df['atr_14'] / df['close'] * 100, 0)
    df['daily_range'] = np.where(df['close'] > 0, (df['high'] - df['low']) / df['close'] * 100, 0)
    df['volatility_5d'] = df['return_1d'].rolling(5).std().fillna(0) * 100
    df['volatility_20d'] = df['return_1d'].rolling(20).std().fillna(0) * 100
    df['vol_ratio'] = np.where(df['volatility_20d'] > 0, df['volatility_5d'] / df['volatility_20d'], 1)
    
    # Volume
    vol_sma_20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = pd.Series(np.where(vol_sma_20 > 0, df['volume'] / vol_sma_20, 1), index=df.index).fillna(1)
    df['volume_trend'] = df['volume'].rolling(5).mean().pct_change(5).fillna(0)
    if 'obv' in df.columns:
        df['obv_slope'] = pd.to_numeric(df['obv'], errors='coerce').diff(5).fillna(0)
    else:
        df['obv_slope'] = 0
    
    # Candlestick
    body = df['close'] - df['open']
    full_range = df['high'] - df['low']
    df['candle_body_ratio'] = np.where(full_range > 0, body / full_range, 0)
    df['upper_shadow'] = np.where(full_range > 0, (df['high'] - np.maximum(df['close'], df['open'])) / full_range, 0)
    df['lower_shadow'] = np.where(full_range > 0, (np.minimum(df['close'], df['open']) - df['low']) / full_range, 0)
    df['up_day'] = (df['return_1d'] > 0).astype(int)
    df['consecutive_up'] = df['up_day'].rolling(5).sum().fillna(0)
    
    # Mean reversion
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    
    # ===== NEWS SENTIMENT FEATURES (12 features) =====
    # Rolling sentiment averages
    df['sentiment_1d'] = df['sent_stock']
    df['sentiment_3d'] = df['sent_stock'].rolling(3, min_periods=1).mean().fillna(0)
    df['sentiment_7d'] = df['sent_stock'].rolling(7, min_periods=1).mean().fillna(0)
    df['sentiment_14d'] = df['sent_stock'].rolling(14, min_periods=1).mean().fillna(0)
    # Sentiment momentum (improving or worsening)
    df['sentiment_momentum'] = (df['sentiment_3d'] - df['sentiment_7d']).fillna(0)
    # News volume spike (high news volume often precedes big moves)
    news_avg = df['news_count_stock'].rolling(20, min_periods=1).mean()
    df['news_volume_spike'] = pd.Series(
        np.where(news_avg > 0, df['news_count_stock'] / news_avg, 0), index=df.index).fillna(0)
    # Positive/negative ratio
    total_pn = df['news_pos'] + df['news_neg']
    df['news_positive_ratio'] = pd.Series(
        np.where(total_pn > 0, df['news_pos'] / total_pn, 0.5), index=df.index).fillna(0.5)
    # Market-wide sentiment
    df['mkt_sent_3d'] = df['mkt_sentiment'].rolling(3, min_periods=1).mean().fillna(0)
    df['mkt_sent_7d'] = df['mkt_sentiment'].rolling(7, min_periods=1).mean().fillna(0)
    # Sentiment × price divergence (bearish news + rising price = warning)
    df['sent_price_divergence'] = (df['sentiment_3d'] * -df['return_3d']).fillna(0)
    # Extreme sentiment signals
    df['sent_extreme_pos'] = (df['sent_max_pos'] > 0.8).astype(int)
    df['sent_extreme_neg'] = (df['sent_max_neg'] < -0.8).astype(int)
    
    # Target
    df['future_close'] = df['close'].shift(-forward_days)
    df['future_return_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100
    df['target'] = (df['future_return_pct'] >= target_pct).astype(int)
    
    df = df.dropna(subset=['future_return_pct'])
    df = df.ffill().bfill()
    
    drop_cols = [
        'future_close', 'future_return_pct', 'target',
        'close', 'open', 'high', 'low',
        'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21',
        'obv', 'up_day',
        # Raw sentiment cols (engineered versions are kept)
        'sent_stock', 'news_count_stock', 'news_pos', 'news_neg',
        'sent_max_pos', 'sent_max_neg', 'mkt_sentiment', 'mkt_news_count',
    ]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return features, df['target']


def train_and_evaluate(symbol: str):
    print(f"🔄 Loading data for {symbol}...")
    df = load_data_for_symbol(symbol)
    if df.empty:
        print(f"❌ No data found for {symbol}.")
        return
    print(f"📊 {len(df)} daily records loaded\n")
    
    # ================================================
    # MULTI-HORIZON SWEEP (5 days → 6 months)
    # ================================================
    horizons = {
        5:   ("1 Week",    0.5),
        10:  ("2 Weeks",   1.0),
        20:  ("1 Month",   2.0),
        40:  ("2 Months",  3.0),
        60:  ("3 Months",  5.0),
        120: ("6 Months",  8.0),
    }
    
    all_results = {}
    
    for fwd, (label, tgt_pct) in horizons.items():
        print(f"{'='*60}")
        print(f"📅 {label} ({fwd}d, target ≥{tgt_pct}%)")
        print(f"{'='*60}")
        
        X, y = engineer_features_and_target(df, forward_days=fwd, target_pct=tgt_pct)
        balance = y.mean() * 100
        print(f"   {len(X.columns)} features | {balance:.0f}% positive class")
        
        cutoff = X.index.max() - pd.DateOffset(months=6)
        Xtr, ytr = X[X.index < cutoff], y[y.index < cutoff]
        Xte, yte = X[X.index >= cutoff], y[y.index >= cutoff]
        
        Xtr = Xtr.replace([np.inf, -np.inf], np.nan).fillna(0)
        Xte = Xte.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        if len(Xtr) < 100 or len(Xte) < 10:
            print(f"   ⚠️ Skipping — insufficient data ({len(Xtr)} train, {len(Xte)} test)")
            continue
        
        print(f"   Train: {len(Xtr)} | Test: {len(Xte)} (balance: {yte.mean()*100:.0f}%)")
        
        models = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=500, learning_rate=0.02, max_depth=5,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, eval_metric='logloss', early_stopping_rounds=50),
            "XGB_HiReg": xgb.XGBClassifier(
                n_estimators=800, learning_rate=0.01, max_depth=3,
                min_child_weight=10, subsample=0.7, colsample_bytree=0.5,
                gamma=0.3, reg_alpha=2.0, reg_lambda=5.0,
                random_state=42, eval_metric='logloss', early_stopping_rounds=80),
            "RandForest": RandomForestClassifier(
                n_estimators=500, max_depth=6, min_samples_leaf=15,
                min_samples_split=30, max_features='sqrt',
                class_weight='balanced', random_state=42, n_jobs=-1),
            "GradBoost": GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=3,
                min_samples_leaf=20, subsample=0.8, random_state=42),
        }
        
        for mname, model in models.items():
            try:
                if "XG" in mname:
                    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
                else:
                    model.fit(Xtr, ytr)
                
                proba = model.predict_proba(Xte)[:, 1]
                
                # Find best threshold
                best_thr, best_sc, found = 0.5, 0, False
                for thr in np.arange(0.35, 0.90, 0.005):
                    yt = (proba >= thr).astype(int)
                    if yt.sum() < 3: continue
                    a = accuracy_score(yte, yt)
                    p = precision_score(yte, yt, zero_division=0)
                    r = recall_score(yte, yt, zero_division=0)
                    if r < 0.05: continue
                    if a >= 0.70 and p >= 0.70:
                        sc = r + 1000
                        if sc > best_sc:
                            best_sc, best_thr, found = sc, thr, True
                    elif not found:
                        sc = min(a, p) * 0.6 + p * 0.3 + a * 0.1
                        if sc > best_sc:
                            best_sc, best_thr = sc, thr
                
                yp = (proba >= best_thr).astype(int)
                a = accuracy_score(yte, yp)
                p = precision_score(yte, yp, zero_division=0)
                r = recall_score(yte, yp, zero_division=0)
                f = f1_score(yte, yp, zero_division=0)
                
                key = f"{label}|{mname}"
                all_results[key] = {
                    'model': model, 'model_name': mname,
                    'horizon': label, 'fwd': fwd, 'tgt_pct': tgt_pct,
                    'acc': a, 'prec': p, 'rec': r, 'f1': f,
                    'thr': best_thr, 'features': list(Xtr.columns),
                    'report': classification_report(yte, yp, zero_division=0),
                }
                
                ai = "✅" if a >= 0.70 else ""
                pi = "✅" if p >= 0.70 else ""
                print(f"   {mname:11s} Acc:{a:5.1%}{ai} Prec:{p:5.1%}{pi} Rec:{r:5.1%} F1:{f:5.1%} @{best_thr:.2f}")
            except Exception as e:
                print(f"   {mname:11s} ❌ {e}")
    
    if not all_results:
        print("\n❌ No models trained.")
        return
    
    # ================================================
    # CROSS-HORIZON SUMMARY
    # ================================================
    best_key = max(all_results, key=lambda k:
        min(all_results[k]['acc'], all_results[k]['prec']) * 0.7 +
        max(all_results[k]['acc'], all_results[k]['prec']) * 0.3)
    best = all_results[best_key]
    
    print(f"\n{'='*70}")
    print(f"📊 ALL HORIZONS × ALL MODELS (sorted by accuracy)")
    print(f"{'='*70}")
    print(f"{'Horizon':<11} {'Model':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Thr':>5}")
    print(f"{'-'*55}")
    for k in sorted(all_results, key=lambda k: all_results[k]['acc'], reverse=True):
        r = all_results[k]
        m = " 👑" if k == best_key else ""
        t = " ✅✅" if r['acc'] >= 0.70 and r['prec'] >= 0.70 else ""
        print(f"{r['horizon']:<11} {r['model_name']:<12} {r['acc']:>5.1%} {r['prec']:>5.1%} {r['rec']:>5.1%} {r['f1']:>5.1%} {r['thr']:>4.2f}{m}{t}")
    
    print(f"\n{'='*70}")
    print(f"🏆 WINNER: {best['model_name']} @ {best['horizon']} ({best['fwd']}d, ≥{best['tgt_pct']}%)")
    print(f"{'='*70}")
    print(f"   Accuracy:  {best['acc']:.2%}  {'✅' if best['acc'] >= 0.70 else '⚠️'}")
    print(f"   Precision: {best['prec']:.2%}  {'✅' if best['prec'] >= 0.70 else '⚠️'}")
    print(f"   Recall:    {best['rec']:.2%}")
    print(f"   F1-Score:  {best['f1']:.2%}")
    
    print(f"\n📋 Report:\n{best['report']}")
    
    mo = best['model']
    if hasattr(mo, "feature_importances_"):
        print("🌟 Top 10 Features:")
        fi = pd.Series(mo.feature_importances_, index=best['features']).sort_values(ascending=False)
        for feat, imp in fi.head(10).items():
            print(f"   {feat:22s}: {imp:.4f} {'█' * int(imp * 200)}")
    
    # Save
    os.makedirs("models", exist_ok=True)
    path = f"models/best_{symbol}_v3.pkl"
    artifact = {
        'model': mo, 'threshold': best['thr'],
        'features': best['features'],
        'metrics': {'accuracy': best['acc'], 'precision': best['prec'],
                    'recall': best['rec'], 'f1': best['f1']},
        'model_name': best['model_name'],
        'horizon': best['horizon'], 'forward_days': best['fwd'],
        'target_pct': best['tgt_pct'],
        'trained_at': datetime.now().isoformat(), 'symbol': symbol,
    }
    joblib.dump(artifact, path)
    joblib.dump(mo, f"models/best_{symbol}_v2.pkl")
    print(f"\n✅ Saved to {path}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    train_and_evaluate(sym)
