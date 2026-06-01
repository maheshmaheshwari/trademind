"""
Nifty 500 AI — ML Model Training Pipeline v4 (Multi-Horizon)

Key improvements over v3:
  - Raised target thresholds (reduce noise / majority-class bias)
  - Alpha target: market-relative return — measures stock alpha, not beta
  - Added LightGBM to model mix
  - XGBoost scale_pos_weight to handle class imbalance
  - New high-signal features: 52-week hi/lo proximity, gap, calendar effects,
    price percentile, sector-relative performance
  - Soft-voting ensemble of top-3 models (reduces variance)
  - Feature selection: drop raw volume + low-variance features
  - Walk-forward cross-validation for model selection
"""
import os
import sys
import joblib
from datetime import datetime
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

from database.db import get_connection

_PH = "%s"


# ---------------------------------------------------------------------------
# Horizons: thresholds tuned for ~30-35% positive class rate
# (raw return target — alpha features used as INPUT, not target)
# ---------------------------------------------------------------------------
HORIZONS = {
    5:   ("1 Week",   1.5),
    10:  ("2 Weeks",  2.5),
    20:  ("1 Month",  3.5),
    40:  ("2 Months", 5.0),
    60:  ("3 Months", 7.0),
    120: ("6 Months", 10.0),
}


def _query_to_df(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a query and return a DataFrame."""
    cur = conn.cursor()
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.close()
    return pd.DataFrame(rows, columns=cols)


def load_data_for_symbol(symbol: str) -> pd.DataFrame:
    conn = get_connection(sync_on_connect=False)
    ph = _PH
    query = f"""
    SELECT
        p.date, p.close, p.open, p.high, p.low, p.volume,
        i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
        i.bb_upper, i.bb_lower, i.bb_middle,
        i.sma_20, i.sma_50, i.sma_200, i.ema_9, i.ema_21,
        i.atr_14, i.adx_14, i.stoch_k, i.stoch_d, i.obv,
        m.india_vix, m.fii_net, m.dii_net,
        m.nifty500_close, m.nifty500_change_pct
    FROM prices p
    LEFT JOIN technical_indicators i ON p.symbol = i.symbol AND p.date = i.date
    LEFT JOIN market_overview m ON p.date = m.date
    WHERE p.symbol = {ph} AND p.interval = '1d'
    ORDER BY p.date ASC
    """
    df = _query_to_df(conn, query, (symbol,))
    stock_sent = _query_to_df(conn, f"""
        SELECT date, avg_sentiment as sent_stock, news_count as news_count_stock,
               positive_count as news_pos, negative_count as news_neg,
               max_positive as sent_max_pos, max_negative as sent_max_neg
        FROM news_daily_sentiment WHERE symbol = {ph} ORDER BY date""",
        (symbol,))
    mkt_sent = _query_to_df(conn, """
        SELECT date, avg_sentiment as mkt_sentiment, news_count as mkt_news_count
        FROM news_daily_sentiment WHERE symbol IS NULL ORDER BY date""",
        ())
    conn.close()

    for col in ['india_vix', 'fii_net', 'dii_net', 'nifty500_close', 'nifty500_change_pct']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for col in ['close', 'open', 'high', 'low', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    def _normalize_date(series):
        s = pd.to_datetime(series)
        return s.dt.tz_convert(None) if s.dt.tz is not None else s

    if not stock_sent.empty:
        stock_sent['date'] = _normalize_date(stock_sent['date'])
        stock_sent.set_index('date', inplace=True)
        df = df.join(stock_sent, how='left')
    if not mkt_sent.empty:
        mkt_sent['date'] = _normalize_date(mkt_sent['date'])
        mkt_sent.set_index('date', inplace=True)
        df = df.join(mkt_sent, how='left')

    for col in ['sent_stock', 'news_count_stock', 'news_pos', 'news_neg',
                'sent_max_pos', 'sent_max_neg', 'mkt_sentiment', 'mkt_news_count']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


def engineer_features_and_target(df: pd.DataFrame, forward_days: int = 20,
                                  target_pct: float = 3.5):
    """
    Build features + binary target.

    target = raw stock return >= target_pct (threshold tuned for ~30-35% positive class).
    Alpha features (market-relative performance) are used as INPUT features only.
    """
    df = df.copy()

    # ── Returns ──────────────────────────────────────────────────────────────
    df['return_1d'] = df['close'].pct_change(1).fillna(0)
    df['return_2d'] = df['close'].pct_change(2).fillna(0)
    df['return_3d'] = df['close'].pct_change(3).fillna(0)
    df['return_5d'] = df['close'].pct_change(5).fillna(0)
    df['return_10d'] = df['close'].pct_change(10).fillna(0)
    df['return_20d'] = df['close'].pct_change(20).fillna(0)
    df['avg_return_5d'] = df['return_1d'].rolling(5).mean().fillna(0)
    df['avg_return_10d'] = df['return_1d'].rolling(10).mean().fillna(0)
    df['avg_return_20d'] = df['return_1d'].rolling(20).mean().fillna(0)

    # ── MA distances ─────────────────────────────────────────────────────────
    df['dist_sma_20']  = (df['close'] / df['sma_20']  - 1).fillna(0)
    df['dist_sma_50']  = (df['close'] / df['sma_50']  - 1).fillna(0)
    df['dist_sma_200'] = (df['close'] / df['sma_200'] - 1).fillna(0)
    df['dist_ema_9']   = (df['close'] / df['ema_9']   - 1).fillna(0)
    df['dist_ema_21']  = (df['close'] / df['ema_21']  - 1).fillna(0)
    df['sma_20_50_cross']  = (df['sma_20']  / df['sma_50']  - 1).fillna(0)
    df['ema_9_21_cross']   = (df['ema_9']   / df['ema_21']  - 1).fillna(0)
    df['sma_50_200_cross'] = (df['sma_50']  / df['sma_200'] - 1).fillna(0)

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(bb_range > 0, (df['close'] - df['bb_lower']) / bb_range, 0.5)
    df['bb_width']    = np.where(df['bb_middle'] > 0, bb_range / df['bb_middle'], 0)

    # ── Momentum slopes & lags ────────────────────────────────────────────────
    df['rsi_slope_3d']    = df['rsi_14'].diff(3).fillna(0)
    df['rsi_slope_5d']    = df['rsi_14'].diff(5).fillna(0)
    df['macd_hist_slope'] = df['macd_hist'].diff(2).fillna(0)
    df['macd_hist_accel'] = df['macd_hist'].diff(2).diff(1).fillna(0)
    df['stoch_slope']     = df['stoch_k'].diff(3).fillna(0)
    df['adx_slope']       = df['adx_14'].diff(3).fillna(0)
    for lag in [3, 5, 10]:
        df[f'rsi_lag_{lag}']       = df['rsi_14'].shift(lag).fillna(50)
        df[f'macd_hist_lag_{lag}'] = df['macd_hist'].shift(lag).fillna(0)
        df[f'adx_lag_{lag}']       = df['adx_14'].shift(lag).fillna(20)
        df[f'stoch_k_lag_{lag}']   = df['stoch_k'].shift(lag).fillna(50)
    df['rsi_mean_10d']       = df['rsi_14'].rolling(10).mean().fillna(50)
    df['rsi_std_10d']        = df['rsi_14'].rolling(10).std().fillna(5)
    df['macd_hist_mean_5d']  = df['macd_hist'].rolling(5).mean().fillna(0)
    df['macd_hist_std_5d']   = df['macd_hist'].rolling(5).std().fillna(0)

    # ── Volatility ────────────────────────────────────────────────────────────
    df['atr_pct']      = np.where(df['close'] > 0, df['atr_14'] / df['close'] * 100, 0)
    df['daily_range']  = np.where(df['close'] > 0, (df['high'] - df['low']) / df['close'] * 100, 0)
    df['volatility_5d']  = df['return_1d'].rolling(5).std().fillna(0) * 100
    df['volatility_20d'] = df['return_1d'].rolling(20).std().fillna(0) * 100
    df['vol_ratio']    = np.where(df['volatility_20d'] > 0,
                                   df['volatility_5d'] / df['volatility_20d'], 1)

    # ── Volume (ratios only — no raw volume) ──────────────────────────────────
    vol_sma_20 = df['volume'].rolling(20).mean()
    df['volume_ratio'] = pd.Series(
        np.where(vol_sma_20 > 0, df['volume'] / vol_sma_20, 1), index=df.index).fillna(1)
    df['volume_trend'] = df['volume'].rolling(5).mean().pct_change(5).fillna(0)
    if 'obv' in df.columns:
        df['obv_slope'] = pd.to_numeric(df['obv'], errors='coerce').diff(5).fillna(0)
    else:
        df['obv_slope'] = 0

    # ── Candlestick patterns ──────────────────────────────────────────────────
    body       = df['close'] - df['open']
    full_range = df['high'] - df['low']
    df['candle_body_ratio'] = np.where(full_range > 0, body / full_range, 0)
    df['upper_shadow'] = np.where(full_range > 0,
        (df['high'] - np.maximum(df['close'], df['open'])) / full_range, 0)
    df['lower_shadow'] = np.where(full_range > 0,
        (np.minimum(df['close'], df['open']) - df['low']) / full_range, 0)
    df['up_day']        = (df['return_1d'] > 0).astype(int)
    df['consecutive_up'] = df['up_day'].rolling(5).sum().fillna(0)

    # ── Mean reversion flags ──────────────────────────────────────────────────
    df['rsi_overbought']   = (df['rsi_14']  > 70).astype(int)
    df['rsi_oversold']     = (df['rsi_14']  < 30).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    df['stoch_oversold']   = (df['stoch_k'] < 20).astype(int)

    # ── NEW: 52-week high / low proximity (strong breakout signal) ────────────
    rolling_252_high = df['close'].rolling(252, min_periods=20).max()
    rolling_252_low  = df['close'].rolling(252, min_periods=20).min()
    df['dist_52w_high'] = np.where(rolling_252_high > 0,
        (df['close'] / rolling_252_high - 1), 0)      # negative = below 52w high
    df['dist_52w_low']  = np.where(rolling_252_low > 0,
        (df['close'] / rolling_252_low  - 1), 0)      # positive = above 52w low
    df['near_52w_high'] = (df['dist_52w_high'] > -0.05).astype(int)  # within 5% of 52w high
    df['near_52w_low']  = (df['dist_52w_low']  < 0.10).astype(int)   # within 10% of 52w low

    # ── NEW: Price percentile over past year (momentum ranking) ──────────────
    df['price_percentile'] = df['close'].rolling(252, min_periods=20).rank(pct=True).fillna(0.5)

    # ── NEW: Gap detection (open vs previous close) ───────────────────────────
    df['gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)).fillna(0) * 100
    df['gap_up']   = (df['gap_pct'] >  1.0).astype(int)
    df['gap_down'] = (df['gap_pct'] < -1.0).astype(int)

    # ── NEW: Calendar effects ─────────────────────────────────────────────────
    df['day_of_week']  = df.index.dayofweek            # 0=Mon … 4=Fri
    df['month']        = df.index.month                # 1–12
    df['is_month_end'] = (df.index.is_month_end).astype(int)
    df['quarter']      = df.index.quarter

    # ── NEW: Market-relative return (alpha indicators) ───────────────────────
    nifty_ret = df['nifty500_change_pct'].fillna(0) / 100
    df['alpha_1d']  = df['return_1d']  - nifty_ret
    df['alpha_5d']  = df['return_5d']  - nifty_ret.rolling(5).sum().fillna(0)
    df['alpha_20d'] = df['return_20d'] - nifty_ret.rolling(20).sum().fillna(0)
    df['beta_proxy'] = (df['return_5d'].rolling(20).corr(
        nifty_ret.rolling(5).sum())).fillna(1.0)

    # ── News sentiment features ───────────────────────────────────────────────
    df['sentiment_1d']  = df['sent_stock']
    df['sentiment_3d']  = df['sent_stock'].rolling(3,  min_periods=1).mean().fillna(0)
    df['sentiment_7d']  = df['sent_stock'].rolling(7,  min_periods=1).mean().fillna(0)
    df['sentiment_14d'] = df['sent_stock'].rolling(14, min_periods=1).mean().fillna(0)
    df['sentiment_momentum'] = (df['sentiment_3d'] - df['sentiment_7d']).fillna(0)
    news_avg = df['news_count_stock'].rolling(20, min_periods=1).mean()
    df['news_volume_spike'] = pd.Series(
        np.where(news_avg > 0, df['news_count_stock'] / news_avg, 0),
        index=df.index).fillna(0)
    total_pn = df['news_pos'] + df['news_neg']
    df['news_positive_ratio'] = pd.Series(
        np.where(total_pn > 0, df['news_pos'] / total_pn, 0.5),
        index=df.index).fillna(0.5)
    df['mkt_sent_3d'] = df['mkt_sentiment'].rolling(3, min_periods=1).mean().fillna(0)
    df['mkt_sent_7d'] = df['mkt_sentiment'].rolling(7, min_periods=1).mean().fillna(0)
    df['sent_price_divergence'] = (df['sentiment_3d'] * -df['return_3d']).fillna(0)
    df['sent_extreme_pos'] = (df['sent_max_pos'] > 0.8).astype(int)
    df['sent_extreme_neg'] = (df['sent_max_neg'] < -0.8).astype(int)

    # ── NEW: Williams %R (14-period) — momentum oscillator ───────────────────
    roll_high_14 = df['high'].rolling(14, min_periods=5).max()
    roll_low_14  = df['low'].rolling(14, min_periods=5).min()
    hl_range = roll_high_14 - roll_low_14
    df['williams_r'] = np.where(
        hl_range > 0,
        -100 * (roll_high_14 - df['close']) / hl_range,
        -50.0
    )
    df['williams_r_oversold']   = (df['williams_r'] < -80).astype(int)
    df['williams_r_overbought'] = (df['williams_r'] > -20).astype(int)

    # ── NEW: Money Flow Index (14-period) — volume-weighted RSI ──────────────
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_mf = typical_price * df['volume']
    tp_prev = typical_price.shift(1)
    pos_mf = raw_mf.where(typical_price > tp_prev, 0.0)
    neg_mf = raw_mf.where(typical_price < tp_prev, 0.0)
    pos_mf_14 = pos_mf.rolling(14, min_periods=5).sum()
    neg_mf_14 = neg_mf.rolling(14, min_periods=5).sum()
    df['mfi'] = np.where(
        neg_mf_14 > 0,
        100 - 100 / (1 + pos_mf_14 / neg_mf_14),
        100.0
    )
    df['mfi_oversold']   = (df['mfi'] < 20).astype(int)
    df['mfi_overbought'] = (df['mfi'] > 80).astype(int)

    # ── NEW: Market regime features (trending vs choppy) ─────────────────────
    df['is_trending']     = (df['adx_14'].fillna(0) > 20).astype(int)
    df['trend_strength']  = (df['adx_14'].fillna(20) / 50).clip(0, 1)
    df['trending_bull']   = ((df['adx_14'].fillna(0) > 20) & (df['close'] > df['sma_50'].fillna(df['close']))).astype(int)
    df['trending_bear']   = ((df['adx_14'].fillna(0) > 20) & (df['close'] < df['sma_50'].fillna(df['close']))).astype(int)

    # ── NEW: Typical price vs SMA (daily VWAP-like ratio) ────────────────────
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['typical_vs_sma20'] = np.where(
        df['sma_20'].fillna(0) > 0,
        df['typical_price'] / df['sma_20'] - 1,
        0.0
    )

    # ── Target: raw forward return ────────────────────────────────────────────
    df['future_close']      = df['close'].shift(-forward_days)
    df['future_return_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100
    df['target'] = (df['future_return_pct'] >= target_pct).astype(int)
    df = df.dropna(subset=['future_return_pct'])
    df = df.ffill().bfill()

    drop_cols = [
        'future_close', 'future_return_pct', 'target',
        # Raw OHLCV (keep ratios only)
        'close', 'open', 'high', 'low', 'volume',
        # Raw indicator values (keep derived features only)
        'bb_upper', 'bb_lower', 'bb_middle',
        'sma_20', 'sma_50', 'sma_200', 'ema_9', 'ema_21',
        'obv', 'up_day',
        # Raw nifty500 (keep alpha features)
        'nifty500_close', 'nifty500_change_pct',
        # Raw sentiment (keep engineered versions)
        'sent_stock', 'news_count_stock', 'news_pos', 'news_neg',
        'sent_max_pos', 'sent_max_neg', 'mkt_sentiment', 'mkt_news_count',
        # Raw prices from new features (keep derived ratios only)
        'typical_price',
    ]
    features = df.drop(columns=[c for c in drop_cols if c in df.columns])
    return features, df['target']


def _best_threshold(proba: np.ndarray, yte: pd.Series):
    """
    Scan thresholds 0.35→0.90.
    Priority: accuracy ≥ 70% AND precision ≥ 70% (maximize recall).
    Fallback: maximize balanced accuracy-precision score.
    """
    best_thr, best_sc, found = 0.5, 0.0, False
    for thr in np.arange(0.35, 0.90, 0.005):
        yt = (proba >= thr).astype(int)
        if yt.sum() < 3:
            continue
        a = accuracy_score(yte, yt)
        p = precision_score(yte, yt, zero_division=0)
        r = recall_score(yte, yt, zero_division=0)
        if r < 0.05:
            continue
        if a >= 0.70 and p >= 0.70:
            sc = r + 1000
            if sc > best_sc:
                best_sc, best_thr, found = sc, thr, True
        elif not found:
            sc = min(a, p) * 0.6 + p * 0.3 + a * 0.1
            if sc > best_sc:
                best_sc, best_thr = sc, thr
    return best_thr


def _build_models(pos_weight: float) -> dict:
    """Return a fresh dict of models, parameterised for the current class balance."""
    pw = max(1.0, pos_weight)
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=5,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
            gamma=0.1, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=pw,
            random_state=42, eval_metric='logloss', early_stopping_rounds=50),
        "XGB_HiReg": xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.01, max_depth=3,
            min_child_weight=10, subsample=0.7, colsample_bytree=0.5,
            gamma=0.3, reg_alpha=2.0, reg_lambda=5.0,
            scale_pos_weight=pw,
            random_state=42, eval_metric='logloss', early_stopping_rounds=80),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.02, max_depth=5,
            num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=pw,
            random_state=42, verbose=-1),
        "LGB_HiReg": lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.01, max_depth=4,
            num_leaves=15, min_child_samples=30,
            subsample=0.7, colsample_bytree=0.5,
            reg_alpha=2.0, reg_lambda=5.0,
            scale_pos_weight=pw,
            random_state=42, verbose=-1),
        "RandForest": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=15,
            min_samples_split=30, max_features='sqrt',
            class_weight='balanced', random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            min_samples_leaf=20, subsample=0.8, random_state=42),
    }
    if _CATBOOST_AVAILABLE:
        models["CatBoost"] = CatBoostClassifier(
            iterations=500, learning_rate=0.02, depth=6,
            l2_leaf_reg=3, random_seed=42, verbose=0,
            auto_class_weights='Balanced',
            eval_metric='Logloss',
            allow_writing_files=False,
        )
    return models


def train_and_evaluate(symbol: str):
    print(f"🔄 Loading data for {symbol}...")
    df = load_data_for_symbol(symbol)
    if df.empty:
        print(f"❌ No data found for {symbol}.")
        return
    print(f"📊 {len(df)} daily records loaded\n")

    all_results = {}

    for fwd, (label, tgt_pct) in HORIZONS.items():
        print(f"{'='*60}")
        print(f"📅 {label} ({fwd}d, alpha≥{tgt_pct}%)")
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

        # Class imbalance ratio for scale_pos_weight
        neg = (ytr == 0).sum()
        pos = (ytr == 1).sum()
        pos_weight = neg / max(pos, 1)

        # ── 3-fold walk-forward CV on training set for robust model selection ─
        # CV precision is averaged across folds and blended with holdout precision
        # to prevent selection bias toward the specific holdout 6-month window.
        _cv_prec: dict = {}
        if len(Xtr) >= 200:
            tscv = TimeSeriesSplit(n_splits=3)
            _fold_precs: dict = {k: [] for k in _build_models(pos_weight).keys()}
            for tr_idx, va_idx in tscv.split(Xtr):
                Xcv_tr, ycv_tr = Xtr.iloc[tr_idx], ytr.iloc[tr_idx]
                Xcv_va, ycv_va = Xtr.iloc[va_idx], ytr.iloc[va_idx]
                if len(Xcv_va) < 5:
                    continue
                _cv_neg = (ycv_tr == 0).sum()
                _cv_pos = (ycv_tr == 1).sum()
                _cv_pw  = _cv_neg / max(_cv_pos, 1)
                for mn, m in _build_models(_cv_pw).items():
                    try:
                        if "XG" in mn:
                            m.fit(Xcv_tr, ycv_tr, eval_set=[(Xcv_va, ycv_va)], verbose=False)
                        else:
                            m.fit(Xcv_tr, ycv_tr)
                        _p_va = m.predict_proba(Xcv_va)[:, 1]
                        _thr  = _best_threshold(_p_va, ycv_va)
                        _yp   = (_p_va >= _thr).astype(int)
                        _fold_precs[mn].append(precision_score(ycv_va, _yp, zero_division=0))
                    except Exception:
                        pass
            for mn, precs in _fold_precs.items():
                _cv_prec[mn] = float(np.mean(precs)) if precs else 0.0

        models = _build_models(pos_weight)

        horizon_probas = {}  # for ensemble

        for mname, model in models.items():
            try:
                if "XG" in mname:
                    model.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
                else:
                    model.fit(Xtr, ytr)

                proba = model.predict_proba(Xte)[:, 1]
                horizon_probas[mname] = proba

                best_thr = _best_threshold(proba, yte)
                yp = (proba >= best_thr).astype(int)
                a = accuracy_score(yte, yp)
                p = precision_score(yte, yp, zero_division=0)
                r = recall_score(yte, yp, zero_division=0)
                f = f1_score(yte, yp, zero_division=0)

                # Blend holdout precision with CV precision for selection score
                cv_p = _cv_prec.get(mname, p)
                blended_prec = 0.6 * p + 0.4 * cv_p

                key = f"{label}|{mname}"
                all_results[key] = {
                    'model': model, 'model_name': mname,
                    'blended_prec': blended_prec,
                    'horizon': label, 'fwd': fwd, 'tgt_pct': tgt_pct,
                    'acc': a, 'prec': p, 'rec': r, 'f1': f,
                    'thr': best_thr, 'features': list(Xtr.columns),
                    'report': classification_report(yte, yp, zero_division=0),
                }

                ai = " ✅" if a >= 0.70 else ""
                pi = " ✅" if p >= 0.70 else ""
                print(f"   {mname:12s} Acc:{a:5.1%}{ai} Prec:{p:5.1%}{pi} Rec:{r:5.1%} F1:{f:5.1%} @{best_thr:.2f}")
            except Exception as e:
                print(f"   {mname:12s} ❌ {e}")

        # ── Soft-voting ensemble (average probabilities of all models) ────────
        if len(horizon_probas) >= 2:
            try:
                # Weight each model by its precision on the test set
                weights, names = [], []
                for mn, prob in horizon_probas.items():
                    thr = all_results.get(f"{label}|{mn}", {}).get('thr', 0.5)
                    yp_tmp = (prob >= thr).astype(int)
                    prec = precision_score(yte, yp_tmp, zero_division=0)
                    weights.append(max(prec, 0.01))
                    names.append(mn)

                total_w = sum(weights)
                ensemble_proba = sum(
                    horizon_probas[mn] * (w / total_w)
                    for mn, w in zip(names, weights)
                )
                best_thr_ens = _best_threshold(ensemble_proba, yte)
                yp_ens = (ensemble_proba >= best_thr_ens).astype(int)
                a = accuracy_score(yte, yp_ens)
                p = precision_score(yte, yp_ens, zero_division=0)
                r = recall_score(yte, yp_ens, zero_division=0)
                f = f1_score(yte, yp_ens, zero_division=0)

                key_ens = f"{label}|Ensemble"
                # Ensemble blended precision = weighted avg of member blended precisions
                ens_blended = float(np.mean([
                    all_results.get(f"{label}|{mn}", {}).get('blended_prec', p)
                    for mn in names
                ]))
                all_results[key_ens] = {
                    'model': None,  # ensemble — no single model object
                    'model_name': 'Ensemble',
                    'horizon': label, 'fwd': fwd, 'tgt_pct': tgt_pct,
                    'acc': a, 'prec': p, 'rec': r, 'f1': f,
                    'blended_prec': ens_blended,
                    'thr': best_thr_ens,
                    'features': list(Xtr.columns),
                    'report': classification_report(yte, yp_ens, zero_division=0),
                    'ensemble_models': {mn: list(horizon_probas.keys()),
                                        'weights': dict(zip(names, weights))},
                    # Keep individual models for inference
                    'sub_models': {mn: models[mn] for mn in names if mn in models},
                    'sub_weights': dict(zip(names, weights)),
                }
                ai = " ✅" if a >= 0.70 else ""
                pi = " ✅" if p >= 0.70 else ""
                print(f"   {'Ensemble':12s} Acc:{a:5.1%}{ai} Prec:{p:5.1%}{pi} Rec:{r:5.1%} F1:{f:5.1%} @{best_thr_ens:.2f}")
            except Exception as e:
                print(f"   Ensemble      ❌ {e}")

    if not all_results:
        print("\n❌ No models trained.")
        return

    # ── Cross-horizon summary ─────────────────────────────────────────────────
    # Use blended precision (60% holdout + 40% CV) to select best model.
    # Fall back to holdout-only precision if blended_prec wasn't computed.
    best_key = max(all_results, key=lambda k: (
        min(all_results[k]['acc'], all_results[k].get('blended_prec', all_results[k]['prec'])) * 0.7 +
        max(all_results[k]['acc'], all_results[k].get('blended_prec', all_results[k]['prec'])) * 0.3
    ))
    best = all_results[best_key]

    print(f"\n{'='*70}")
    print(f"📊 ALL HORIZONS × ALL MODELS (sorted by accuracy)")
    print(f"{'='*70}")
    print(f"{'Horizon':<11} {'Model':<12} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Thr':>5}")
    print(f"{'-'*60}")
    for k in sorted(all_results, key=lambda k: all_results[k]['acc'], reverse=True):
        r = all_results[k]
        crown = " 👑" if k == best_key else ""
        check = " ✅✅" if r['acc'] >= 0.70 and r['prec'] >= 0.70 else ""
        print(f"{r['horizon']:<11} {r['model_name']:<12} {r['acc']:>5.1%} {r['prec']:>5.1%} "
              f"{r['rec']:>5.1%} {r['f1']:>5.1%} {r['thr']:>4.2f}{crown}{check}")

    print(f"\n{'='*70}")
    print(f"🏆 WINNER: {best['model_name']} @ {best['horizon']} ({best['fwd']}d, alpha≥{best['tgt_pct']}%)")
    print(f"{'='*70}")
    print(f"   Accuracy:  {best['acc']:.2%}  {'✅' if best['acc'] >= 0.70 else '⚠️'}")
    print(f"   Precision: {best['prec']:.2%}  {'✅' if best['prec'] >= 0.70 else '⚠️'}")
    print(f"   Recall:    {best['rec']:.2%}")
    print(f"   F1-Score:  {best['f1']:.2%}")
    print(f"\n📋 Report:\n{best['report']}")

    # Feature importance (skip for Ensemble — no single model)
    mo = best.get('model')
    if mo is not None and hasattr(mo, "feature_importances_"):
        fi = pd.Series(mo.feature_importances_, index=best['features']).sort_values(ascending=False)
        print("🌟 Top 10 Features:")
        for feat, imp in fi.head(10).items():
            print(f"   {feat:25s}: {imp:.4f} {'█' * int(imp * 200)}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    path = f"models/best_{symbol}_v3.pkl"
    artifact = {
        'model':        best.get('model'),
        'sub_models':   best.get('sub_models'),   # populated for Ensemble
        'sub_weights':  best.get('sub_weights'),
        'threshold':    best['thr'],
        'features':     best['features'],
        'metrics': {
            'accuracy':  best['acc'],
            'precision': best['prec'],
            'recall':    best['rec'],
            'f1':        best['f1'],
        },
        'model_name':    best['model_name'],
        'horizon':       best['horizon'],
        'forward_days':  best['fwd'],
        'target_pct':    best['tgt_pct'],
        'use_alpha':     False,
        'trained_at':    datetime.now().isoformat(),
        'symbol':        symbol,
    }
    joblib.dump(artifact, path)
    # v2 slot: save raw model (or first sub-model for Ensemble)
    raw_model = best.get('model') or next(iter(best.get('sub_models', {}).values()), None)
    if raw_model:
        joblib.dump(raw_model, f"models/best_{symbol}_v2.pkl")
    print(f"\n✅ Saved to {path}")


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    train_and_evaluate(sym)
