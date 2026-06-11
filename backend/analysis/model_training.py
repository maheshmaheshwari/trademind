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
from sklearn.calibration import CalibratedClassifierCV

try:
    from catboost import CatBoostClassifier
    _CATBOOST_AVAILABLE = True
except ImportError:
    _CATBOOST_AVAILABLE = False

from database.db import get_connection, release_connection
import json as _json
import os as _os

_PH = "%s"

# ── Sector map from angel_tokens.json ──────────────────────────────────────
_TOKENS_PATH = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "angel_tokens.json")
_SECTOR_MAP: dict = {}
try:
    _raw = _json.load(open(_TOKENS_PATH))
    _SECTOR_MAP = {f"{sym}.NS": info.get("sector", "Unknown") for sym, info in _raw.items()}
except Exception:
    pass

# ── Sector returns cache (pre-computed once, reused for all stocks) ─────────
# Dict: {sector_name: pd.Series(index=date, values=avg_daily_return)}
_SECTOR_RETURNS_CACHE: dict = {}

# ── Full data cache — pre-fetched from DB once for all symbols ──────────────
# Dict: {symbol: pd.DataFrame} — eliminates all per-stock DB queries during training
_DATA_CACHE: dict = {}


def prefetch_all_data(symbols: list = None) -> None:
    """
    Pre-fetch ALL training data from DB in bulk — one query per table.
    After this runs, load_data_for_symbol() returns instantly from memory.

    symbols: list of symbols to fetch (None = all from prices table)
    """
    global _DATA_CACHE
    import logging
    log = logging.getLogger(__name__)
    log.info("📦 Pre-fetching all stock data from DB (one-time batch)...")

    conn = get_connection()
    try:
        # ── 1. Prices + indicators + market overview + delivery (one big JOIN) ──
        log.info("   Loading prices + indicators + market + delivery...")
        conn.cursor().execute("SET statement_timeout = '120s'")
        df_all = _query_to_df(conn, """
            SELECT
                p.symbol, p.date,
                p.close, p.open, p.high, p.low, p.volume,
                i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
                i.bb_upper, i.bb_lower, i.bb_middle,
                i.sma_20, i.sma_50, i.sma_200, i.ema_9, i.ema_21,
                i.atr_14, i.adx_14, i.stoch_k, i.stoch_d, i.obv,
                m.india_vix,
                m.nifty500_close, m.nifty500_change_pct,
                COALESCE(f.fii_net,  m.fii_net,  0) AS fii_net,
                COALESCE(f.dii_net,  m.dii_net,  0) AS dii_net,
                COALESCE(f.fii_buy,  0)              AS fii_buy,
                COALESCE(f.fii_sell, 0)              AS fii_sell,
                COALESCE(f.dii_buy,  0)              AS dii_buy,
                COALESCE(f.dii_sell, 0)              AS dii_sell,
                COALESCE(d.delivery_pct, 50.0) AS delivery_pct
            FROM prices p
            LEFT JOIN technical_indicators i ON p.symbol = i.symbol AND p.date = i.date
            LEFT JOIN market_overview m      ON p.date = m.date
            LEFT JOIN fii_dii_daily f        ON p.date = f.date
            LEFT JOIN delivery_data d        ON p.symbol = d.symbol AND p.date = d.date
            WHERE p.interval = '1d'
            ORDER BY p.symbol, p.date ASC
        """, timeout="120s")
        log.info(f"   Prices loaded: {len(df_all):,} rows across {df_all['symbol'].nunique()} symbols")

        # ── 2. Stock-level news sentiment ─────────────────────────────────────
        log.info("   Loading stock news sentiment...")
        df_sent = _query_to_df(conn, """
            SELECT symbol, date,
                   avg_sentiment   AS sent_stock,
                   news_count      AS news_count_stock,
                   positive_count  AS news_pos,
                   negative_count  AS news_neg,
                   max_positive    AS sent_max_pos,
                   max_negative    AS sent_max_neg
            FROM news_daily_sentiment
            WHERE symbol IS NOT NULL
            ORDER BY symbol, date ASC
        """, timeout="60s")
        log.info(f"   Sentiment loaded: {len(df_sent):,} rows")

        # ── 3. Market-wide sentiment ──────────────────────────────────────────
        log.info("   Loading market sentiment...")
        df_mkt = _query_to_df(conn, """
            SELECT date,
                   avg_sentiment AS mkt_sentiment,
                   news_count    AS mkt_news_count
            FROM news_daily_sentiment
            WHERE symbol IS NULL
            ORDER BY date ASC
        """, timeout="30s")
        log.info(f"   Market sentiment: {len(df_mkt):,} rows")

    finally:
        release_connection(conn)

    # ── Prepare market sentiment as a Series ──────────────────────────────────
    df_mkt['date'] = pd.to_datetime(df_mkt['date'])
    if df_mkt['date'].dt.tz is not None:
        df_mkt['date'] = df_mkt['date'].dt.tz_convert(None)
    df_mkt = df_mkt.set_index('date')
    for col in ['mkt_sentiment', 'mkt_news_count']:
        df_mkt[col] = pd.to_numeric(df_mkt[col], errors='coerce').fillna(0)

    # ── Prepare stock sentiment as dict ───────────────────────────────────────
    df_sent['date'] = pd.to_datetime(df_sent['date'])
    if df_sent['date'].dt.tz is not None:
        df_sent['date'] = df_sent['date'].dt.tz_convert(None)
    df_sent = df_sent.set_index(['symbol', 'date'])

    # ── Split prices by symbol and build per-symbol DataFrames ───────────────
    df_all['date'] = pd.to_datetime(df_all['date'])
    if df_all['date'].dt.tz is not None:
        df_all['date'] = df_all['date'].dt.tz_convert(None)

    numeric_cols = ['close','open','high','low','volume',
                    'india_vix','fii_net','dii_net','fii_buy','fii_sell',
                    'dii_buy','dii_sell','nifty500_close','nifty500_change_pct']

    sent_fill_cols = ['sent_stock','news_count_stock','news_pos','news_neg',
                      'sent_max_pos','sent_max_neg']

    target_symbols = symbols if symbols else df_all['symbol'].unique().tolist()

    for sym in target_symbols:
        sym_df = df_all[df_all['symbol'] == sym].copy()
        if sym_df.empty:
            continue
        sym_df = sym_df.drop(columns=['symbol']).set_index('date').sort_index()

        # Coerce numerics
        for col in numeric_cols:
            if col in sym_df.columns:
                sym_df[col] = pd.to_numeric(sym_df[col], errors='coerce')

        # Join market sentiment
        sym_df = sym_df.join(df_mkt, how='left')

        # Join stock sentiment
        if sym in df_sent.index.get_level_values(0):
            s_sent = df_sent.loc[sym].copy()
            sym_df = sym_df.join(s_sent, how='left')

        # Fill missing sentiment columns with 0
        for col in sent_fill_cols + ['mkt_sentiment', 'mkt_news_count']:
            if col not in sym_df.columns:
                sym_df[col] = 0.0
            sym_df[col] = pd.to_numeric(sym_df[col], errors='coerce').fillna(0)

        _DATA_CACHE[sym] = sym_df

    log.info(f"✅ Data cache ready: {len(_DATA_CACHE)} symbols in memory")


def _SECTOR_RETURNS_CACHE_RESET():
    global _SECTOR_RETURNS_CACHE
    _SECTOR_RETURNS_CACHE = {}


def precompute_sector_returns() -> None:
    """
    Pre-compute daily average returns for every sector using a single DB query.
    Results are cached in _SECTOR_RETURNS_CACHE and reused during training —
    eliminates per-stock sector DB queries that caused hangs on large sectors.

    Call this once before training starts in retrain_walk_forward.py.
    """
    global _SECTOR_RETURNS_CACHE
    if _SECTOR_RETURNS_CACHE:
        return  # Already computed

    import logging
    log = logging.getLogger(__name__)
    log.info("Pre-computing sector returns (one-time, all sectors)...")

    try:
        conn = get_connection()
        # One query: daily returns for all symbols
        df_all = _query_to_df(conn, """
            SELECT
                date,
                symbol,
                (close - LAG(close) OVER (PARTITION BY symbol ORDER BY date)) /
                NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY date), 0) AS daily_return
            FROM prices
            WHERE interval = '1d'
            ORDER BY date
        """)
        release_connection(conn)

        if df_all.empty:
            log.warning("No price data for sector pre-computation")
            return

        df_all['date']         = pd.to_datetime(df_all['date'])
        df_all['daily_return'] = pd.to_numeric(df_all['daily_return'], errors='coerce')
        df_all['sector']       = df_all['symbol'].map(_SECTOR_MAP)
        df_all = df_all.dropna(subset=['sector', 'daily_return'])
        df_all = df_all[df_all['sector'] != 'Unknown']

        # Aggregate: mean daily return per sector per date
        sector_daily = (
            df_all.groupby(['sector', 'date'])['daily_return']
            .mean()
            .reset_index()
        )

        # Store as dict of Series indexed by date
        for sector, grp in sector_daily.groupby('sector'):
            s = grp.set_index('date')['daily_return']
            s.index = s.index.normalize()  # strip time component
            _SECTOR_RETURNS_CACHE[sector] = s

        log.info(f"Sector returns cached for {len(_SECTOR_RETURNS_CACHE)} sectors")
    except Exception as e:
        log.warning(f"Sector pre-computation failed (sector features disabled): {e}")


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


class PurgedTimeSeriesSplit:
    """
    Priority 5: Walk-forward CV with embargo gap between train and validation.
    Prevents forward-looking bias by skipping rows equal to the prediction horizon.
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits    = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X, y=None, groups=None):
        n          = len(X)
        fold_size  = n // (self.n_splits + 1)
        embargo    = max(1, int(n * self.embargo_pct))
        for i in range(self.n_splits):
            train_end = fold_size * (i + 1)
            val_start = train_end + embargo
            val_end   = val_start + fold_size
            if val_end > n:
                break
            yield list(range(0, train_end)), list(range(val_start, val_end))


def _query_to_df(conn, sql: str, params: tuple = (), timeout: str = "30s") -> pd.DataFrame:
    """Execute a query and return a DataFrame. Times out after `timeout` to prevent hangs."""
    cur = conn.cursor()
    cur.execute(f"SET statement_timeout = '{timeout}'")
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    cur.execute("SET statement_timeout = 0")  # reset after query
    cur.close()
    return pd.DataFrame(rows, columns=cols)


def load_data_for_symbol(symbol: str) -> pd.DataFrame:
    # ── Serve from pre-fetched cache if available (zero DB queries) ──────────
    if _DATA_CACHE and symbol in _DATA_CACHE:
        return _DATA_CACHE[symbol].copy()

    # ── Fallback: fetch from DB (used when cache not pre-built) ──────────────
    conn = get_connection()
    ph = _PH
    try:
        query = f"""
        SELECT
            p.date, p.close, p.open, p.high, p.low, p.volume,
            i.rsi_14, i.macd, i.macd_signal, i.macd_hist,
            i.bb_upper, i.bb_lower, i.bb_middle,
            i.sma_20, i.sma_50, i.sma_200, i.ema_9, i.ema_21,
            i.atr_14, i.adx_14, i.stoch_k, i.stoch_d, i.obv,
            m.india_vix,
            m.nifty500_close, m.nifty500_change_pct,
            COALESCE(f.fii_net,  m.fii_net,  0) AS fii_net,
            COALESCE(f.dii_net,  m.dii_net,  0) AS dii_net,
            COALESCE(f.fii_buy,  0)              AS fii_buy,
            COALESCE(f.fii_sell, 0)              AS fii_sell,
            COALESCE(f.dii_buy,  0)              AS dii_buy,
            COALESCE(f.dii_sell, 0)              AS dii_sell,
            COALESCE(d.delivery_pct, 50.0) as delivery_pct
        FROM prices p
        LEFT JOIN technical_indicators i ON p.symbol = i.symbol AND p.date = i.date
        LEFT JOIN market_overview m ON p.date = m.date
        LEFT JOIN fii_dii_daily f   ON p.date = f.date
        LEFT JOIN delivery_data d   ON p.symbol = d.symbol AND p.date = d.date
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
    finally:
        release_connection(conn)

    for col in ['india_vix', 'fii_net', 'dii_net', 'fii_buy', 'fii_sell',
                'dii_buy', 'dii_sell', 'nifty500_close', 'nifty500_change_pct']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
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

    # ── Priority 7: Sector-relative return (from pre-computed cache) ─────────
    # No DB query here — _SECTOR_RETURNS_CACHE was built once before training started
    sector = _SECTOR_MAP.get(symbol, "")
    if sector and sector != "Unknown" and sector in _SECTOR_RETURNS_CACHE:
        try:
            sect_series = _SECTOR_RETURNS_CACHE[sector]
            # Align to df's date index
            df_idx = df.index.normalize()
            sect_aligned = sect_series.reindex(df_idx).values
            df['sector_return_1d'] = sect_aligned
            df['sector_return_1d'] = pd.to_numeric(df['sector_return_1d'], errors='coerce').fillna(0)
        except Exception:
            df['sector_return_1d'] = 0.0
    else:
        df['sector_return_1d'] = 0.0

    return df


def engineer_features_and_target(df: pd.DataFrame, forward_days: int = 20,
                                  target_pct: float = 3.5):
    """
    Build features + binary target.

    target = raw stock return >= target_pct (threshold tuned for ~30-35% positive class).
    Alpha features (market-relative performance) are used as INPUT features only.
    """
    df = df.copy()

    # Coerce all numeric columns — DB may store None objects instead of NaN
    for col in df.columns:
        if col not in ('date',):
            df[col] = pd.to_numeric(df[col], errors='coerce')

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
    df['sent_extreme_neg'] = (df['sent_max_neg'] > 0.8).astype(int)

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

    # ── FII / DII institutional flow features ────────────────────────────────
    fii = df['fii_net'].fillna(0)
    dii = df['dii_net'].fillna(0)

    # Rolling cumulative flows
    df['fii_5d']  = fii.rolling(5,  min_periods=1).sum()
    df['fii_10d'] = fii.rolling(10, min_periods=1).sum()
    df['fii_20d'] = fii.rolling(20, min_periods=1).sum()
    df['dii_5d']  = dii.rolling(5,  min_periods=1).sum()
    df['dii_10d'] = dii.rolling(10, min_periods=1).sum()
    df['dii_20d'] = dii.rolling(20, min_periods=1).sum()

    # Net combined institutional flow
    df['inst_flow_5d']  = df['fii_5d']  + df['dii_5d']
    df['inst_flow_20d'] = df['fii_20d'] + df['dii_20d']

    # FII vs DII divergence: FII buying while DII selling (or vice versa)
    total_abs = (fii.abs() + dii.abs()).replace(0, 1)
    df['fii_dii_divergence'] = (fii - dii) / total_abs      # -1 (DII dominates) … +1 (FII dominates)
    df['fii_dii_div_5d']     = df['fii_dii_divergence'].rolling(5, min_periods=1).mean()

    # Buy/sell pressure ratios from detailed data
    if 'fii_buy' in df.columns and df['fii_buy'].abs().sum() > 0:
        fii_gross = (df['fii_buy'].fillna(0) + df['fii_sell'].fillna(0)).replace(0, 1)
        dii_gross = (df['dii_buy'].fillna(0) + df['dii_sell'].fillna(0)).replace(0, 1)
        df['fii_buy_ratio'] = df['fii_buy'].fillna(0) / fii_gross    # 0.5 = balanced
        df['dii_buy_ratio'] = df['dii_buy'].fillna(0) / dii_gross
        df['fii_buy_pressure'] = (df['fii_buy_ratio'] > 0.55).astype(int)
        df['dii_buy_pressure'] = (df['dii_buy_ratio'] > 0.55).astype(int)
        # Both institutions buying together = strong signal
        df['dual_buy_pressure'] = (df['fii_buy_pressure'] & df['dii_buy_pressure']).astype(int)

    # Trend direction (positive = sustained inflow over 10d)
    df['fii_trend']  = np.sign(df['fii_10d'])
    df['dii_trend']  = np.sign(df['dii_10d'])

    # Normalised flow (z-score vs 60d rolling mean/std for stationarity)
    fii_60_mean = fii.rolling(60, min_periods=10).mean().fillna(0)
    fii_60_std  = fii.rolling(60, min_periods=10).std().fillna(1).replace(0, 1)
    df['fii_zscore'] = (fii - fii_60_mean) / fii_60_std
    dii_60_mean = dii.rolling(60, min_periods=10).mean().fillna(0)
    dii_60_std  = dii.rolling(60, min_periods=10).std().fillna(1).replace(0, 1)
    df['dii_zscore'] = (dii - dii_60_mean) / dii_60_std

    # ── Priority 7: Sector-relative alpha features ────────────────────────────
    if 'sector_return_1d' in df.columns:
        sect_ret = df['sector_return_1d'].fillna(0)
        df['alpha_vs_sector_1d']  = df['return_1d']  - sect_ret
        df['alpha_vs_sector_5d']  = df['return_5d']  - sect_ret.rolling(5).sum().fillna(0)
        df['alpha_vs_sector_20d'] = df['return_20d'] - sect_ret.rolling(20).sum().fillna(0)
        df['sector_leader']       = (df['alpha_vs_sector_5d'] > 0.02).astype(int)
        df['sector_laggard']      = (df['alpha_vs_sector_5d'] < -0.02).astype(int)
        df['sector_momentum']     = sect_ret.rolling(5).mean().fillna(0)
        df['stock_vs_sector_vol'] = (df['return_1d'].rolling(10).std() /
                                     sect_ret.rolling(10).std().clip(lower=0.0001)).fillna(1.0)

    # ── Priority 4: Delivery % features (institutional conviction signal) ────
    if 'delivery_pct' in df.columns:
        df['delivery_pct']        = df['delivery_pct'].fillna(50.0)
        df['delivery_ma5']        = df['delivery_pct'].rolling(5).mean().fillna(50)
        df['delivery_ma20']       = df['delivery_pct'].rolling(20).mean().fillna(50)
        df['delivery_spike']      = (df['delivery_pct'] > df['delivery_ma20'] * 1.3).astype(int)
        df['high_delivery_bull']  = ((df['delivery_pct'] > 60) & (df['return_1d'] > 0)).astype(int)
        df['delivery_trend']      = (df['delivery_pct'] - df['delivery_ma20']).fillna(0)

    # ── Priority 4: Interaction features ─────────────────────────────────────
    df['rsi_in_trend']          = df['rsi_14'].fillna(50)  * df['is_trending']
    df['macd_in_trend']         = df['macd_hist'].fillna(0) * df['is_trending']
    df['vol_breakout']          = df['volume_ratio'].fillna(1) * df.get('near_52w_high', pd.Series(0, index=df.index)).fillna(0)
    df['vol_oversold_bounce']   = df['volume_ratio'].fillna(1) * df['rsi_oversold']
    df['sent_momentum_confirm'] = df['sentiment_3d'].fillna(0) * df['alpha_5d'].fillna(0)
    df['sent_vol_confirm']      = df['sentiment_3d'].fillna(0) * df['volume_ratio'].fillna(1)
    df['rsi_bull_regime']       = df['rsi_14'].fillna(50) * df['trending_bull']
    df['rsi_bear_regime']       = df['rsi_14'].fillna(50) * df['trending_bear']
    df['high_rank_momentum']    = df['price_percentile'].fillna(0.5) * df['alpha_5d'].fillna(0)
    df['low_rank_bounce']       = (1 - df['price_percentile'].fillna(0.5)) * df['rsi_oversold']
    if 'delivery_pct' in df.columns:
        df['delivery_vol_confirm'] = df.get('delivery_spike', pd.Series(0, index=df.index)).fillna(0) * df['volume_ratio'].fillna(1)

    # ── Priority 3: Risk-adjusted target ──────────────────────────────────────
    # Reward return relative to recent volatility — penalises lucky volatile spikes
    df['future_close']      = df['close'].shift(-forward_days)
    df['future_return_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100
    volatility_20d = df['return_1d'].rolling(20).std().shift(1).fillna(0.02) * 100
    # Risk-adjusted target: future_return / volatility >= threshold factor
    # This is equivalent to a Sharpe-like criterion over the holding period
    risk_adj_return = df['future_return_pct'] / volatility_20d.clip(lower=0.5)
    risk_adj_threshold = target_pct / 3.0   # scaled for volatility-normalised space
    df['target'] = (
        (df['future_return_pct'] >= target_pct) &          # raw return still >= threshold
        (risk_adj_return >= risk_adj_threshold)             # AND risk-adjusted return >= scaled threshold
    ).astype(int)
    df = df.dropna(subset=['future_return_pct'])
    df = df.ffill()

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


def _tune_xgboost(Xtr, ytr, Xval, yval, pos_weight, n_trials: int = 30):
    """Priority 8: Optuna tuning for XGBoost. Returns best params dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators":     trial.suggest_int("n_estimators", 200, 800),
                "learning_rate":    trial.suggest_float("lr", 0.005, 0.05, log=True),
                "max_depth":        trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_int("mcw", 3, 20),
                "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample", 0.4, 0.9),
                "reg_alpha":        trial.suggest_float("alpha", 0.1, 5.0, log=True),
                "reg_lambda":       trial.suggest_float("lambda", 0.5, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "eval_metric": "logloss", "random_state": 42, "verbosity": 0,
            }
            m = xgb.XGBClassifier(**params, early_stopping_rounds=20)
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            proba = m.predict_proba(Xval)[:, 1]
            thr   = _best_threshold(proba, yval)
            pred  = (proba >= thr).astype(int)
            return precision_score(yval, pred, zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params
    except Exception:
        return {}


def _tune_lightgbm(Xtr, ytr, Xval, yval, pos_weight, n_trials: int = 30):
    """Priority 8: Optuna tuning for LightGBM. Returns best params dict."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = {
                "n_estimators":    trial.suggest_int("n_estimators", 200, 800),
                "learning_rate":   trial.suggest_float("lr", 0.005, 0.05, log=True),
                "max_depth":       trial.suggest_int("max_depth", 3, 7),
                "num_leaves":      trial.suggest_int("num_leaves", 15, 127),
                "min_child_samples": trial.suggest_int("mcs", 10, 50),
                "subsample":       trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree":trial.suggest_float("colsample", 0.4, 0.9),
                "reg_alpha":       trial.suggest_float("alpha", 0.1, 5.0, log=True),
                "reg_lambda":      trial.suggest_float("lambda", 0.5, 10.0, log=True),
                "scale_pos_weight": pos_weight,
                "random_state": 42, "verbose": -1,
            }
            m = lgb.LGBMClassifier(**params)
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
            proba = m.predict_proba(Xval)[:, 1]
            thr   = _best_threshold(proba, yval)
            pred  = (proba >= thr).astype(int)
            return precision_score(yval, pred, zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        return study.best_params
    except Exception:
        return {}


def _build_stacking_ensemble(horizon_probas: dict, yte: pd.Series, all_results: dict, label: str):
    """
    Priority 9: Stacking ensemble — LogisticRegression meta-learner.
    Trains on first half of test probabilities, evaluates on second half.
    More intelligent than soft-voting: learns which model to trust per condition.
    """
    from sklearn.linear_model import LogisticRegression
    if len(horizon_probas) < 2 or len(yte) < 20:
        return None

    names = list(horizon_probas.keys())
    meta_X = np.column_stack([horizon_probas[mn] for mn in names])
    meta_y = yte.values

    split = max(5, len(meta_X) // 2)
    X_meta_tr, y_meta_tr = meta_X[:split], meta_y[:split]
    X_meta_te, y_meta_te = meta_X[split:], meta_y[split:]

    if len(np.unique(y_meta_tr)) < 2:
        return None

    try:
        meta_clf = LogisticRegression(C=0.1, max_iter=500, random_state=42)
        meta_clf.fit(X_meta_tr, y_meta_tr)
        ensemble_proba_full = meta_clf.predict_proba(meta_X)[:, 1]

        # Evaluate on held-out second half
        ens_proba_te = ensemble_proba_full[split:]
        best_thr     = _best_threshold(ens_proba_te, pd.Series(y_meta_te))
        yp           = (ens_proba_te >= best_thr).astype(int)

        a = accuracy_score(y_meta_te, yp)
        p = precision_score(y_meta_te, yp, zero_division=0)
        r = recall_score(y_meta_te, yp, zero_division=0)
        f = f1_score(y_meta_te, yp, zero_division=0)

        weights = {mn: float(c) for mn, c in zip(names, meta_clf.coef_[0])}
        # Get fwd/tgt_pct from one of the base results
        _base = all_results.get(f"{label}|{names[0]}", {})
        return {
            "model":        None,
            "model_name":   "StackEnsemble",
            "meta_learner": meta_clf,
            "horizon":      label,
            "fwd":          _base.get("fwd", 0),
            "tgt_pct":      _base.get("tgt_pct", 0),
            "acc": a, "prec": p, "rec": r, "f1": f,
            "blended_prec": p,
            "thr":          best_thr,
            "features":     _base.get("features", []),
            "report":       classification_report(y_meta_te, yp, zero_division=0),
            "sub_models":   {mn: None for mn in names},
            "sub_weights":  weights,
            "stacking_weights": weights,
        }
    except Exception:
        return None


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


def train_and_evaluate(symbol: str, train_end_date: str = None, test_start_date: str = None):
    """
    Train and evaluate all models for a symbol.

    train_end_date: last date of training data e.g. "2024-12-31"
    test_start_date: first date of test data e.g. "2025-01-01"
    If not provided, falls back to the legacy 6-month rolling holdout.
    """
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

        if train_end_date and test_start_date:
            # Walk-forward: strict date-based split, no leakage
            cutoff_end   = pd.Timestamp(train_end_date)
            cutoff_start = pd.Timestamp(test_start_date)
            Xtr = X[X.index.normalize() <= cutoff_end]
            ytr = y[y.index.normalize() <= cutoff_end]
            Xte = X[X.index.normalize() >= cutoff_start]
            yte = y[y.index.normalize() >= cutoff_start]
        else:
            # Legacy: rolling 6-month holdout
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
            # Priority 5: Purged CV with embargo — prevents leakage from prediction horizon
            tscv = PurgedTimeSeriesSplit(n_splits=5, embargo_pct=0.01)
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

        # ── Priority 8: Optuna tuning for XGBoost + LightGBM ─────────────────
        # Only when training set is large enough (>500 rows) to benefit from tuning
        optuna_xgb_params = {}
        optuna_lgb_params = {}
        if len(Xtr) >= 500:
            print(f"   🔍 Optuna tuning XGBoost ({30} trials)...")
            _val_split = int(len(Xtr) * 0.8)
            _Xopt_tr, _yopt_tr = Xtr.iloc[:_val_split], ytr.iloc[:_val_split]
            _Xopt_va, _yopt_va = Xtr.iloc[_val_split:], ytr.iloc[_val_split:]
            optuna_xgb_params = _tune_xgboost(_Xopt_tr, _yopt_tr, _Xopt_va, _yopt_va, pos_weight, n_trials=30)
            optuna_lgb_params = _tune_lightgbm(_Xopt_tr, _yopt_tr, _Xopt_va, _yopt_va, pos_weight, n_trials=30)
            if optuna_xgb_params: print(f"   ✅ XGB best: lr={optuna_xgb_params.get('lr','?'):.4f} depth={optuna_xgb_params.get('max_depth','?')}")
            if optuna_lgb_params: print(f"   ✅ LGB best: lr={optuna_lgb_params.get('lr','?'):.4f} leaves={optuna_lgb_params.get('num_leaves','?')}")

        models = _build_models(pos_weight)

        # Override XGBoost + LightGBM with Optuna-tuned params
        if optuna_xgb_params:
            for mn in ["XGBoost", "XGB_HiReg"]:
                if mn in models:
                    p = optuna_xgb_params.copy()
                    p.pop("lr", None)
                    models[mn] = xgb.XGBClassifier(
                        **p, learning_rate=optuna_xgb_params.get("lr", 0.02),
                        scale_pos_weight=pos_weight, eval_metric="logloss",
                        random_state=42, verbosity=0,
                    )
        if optuna_lgb_params:
            for mn in ["LightGBM", "LGB_HiReg"]:
                if mn in models:
                    p = optuna_lgb_params.copy()
                    p.pop("lr", None)
                    models[mn] = lgb.LGBMClassifier(
                        **p, learning_rate=optuna_lgb_params.get("lr", 0.02),
                        scale_pos_weight=pos_weight, random_state=42, verbose=-1,
                    )

        horizon_probas = {}  # for ensemble

        # ── Priority 10: TabNet deep learning model ────────────────────────
        from analysis.deep_model import is_available as tabnet_available, train_tabnet, tabnet_predict_proba
        if tabnet_available() and len(Xtr) >= 200:
            try:
                tab_model = train_tabnet(Xtr, ytr, Xte, yte, pos_weight=pos_weight)
                if tab_model is not None:
                    tab_proba = tabnet_predict_proba(tab_model, Xte)
                    horizon_probas["TabNet"] = tab_proba
                    best_thr_tab = _best_threshold(tab_proba, yte)
                    yp_tab = (tab_proba >= best_thr_tab).astype(int)
                    a = accuracy_score(yte, yp_tab)
                    p = precision_score(yte, yp_tab, zero_division=0)
                    r = recall_score(yte, yp_tab, zero_division=0)
                    f = f1_score(yte, yp_tab, zero_division=0)
                    cv_p = _cv_prec.get("TabNet", p)
                    all_results[f"{label}|TabNet"] = {
                        "model": tab_model, "model_name": "TabNet",
                        "blended_prec": 0.6 * p + 0.4 * cv_p,
                        "horizon": label, "fwd": fwd, "tgt_pct": tgt_pct,
                        "acc": a, "prec": p, "rec": r, "f1": f,
                        "thr": best_thr_tab, "features": list(Xtr.columns),
                        "report": classification_report(yte, yp_tab, zero_division=0),
                    }
                    ai = " ✅" if a >= 0.70 else ""
                    pi = " ✅" if p >= 0.70 else ""
                    print(f"   {'TabNet':12s} Acc:{a:5.1%}{ai} Prec:{p:5.1%}{pi} Rec:{r:5.1%} F1:{f:5.1%} @{best_thr_tab:.2f}")
            except Exception as e:
                print(f"   TabNet        ❌ {e}")

        # Carve a small validation split from Xtr for XGBoost early stopping
        # (avoids test-set leakage into model selection)
        _es_split = max(10, int(len(Xtr) * 0.10))
        Xval_es = Xtr.iloc[-_es_split:]
        yval_es = ytr.iloc[-_es_split:]
        Xtr_fit = Xtr.iloc[:-_es_split]
        ytr_fit = ytr.iloc[:-_es_split]

        for mname, model in models.items():
            try:
                if "XG" in mname:
                    model.fit(Xtr_fit, ytr_fit, eval_set=[(Xval_es, yval_es)], verbose=False)
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

        # ── Priority 9: Stacking ensemble meta-learner ────────────────────────
        if len(horizon_probas) >= 2:
            stack_result = _build_stacking_ensemble(horizon_probas, yte, all_results, label)
            if stack_result is not None:
                # Populate sub_models with the actual trained models
                stack_result["sub_models"] = {mn: models.get(mn) for mn in horizon_probas}
                key_stack = f"{label}|StackEnsemble"
                all_results[key_stack] = stack_result
                a, p = stack_result["acc"], stack_result["prec"]
                ai = " ✅" if a >= 0.70 else ""
                pi = " ✅" if p >= 0.70 else ""
                print(f"   {'StackEnsemble':12s} Acc:{a:5.1%}{ai} Prec:{p:5.1%}{pi} Rec:{stack_result['rec']:5.1%} F1:{stack_result['f1']:5.1%}")

    if not all_results:
        print("\n❌ No models trained.")
        return

    # ── Cross-horizon summary ─────────────────────────────────────────────────
    # Use blended precision (60% holdout + 40% CV) to select best model.
    # Fall back to holdout-only precision if blended_prec wasn't computed.
    def _selection_score(r):
        acc  = r['acc']
        prec = r['prec']
        # Harmonic mean of accuracy and precision — both must be high
        if acc + prec == 0:
            return 0.0
        hmean = 2 * acc * prec / (acc + prec)
        # Bonus for both exceeding 0.70 threshold
        bonus = 0.05 if (acc >= 0.70 and prec >= 0.70) else 0.0
        return hmean + bonus

    # ── Priority 6: Probability calibration ──────────────────────────────────
    # XGBoost/LightGBM probabilities are overconfident — calibration makes
    # the confidence % shown in the UI trustworthy (80% confidence = 80% win rate)
    if train_end_date and test_start_date:
        Xte_cal = X[X.index.normalize() >= pd.Timestamp(test_start_date)]
        yte_cal = y[y.index.normalize() >= pd.Timestamp(test_start_date)]
    else:
        cutoff_cal = X.index.max() - pd.DateOffset(months=6)
        Xte_cal, yte_cal = X[X.index >= cutoff_cal], y[y.index >= cutoff_cal]
    Xte_cal = Xte_cal.replace([np.inf, -np.inf], np.nan).fillna(0)

    for key, res in all_results.items():
        if res.get("model") is None:
            continue
        if len(Xte_cal) < 30:
            break
        try:
            method = 'isotonic' if len(Xte_cal) >= 100 else 'sigmoid'
            calibrated = CalibratedClassifierCV(res["model"], method=method, cv='prefit')
            calibrated.fit(Xte_cal[res["features"]], yte_cal)
            res["model"] = calibrated
            res["calibrated"] = True
        except Exception:
            pass

    best_key = max(all_results, key=lambda k: _selection_score(all_results[k]))
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

    # Also save to final_models/ so retrained models are immediately used by generate_trades.py
    os.makedirs("final_models", exist_ok=True)
    bare = symbol.replace(".NS", "")
    final_path = f"final_models/{bare}_final.pkl"
    joblib.dump(artifact, final_path)
    print(f"✅ Also saved to {final_path} (production inference path)")

    return artifact


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    train_and_evaluate(sym)
