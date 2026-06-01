# Model Accuracy Improvements — Implementation Plan

**File:** `analysis/model_training.py` (v4 → v5)  
**Status:** Planned  
**Priority order:** Data → Features → Target → Models → CV → Calibration

---

## Summary of Changes

| Priority | Change | File(s) | Effort | Prerequisite |
|---|---|---|---|---|
| 1 | Retrain after GDELT news data | `model_training.py` | Low | GDELT bootstrap |
| 2 | NSE Delivery % collector | `collectors/delivery_collector.py` (new) | Medium | None |
| 3 | Risk-adjusted target definition | `model_training.py` | Low | None |
| 4 | CatBoost + interaction features | `model_training.py` | Low | `pip install catboost` |
| 5 | Purged walk-forward CV (5 folds) | `model_training.py` | Medium | None |
| 6 | Post-training probability calibration | `model_training.py` | Low | None |
| 7 | Sector-relative alpha features | `model_training.py` + DB | Medium | Sector map |
| 8 | Optuna hyperparameter tuning | `model_training.py` | Medium | `pip install optuna` |
| 9 | Stacking ensemble meta-learner | `model_training.py` | Medium | None |
| 10 | TabNet / LSTM | `analysis/deep_model.py` (new) | High | `pip install pytorch-tabnet` |

---

## Priority 1 — Retrain After GDELT News Data

**Why:** All 10+ sentiment features (`sentiment_1d`, `sentiment_3d`, `sentiment_7d`, etc.) are currently **zeroed out** for all 498 stocks. Models are blind to news signal entirely.

**When:** After GDELT bootstrap completes (restart tomorrow after 5:30 AM IST).

**Command:**
```bash
cd backend && source venv/bin/activate
python -c "
import json
tokens = json.load(open('data/angel_tokens.json'))
from analysis.model_training import train_and_evaluate
for symbol in tokens:
    try:
        train_and_evaluate(f'{symbol}.NS')
    except Exception as e:
        print(f'FAILED {symbol}: {e}')
"
```

**Expected improvement:** +3–8% precision on 1M/3M/6M horizons where sentiment signal is strongest.

---

## Priority 2 — NSE Delivery % Collector (New File)

**Why:** Delivery % = what fraction of traded volume was actual delivery (not squared off intraday). High delivery % signals institutional conviction. One of the strongest predictors for Indian equities.

**Source:** NSE bhavcopy (free, daily, no API key).

**New file:** `backend/collectors/delivery_collector.py`

```python
"""
NSE Delivery % Collector

Downloads NSE bhavcopy (daily delivery data) and stores delivery_pct per stock.
Source: https://archives.nseindia.com/products/content/sec_bhavdata_full_{DD-Mon-YYYY}.csv

Schedule: Daily after 6 PM IST (NSE uploads ~5:30 PM)
"""
import requests
import pandas as pd
from datetime import datetime, timedelta
from database.db import get_connection, _execute

NSE_BHAVCOPY_URL = (
    "https://archives.nseindia.com/products/content/"
    "sec_bhavdata_full_{date}.csv"
)
HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_delivery_data(date: datetime) -> pd.DataFrame:
    date_str = date.strftime("%d-%b-%Y").upper()  # e.g. 01-JUN-2026
    url = NSE_BHAVCOPY_URL.format(date=date_str)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(resp.text))
    df.columns = df.columns.str.strip()
    # Columns: SYMBOL, SERIES, DELIV_QTY, DELIV_PER, TTL_TRD_QNTY, ...
    df = df[df['SERIES'] == 'EQ'][['SYMBOL', 'DELIV_PER', 'TTL_TRD_QNTY']].copy()
    df.columns = ['symbol', 'delivery_pct', 'total_volume']
    df['symbol'] = df['symbol'].str.strip() + '.NS'
    df['delivery_pct'] = pd.to_numeric(df['delivery_pct'], errors='coerce')
    df['date'] = date.strftime('%Y-%m-%d')
    return df.dropna(subset=['delivery_pct'])

def store_delivery_data(df: pd.DataFrame):
    conn = get_connection()
    for _, row in df.iterrows():
        _execute(conn,
            """INSERT INTO delivery_data (symbol, date, delivery_pct, total_volume)
               VALUES (?, ?, ?, ?)
               ON CONFLICT (symbol, date) DO UPDATE SET
               delivery_pct = EXCLUDED.delivery_pct""",
            (row['symbol'], row['date'], row['delivery_pct'], row['total_volume'])
        )
    conn.commit()
    conn.close()

def backfill_delivery(days: int = 400):
    today = datetime.now()
    for i in range(days):
        dt = today - timedelta(days=i)
        if dt.weekday() >= 5:  # skip weekends
            continue
        try:
            df = fetch_delivery_data(dt)
            if not df.empty:
                store_delivery_data(df)
                print(f"  {dt.strftime('%Y-%m-%d')}: {len(df)} stocks")
        except Exception as e:
            print(f"  {dt.strftime('%Y-%m-%d')}: {e}")
```

**DB schema addition** (`schema_pg.py`):
```sql
CREATE TABLE IF NOT EXISTS delivery_data (
    symbol       TEXT NOT NULL,
    date         DATE NOT NULL,
    delivery_pct FLOAT,
    total_volume BIGINT,
    PRIMARY KEY (symbol, date)
);
```

**Feature additions** in `engineer_features_and_target()`:
```python
# Join delivery_pct from delivery_data table
df['delivery_pct'] = ...  # merged from DB
df['delivery_pct_ma5']   = df['delivery_pct'].rolling(5).mean().fillna(50)
df['delivery_pct_ma20']  = df['delivery_pct'].rolling(20).mean().fillna(50)
df['delivery_spike']     = (df['delivery_pct'] > df['delivery_pct_ma20'] * 1.3).astype(int)
df['high_delivery_bull'] = ((df['delivery_pct'] > 60) & (df['return_1d'] > 0)).astype(int)
```

---

## Priority 3 — Risk-Adjusted Target Definition

**Why:** Current target (`return >= fixed_threshold`) rewards volatile stocks with lucky spikes. A risk-adjusted target rewards consistent, low-volatility outperformance.

**Change in `engineer_features_and_target()`:**

```python
# CURRENT (v4):
df['target'] = (df['future_return_pct'] >= target_pct).astype(int)

# OPTION A — Risk-adjusted (v5 recommended):
# Reward return relative to recent volatility
df['target'] = (
    df['future_return_pct'] / df['volatility_20d'].clip(lower=1.0) >= 0.4
).astype(int)

# OPTION B — Alpha target (beat Nifty 500):
nifty_fwd = df['nifty500_change_pct'].shift(-forward_days).fillna(0)
df['future_alpha'] = df['future_return_pct'] - nifty_fwd
df['target'] = (df['future_alpha'] >= target_pct * 0.6).astype(int)

# OPTION C — Multi-class (most information, hardest to train):
df['target'] = pd.cut(
    df['future_return_pct'],
    bins=[-np.inf, -target_pct, target_pct, np.inf],
    labels=[0, 1, 2]  # 0=SELL, 1=HOLD, 2=BUY
).astype(int)
```

**Recommended:** Start with Option A (risk-adjusted). Switch to Option B after delivery data is available.

---

## Priority 4 — CatBoost + Interaction Features

### Install CatBoost
```bash
pip install catboost
```
CatBoost is already imported in `model_training.py` with `_CATBOOST_AVAILABLE` flag — it will activate automatically once installed.

### Interaction Features (add to `engineer_features_and_target()`)

```python
# ── Interaction features ──────────────────────────────────────────────────
# Momentum confirmed by trend
df['rsi_in_trend']          = df['rsi_14'].fillna(50) * df['is_trending']
df['macd_in_trend']         = df['macd_hist'].fillna(0) * df['is_trending']

# Volume confirms breakout
df['vol_breakout']          = df['volume_ratio'] * df['near_52w_high']
df['vol_oversold_bounce']   = df['volume_ratio'] * df['rsi_oversold']

# Sentiment confirms price momentum
df['sent_momentum_confirm'] = df['sentiment_3d'].fillna(0) * df['alpha_5d']
df['sent_vol_confirm']      = df['sentiment_3d'].fillna(0) * df['volume_ratio']

# Delivery confirms move (once delivery_pct available)
# df['delivery_vol_confirm'] = df['delivery_spike'] * df['volume_ratio']

# Regime-conditioned RSI
df['rsi_bull_regime']       = df['rsi_14'].fillna(50) * df['trending_bull']
df['rsi_bear_regime']       = df['rsi_14'].fillna(50) * df['trending_bear']

# Price level + momentum
df['high_rank_momentum']    = df['price_percentile'] * df['alpha_5d']
df['low_rank_bounce']       = (1 - df['price_percentile']) * df['rsi_oversold']
```

---

## Priority 5 — Purged Walk-Forward CV with Embargo

**Why:** Current 3-fold `TimeSeriesSplit` leaks future information — the validation fold immediately follows training, but in financial ML you need an **embargo period** equal to the prediction horizon to prevent forward-looking bias.

**Replace in `train_and_evaluate()`:**

```python
# CURRENT (leaky):
tscv = TimeSeriesSplit(n_splits=3)

# REPLACEMENT — PurgedTimeSeriesSplit:
class PurgedTimeSeriesSplit:
    """Walk-forward CV with embargo gap between train and validation."""
    def __init__(self, n_splits=5, embargo_pct=0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct  # % of dataset to embargo

    def split(self, X, y=None, groups=None):
        n = len(X)
        fold_size = n // (self.n_splits + 1)
        embargo = int(n * self.embargo_pct)

        for i in range(self.n_splits):
            train_end  = fold_size * (i + 1)
            val_start  = train_end + embargo          # skip embargo
            val_end    = val_start + fold_size

            if val_end > n:
                break

            train_idx = list(range(0, train_end))
            val_idx   = list(range(val_start, val_end))
            yield train_idx, val_idx

# Usage (5 folds, 1% embargo):
tscv = PurgedTimeSeriesSplit(n_splits=5, embargo_pct=0.01)
```

**Change folds:** `n_splits=3` → `n_splits=5`

---

## Priority 6 — Post-Training Probability Calibration

**Why:** XGBoost/LightGBM probabilities are systematically overconfident. A model predicting 80% confidence doesn't win 80% of the time. Calibration fixes this so the confidence % shown in the UI is trustworthy.

**Add after best model selection in `train_and_evaluate()`:**

```python
from sklearn.calibration import CalibratedClassifierCV

# After selecting best_model:
if len(Xte) >= 30:  # need enough samples to calibrate
    calibrated = CalibratedClassifierCV(
        best_result['model'],
        method='isotonic',  # 'sigmoid' for small datasets
        cv='prefit'
    )
    calibrated.fit(Xte, yte)
    best_result['model'] = calibrated
    best_result['calibrated'] = True
```

---

## Priority 7 — Sector-Relative Alpha Features

**Why:** A stock +5% when its sector is +8% is actually underperforming. Current alpha is only vs Nifty 500.

**Sector map** — add to `data/sector_map.json`:
```json
{
  "HDFCBANK": "Banking",
  "ICICIBANK": "Banking",
  "TCS": "IT",
  "INFY": "IT",
  ...
}
```

**Feature additions:**
```python
# In load_data_for_symbol(), also load sector average return:
sector_ret = _query_to_df(conn, """
    SELECT p.date,
           AVG(p.close / LAG(p.close) OVER (PARTITION BY p.symbol ORDER BY p.date) - 1) as sector_return
    FROM prices p
    WHERE p.symbol IN (SELECT symbol FROM sector_members WHERE sector = ?)
    AND p.interval = '1d'
    GROUP BY p.date
""", (sector,))

# New features:
df['sector_return'] = ...  # merged
df['alpha_vs_sector_1d']  = df['return_1d']  - df['sector_return']
df['alpha_vs_sector_5d']  = df['return_5d']  - df['sector_return'].rolling(5).sum()
df['alpha_vs_sector_20d'] = df['return_20d'] - df['sector_return'].rolling(20).sum()
df['sector_leader']       = (df['alpha_vs_sector_5d'] > 0.02).astype(int)
df['sector_laggard']      = (df['alpha_vs_sector_5d'] < -0.02).astype(int)
```

---

## Priority 8 — Optuna Hyperparameter Tuning

**Why:** Current hyperparameters are fixed for all 498 stocks. A large-cap bank stock has very different optimal parameters than a small-cap FMCG stock.

**Install:**
```bash
pip install optuna
```

**Add to `model_training.py`:**
```python
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

def _tune_xgboost(X_tr, y_tr, X_val, y_val, pos_weight, n_trials=30):
    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 200, 1000),
            'learning_rate':     trial.suggest_float('lr', 0.005, 0.05, log=True),
            'max_depth':         trial.suggest_int('max_depth', 3, 7),
            'min_child_weight':  trial.suggest_int('mcw', 3, 20),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample', 0.4, 0.9),
            'reg_alpha':         trial.suggest_float('alpha', 0.1, 5.0, log=True),
            'reg_lambda':        trial.suggest_float('lambda', 0.5, 10.0, log=True),
            'scale_pos_weight':  pos_weight,
            'eval_metric': 'logloss', 'random_state': 42,
        }
        m = xgb.XGBClassifier(**params, early_stopping_rounds=30)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        proba = m.predict_proba(X_val)[:, 1]
        thr = _best_threshold(proba, y_val)
        pred = (proba >= thr).astype(int)
        return precision_score(y_val, pred, zero_division=0)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params
```

**Note:** Only run Optuna for stocks with >500 training rows (smaller datasets overfit to the tuned params).

---

## Priority 9 — Stacking Ensemble Meta-Learner

**Why:** Current soft-voting averages all model probabilities equally. A meta-learner learns which model to trust for which market condition.

**Replace soft-voting in `train_and_evaluate()`:**

```python
from sklearn.linear_model import LogisticRegression

# CURRENT soft-voting:
ensemble_proba = np.average(list(horizon_probas.values()), weights=weights, axis=0)

# REPLACEMENT — stacking:
# Stack probabilities as features for a meta-learner
meta_X = np.column_stack(list(horizon_probas.values()))  # shape: (n_test, n_models)
meta_y = yte.values

# Train meta-learner on first half of test, evaluate on second half
split = len(meta_X) // 2
meta_clf = LogisticRegression(C=0.1, max_iter=500)
meta_clf.fit(meta_X[:split], meta_y[:split])
ensemble_proba = meta_clf.predict_proba(meta_X)[:, 1]
```

---

## Priority 10 — TabNet (Deep Learning for Tabular Data)

**Why:** TabNet uses attention to select which features matter for each prediction — effectively learns feature interactions automatically. Often outperforms GBM on financial data with 100+ features.

**Install:**
```bash
pip install pytorch-tabnet
```

**New file:** `analysis/deep_model.py`

```python
"""
TabNet model for per-stock prediction — supplement to GBM ensemble.
"""
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np

def train_tabnet(X_tr, y_tr, X_val, y_val, pos_weight):
    clf = TabNetClassifier(
        n_d=16, n_a=16,            # embedding dimensions
        n_steps=5,                  # attention steps
        gamma=1.3,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
        scheduler_params=dict(step_size=50, gamma=0.9),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        verbose=0,
    )
    clf.fit(
        X_tr.values, y_tr.values,
        eval_set=[(X_val.values, y_val.values)],
        eval_metric=['logloss'],
        max_epochs=200,
        patience=20,
        weights={0: 1.0, 1: float(pos_weight)},
        batch_size=256,
        virtual_batch_size=128,
    )
    return clf
```

---

## Implementation Order & Timeline

```
Week 1 (Now)
├── Tonight:  Restart GDELT after 5:30 AM IST
├── Day 1:    Priority 3 — Risk-adjusted target (30 min code change)
├── Day 1:    Priority 4 — Install CatBoost + interaction features (1 hour)
├── Day 2:    Priority 2 — NSE delivery % collector (2 hours)
└── Day 2:    Priority 5 — Purged CV + 5 folds (1 hour)

Week 2
├── Day 3:    Priority 6 — Probability calibration (30 min)
├── Day 3:    Priority 7 — Sector-relative alpha (2 hours)
├── Day 4:    Retrain all 498 models with v5 pipeline
└── Day 5:    Evaluate: compare v4 vs v5 precision/recall

Week 3
├── Priority 8 — Optuna tuning (run overnight, slow)
├── Priority 9 — Stacking ensemble
└── Priority 10 — TabNet (optional, GPU recommended)
```

---

## Expected Accuracy Gains (Cumulative)

| After completing | Expected precision gain (1M horizon) |
|---|---|
| Priority 1 (GDELT retrain) | +3–8% |
| + Priority 3 (risk-adjusted target) | +2–4% |
| + Priority 4 (CatBoost + interactions) | +1–3% |
| + Priority 5 (purged CV) | +1–2% (less overfitting) |
| + Priority 2 (delivery %) | +3–6% |
| + Priority 6 (calibration) | Confidence scores trustworthy |
| + Priority 8 (Optuna) | +2–4% |
| **Total estimated gain** | **~12–27% precision improvement** |

---

## Dependencies to Install

```bash
cd backend && source venv/bin/activate
pip install catboost optuna pytorch-tabnet shap
```

| Package | Used for |
|---|---|
| `catboost` | CatBoost model (Priority 4) |
| `optuna` | Hyperparameter tuning (Priority 8) |
| `pytorch-tabnet` | TabNet deep model (Priority 10) |
| `shap` | Feature importance analysis (debugging) |
