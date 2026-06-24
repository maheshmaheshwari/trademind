# Plan: Corporate Action Price Adjustment in Model Training

**Status:** PLANNED  
**Date:** 2026-06-19  
**Scope:** Read-time backward price adjustment for splits and bonus issues during ML model training. No stored prices or stored indicators are modified.

---

## 1. Problem Statement

The `prices` table stores raw Angel One candle prices — unadjusted for corporate actions. When a stock split or bonus occurs on date D:

- All rows **before D** have the old (pre-event) price
- All rows **from D onward** have the new (post-event) price
- The gap creates a false return signal on day D (e.g. −50% for a 1:1 bonus)

This corrupts features computed across the event boundary:

| Feature group | Corrupted window |
|---|---|
| `return_1d`, `return_5d`, …, `return_20d` | 1–20 rows around D |
| `volatility_5d`, `volatility_20d` | 5–20 rows |
| `SMA 20/50/200`, `EMA 9/21` | up to 200 rows |
| `dist_sma_20/50/200` | up to 200 rows |
| `Bollinger`, `ATR` | ~20 rows |
| `RSI`, `MACD`, `Stoch` | 14–26 rows |
| `52w hi/lo`, `price_percentile` | up to 252 rows |
| **Training target** `future_return_pct` | any row whose horizon window crosses D |

**89 events (2023–2026) across ~75 stocks** currently exist in `corporate_actions`. Major affected stocks include HDFCBANK, RELIANCE, BAJFINANCE, WIPRO, KOTAKBANK, NESTLEIND.

---

## 2. Data Available

`corporate_actions` table (populated daily via `trendlyne_collector.py`):

```
nse_symbol   ex_date     event_type  ratio  adj_factor
HDFCBANK.NS  2025-08-26  Bonus       1:1    0.5
BAJFINANCE.NS 2025-06-16 Bonus       4:1    0.2
BAJFINANCE.NS 2025-06-16 Split       2:1    0.5
KOTAKBANK.NS  2026-01-14 Split       5:1    0.2
RELIANCE.NS   2024-10-28 Bonus       1:1    0.5
NESTLEIND.NS  2024-01-05 Split       10:1   0.1
```

**`adj_factor` meaning:**
- All OHLC prices **before** `ex_date` must be multiplied by `adj_factor` to be comparable to post-event prices
- If multiple events exist for a symbol, apply oldest → newest (compounding)

Example — BAJFINANCE has both Bonus 4:1 and Split 2:1 on same day:
- Combined adj_factor = 0.2 × 0.5 = 0.1
- Any price before 2025-06-16 × 0.1 = adjusted price

---

## 3. Decision: What NOT to Do

| Approach | Problem |
|---|---|
| Modify stored `prices` table | Breaks stored `technical_indicators` (computed from raw prices); risky |
| Modify stored `technical_indicators` | ~480 symbols × 200+ days = massive recompute; stored SMA now inconsistent with stored prices |
| Ignore splits entirely | Current state — model sees −50% return as valid signal |
| Only drop split-day rows | Doesn't fix the 200-day SMA window contamination |

**Decision: Apply adjustment at read time in `model_training.py`, never touch stored data.**

---

## 4. Chosen Approach

### Two-path loading in `prefetch_all_data()`

```
Symbol has corporate_actions?
│
├── NO  →  Use stored prices + stored indicators as-is (current fast path, ~480 symbols)
│
└── YES →  Path B:
            1. Apply backward OHLC adjustment from corporate_actions
            2. Recompute all indicators in-memory from adjusted prices
            3. Exclude rows within [ex_date - max_horizon, ex_date + 5] from target
            4. Use adjusted data for feature engineering
```

**~75 symbols** take path B (those with events since 2023). The other ~400 take the existing fast path unchanged.

### Why recompute indicators for affected symbols?

The stored `technical_indicators` (SMA20/50/200, EMA, RSI, etc.) were computed from raw prices. If we adjust `close` but keep stored SMA values, then:

```
dist_sma_20 = adjusted_close / raw_sma_20 - 1  ← wrong: numerator and denominator on different scales
```

For path B symbols, we throw away stored indicators and recompute from adjusted OHLC using the same `ta` library already used in `indicators.py`.

---

## 5. Files to Modify

### 5.1 `database/db.py`
Add one function:
```python
def get_all_corporate_actions() -> Dict[str, List[Dict]]:
    """
    Load all rows from corporate_actions, grouped by nse_symbol.
    Returns: {nse_symbol: [{"ex_date": date, "adj_factor": float, "event_type": str}, ...]}
    Ordered ex_date ASC per symbol so compound adjustment applies correctly.
    """
```

Single DB query at training start — cached in memory for the full training run.

---

### 5.2 `analysis/indicators.py`
Add one function:
```python
def compute_indicators_for_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all technical indicators on an in-memory price DataFrame.

    Input df must have columns: open, high, low, close, volume (DatetimeIndex).
    Returns df with indicator columns added (same as stored in technical_indicators table):
        rsi_14, macd, macd_signal, macd_hist,
        bb_upper, bb_lower, bb_middle,
        sma_20, sma_50, sma_200, ema_9, ema_21,
        atr_14, adx_14, stoch_k, stoch_d, obv
    """
```

This mirrors the existing `calculate_indicators()` logic but works on a DataFrame instead of fetching from DB.

---

### 5.3 `analysis/model_training.py`
Three additions:

#### A. `_apply_price_adjustment(sym_df, actions) → pd.DataFrame`
```python
def _apply_price_adjustment(sym_df: pd.DataFrame, actions: List[Dict]) -> pd.DataFrame:
    """
    Apply backward OHLC adjustment for all corporate actions, oldest→newest.

    For each event:
        all rows where index < ex_date → multiply open/high/low/close by adj_factor

    Volume: multiplied by (1/adj_factor) for splits to normalise traded quantity.
            Left unchanged for bonus issues (bonus doesn't affect share lot sizes
            in the same way and volume_ratio self-normalises over the 20d window).
    """
```

#### B. ~~`_get_exclusion_mask()`~~ — NOT NEEDED

An exclusion window is not required. With full-series backward adjustment applied to `close`, `future_close = df['close'].shift(-forward_days)` is also on the adjusted scale for all rows — including rows whose target horizon crosses `ex_date`.

Proof for T = ex_date − 50, forward_days = 120, adj = 0.5 (HDFCBANK bonus):
```
close[T]        = raw_close × 0.5  = 900   (adjusted)
future_close[T] = close[ex_date+70] = 950  (post-event, already on adjusted scale)
future_return   = (950 − 900) / 900 = 5.56%  ← economically correct
```

Same for `return_1d` on ex_date itself:
```
close[ex_date]     = 905  (post-event)
close[ex_date − 1] = 1800 × 0.5 = 900  (adjusted)
return_1d          = (905 − 900) / 900 = +0.56%  ← no −50% spike
```

Zero training rows are lost. The only optional exclusion is `ex_date ± 1` for volume spike noise, but the price signal is clean.

#### C. `_recompute_indicators(sym_df) → pd.DataFrame`

**Critical:** after `_apply_price_adjustment()`, `close` is adjusted but stored indicator columns (`sma_20`, `rsi_14`, etc.) still hold raw DB values — a scale mismatch. `_recompute_indicators` must **DROP** all stored indicator columns first, then assign freshly computed ones:

```python
STORED_INDICATOR_COLS = [
    "sma_20", "sma_50", "sma_200", "ema_9", "ema_21",
    "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_middle",
    "atr_14", "adx_14", "stoch_k", "stoch_d", "obv",
]

def _recompute_indicators(sym_df: pd.DataFrame) -> pd.DataFrame:
    # Drop stored (raw-price) indicator columns to prevent scale mismatch
    sym_df = sym_df.drop(columns=[c for c in STORED_INDICATOR_COLS if c in sym_df.columns])
    # Recompute from adjusted open/high/low/close/volume
    computed = compute_indicators_for_df(sym_df)
    return sym_df.join(computed)
```

After this call: `close` (adjusted) and `sma_20` (recomputed from adjusted prices) are on the same scale.

#### D. Modify `prefetch_all_data()` to call the above

In the existing loop over `target_symbols`:
```python
# After building sym_df, before adding to _DATA_CACHE:

actions = corp_actions_map.get(f"{sym}.NS", [])
if actions:
    sym_df = _apply_price_adjustment(sym_df, actions)
    sym_df = _recompute_indicators(sym_df)   # drops raw stored cols, adds adjusted cols

_DATA_CACHE[sym] = sym_df
```

No change to `build_features_and_target()` — all rows are used, zero excluded.

---

## 6. Adjustment Logic Detail

### 6.1 Single event (HDFCBANK, Bonus 1:1, 2025-08-26, adj=0.5)

```
Date        Raw Close   Adjusted Close
2025-08-20  1750.00  →  875.00   (× 0.5)
2025-08-25  1800.00  →  900.00   (× 0.5)
2025-08-26  905.00  →  905.00   (no change — post-event)
2025-08-27  910.00  →  910.00   (no change)
```

`return_1d` on 2025-08-26 was previously (905−1800)/1800 = **−49.7%** (false crash)  
After adjustment: (905−900)/900 = **+0.56%** (correct)

### 6.2 Multiple events (BAJFINANCE, Bonus 4:1 + Split 2:1, 2025-06-16, combined adj=0.1)

Both events on same date. Apply oldest first in loop — since same date, combined multiplier = 0.2 × 0.5 = 0.1.

```
Date        Raw Close   Adjusted Close
2025-06-13  8000.00  →  800.00   (× 0.1)
2025-06-16   820.00  →  820.00   (post-event)
```

### 6.3 Multiple events on different dates (CGCL: Bonus 1:1 + Split 2:1 both 2024-03-05, 360ONE: Split 2024-03-02 + Bonus 2024-03-02)

Same handling as 6.2 — same date events compound.

### 6.4 Older event then newer event (hypothetical)

```
Events: Split 5:1 on 2023-06-01, Bonus 1:1 on 2024-09-15
```

Apply oldest first:
- All rows before 2023-06-01: multiply by 0.2
- All rows before 2024-09-15 (including pre-2023): multiply by 0.5
- Net for rows before 2023-06-01: × 0.2 × 0.5 = × 0.1
- Rows between 2023-06-01 and 2024-09-15: × 0.5 only

---

## 7. Volume Adjustment

| Event type | Pre-event volume adjustment | Reason |
|---|---|---|
| Split N:1 | × N (i.e. × 1/adj_factor) | 1 old share = N new shares; equivalent traded volume normalised |
| Bonus | No change | Bonus shares don't alter lot sizes; `volume_ratio` self-normalises over 20d rolling mean |

Volume adjustment only affects `volume_ratio` and `volume_trend` features. Since these are ratio-based, partial adjustment is acceptable.

---

## 8. Training Row Impact

**Zero rows lost.** Full-series backward adjustment means every row — including those whose target horizon crosses `ex_date` — produces a correct, economically meaningful `future_return_pct`.

The only optional micro-exclusion is `ex_date ± 1` (2 rows per event) to drop volume spike noise. Even this is cosmetic, not required for correctness.

---

## 9. Implementation Order

| Step | File | Function | Complexity |
|---|---|---|---|
| 1 | `database/db.py` | `get_all_corporate_actions()` | Low |
| 2 | `analysis/indicators.py` | `compute_indicators_for_df()` | Medium |
| 3 | `analysis/model_training.py` | `_apply_price_adjustment()` | Low |
| 4 | `analysis/model_training.py` | `_recompute_indicators()` (wrapper) | Low |
| 5 | `analysis/model_training.py` | Modify `prefetch_all_data()` | Low |
| 6 | — | Test on HDFCBANK, BAJFINANCE | Validation |

---

## 10. Validation Plan

After implementation, verify with a single symbol before full retrain:

```python
# Before adjustment:
# HDFCBANK 2025-08-26: return_1d should be ~−50%
# After adjustment:
# HDFCBANK 2025-08-26: return_1d should be ~+0.5%

from analysis.model_training import prefetch_all_data, _DATA_CACHE
prefetch_all_data(symbols=["HDFCBANK"])
df = _DATA_CACHE["HDFCBANK"]
row = df.loc["2025-08-26"]
print(f"return_1d on ex_date: {row['return_1d']:.4f}")    # expect ~0.005, not −0.50
print(f"dist_sma_20 on ex_date: {row['dist_sma_20']:.4f}")  # expect small value
print(f"Total rows (all used): {len(df)}")               # should be same as before adjustment
```

---

## 11. What This Does NOT Fix

- **Stored `technical_indicators` table** — remains raw/unadjusted (used only for non-affected symbols in fast path)
- **Trade signals generated from current live price** — signal generation uses the most recent close, which is always post-event, no adjustment needed
- **Backtesting** — if a backtester reads stored prices, it will still see the raw prices. Adjustment only applies during ML training.
- **Dividends** — not handled (small impact; dividend-adjusted price ≈ raw price for Indian stocks where dividends are small)

---

## 12. Expected Model Impact

Stocks where existing models are most likely distorted:

| Stock | Event | adj_factor | Rows corrupted (est.) |
|---|---|---|---|
| NESTLEIND | Split 10:1, 2024-01-05 | 0.1 | ~200 (SMA200 window) |
| HDFCBANK | Bonus 1:1, 2025-08-26 | 0.5 | ~200 |
| BAJFINANCE | Bonus+Split, 2025-06-16 | 0.1 | ~200 |
| KOTAKBANK | Split 5:1, 2026-01-14 | 0.2 | ~200 (very recent) |
| RELIANCE | Bonus 1:1, 2024-10-28 | 0.5 | ~200 |
| WIPRO | Bonus 1:1, 2024-12-03 | 0.5 | ~180 |

These are high-coverage stocks with many news articles and signal interest — fixing their training data will have the most visible impact on signal quality.
