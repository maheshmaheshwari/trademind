# Issue: Newly Listed Stocks — No Training Data in Walk-Forward Window

## Summary

23 stocks in the Nifty 500 universe either have **zero** or **fewer than 60 rows** in the
training window (Jan 2023 – Dec 2024). The walk-forward retrainer marks them as `no_data`
or `below_threshold` and skips them. Their models are never saved; the signal generator
falls back to the rules-based engine for these stocks.

Data IS available for these stocks online — the problem is that they were listed (IPO,
demerger, or corporate restructuring) **after** the training cutoff, so our price collector
never captured their history in the training window.

---

## Root Cause

`retrain_walk_forward.py` uses a fixed train/test split:

```
Train : Jan 2023 – Dec 2024
Test  : Jan 2025 – present
```

Any stock whose NSE listing date falls after Jan 2025 has 0 training rows.
Any stock listed in late 2024 has fewer than 60 trading days in the training window,
which is below the minimum the ML pipeline needs to build reliable features (rolling
windows of 5, 20, 50, 200 days all fail or return NaN for every row).

---

## Affected Stocks

### Group A — Zero training-window rows (listed after Jan 2025)

| Symbol | Total Rows | First Available Date | Notes |
|---|---|---|---|
| JSWCEMENT.NS | 205 | 2025-08-14 | JSW Cement IPO |
| ABLBL.NS | 243 | 2025-06-23 | |
| ENRIN.NS | 245 | 2025-06-19 | |
| AEGISVOPAK.NS | 258 | 2025-06-02 | Aegis-Vopak merger entity |
| THELEELA.NS | 258 | 2025-06-02 | The Leela Hotels spinoff |
| ATHERENERG.NS | 277 | 2025-05-06 | Ather Energy IPO |
| HEXT.NS | 324 | 2025-02-19 | |
| AGARWALEYE.NS | 335 | 2025-02-04 | |
| ITCHOTELS.NS | 340 | 2025-01-29 | ITC Hotels demerger from ITC |
| ONESOURCE.NS | 343 | 2025-01-24 | |

### Group B — Fewer than 60 training-window rows (listed Oct–Dec 2024)

| Symbol | Total Rows | Train Rows | First Date |
|---|---|---|---|
| VENTIVE.NS | 361 | 1 | 2024-12-30 |
| IGIL.NS | 366 | 6 | 2024-12-20 |
| IKS.NS | 367 | 7 | 2024-12-19 |
| VMM.NS | 368 | 8 | 2024-12-18 |
| SAILIFE.NS | 368 | 8 | 2024-12-18 |
| NTPCGREEN.NS | 383 | 23 | 2024-11-27 |
| NIVABUPA.NS | 390 | 30 | 2024-11-14 |
| ACMESOLAR.NS | 391 | 31 | 2024-11-13 |
| SWIGGY.NS | 391 | 31 | 2024-11-13 |
| SAGILITY.NS | 392 | 32 | 2024-11-12 |
| AFCONS.NS | 398 | 38 | 2024-11-04 |
| WAAREEENER.NS | 325 | 43 | 2024-10-28 |
| HYUNDAI.NS | 407 | 47 | 2024-10-22 |

**Total affected: 23 stocks**

---

## Why This Happens (Not a Bug in the Collector)

The collector correctly captures data from the stock's actual NSE listing date onward.
There is no pre-listing price history to collect for a stock that did not exist on NSE
before its IPO date. The issue is structural: the fixed training window predates these
listings.

---

## Data Source: Can Angel One Fill the Gap?

**Short answer: No — and neither can yfinance.** Pre-listing price data does not exist
anywhere because these stocks were not traded before their IPO date.

Angel One's `getCandleData` endpoint returns data from a stock's NSE listing date onward —
the same window our `angel_collector.py` already captures. There are no "missing" rows to
backfill for the training window; the data simply never existed for that period.

**Exception — Group B stocks (listed Oct–Dec 2024):** Angel One *may* have a handful of
early trading days that our collector missed if it wasn't configured for the symbol at
listing time. A targeted backfill via `angel_collector.py getCandleData` for each Group B
symbol back to its listing date is worth doing, but it adds at most a few weeks of rows —
not enough to fix Group B's thin training sets on its own.

**yfinance** has the same limitation. It reflects actual NSE trading history; for a stock
listed in Aug 2025, it starts in Aug 2025 regardless of source.

---

## Decided Approach (not yet implemented)

### Adaptive 80/20 split for newly listed stocks

When a stock has insufficient data in the fixed training window, detect it early and switch
to a per-stock chronological 80/20 split instead of failing with `no_data`:

```
train = first 80% of available rows (sorted by date)
test  = last 20% of available rows
```

**Where to detect**: in `retrain_walk_forward.py`, before calling `train_and_evaluate()`,
check if `train_rows < MIN_TRAIN_ROWS` (e.g. < 60). If so, pass `train_end_date=None`
and `adaptive_split=True` to `train_and_evaluate()`.

**Where to implement**: `analysis/model_training.py` → `train_and_evaluate()` accepts
`adaptive_split: bool = False`. When True, ignores `train_end_date`/`test_start_date` and
splits the stock's own data 80/20 by row index.

**Expected outcome per group:**

| Group | Example | Available Rows | Train (80%) | Test (20%) |
|---|---|---|---|---|
| A (post-Jan 2025) | JSWCEMENT.NS | 205 | ~164 | ~41 |
| A (post-Jan 2025) | SWIGGY.NS (if listed 2025) | ~300 | ~240 | ~60 |
| B (Oct–Dec 2024) | HYUNDAI.NS | 407 | ~326 | ~81 |
| B (Oct–Dec 2024) | NTPCGREEN.NS | 383 | ~306 | ~77 |

**Trade-offs:**
- Models for Group A stocks will be weaker — 164 rows is enough for XGBoost/RF but
  rolling features (SMA-200, 52-week hi/lo) will be NaN for the first 200 rows, so
  effective feature coverage is limited. These models should be treated as indicative only.
- Test periods differ across stocks — not directly comparable to the fixed-window cohort.
  Keep a `split_type` field in the artifact (`"fixed"` vs `"adaptive"`) so downstream
  code can filter or weight accordingly.
- Self-correcting: each retrain cycle adds ~125 more trading days. By Dec 2026, Group B
  stocks will have 530+ rows and Group A stocks 330+ rows — enough for a meaningful fixed
  or adaptive split.

**Do not implement now** — implement after current retrain (Jun 2026) completes.
Rolling `TRAIN_END` (see below) handles graduation automatically so no separate
"remove label" step is needed.

---

## Rolling TRAIN_END with Walk-Forward Aggregation (decided — implement after current retrain)

Rather than comparing metrics across retrains (which is misleading because the test window
shifts), each retrain **accumulates** test results into the artifact. This gives a
weighted, multi-year performance estimate that grows more reliable each cycle.

### How it works

```
Retrain Dec 2025: train Jan 2023–Dec 2024, test Jan–Dec 2025 (380 samples, acc=77%)
Retrain Dec 2026: train Jan 2023–Dec 2025, test Jan–Dec 2026 (252 samples, acc=65%)
Aggregated:       (77%×380 + 65%×252) / 632  →  72.5%  ← stable multi-year estimate
```

Each `.pkl` artifact stores a `test_history` list (one entry per retrain cycle) alongside
the existing `metrics` dict. On every retrain, `train_and_evaluate()` loads the previous
artifact, appends the new period's result, and recomputes a sample-weighted aggregate.

```python
# New fields added to artifact:
artifact["test_history"] = [
    {"train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31",
     "accuracy": 0.77, "precision": 0.74, "n_samples": 380},
    {"train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-06-16",
     "accuracy": 0.65, "precision": 0.68, "n_samples": 120},
]
artifact["aggregated_metrics"] = {
    "accuracy":  0.745,   # weighted by n_samples across all periods
    "precision": 0.726,
    "total_samples": 500,
    "periods": 2,
}
```

### Why this is better than comparison

- A model consistently at 72% across 3 market years is more trustworthy than one that
  scored 90% in a bull year and 55% in a correction.
- Aggregated metrics absorb different market regimes — each test year adds a distinct
  economic environment (2025 recovery, 2026 data, etc.).
- Newly listed stocks that start on adaptive split accumulate their own history —
  by their third retrain they have a multi-period aggregate just like established stocks.

### Files that need to change

| File | Change |
|---|---|
| `analysis/model_training.py` | Load existing artifact before training; append new period to `test_history`; compute `aggregated_metrics` (sample-weighted); save both alongside existing `metrics` |
| `analysis/signals.py` | Read `artifact.get("aggregated_metrics") or artifact["metrics"]` for display and buy_prob thresholds — fallback keeps old single-run artifacts working |
| `retrain_walk_forward.py` | Roll `TRAIN_END` / `TEST_START` dynamically; `already_trained()` check against current `TRAIN_END` is unchanged |

`already_trained()`, the resume logic, and the CSV results file need no changes.

### Graduation of newly listed stocks with this approach

With a rolling `TRAIN_END`, adaptive-split stocks graduate automatically:

| Stock group | Listed | Graduates to fixed window |
|---|---|---|
| Group B (Oct–Dec 2024) | Oct–Dec 2024 | Dec 2026 retrain (TRAIN_END = 2025-12-31, ~250 train rows) |
| Group A early (Jan–Jun 2025) | Jan–Jun 2025 | Dec 2027 retrain (TRAIN_END = 2026-12-31) |
| Group A late (Jul–Dec 2025) | Jul–Dec 2025 | Dec 2027 retrain (TRAIN_END = 2026-12-31, ~125 rows — may still need adaptive) |

The `adaptive_split` flag in `train_and_evaluate()` is checked at runtime against the
current `TRAIN_END`, so graduation is automatic — no manual label removal needed.

---

## Rejected Options

| Option | Reason rejected |
|---|---|
| Yahoo Finance backfill | Same listing-date limitation as Angel One. No new training rows. |
| Angel One backfill | Already our source. May recover a few missed early days for Group B but doesn't solve the structural gap. |
| Parent company proxy (for demergers) | High manual maintenance, price behaviour diverges post-demerger. Not generalizable. |
| Skip with better logging only | These 23 stocks never get ML signals — rules-based fallback is worse than a thin adaptive model. |

---

## How to Re-check After Future Data Collection

```python
# Run this to see current training-window coverage per stock
SELECT symbol,
       SUM(CASE WHEN date < '2025-01-01' THEN 1 ELSE 0 END) AS train_rows,
       COUNT(*) AS total_rows,
       MIN(date) AS first_date
FROM prices
WHERE interval = '1d'
GROUP BY symbol
HAVING SUM(CASE WHEN date < '2025-01-01' THEN 1 ELSE 0 END) < 60
ORDER BY train_rows;
```
