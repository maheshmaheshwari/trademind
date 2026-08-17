"""
TradeMind AI — Walk-Forward Model Retraining

Trains all ~500 Nifty 500 stocks using a strict time-based (walk-forward) split.
The cutoff is ROLLING, not fixed — TRAIN_END is the last month-end TRAIN_GAP_MONTHS
(default 6) before the current month, so every run trains on all available history
up to ~6 months ago and tests on the most recent ~6 months. For a run in Jul 2026:
  Train : all history through 2026-01-31
  Test  : 2026-02-01 – present   (held-out, ~6 months)
(Override with TRAIN_END_OVERRIDE / TRAIN_GAP_MONTHS. See the TRAIN_END config below.)

For each stock, trains 6-7 models × 6 horizons, selects the best
by harmonic mean of accuracy + precision on the held-out test set,
and saves it to final_models/{SYMBOL}_final.pkl.

Per-symbol metrics are written to the model_training_stats DB table (read by
/api/backtest/summary) and also logged to data/retrain_results.csv for local review.

Usage:
    PYTHONPATH=. python retrain_walk_forward.py
    PYTHONPATH=. python retrain_walk_forward.py --symbol HDFCBANK
    PYTHONPATH=. python retrain_walk_forward.py --workers 4   (parallel)
    PYTHONPATH=. python retrain_walk_forward.py --resume      (skip done)
"""

import argparse
import calendar
import csv
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import date, datetime
from typing import Optional, Tuple

import joblib
import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BACKEND_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join("logs", f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
    ]
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
# ROLLING training window (was frozen at 2024-12-31). The signal backtester
# (scripts/backtest_signals.py) showed models decay ~12 months after their
# training cutoff — a frozen 2024-12-31 model was strongly +alpha through 2025
# then LOST money in 2026, while a model retrained through 2025-12-31 was +alpha
# on the same 2026 window (A/B: frozen −15%..−23% CAGR vs rolling +4%..+19%).
# So TRAIN_END now rolls forward each run: train through the last month-end that
# is TRAIN_GAP_MONTHS before the current month, test on the recent tail. This
# keeps every weekly retrain's models fresh instead of re-fitting stale data.
#
# Gap = 9 months. It was 6, and the comment here claimed that "keeps the longest
# (6-month) horizon testable". It did exactly the opposite, and silently: the
# 6-month horizon has produced ZERO signals since 2026-07-29 and every gha_*
# retrain has picked it 0 times, while the local backfill run picked it for 133
# symbols (audit #4).
#
# The arithmetic. A 120-trading-day forward return means engineer_features_and_
# target() can only produce rows up to ~120 trading days before the last price
# bar. Six calendar months of gap is almost exactly those 120 days, so X ends
# BEFORE the test window opens and the horizon gets no test rows at all —
# train_and_evaluate() then hits `len(Xte) < 10` and skips it. Measured on
# RELIANCE.NS with prices through 2026-08-14:
#
#     gap   TRAIN_END     6M test rows   verdict
#       6   2026-02-28               0   SKIPPED  (X ends 2026-02-17)
#       7   2026-01-31              13   passes the guard, but on noise
#       8   2025-12-31              33
#       9   2025-11-30              55   <- chosen
#      10   2025-10-31              74
#      12   2025-08-31             117
#
# 9 buys the 6M horizon a real out-of-sample window at the cost of models three
# months staler at the cutoff. That is still well inside the fresh zone the A/B
# identified — it found a FROZEN 2024-12-31 model losing money through 2026
# (−15%..−23% CAGR) versus a rolling one at +4%..+19%; the problem there was
# staleness measured in years, not one extra quarter.
#
# Shorter horizons are unaffected except for having more test rows.
#
# Overridable via env for pinning/experiments:
#   TRAIN_END_OVERRIDE=2024-12-31   (force a fixed cutoff)
#   TRAIN_GAP_MONTHS=9              (months of recent test tail to hold out)
TRAIN_GAP_MONTHS = int(os.environ.get("TRAIN_GAP_MONTHS", "9"))


def _rolling_train_end(gap_months: int = TRAIN_GAP_MONTHS) -> str:
    """Last day of the month `gap_months` before the current month (ISO date)."""
    today = date.today()
    m = today.month - gap_months
    y = today.year
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, calendar.monthrange(y, m)[1]).isoformat()


def _next_day(iso: str) -> str:
    from datetime import timedelta
    return (datetime.strptime(iso, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()


TRAIN_END   = os.environ.get("TRAIN_END_OVERRIDE") or _rolling_train_end()
TEST_START  = _next_day(TRAIN_END)   # first day of test (no overlap)
FINAL_DIR   = "final_models"
RESULTS_CSV = os.path.join("data", "retrain_results.csv")
# Quality tiers, NOT a save gate. Every symbol's freshly-trained best model is
# always saved (100% coverage, no stale leftovers) — the real tradeability gate
# lives in scripts/generate_trades.py, which already requires acc>=0.70 (BUY) /
# 0.80 (STRONG BUY) before emitting a buy signal. The tier below is recorded on
# the artifact so signal generation and the UI can see model quality at a glance.
QUALITY_HIGH_ACC   = 0.70    # acc>=this (with decent precision) → BUY-eligible
QUALITY_MED_ACC    = 0.60    # acc>=this → "medium" tier
QUALITY_MIN_PREC   = 0.60    # precision floor for high/medium tiers


def quality_tier(acc: float, prec: float) -> str:
    """Classify a model by its walk-forward test metrics (informational)."""
    if acc >= QUALITY_HIGH_ACC and prec >= QUALITY_MIN_PREC:
        return "high"
    if acc >= QUALITY_MED_ACC and prec >= QUALITY_MIN_PREC:
        return "medium"
    return "low"

os.makedirs(FINAL_DIR, exist_ok=True)
os.makedirs("data",    exist_ok=True)
os.makedirs("logs",    exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def already_trained(symbol: str) -> bool:
    """True if a fresh walk-forward model already exists for this symbol."""
    path = os.path.join(FINAL_DIR, f"{symbol}_final.pkl")
    if not os.path.exists(path):
        return False
    try:
        art = joblib.load(path)
        # Only count it if it was trained with the new walk-forward cutoff
        return art.get("train_end_date") == TRAIN_END
    except Exception:
        return False


def train_one(symbol: str) -> dict:
    """
    Train all models for one symbol using walk-forward split.
    Returns a result dict for CSV logging.
    """
    from analysis.model_training import train_and_evaluate
    t0 = time.time()
    result = {
        "symbol": symbol, "status": "error",
        "best_model": "", "horizon": "",
        "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "train_rows": 0, "test_rows": 0,
        "elapsed_s": 0, "corp_adj": False, "error": "",
    }
    try:
        art = train_and_evaluate(
            symbol,
            train_end_date=TRAIN_END,
            test_start_date=TEST_START,
        )
        elapsed = round(time.time() - t0, 1)
        result["elapsed_s"] = elapsed

        if art is None:
            result["status"] = "no_data"
            return result

        m = art.get("metrics", {})
        acc  = m.get("accuracy",  0.0)
        prec = m.get("precision", 0.0)
        tier = quality_tier(acc, prec)

        # No save gate: always persist the fresh best model so every symbol has a
        # current model. Quality is enforced downstream at signal time.
        # Tag with walk-forward metadata and save
        art["quality_tier"]    = tier
        art["train_end_date"]  = TRAIN_END
        art["test_start_date"] = TEST_START
        art["retrained_at"]    = datetime.now().isoformat()
        art["retrained_date"]  = datetime.now().strftime("%Y-%m-%d")
        # Flag whether corporate action price adjustment was applied to training data.
        # Check _DATA_CACHE to see if this symbol's prices were adjusted.
        from analysis.model_training import _DATA_CACHE
        from database.db import get_all_corporate_actions
        _corp_actions = get_all_corporate_actions()
        art["corp_adj"] = symbol in _corp_actions and len(_corp_actions[symbol]) > 0
        result["corp_adj"] = art["corp_adj"]

        out_path = os.path.join(FINAL_DIR, f"{symbol}_final.pkl")

        # Archive the existing model before overwriting it.
        # Name: previous_models/{symbol}_final_{YYYY-MM-DD}.pkl  using the old model's retrained_date.
        prev_dir = os.path.join(_BACKEND_DIR, "model_archives", "previous_models")
        os.makedirs(prev_dir, exist_ok=True)
        if os.path.exists(out_path):
            try:
                old_art  = joblib.load(out_path)
                old_date = old_art.get("retrained_date") or old_art.get("retrained_at", "unknown")[:10]
            except Exception:
                old_date = "unknown"
            archive_path = os.path.join(prev_dir, f"{symbol}_final_{old_date}.pkl")
            import shutil
            shutil.move(out_path, archive_path)

        joblib.dump(art, out_path)

        # status "ok" = saved (model is on disk); quality_tier carries the grade.
        result.update({
            "status":     "ok",
            "best_model": art.get("model_name", ""),
            "horizon":    art.get("horizon", ""),
            "accuracy":   round(acc, 4),
            "precision":  round(prec, 4),
            "recall":     round(m.get("recall", 0.0), 4),
            "f1":         round(m.get("f1", 0.0), 4),
            "quality_tier": tier,
        })
        logger.info(f"{'✅' if tier != 'low' else '🟡'} {symbol}: {art.get('model_name')} | "
                    f"{art.get('horizon')} acc={acc:.1%} prec={prec:.1%} [{tier}] ({elapsed}s)")
    except Exception as e:
        result["error"]   = str(e)
        result["elapsed_s"] = round(time.time() - t0, 1)
        logger.error(f"❌ {symbol}: {e}\n{traceback.format_exc()}")

    return result


def _worker(args):
    symbol = args
    return train_one(symbol)


# ── Main ───────────────────────────────────────────────────────────────────────

def _cached_symbols() -> set:
    """Symbols currently resident in the training cache."""
    from analysis.model_training import _DATA_CACHE
    return set(_DATA_CACHE)


def retrain_all(symbol_filter: Optional[str] = None,
                workers: int = 1,
                resume: bool = True,
                shard: Optional[Tuple[int, int]] = None,
                training_data: Optional[str] = None):
    from database.db import get_active_universe

    # Current Nifty 500 constituents only. This used to be
    # `SELECT DISTINCT symbol FROM prices`, which is ~537: de-indexed names
    # whose price history is deliberately retained, plus the index tickers
    # (^NSEI, ^BSESN, ^INDIAVIX, ^CRSLDX) that are benchmarks rather than
    # stocks — all of them were being trained.
    all_symbols = get_active_universe()

    if symbol_filter:
        all_symbols = [s for s in all_symbols if s == symbol_filter]
        if not all_symbols:
            logger.error(f"Symbol {symbol_filter} not found in DB")
            sys.exit(1)

    if shard:
        # Round-robin split so shards stay balanced regardless of list size
        idx, n = shard
        all_symbols = all_symbols[idx::n]
        logger.info(f"Shard {idx + 1}/{n}: {len(all_symbols)} symbols")

    if resume:
        pending = [s for s in all_symbols if not already_trained(s)]
        skipped = len(all_symbols) - len(pending)
        logger.info(f"Resume mode: {skipped} already done, {len(pending)} to go")
    else:
        pending = all_symbols

    total = len(pending)
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 Walk-Forward Retraining")
    logger.info(f"   Train: Jan 2023 – {TRAIN_END}  |  Test: {TEST_START} – present")
    logger.info(f"   Stocks: {total}  |  Workers: {workers}")
    logger.info(f"{'='*70}\n")

    if training_data:
        # Cache exported by an earlier job (see model_training.export_training_data).
        # This shard reads prices ZERO times: both full-history scans it would
        # otherwise run — the prefetch join and the unfiltered sector-returns
        # window query — were done once, sequentially, by the export job.
        logger.info(f"📦 Loading training cache from {training_data} (no DB scans)...")
        from analysis.model_training import load_training_data
        n = load_training_data(training_data)
        missing = [s for s in pending if s not in _cached_symbols()]
        if missing:
            # Fail loudly. Silently training a subset would look like success
            # while quietly dropping symbols from the model set.
            logger.error(f"Training cache is missing {len(missing)} of this shard's "
                         f"{len(pending)} symbols, e.g. {missing[:5]}")
            logger.error("The export's shard slicing must match this job's — "
                         "check --shard i/N is identical on both sides.")
            sys.exit(1)
        logger.info(f"✅ {n} symbols loaded from cache — starting training\n")
    else:
        # Pre-fetch ALL stock data from DB once — zero DB queries during training
        logger.info("⏳ Pre-fetching all stock data from DB...")
        from analysis.model_training import prefetch_all_data, precompute_sector_returns
        prefetch_all_data(symbols=pending)     # only fetch symbols we'll actually train
        precompute_sector_returns()            # sector returns from the cached price data
        logger.info("✅ All data cached — starting training (zero DB queries from here)\n")

    results = []
    done = 0
    ok = 0
    failed = []

    # Write CSV header
    csv_fields = ["run_id","symbol","status","best_model","horizon",
                  "accuracy","precision","recall","f1",
                  "train_rows","test_rows","elapsed_s","corp_adj","error"]
    write_header = not os.path.exists(RESULTS_CSV)
    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

    # Shared run identifier — all rows from this invocation share the same run_id
    # so you can filter retrain_results.csv by run_id to see only the latest run.
    # RETRAIN_RUN_ID lets parallel shards of one logical run share a single id.
    RUN_ID = os.environ.get("RETRAIN_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")

    WORKER_TIMEOUT = 600  # 10 minutes max per stock — auto-skip if stuck

    def _run_with_pool(symbol_list):
        """Run symbol_list through ProcessPoolExecutor. Returns (done, ok, failed, results)."""
        _done, _ok, _failed, _results = 0, 0, [], []
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(_worker, s): s for s in symbol_list}
                for fut in as_completed(futs):
                    sym = futs[fut]
                    try:
                        r = fut.result(timeout=WORKER_TIMEOUT)
                    except Exception as e:
                        err_msg = f"timeout after {WORKER_TIMEOUT}s" if "TimeoutError" in type(e).__name__ else str(e)
                        logger.error(f"❌ {sym}: {err_msg} — skipping")
                        r = {"symbol": sym, "status": "error", "error": err_msg,
                             "best_model": "", "horizon": "", "accuracy": 0,
                             "precision": 0, "recall": 0, "f1": 0,
                             "train_rows": 0, "test_rows": 0, "elapsed_s": WORKER_TIMEOUT}
                        _failed.append(sym)
                    _results.append(r)
                    writer.writerow({k: r.get(k, "") for k in csv_fields
                                     if k not in ("run_id",)} | {"run_id": RUN_ID})
                    csv_fh.flush()
                    _done += 1
                    if r["status"] == "ok":
                        _ok += 1
                    pct = (_done + done) / total * 100
                    logger.info(f"[{_done + done}/{total} {pct:.0f}%] {r['symbol']} → {r['status']}")
        except BrokenProcessPool as exc:
            # A worker was killed (SIGSEGV / OOM / signal) — the pool cannot recover.
            # Identify which symbols never returned a result and fall through to single-threaded mode.
            finished = {r["symbol"] for r in _results}
            orphaned = [s for s in symbol_list if s not in finished]
            logger.error(
                f"⚠️  ProcessPool broken ({exc}). "
                f"{len(orphaned)} symbols never returned — retrying in single-threaded mode."
            )
            return _done, _ok, _failed, _results, orphaned
        return _done, _ok, _failed, _results, []

    if workers > 1:
        pool_done, pool_ok, pool_failed, pool_results, orphaned = _run_with_pool(pending)
        done += pool_done
        ok += pool_ok
        failed.extend(pool_failed)
        results.extend(pool_results)

        # Re-run orphaned symbols single-threaded (no OpenMP conflicts possible)
        if orphaned:
            logger.info(f"\n🔁 Single-threaded fallback for {len(orphaned)} orphaned stocks...")
            for sym in orphaned:
                r = train_one(sym)
                results.append(r)
                writer.writerow({k: r.get(k, "") for k in csv_fields})
                csv_fh.flush()
                done += 1
                if r["status"] == "ok":
                    ok += 1
                else:
                    failed.append(r["symbol"])
                pct = done / total * 100
                logger.info(f"[{done}/{total} {pct:.0f}%] {r['symbol']} → {r['status']}")
    else:
        for i, symbol in enumerate(pending, 1):
            r = train_one(symbol)
            results.append(r)
            writer.writerow({k: r.get(k,"") for k in csv_fields})
            csv_fh.flush()
            done += 1
            if r["status"] == "ok": ok += 1
            else: failed.append(r["symbol"])
            pct = done / total * 100
            eta_min = (total - done) * (sum(r2["elapsed_s"] for r2 in results) / done) / 60
            logger.info(f"[{done}/{total} {pct:.0f}%] ETA ~{eta_min:.0f}min")

    csv_fh.close()

    # Persist per-symbol metrics to the DB (the source /api/backtest/summary
    # reads). Each shard upserts its own symbols under the shared RETRAIN_RUN_ID,
    # so the route sees one coherent run. Non-fatal: the models are already saved
    # to disk — a DB hiccup here must not fail the retrain.
    try:
        from database.db import insert_model_training_stats
        n = insert_model_training_stats(RUN_ID, results)
        logger.info(f"📥 Wrote {n} training-stat rows to DB (run_id={RUN_ID})")
    except Exception as e:
        logger.error(f"⚠️  Failed to write training stats to DB: {e}")

    # Summary
    from collections import Counter
    tiers = Counter(r.get("quality_tier") for r in results if r["status"] == "ok")
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Retraining Complete")
    logger.info(f"   ✅ Saved  : {ok}   "
                f"(high={tiers.get('high',0)} medium={tiers.get('medium',0)} low={tiers.get('low',0)})")
    logger.info(f"   ⚠️  No data / error / timeout (not saved): {len(failed)}")
    if failed:
        logger.info(f"   Not saved: {', '.join(failed[:20])}")
    logger.info(f"   📄 Results: {RESULTS_CSV}")

    # Print top 20 by accuracy
    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: (r["accuracy"] + r["precision"]) / 2, reverse=True)
    if ok_results:
        logger.info(f"\n🏆 Top models by avg(accuracy, precision):")
        logger.info(f"  {'Symbol':<20} {'Model':<12} {'Horizon':<10} {'Acc':>6} {'Prec':>6}")
        logger.info(f"  {'-'*60}")
        for r in ok_results[:20]:
            logger.info(f"  {r['symbol']:<20} {r['best_model']:<12} {r['horizon']:<10} "
                        f"{r['accuracy']:>5.1%} {r['precision']:>6.1%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Walk-forward model retraining")
    parser.add_argument("--symbol",  type=str, default=None, help="Single symbol e.g. HDFCBANK.NS")
    parser.add_argument("--workers", type=int, default=2,    help="Parallel processes (default 2)")
    parser.add_argument("--resume",  action="store_true", default=True,
                        help="Skip already-trained symbols (default True)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Retrain all even if already done")
    args = parser.parse_args()

    retrain_all(
        symbol_filter=args.symbol,
        workers=args.workers,
        resume=args.resume,
    )
