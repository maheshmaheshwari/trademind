"""
TradeMind AI — Walk-Forward Model Retraining

Trains all 499 Nifty 500 stocks using a strict time-based split:
  Train : Jan 2023 – Dec 2024  (2 years, ~504 trading days)
  Test  : Jan 2025 – present   (1 year+, ~330 trading days)

For each stock, trains 6-7 models × 6 horizons, selects the best
by harmonic mean of accuracy + precision on the held-out test set,
and saves it to final_models/{SYMBOL}_final.pkl.

Results logged to data/retrain_results.csv for review.

Usage:
    PYTHONPATH=. python retrain_walk_forward.py
    PYTHONPATH=. python retrain_walk_forward.py --symbol HDFCBANK
    PYTHONPATH=. python retrain_walk_forward.py --workers 4   (parallel)
    PYTHONPATH=. python retrain_walk_forward.py --resume      (skip done)
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime
from typing import Optional

import joblib
import numpy as np

# ── Path bootstrap ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
TRAIN_END   = "2024-12-31"   # last day of training
TEST_START  = "2025-01-01"   # first day of test (no overlap)
FINAL_DIR   = "final_models"
RESULTS_CSV = os.path.join("data", "retrain_results.csv")
MIN_ACC     = 0.60           # minimum acceptable accuracy to save model
MIN_PREC    = 0.60           # minimum acceptable precision to save model

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
        "elapsed_s": 0, "error": "",
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

        if acc < MIN_ACC or prec < MIN_PREC:
            result.update({"status": "below_threshold",
                           "best_model": art.get("model_name",""),
                           "horizon":    art.get("horizon",""),
                           "accuracy":   round(acc,4),
                           "precision":  round(prec,4)})
            logger.warning(f"{symbol}: below threshold (acc={acc:.1%} prec={prec:.1%}) — not saved")
            return result

        # Tag with walk-forward metadata and save
        art["train_end_date"]  = TRAIN_END
        art["test_start_date"] = TEST_START
        art["retrained_at"]    = datetime.now().isoformat()

        out_path = os.path.join(FINAL_DIR, f"{symbol}_final.pkl")
        joblib.dump(art, out_path)

        result.update({
            "status":     "ok",
            "best_model": art.get("model_name", ""),
            "horizon":    art.get("horizon", ""),
            "accuracy":   round(acc, 4),
            "precision":  round(prec, 4),
            "recall":     round(m.get("recall", 0.0), 4),
            "f1":         round(m.get("f1", 0.0), 4),
        })
        logger.info(f"✅ {symbol}: {art.get('model_name')} | {art.get('horizon')} "
                    f"acc={acc:.1%} prec={prec:.1%} ({elapsed}s)")
    except Exception as e:
        result["error"]   = str(e)
        result["elapsed_s"] = round(time.time() - t0, 1)
        logger.error(f"❌ {symbol}: {e}\n{traceback.format_exc()}")

    return result


def _worker(args):
    symbol = args
    return train_one(symbol)


# ── Main ───────────────────────────────────────────────────────────────────────

def retrain_all(symbol_filter: Optional[str] = None,
                workers: int = 1,
                resume: bool = True):
    from database.db import get_connection, release_connection, _execute

    # Load all symbols from DB
    conn = get_connection()
    cur  = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d' ORDER BY symbol")
    all_symbols = [r[0] for r in cur.fetchall()]
    release_connection(conn)

    if symbol_filter:
        all_symbols = [s for s in all_symbols if s == symbol_filter]
        if not all_symbols:
            logger.error(f"Symbol {symbol_filter} not found in DB")
            sys.exit(1)

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
    csv_fields = ["symbol","status","best_model","horizon",
                  "accuracy","precision","recall","f1",
                  "train_rows","test_rows","elapsed_s","error"]
    write_header = not os.path.exists(RESULTS_CSV)
    csv_fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=csv_fields)
    if write_header:
        writer.writeheader()

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
                    writer.writerow({k: r.get(k, "") for k in csv_fields})
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

    # Summary
    logger.info(f"\n{'='*70}")
    logger.info(f"📊 Retraining Complete")
    logger.info(f"   ✅ Saved  : {ok}")
    logger.info(f"   ⚠️  Below threshold / no data / timeout: {len(failed)}")
    if failed:
        logger.info(f"   Failed: {', '.join(failed[:20])}")
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
