"""
TradeMind AI — Weekly Retrain Pipeline

Runs every Friday at 22:00 IST (scheduled via APScheduler).
Finishes ~04:00 IST Saturday so Monday opens with fresh model predictions.
Can also be triggered manually at any time.

Pipeline:
  1. Wait for today's EOD prices to be fully collected (polls DB)
  2. Retrain all 502 models — walk-forward, single-threaded (workers=1)
     IMPORTANT: workers must stay at 1. ProcessPoolExecutor spawns fresh
     processes that don't share the in-memory data cache, causing per-symbol
     DB round-trips that hang on the cloud Timescale connection.
  3. Regenerate trade signals from fresh models

Logs: logs/YYYY-MM-DD/weekly_retrain.log

Usage:
    PYTHONPATH=. python weekly_retrain_pipeline.py
    PYTHONPATH=. python weekly_retrain_pipeline.py --skip-wait   (skip price wait)
    PYTHONPATH=. python weekly_retrain_pipeline.py --no-resume   (retrain all, even if already done)
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

# ── Path bootstrap ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Date-stamped log file ───────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
_LOG_DATE_DIR = os.path.join("logs", datetime.now().strftime("%Y-%m-%d"))
os.makedirs(_LOG_DATE_DIR, exist_ok=True)
LOG_FILE = os.path.join(_LOG_DATE_DIR, "weekly_retrain.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────────
MIN_SYMBOLS_EXPECTED = 450   # at least this many stocks must have today's prices
WAIT_POLL_SECONDS    = 120   # check DB every 2 minutes while waiting
WAIT_MAX_HOURS       = 3     # give up waiting after 3 hours


def _latest_price_date() -> Tuple[Optional[date], int]:
    """Return (latest price date, symbol count for that date)."""
    from database.db import get_connection, _execute, release_connection
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT MAX(date) as d FROM prices WHERE interval = '1d'"
        )
        row = cur.fetchone()
        latest = row[0] if row and row[0] else None
        if latest is None:
            return None, 0
        cur2 = _execute(conn,
            "SELECT COUNT(DISTINCT symbol) as cnt FROM prices WHERE date = ? AND interval = '1d'",
            (latest,)
        )
        cnt = (cur2.fetchone() or [0])[0]
        return latest, int(cnt)
    finally:
        release_connection(conn)


def wait_for_eod_prices(skip: bool = False) -> bool:
    """
    Poll the DB until today's EOD prices are fully loaded.
    Returns True when ready, False if timed out.
    """
    if skip:
        logger.info("⏭  --skip-wait flag set — skipping price wait")
        return True

    today = date.today()
    deadline = datetime.now() + timedelta(hours=WAIT_MAX_HOURS)

    logger.info(f"⏳ Waiting for today's ({today}) EOD prices ({MIN_SYMBOLS_EXPECTED}+ symbols)…")
    while datetime.now() < deadline:
        latest, cnt = _latest_price_date()
        if latest == today and cnt >= MIN_SYMBOLS_EXPECTED:
            logger.info(f"✅ EOD prices ready: {cnt} symbols for {today}")
            return True

        remaining = int((deadline - datetime.now()).total_seconds() / 60)
        logger.info(
            f"   Latest prices: {latest} ({cnt} symbols) — "
            f"need {today} with {MIN_SYMBOLS_EXPECTED}+. "
            f"Retrying in {WAIT_POLL_SECONDS}s (timeout in {remaining}m)…"
        )
        time.sleep(WAIT_POLL_SECONDS)

    logger.error(
        f"❌ Timed out after {WAIT_MAX_HOURS}h waiting for today's EOD prices. "
        "Run collect_eod_data_job() manually first."
    )
    return False


def run_retrain(workers: int, resume: bool) -> bool:
    """Retrain all 499 models. Returns True on success."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("🤖 STEP 2 — Model Retraining")
    logger.info(f"   Workers: {workers}  |  Resume: {resume}")
    logger.info("=" * 70)
    t0 = time.time()
    try:
        from retrain_walk_forward import retrain_all
        retrain_all(workers=workers, resume=resume)
        elapsed = (time.time() - t0) / 3600
        logger.info(f"✅ Retraining complete in {elapsed:.1f}h")
        return True
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)
        return False


def run_generate_signals() -> bool:
    """Generate fresh trade signals from the newly trained models. Returns True on success."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("📡 STEP 3 — Trade Signal Generation")
    logger.info("=" * 70)
    t0 = time.time()
    try:
        from generate_trades import generate_signals
        generate_signals()
        elapsed = round(time.time() - t0)
        logger.info(f"✅ Signals generated in {elapsed}s")
        return True
    except Exception as e:
        logger.error(f"❌ Signal generation failed: {e}", exc_info=True)
        return False


def run_pipeline(workers: int = 1, resume: bool = False, skip_wait: bool = False):
    start = datetime.now()
    logger.info("")
    logger.info("=" * 70)
    logger.info("🚀 TradeMind AI — Weekly Retrain Pipeline")
    logger.info(f"   Started : {start.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Log     : {LOG_FILE}")
    logger.info("=" * 70)

    # ── Step 1: Wait for EOD prices ─────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("📥 STEP 1 — Wait for EOD Prices")
    logger.info("=" * 70)
    if not wait_for_eod_prices(skip=skip_wait):
        logger.error("Pipeline aborted — EOD prices not ready.")
        sys.exit(1)

    # ── Step 2: Retrain models ───────────────────────────────────────────────────
    retrain_ok = run_retrain(workers=workers, resume=resume)
    if not retrain_ok:
        logger.error("Pipeline aborted — retraining failed.")
        sys.exit(1)

    # ── Step 3: Generate signals ─────────────────────────────────────────────────
    signals_ok = run_generate_signals()

    # ── Summary ──────────────────────────────────────────────────────────────────
    elapsed_total = (datetime.now() - start).total_seconds() / 3600
    logger.info("")
    logger.info("=" * 70)
    logger.info("📊 Pipeline Summary")
    logger.info(f"   EOD prices  : ✅")
    logger.info(f"   Retraining  : {'✅' if retrain_ok  else '❌'}")
    logger.info(f"   Signals     : {'✅' if signals_ok  else '❌'}")
    logger.info(f"   Total time  : {elapsed_total:.1f}h")
    logger.info(f"   Log file    : {LOG_FILE}")
    logger.info("=" * 70)

    if not signals_ok:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly retrain + signal generation pipeline")
    parser.add_argument("--workers",    type=int,  default=1,
                        help="Training workers (default 1 — must stay 1 for cache to work)")
    parser.add_argument("--skip-wait",  action="store_true", default=False,
                        help="Skip waiting for EOD prices (use existing latest data)")
    parser.add_argument("--no-resume",  action="store_true", default=False,
                        help="Retrain all stocks even if already done today")
    args = parser.parse_args()

    run_pipeline(
        workers=args.workers,
        resume=not args.no_resume,
        skip_wait=args.skip_wait or True,  # Friday night — EOD prices already collected
    )
