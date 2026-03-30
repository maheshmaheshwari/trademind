"""
Nifty 500 AI — Automated Scheduler

Uses APScheduler to run data collection, indicator calculation,
news fetching, and signal generation at scheduled intervals.

Schedule:
    DAILY (4:00 PM IST, after market close):
        - End-of-day price collection
        - Technical indicator recalculation
        - News collection + sentiment scoring
        - FII/DII data collection
        - Signal generation
        - CSV backup

    HOURLY (9 AM - 4 PM IST, weekdays):
        - Intraday data collection (Nifty 50)
        - News sentiment refresh
        - Market overview update

    WEEKLY (Sunday 8 PM IST):
        - Cleanup old intraday data (>30 days)
        - Data integrity check

Usage:
    from scheduler.jobs import start_scheduler
    start_scheduler()  # Blocks and runs forever
"""

import logging
from datetime import datetime

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def collect_eod_data_job():
    """Daily job: collect end-of-day prices for all stocks via Angel One (smart date detection)."""
    logger.info("⏰ Running EOD data collection (Angel One)...")
    try:
        from update_stocks_angel import main as run_eod
        import sys
        sys.argv = ["update_stocks_angel.py", "--days", "2"]
        run_eod()
        logger.info("EOD collection done")
    except Exception as e:
        logger.error(f"EOD collection failed: {e}")


def calculate_indicators_job():
    """Daily job: recalculate all technical indicators."""
    logger.info("⏰ Running indicator calculation...")
    try:
        from analysis.signals import process_all_stocks
        result = process_all_stocks()
        logger.info(f"Indicators done: processed {result['processed']} stocks")
    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")


def collect_news_job():
    """Daily/Hourly job: fetch and score latest news."""
    logger.info("⏰ Running news collection...")
    try:
        from collectors.news_collector import collect_all_news
        from analysis.sentiment import score_and_update_news

        result = collect_all_news(save_to_db=True)
        logger.info(f"News collection done: {result['total']} articles")

        # Score sentiment for new articles
        if result.get("articles"):
            scored = score_and_update_news(result["articles"])
            logger.info(f"Sentiment scored for {len(scored)} articles")
    except Exception as e:
        logger.error(f"News collection failed: {e}")


def collect_fii_data_job():
    """Daily job: scrape FII/DII flow data from NSE."""
    logger.info("⏰ Running FII/DII collection...")
    try:
        from collectors.fii_collector import collect_fii_dii_data
        from database.db import insert_market_overview

        data = collect_fii_dii_data()
        if data:
            insert_market_overview({
                "date": data["date"],
                "fii_net": data["fii_net"],
                "dii_net": data["dii_net"],
            })
            logger.info(f"FII/DII data stored: FII={data['fii_net']}Cr, DII={data['dii_net']}Cr")
    except Exception as e:
        logger.error(f"FII/DII collection failed: {e}")


def collect_index_data_job():
    """Daily job: collect index data (Nifty 50, 500, Sensex, VIX)."""
    logger.info("⏰ Running index data collection...")
    try:
        from collectors.price_collector import collect_index_data
        result = collect_index_data(period="5d")
        logger.info(f"Index data done: {result}")
    except Exception as e:
        logger.error(f"Index data collection failed: {e}")


def cleanup_old_data_job():
    """Weekly job: remove intraday data older than 30 days."""
    logger.info("⏰ Running data cleanup...")
    try:
        from database.db import get_connection
        conn = get_connection()
        try:
            cursor = conn.execute(
                """DELETE FROM prices
                WHERE interval != '1d'
                AND date < date('now', '-30 days')"""
            )
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleanup: deleted {deleted} old intraday rows")
        finally:
            conn.close()
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")


def verify_data_integrity_job():
    """Weekly job: check for data gaps and report issues."""
    logger.info("⏰ Running data integrity check...")
    try:
        from database.db import get_db_stats
        stats = get_db_stats()
        logger.info(f"Data integrity — DB stats: {stats}")

        # Check for potentially missing data
        if stats.get("prices", 0) == 0:
            logger.warning("⚠️ No price data in database!")
        if stats.get("technical_indicators", 0) == 0:
            logger.warning("⚠️ No indicators calculated!")
    except Exception as e:
        logger.error(f"Data integrity check failed: {e}")


def generate_trade_signals_job():
    """Daily job: generate AI trade signals from trained models."""
    logger.info("⏰ Running trade signal generation...")
    try:
        from generate_trades import generate_signals
        result = generate_signals()
        logger.info(f"Trade signals generated: {result}")
    except Exception as e:
        logger.error(f"Trade signal generation failed: {e}")


def sync_to_turso_job():
    """Daily EOD job: sync local trade_signals to Turso cloud."""
    logger.info("⏰ Running EOD Turso sync...")
    try:
        from database.db import sync_trade_signals_to_turso
        count = sync_trade_signals_to_turso()
        logger.info(f"EOD sync done: {count} trade signals pushed to Turso")
    except Exception as e:
        logger.error(f"EOD Turso sync failed: {e}")


def intraday_price_fetch_job():
    """Every 30 min: fetch intraday 30-min candle data for stocks with open positions."""
    logger.info("⏰ Fetching intraday 30-min candle data...")
    try:
        from collectors.ltp_fetcher import fetch_intraday_30min
        count = fetch_intraday_30min()
        logger.info(f"Intraday fetch done: {count} candles saved")
    except Exception as e:
        logger.error(f"Intraday price fetch failed: {e}")


def price_monitor_job():
    """Every 5 min: check SL/Target triggers using live LTP from Angel One."""
    logger.info("⏰ Running price monitor (SL/Target check)...")
    try:
        from trading.price_monitor import run_monitor
        triggered = run_monitor()
        if triggered:
            logger.warning(f"⚡ {len(triggered)} positions auto-closed!")
    except Exception as e:
        logger.error(f"Price monitor failed: {e}")


def collect_index_data_eod_job():
    """Daily job: collect NIFTY50, NIFTY500, SENSEX, INDIA VIX via Angel One."""
    logger.info("⏰ Running index + market overview collection...")
    try:
        from collectors.index_collector import collect_index_daily
        collect_index_daily()
        logger.info("Index data collection done")
    except Exception as e:
        logger.error(f"Index data collection failed: {e}")


def collect_intraday_30m_job():
    """Every 30 min during market hours: fetch 30-min candles for open positions."""
    logger.info("⏰ Running intraday 30-min collection...")
    try:
        from collectors.intraday_collector import collect_intraday
        count = collect_intraday(interval="THIRTY_MINUTE")
        logger.info(f"Intraday 30m done: {count} candles saved")
    except Exception as e:
        logger.error(f"Intraday 30m collection failed: {e}")


def collect_av_news_job():
    """Daily job: fetch Alpha Vantage news sentiment (25 stocks/day free tier)."""
    logger.info("⏰ Running Alpha Vantage news collection...")
    try:
        from collectors.alphavantage_collector import collect_av_batch
        result = collect_av_batch()
        logger.info(f"Alpha Vantage news done: {result}")
    except Exception as e:
        logger.error(f"Alpha Vantage news collection failed: {e}")


def score_pending_news_job():
    """Hourly job: run FinBERT on any unscored news articles."""
    logger.info("⏰ Scoring pending news with FinBERT...")
    try:
        from collectors.gdelt_collector import score_pending_news
        count = score_pending_news(batch_limit=500)
        logger.info(f"FinBERT scoring done: {count} articles scored")
    except Exception as e:
        logger.error(f"News scoring failed: {e}")


def sync_gtt_status_job():
    """Every 5 min: sync GTT rule statuses from Angel One to local DB."""
    logger.info("⏰ Syncing GTT statuses from Angel One...")
    try:
        from trading.gtt_manager import sync_gtt_statuses
        triggered = sync_gtt_statuses()
        if triggered:
            for t in triggered:
                logger.warning(f"⚡ GTT triggered: {t['symbol']} {t.get('trigger')} → P&L: ₹{t['pnl']:+,.2f}")
        else:
            logger.info("No GTT triggers detected")
    except Exception as e:
        logger.error(f"GTT sync failed: {e}")


def start_scheduler() -> None:
    """
    Start the APScheduler with all configured jobs.

    This blocks the main thread and runs forever.
    Press Ctrl+C to stop.

    Schedule (all times in IST = UTC+5:30):
        - Daily at 16:00 IST: EOD collection → indicators → news → signals
        - Hourly 9-16 IST weekdays: intraday + news refresh
        - Weekly Sunday 20:00 IST: cleanup + integrity check
    """
    import os
    import time as _time
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED

    # ==========================================
    # CONFIGURE FILE + CONSOLE LOGGING
    # ==========================================
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "scheduler.log")

    # Root logger config
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    root_logger.addHandler(console_handler)

    # File handler (rotates implicitly by appending)
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root_logger.addHandler(file_handler)

    logger.info("=" * 60)
    logger.info("🚀 TradeMind AI — Scheduler Starting")
    logger.info(f"📁 Log file: {log_file}")
    logger.info(f"🕐 Current time (IST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # ==========================================
    # EVENT LISTENER — log every job execution
    # ==========================================
    _job_start_times: dict = {}

    def job_listener(event):
        job_id = event.job_id
        job = scheduler.get_job(job_id)
        job_name = job.name if job else job_id

        if event.code == EVENT_JOB_EXECUTED:
            elapsed = _time.time() - _job_start_times.pop(job_id, _time.time())
            logger.info(f"✅ {job_name} completed in {elapsed:.1f}s")
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"❌ {job_name} FAILED: {event.exception}")
            logger.error(f"   Traceback: {event.traceback}")
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"⏭️  {job_name} MISSED (scheduled time passed)")

    def before_job(event):
        _job_start_times[event.job_id] = _time.time()

    from apscheduler.events import EVENT_JOB_SUBMITTED
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    scheduler.add_listener(before_job, EVENT_JOB_SUBMITTED)

    # Delegate to _add_all_jobs so blocking and background schedulers stay in sync
    _add_all_jobs(scheduler)

    # ==========================================
    # PRINT SCHEDULE TABLE WITH NEXT RUN TIMES
    # ==========================================
    logger.info("")
    logger.info("📅 Scheduled Jobs:")
    logger.info("-" * 75)
    logger.info(f"  {'Job Name':<35s} {'Schedule':<25s} {'Next Run':<15s}")
    logger.info("-" * 75)
    for job in sorted(scheduler.get_jobs(), key=lambda j: j.next_run_time or datetime.max):
        next_run = job.next_run_time.strftime("%H:%M:%S") if job.next_run_time else "N/A"
        trigger_str = str(job.trigger)[:24]
        logger.info(f"  ⏰ {job.name:<33s} {trigger_str:<25s} {next_run}")
    logger.info("-" * 75)
    logger.info(f"  Total: {len(scheduler.get_jobs())} jobs")
    logger.info("")
    logger.info("🟢 Scheduler running. Press Ctrl+C to stop.")
    logger.info("")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("⏹  Scheduler stopped by user.")
        scheduler.shutdown()


# ==========================================
# Global reference for background scheduler
# ==========================================
_bg_scheduler: BackgroundScheduler | None = None


def _add_all_jobs(scheduler):
    """Add all scheduled jobs to any scheduler instance (shared between blocking and background)."""
    # DAILY JOBS — after market close (IST)
    # 15:35 — EOD prices (smart date detection, ~17 min for 499 stocks)
    scheduler.add_job(collect_eod_data_job, CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="eod_data", name="EOD Price Collection", misfire_grace_time=3600, replace_existing=True)
    # 16:00 — Index data (NIFTY50/500, SENSEX, VIX via Angel One)
    scheduler.add_job(collect_index_data_eod_job, CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="index_data_eod", name="Index & Market Overview", misfire_grace_time=3600, replace_existing=True)
    # 16:05 — Legacy index collector (price_collector.py fallback)
    scheduler.add_job(collect_index_data_job, CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="index_data", name="Index Data Collection (legacy)", misfire_grace_time=3600, replace_existing=True)
    # 16:15 — Technical indicators
    scheduler.add_job(calculate_indicators_job, CronTrigger(hour=16, minute=15, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="indicators", name="Calculate Indicators", misfire_grace_time=3600, replace_existing=True)
    # 16:30 — RSS + NewsAPI live news
    scheduler.add_job(collect_news_job, CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="daily_news", name="Daily News Collection", misfire_grace_time=3600, replace_existing=True)
    # 16:45 — Alpha Vantage news (25 stocks/day free tier)
    scheduler.add_job(collect_av_news_job, CronTrigger(hour=16, minute=45, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="av_news", name="Alpha Vantage News", misfire_grace_time=3600, replace_existing=True)
    # 17:00 — FII/DII data
    scheduler.add_job(collect_fii_data_job, CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="fii_dii", name="FII/DII Data", misfire_grace_time=3600, replace_existing=True)
    # 17:15 — Trade signal generation
    scheduler.add_job(generate_trade_signals_job, CronTrigger(hour=17, minute=15, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="trade_signals", name="Generate Trade Signals", misfire_grace_time=3600, replace_existing=True)
    # 20:00 — Turso cloud sync
    scheduler.add_job(sync_to_turso_job, CronTrigger(hour=20, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="eod_turso_sync", name="EOD Turso Cloud Sync", misfire_grace_time=3600, replace_existing=True)

    # HOURLY JOBS — 9 AM–4 PM IST weekdays
    # Every hour: RSS news refresh
    scheduler.add_job(collect_news_job, CronTrigger(hour="9-16", minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="hourly_news", name="Hourly News Refresh", misfire_grace_time=1800, replace_existing=True)
    # Every hour: score unscored news with FinBERT
    scheduler.add_job(score_pending_news_job, CronTrigger(hour="9-20", minute=5, timezone="Asia/Kolkata"), id="score_news", name="FinBERT News Scoring", misfire_grace_time=1800, replace_existing=True)

    # INTRADAY JOBS — market hours (9:15–15:30 IST)
    # Every 30 min: 30m candles for open positions (new dedicated collector)
    scheduler.add_job(collect_intraday_30m_job, CronTrigger(hour="9-15", minute="*/30", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="intraday_30min_new", name="Intraday 30m (positions)", misfire_grace_time=600, replace_existing=True)
    # Every 30 min: legacy 30m candle fetch
    scheduler.add_job(intraday_price_fetch_job, CronTrigger(hour="9-15", minute="15,45", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="intraday_30min", name="Intraday 30-Min Candle Fetch", misfire_grace_time=600, replace_existing=True)
    # Every 5 min: SL/Target price monitor
    scheduler.add_job(price_monitor_job, CronTrigger(hour="9-15", minute="*/5", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="price_monitor", name="Price Monitor (SL/Target)", misfire_grace_time=300, replace_existing=True)
    # Every 5 min: GTT status sync
    scheduler.add_job(sync_gtt_status_job, CronTrigger(hour="9-15", minute="*/5", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="gtt_sync", name="GTT Status Sync (Angel One)", misfire_grace_time=300, replace_existing=True)

    # WEEKLY JOBS — Sunday 8 PM IST
    scheduler.add_job(cleanup_old_data_job, CronTrigger(day_of_week="sun", hour=20, minute=0, timezone="Asia/Kolkata"), id="cleanup", name="Weekly Data Cleanup", misfire_grace_time=7200, replace_existing=True)
    scheduler.add_job(verify_data_integrity_job, CronTrigger(day_of_week="sun", hour=20, minute=30, timezone="Asia/Kolkata"), id="integrity", name="Weekly Data Integrity Check", misfire_grace_time=7200, replace_existing=True)


def start_background_scheduler() -> BackgroundScheduler | None:
    """
    Start a non-blocking BackgroundScheduler inside the API server process.

    Unlike start_scheduler() (which blocks), this runs in a daemon thread
    and is safe to call from FastAPI's startup event.

    Returns the scheduler instance (or None if already running).
    """
    import os
    import time as _time
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_SUBMITTED

    global _bg_scheduler

    if _bg_scheduler and _bg_scheduler.running:
        logger.warning("Background scheduler already running — skipping")
        return _bg_scheduler

    # ==========================================
    # LOGGING SETUP
    # ==========================================
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "scheduler.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add file handler to scheduler logger
    sched_logger = logging.getLogger("scheduler")
    sched_logger.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and getattr(h, 'baseFilename', '').endswith('scheduler.log') for h in sched_logger.handlers):
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        sched_logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("🚀 TradeMind AI — Background Scheduler Starting (inside API)")
    logger.info(f"📁 Log file: {log_file}")
    logger.info(f"🕐 Current time (IST): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    _bg_scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

    # Event listeners
    _job_start_times: dict = {}

    def job_listener(event):
        job_id = event.job_id
        job = _bg_scheduler.get_job(job_id) if _bg_scheduler else None
        job_name = job.name if job else job_id

        if event.code == EVENT_JOB_EXECUTED:
            elapsed = _time.time() - _job_start_times.pop(job_id, _time.time())
            logger.info(f"✅ {job_name} completed in {elapsed:.1f}s")
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"❌ {job_name} FAILED: {event.exception}")
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"⏭️  {job_name} MISSED (scheduled time passed)")

    def before_job(event):
        _job_start_times[event.job_id] = _time.time()

    _bg_scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED)
    _bg_scheduler.add_listener(before_job, EVENT_JOB_SUBMITTED)

    # Add all jobs
    _add_all_jobs(_bg_scheduler)

    # Print schedule table
    logger.info("")
    logger.info("📅 Scheduled Jobs (Background Mode):")
    logger.info("-" * 75)
    logger.info(f"  {'Job Name':<35s} {'Schedule':<25s} {'Next Run':<15s}")
    logger.info("-" * 75)
    # Jobs don't have next_run_time until started, so start first
    _bg_scheduler.start()

    for job in sorted(_bg_scheduler.get_jobs(), key=lambda j: j.next_run_time or datetime.max):
        next_run = job.next_run_time.strftime("%H:%M:%S") if job.next_run_time else "N/A"
        trigger_str = str(job.trigger)[:24]
        logger.info(f"  ⏰ {job.name:<33s} {trigger_str:<25s} {next_run}")
    logger.info("-" * 75)
    logger.info(f"  Total: {len(_bg_scheduler.get_jobs())} jobs")
    logger.info("")
    logger.info("🟢 Background scheduler running in API server.")

    return _bg_scheduler


def stop_background_scheduler():
    """Stop the background scheduler gracefully."""
    global _bg_scheduler
    if _bg_scheduler and _bg_scheduler.running:
        _bg_scheduler.shutdown(wait=False)
        logger.info("⏹  Background scheduler stopped.")
        _bg_scheduler = None
