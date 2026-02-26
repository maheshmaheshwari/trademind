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
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)


def collect_eod_data_job():
    """Daily job: collect end-of-day prices for all stocks."""
    logger.info("⏰ Running EOD data collection...")
    try:
        from collectors.price_collector import collect_eod_data
        result = collect_eod_data()
        logger.info(f"EOD collection done: {result}")
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
    scheduler = BlockingScheduler(timezone="Asia/Kolkata")

    # ==========================================
    # DAILY JOBS — 4:00 PM IST (after market close)
    # ==========================================
    scheduler.add_job(
        collect_eod_data_job,
        CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="eod_data",
        name="EOD Price Collection",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        collect_index_data_job,
        CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="index_data",
        name="Index Data Collection",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        calculate_indicators_job,
        CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="indicators",
        name="Calculate Indicators",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        collect_news_job,
        CronTrigger(hour=16, minute=45, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="daily_news",
        name="Daily News Collection",
        misfire_grace_time=3600,
    )

    scheduler.add_job(
        collect_fii_data_job,
        CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="fii_dii",
        name="FII/DII Data",
        misfire_grace_time=3600,
    )

    # ==========================================
    # HOURLY JOBS — 9 AM to 4 PM IST, weekdays
    # ==========================================
    scheduler.add_job(
        collect_news_job,
        CronTrigger(hour="9-16", minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="hourly_news",
        name="Hourly News Refresh",
        misfire_grace_time=1800,
    )

    # ==========================================
    # WEEKLY JOBS — Sunday 8 PM IST
    # ==========================================
    scheduler.add_job(
        cleanup_old_data_job,
        CronTrigger(day_of_week="sun", hour=20, minute=0, timezone="Asia/Kolkata"),
        id="cleanup",
        name="Weekly Data Cleanup",
        misfire_grace_time=7200,
    )

    scheduler.add_job(
        verify_data_integrity_job,
        CronTrigger(day_of_week="sun", hour=20, minute=30, timezone="Asia/Kolkata"),
        id="integrity",
        name="Weekly Data Integrity Check",
        misfire_grace_time=7200,
    )

    # ==========================================
    # TRADE SIGNAL JOBS
    # ==========================================
    scheduler.add_job(
        generate_trade_signals_job,
        CronTrigger(hour=17, minute=15, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="trade_signals",
        name="Generate Trade Signals",
        misfire_grace_time=3600,
    )

    # ==========================================
    # EOD TURSO SYNC
    # ==========================================
    scheduler.add_job(
        sync_to_turso_job,
        CronTrigger(hour=20, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="eod_turso_sync",
        name="EOD Turso Cloud Sync",
        misfire_grace_time=3600,
    )

    # Print schedule
    print("\n📅 Scheduler Jobs:")
    print("=" * 60)
    for job in scheduler.get_jobs():
        print(f"  ⏰ {job.name:30s} — {job.trigger}")
    print("=" * 60)
    print("\n🚀 Scheduler started. Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n⏹  Scheduler stopped.")
        scheduler.shutdown()
