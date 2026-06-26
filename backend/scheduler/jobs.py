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
import pytz
from datetime import datetime, timedelta
from typing import Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# ──────────────────────────────────────────────────────────────────────────────
# Scheduler log helpers — write job run state to DB
# ──────────────────────────────────────────────────────────────────────────────

def _scheduler_log_write(
    job_id: str, job_name: str, scheduled_at,
    status: str, attempt: int = 0,
    error_msg: str = None,
    started_at=None, completed_at=None,
):
    """Upsert a scheduler_log row. Silently swallows errors so it never breaks a job."""
    try:
        from database.db import get_connection, release_connection, _execute
        conn = get_connection()
        try:
            _execute(conn, """
                INSERT INTO scheduler_log
                    (job_id, job_name, scheduled_at, status, attempt, error_msg, started_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (job_id, scheduled_at) DO UPDATE SET
                    status       = EXCLUDED.status,
                    attempt      = EXCLUDED.attempt,
                    error_msg    = EXCLUDED.error_msg,
                    started_at   = COALESCE(EXCLUDED.started_at,   scheduler_log.started_at),
                    completed_at = COALESCE(EXCLUDED.completed_at, scheduler_log.completed_at)
            """, (job_id, job_name, scheduled_at, status, attempt,
                  error_msg, started_at, completed_at))
            conn.commit()
        finally:
            release_connection(conn)
    except Exception as exc:
        logger.debug("scheduler_log write skipped: %s", exc)

# ──────────────────────────────────────────────────────────────────────────────
# Recovery queue — run missed / failed jobs on startup (FIFO, single worker)
# ──────────────────────────────────────────────────────────────────────────────

# Jobs eligible for recovery. High-frequency intraday jobs are excluded —
# recovering stale 5-min SL checks or 30-min candles is pointless.
# Populated after the job functions are defined (see bottom of file).
RECOVERABLE_JOBS: dict = {}
# job_id → (job_name, cron_hour, cron_minute, fn, day_of_week, lookback_hours)
# day_of_week: "mon-fri" for weekday-only jobs, or a specific day e.g. "fri" for weekly jobs
# lookback_hours: how far back to search for a missed fire (24h for daily, 80h for weekly)


_DAY_NAME_TO_WEEKDAY = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3,
    "fri": 4, "sat": 5, "sun": 6,
}


def run_recovery_queue():
    """
    Called 30s after server startup. Finds jobs that should have fired within
    their lookback window but have no 'done' record, then runs them FIFO.

    Each RECOVERABLE_JOBS entry specifies its own lookback_hours:
      - Daily weekday jobs: 24h  (check today + yesterday)
      - Weekly Friday retrain:  80h  (covers full weekend downtime:
        Friday 22:00 → Monday 06:00 = 56h, with 24h margin)

    Retry logic:
        attempt 1 fails → status='failed', retried on next startup
        attempt 2 fails → status='failed'
        attempt 3 fails → status='permanently_failed', skipped forever

    Stale 'running' entries (started >30 min ago) are reset to 'failed'
    so an interrupted job doesn't block its own retry.
    """
    if not RECOVERABLE_JOBS:
        return

    from database.db import get_connection, release_connection, _execute

    now_ist = datetime.now(IST)

    # Reset stale 'running' entries
    try:
        conn = get_connection()
        try:
            _execute(conn, """
                UPDATE scheduler_log SET status = 'failed'
                WHERE status = 'running'
                  AND started_at < ?
            """, (now_ist - timedelta(minutes=30),))
            conn.commit()
        finally:
            release_connection(conn)
    except Exception as exc:
        logger.debug("Stale running reset skipped: %s", exc)

    recovery_tasks = []

    try:
        conn = get_connection()
        try:
            for job_id, entry in RECOVERABLE_JOBS.items():
                job_name, cron_hour, cron_minute, fn = entry[0], entry[1], entry[2], entry[3]
                day_of_week  = entry[4] if len(entry) > 4 else "mon-fri"
                lookback_hrs = entry[5] if len(entry) > 5 else 24

                lookback = now_ist - timedelta(hours=lookback_hrs)

                # Resolve allowed weekdays from day_of_week spec
                if "-" in day_of_week:
                    start_d, end_d = day_of_week.split("-")
                    allowed_weekdays = set(range(
                        _DAY_NAME_TO_WEEKDAY[start_d],
                        _DAY_NAME_TO_WEEKDAY[end_d] + 1,
                    ))
                else:
                    allowed_weekdays = {_DAY_NAME_TO_WEEKDAY[day_of_week]}

                # Walk back far enough to cover the full lookback window
                max_days_back = lookback_hrs // 24 + 2
                for delta_days in range(int(max_days_back) + 1):
                    candidate_date = (now_ist - timedelta(days=delta_days)).date()
                    if candidate_date.weekday() not in allowed_weekdays:
                        continue
                    scheduled = IST.localize(datetime(
                        candidate_date.year, candidate_date.month, candidate_date.day,
                        cron_hour, cron_minute,
                    ))
                    if scheduled > now_ist or scheduled < lookback:
                        continue

                    # Check scheduler_log for this fire time (±10 min before, +8h after for long jobs)
                    row = _execute(conn, """
                        SELECT status, attempt FROM scheduler_log
                        WHERE job_id = ?
                          AND scheduled_at BETWEEN ? AND ?
                        ORDER BY scheduled_at DESC LIMIT 1
                    """, (job_id,
                          scheduled - timedelta(minutes=10),
                          scheduled + timedelta(hours=8))).fetchone()

                    if row and row[0] in ('done', 'running', 'permanently_failed'):
                        continue
                    attempt = int(row[1]) if row else 0
                    if attempt >= 3:
                        continue
                    recovery_tasks.append((job_id, job_name, scheduled, fn, attempt))
        finally:
            release_connection(conn)
    except Exception as exc:
        logger.error("Recovery queue DB read failed: %s", exc)
        return

    if not recovery_tasks:
        logger.info("🔄 Recovery queue: no missed jobs (all caught up)")
        return

    recovery_tasks.sort(key=lambda x: x[2])   # FIFO by scheduled_at
    logger.info("🔄 Recovery queue: %d missed job(s) to run", len(recovery_tasks))

    for job_id, job_name, scheduled_at, fn, attempt in recovery_tasks:
        new_attempt = attempt + 1
        logger.info("  ▶ Recovering [%d/3]: %s (was due %s IST)",
                    new_attempt, job_name, scheduled_at.strftime("%Y-%m-%d %H:%M"))
        _scheduler_log_write(job_id, job_name, scheduled_at, "running",
                             attempt=new_attempt, started_at=datetime.now(IST))
        try:
            fn()
            _scheduler_log_write(job_id, job_name, scheduled_at, "done",
                                 attempt=new_attempt, completed_at=datetime.now(IST))
            logger.info("  ✅ Recovered: %s", job_name)
        except Exception as exc:
            status = "permanently_failed" if new_attempt >= 3 else "failed"
            _scheduler_log_write(job_id, job_name, scheduled_at, status,
                                 attempt=new_attempt, error_msg=str(exc))
            logger.error("  ❌ Recovery failed [%d/3]: %s — %s", new_attempt, job_name, exc)
            if status == "permanently_failed":
                logger.error("  🚫 %s permanently failed — manual run required", job_name)


def collect_eod_data_job():
    """
    Daily job: EOD prices → indicators → trade signals (chained in order).
    Guarantees each step only starts after the previous one completes,
    regardless of how long each step takes.
    """
    # ── Step 1: EOD prices ───────────────────────────────────────────────────
    logger.info("⏰ [1/3] EOD price collection starting...")
    try:
        from scripts.update_stocks_angel import main as run_eod
        run_eod(days=2)
        logger.info("✅ [1/3] EOD prices done")
    except BaseException as e:
        logger.error(f"❌ [1/3] EOD collection failed: {e} — aborting chain")
        raise RuntimeError(f"EOD price collection failed: {e}") from e

    # ── Step 2: Technical indicators ─────────────────────────────────────────
    logger.info("⏰ [2/3] Technical indicators starting...")
    try:
        from analysis.signals import process_all_stocks
        result = process_all_stocks()
        logger.info(f"✅ [2/3] Indicators done ({result['processed']} stocks)")
    except BaseException as e:
        logger.error(f"❌ [2/3] Indicators failed: {e} — aborting chain")
        raise RuntimeError(f"Indicators failed: {e}") from e

    # ── Step 3: Trade signal generation ──────────────────────────────────────
    logger.info("⏰ [3/3] Trade signal generation starting...")
    try:
        from scripts.generate_trades import generate_signals
        generate_signals()
        logger.info("✅ [3/3] Trade signals done — EOD pipeline complete")
    except BaseException as e:
        logger.error(f"❌ [3/3] Trade signal generation failed: {e}")
        raise RuntimeError(f"Signal generation failed: {e}") from e


def collect_yfinance_news_job():
    """Daily job: fetch ~10 recent articles per stock from yfinance."""
    logger.info("⏰ Running yfinance news collection...")
    try:
        from collectors.yfinance_news_collector import collect_all
        result = collect_all()
        logger.info(f"yfinance news done: {result['total']} new articles")
    except Exception as e:
        logger.error(f"yfinance news collection failed: {e}")


def collect_nse_announcements_job():
    """Daily job: fetch last 2 days of NSE corporate announcements for all 499 stocks."""
    logger.info("⏰ Running NSE corporate announcements collection...")
    try:
        from collectors.nse_announcements_collector import collect_daily
        result = collect_daily(lookback_days=2)
        logger.info(f"NSE announcements done: {result['total_rows']} rows, {result['processed']} stocks")
    except Exception as e:
        logger.error(f"NSE announcements collection failed: {e}")


def collect_corporate_actions_job():
    """Daily EOD job: scrape Trendlyne for new Nifty 500 bonus/split events."""
    logger.info("Running Trendlyne corporate actions collection...")
    try:
        from collectors.trendlyne_collector import collect_corporate_actions
        result = collect_corporate_actions(lookback_days=7)
        logger.info(
            f"Corporate actions done: {result['scraped']} scraped, "
            f"{result['new']} new, {result['skipped']} already in DB"
        )
        if result["new"]:
            logger.info(f"NEW corporate action(s) detected — {result['new']} event(s) added")
    except Exception as e:
        logger.error(f"Corporate actions collection failed: {e}")


def collect_delivery_job():
    """Daily job: fetch NSE delivery % bhavcopy after market close."""
    logger.info("⏰ Running NSE delivery % collection...")
    try:
        from collectors.delivery_collector import collect_today
        n = collect_today()
        logger.info(f"Delivery % done: {n} records stored")
    except Exception as e:
        logger.error(f"Delivery collection failed: {e}")


def collect_rss_news_job():
    """Daily job: scrape market-wide news from ET, Moneycontrol, Business Standard RSS."""
    logger.info("⏰ Running RSS news collection...")
    try:
        from collectors.rss_collector import collect_all_rss
        result = collect_all_rss()
        logger.info(f"RSS news done: {result['total']} new articles")
    except Exception as e:
        logger.error(f"RSS news collection failed: {e}")


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
    """
    Step 1 — 17:00 IST: Fetch NSE live cash-market FII/DII (net Cr) and
    store in both fii_dii_daily and market_overview.

    NSE publishes cash-market FII/DII data via their live API shortly after
    market close. This gives the net buy/sell in Crores.
    """
    logger.info("⏰ [Step 1] FII/DII live cash-market collection…")
    try:
        from collectors.fii_collector import collect_fii_dii_data
        from database.db import insert_market_overview

        data = collect_fii_dii_data()   # writes to fii_dii_daily + returns dict
        if data:
            insert_market_overview({
                "date":    data["date"],
                "fii_net": data["fii_net"],
                "dii_net": data["dii_net"],
            })
            logger.info(
                "FII/DII cash-market stored: date=%s  FII=%.2f Cr  DII=%.2f Cr",
                data["date"], data["fii_net"], data["dii_net"],
            )
        else:
            logger.warning("FII/DII live API returned no data")
    except Exception as e:
        logger.error("FII/DII live collection failed: %s", e, exc_info=True)


def collect_fii_fo_job():
    """
    Step 2 — 17:30 IST: Fetch NSE F&O participant volume data for today and
    upsert into fii_dii_daily (adds buy/sell breakdown + F&O net contracts).

    NSE archives publish fao_participant_vol_DDMMYYYY.csv ~17:15–17:30 IST.
    If today's file isn't ready yet this job retries at 18:00 via the misfire
    window and is safe to call multiple times (upsert).
    """
    logger.info("⏰ [Step 2] FII/DII F&O participant volume collection…")
    try:
        import requests
        from datetime import date
        from database.db import get_connection, release_connection, _execute

        today = date.today()
        ds = today.strftime("%d%m%Y")
        url = f"https://archives.nseindia.com/content/nsccl/fao_participant_vol_{ds}.csv"

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer":    "https://www.nseindia.com/",
        })
        session.get("https://www.nseindia.com/", timeout=10)

        r = session.get(url, timeout=15)
        if r.status_code != 200:
            logger.warning("F&O participant file not ready yet: %s → %s", ds, r.status_code)
            return

        import csv, io
        lines = r.text.strip().split("\n")
        reader = csv.reader(lines[1:])
        headers = [h.strip().lower() for h in next(reader)]
        long_idx  = next(i for i, h in enumerate(headers) if "total long"  in h)
        short_idx = next(i for i, h in enumerate(headers) if "total short" in h)

        fii_buy = fii_sell = dii_buy = dii_sell = 0.0
        for row in reader:
            if not row: continue
            client = row[0].strip().upper()
            if "FII" in client or "FPI" in client:
                fii_buy  = float(row[long_idx].strip().replace(",", ""))
                fii_sell = float(row[short_idx].strip().replace(",", ""))
            elif "DII" in client:
                dii_buy  = float(row[long_idx].strip().replace(",", ""))
                dii_sell = float(row[short_idx].strip().replace(",", ""))

        fii_net = round(fii_buy  - fii_sell, 2)
        dii_net = round(dii_buy  - dii_sell, 2)

        conn = get_connection()
        try:
            _execute(conn, """
                INSERT INTO fii_dii_daily
                    (date, fii_net, dii_net, fii_buy, fii_sell, dii_buy, dii_sell, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (date) DO UPDATE SET
                    fii_buy  = EXCLUDED.fii_buy,
                    fii_sell = EXCLUDED.fii_sell,
                    dii_buy  = EXCLUDED.dii_buy,
                    dii_sell = EXCLUDED.dii_sell,
                    source   = CASE
                        WHEN fii_dii_daily.source = 'nse_live' THEN 'nse_live+fo'
                        ELSE EXCLUDED.source
                    END
            """, (
                str(today),
                fii_net, dii_net,
                fii_buy, fii_sell,
                dii_buy, dii_sell,
                "nse_fo_vol",
            ))
            conn.commit()
        finally:
            release_connection(conn)

        logger.info(
            "FII/DII F&O stored: date=%s  FII_net=%+.0f  DII_net=%+.0f (contracts)",
            today, fii_net, dii_net,
        )
    except Exception as e:
        logger.error("FII/DII F&O collection failed: %s", e, exc_info=True)


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
        from database.db import get_connection, _execute
        conn = get_connection()
        try:
            cursor = _execute(conn,
                "DELETE FROM prices WHERE interval != '1d' AND date < NOW() - INTERVAL '30 days'"
            )
            deleted = cursor.rowcount
            conn.commit()
            logger.info(f"Cleanup: deleted {deleted} old intraday rows")
        finally:
            release_connection(conn)
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


def generate_trade_signals_job(force: bool = False):
    """
    Daily job: generate AI trade signals from trained models.

    Only runs after market close (15:35 IST) to ensure signals are based on
    final EOD prices, not incomplete intraday candles. Pass force=True to
    override this guard (e.g. for manual backfill runs).
    """
    now_ist = datetime.now(IST)
    market_close_ist = now_ist.replace(hour=15, minute=35, second=0, microsecond=0)
    if not force and now_ist < market_close_ist:
        logger.warning(
            "⚠️  Signal generation skipped — market still open (%s IST). "
            "Signals are only generated after 15:35 IST using final EOD prices. "
            "Use force=True to override.",
            now_ist.strftime("%H:%M"),
        )
        return

    logger.info("⏰ Running trade signal generation...")
    try:
        from scripts.generate_trades import generate_signals
        result = generate_signals()
        logger.info(f"Trade signals generated: {result}")
    except Exception as e:
        logger.error(f"Trade signal generation failed: {e}")




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
    """Hourly job: run FinBERT batch inference on any unscored news articles."""
    logger.info("⏰ Scoring pending news with FinBERT...")
    try:
        from collectors.gdelt_collector import score_pending_news
        count = score_pending_news(batch_limit=2000)
        logger.info(f"FinBERT scoring done: {count} articles scored")
    except Exception as e:
        logger.error(f"News scoring failed: {e}")


def score_pending_news_nightly_job():
    """Nightly high-capacity scoring job: clear backlog with a large batch."""
    logger.info("⏰ Nightly FinBERT scoring (high-capacity)...")
    try:
        from collectors.gdelt_collector import score_pending_news
        count = score_pending_news(batch_limit=5000)
        logger.info(f"Nightly FinBERT scoring done: {count} articles scored")
    except Exception as e:
        logger.error(f"Nightly news scoring failed: {e}")


def notify_signal_changes_job():
    """Post-EOD: create notifications for watchlist stocks whose signal changed today."""
    logger.info("⏰ Checking for signal changes on watchlisted stocks...")
    try:
        from database.db import get_connection, _rows_to_dicts, _execute, insert_notification, release_connection
        conn = get_connection()
        try:
            cur = _execute(conn, """
                SELECT DISTINCT ON (w.user_id, s.symbol)
                    w.user_id,
                    s.symbol,
                    s.signal      AS new_signal,
                    s.confidence,
                    LAG(s.signal) OVER (PARTITION BY s.symbol ORDER BY s.generated_at) AS prev_signal
                FROM watchlist w
                JOIN trade_signals s ON s.symbol = w.symbol
                WHERE s.generated_at >= NOW() - INTERVAL '2 days'
                ORDER BY w.user_id, s.symbol, s.generated_at DESC
            """)
            rows = _rows_to_dicts(cur)
        finally:
            release_connection(conn)
        fired = 0
        for row in rows:
            if row.get("prev_signal") and row["new_signal"] != row["prev_signal"]:
                conf = row.get("confidence") or 0
                insert_notification(
                    user_id=row["user_id"],
                    type="signal",
                    title=f"{row['symbol']} signal changed",
                    message=f"{row['prev_signal']} → {row['new_signal']} ({conf:.0%} confidence)",
                    icon="TrendingUp",
                    color="#3B82F6",
                )
                fired += 1
        logger.info(f"Signal change notifications fired: {fired}")
    except Exception as e:
        logger.error(f"Signal change notifier failed: {e}")


def price_alert_job():
    """Hourly: check watchlist price alerts and fire notifications when thresholds are crossed."""
    logger.info("⏰ Checking price alerts...")
    try:
        from database.db import get_connection, _rows_to_dicts, _execute, get_latest_indicators, insert_notification, release_connection
        conn = get_connection()
        try:
            cur = _execute(conn,
                "SELECT user_id, symbol, alert_above, alert_below FROM watchlist "
                "WHERE alert_above IS NOT NULL OR alert_below IS NOT NULL"
            )
            alerts = _rows_to_dicts(cur)
        finally:
            release_connection(conn)

        fired = 0
        for alert in alerts:
            ind = get_latest_indicators(alert["symbol"])
            if not ind:
                continue
            price = ind.get("close") or ind.get("ltp")
            if not price:
                continue

            # Cooldown: skip if a price notification for this user+symbol was sent in the last 24h
            conn2 = get_connection()
            try:
                recent = _execute(conn2,
                    """SELECT 1 FROM notifications
                       WHERE user_id = ? AND type = 'price'
                         AND title LIKE ?
                         AND created_at >= NOW() - INTERVAL '24 hours'
                       LIMIT 1""",
                    (alert["user_id"], f"{alert['symbol']}%"),
                ).fetchone()
            finally:
                release_connection(conn2)

            if recent:
                continue

            if alert["alert_above"] and price >= alert["alert_above"]:
                insert_notification(
                    user_id=alert["user_id"], type="price",
                    title=f"{alert['symbol']} above ₹{alert['alert_above']:,.2f}",
                    message=f"Current price ₹{price:,.2f}",
                    icon="ArrowUp", color="#10B981",
                )
                fired += 1
            elif alert["alert_below"] and price <= alert["alert_below"]:
                insert_notification(
                    user_id=alert["user_id"], type="price",
                    title=f"{alert['symbol']} below ₹{alert['alert_below']:,.2f}",
                    message=f"Current price ₹{price:,.2f}",
                    icon="ArrowDown", color="#EF4444",
                )
                fired += 1
        logger.info(f"Price alert notifications fired: {fired}")
    except Exception as e:
        logger.error(f"Price alert job failed: {e}")


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


def weekly_retrain_job():
    """
    Friday 22:00 IST: retrain all 502 models with the week's latest data,
    then regenerate trade signals so Monday opens with fresh predictions.

    workers=1 is required — ProcessPoolExecutor workers start as fresh
    processes without the pre-fetched data cache, causing per-symbol DB
    queries that hang on the cloud connection. Single-threaded keeps
    everything in one process where the cache is live.

    resume=False ensures all 502 symbols are retrained, not just new ones.
    skip_wait=True skips the EOD price poll (prices already collected
    by the earlier EOD job at 15:35 IST).
    """
    logger.info("⏰ Starting Friday night model retrain pipeline...")
    try:
        from scripts.weekly_retrain_pipeline import run_pipeline
        run_pipeline(workers=1, resume=False, skip_wait=True)
        logger.info("✅ Weekly retrain pipeline complete — signals ready for Monday")
    except Exception as e:
        logger.error(f"❌ Weekly retrain pipeline failed: {e}", exc_info=True)


def sync_autopilot_job():
    """Every 5 min: check GTT statuses for EXECUTED autopilot mandates and settle them."""
    logger.info("⏰ Syncing autopilot mandate statuses...")
    try:
        from trading.gtt_manager import sync_autopilot_statuses
        settled = sync_autopilot_statuses()
        if settled:
            for s in settled:
                logger.warning(
                    f"⚡ Autopilot settled: {s['symbol']} → {s['status']} "
                    f"P&L=₹{s['actual_pnl']:+,.2f}"
                )
        else:
            logger.debug("No autopilot mandates settled this cycle")
    except Exception as e:
        logger.error(f"Autopilot sync failed: {e}")


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
        scheduled_at = getattr(event, "scheduled_run_time", datetime.now(IST))

        if event.code == EVENT_JOB_EXECUTED:
            elapsed = _time.time() - _job_start_times.pop(job_id, _time.time())
            logger.info(f"✅ {job_name} completed in {elapsed:.1f}s")
            _scheduler_log_write(job_id, job_name, scheduled_at, "done",
                                 completed_at=datetime.now(IST))
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"❌ {job_name} FAILED: {event.exception}")
            logger.error(f"   Traceback: {event.traceback}")
            _scheduler_log_write(job_id, job_name, scheduled_at, "failed",
                                 error_msg=str(event.exception))
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"⏭️  {job_name} MISSED (scheduled time passed)")
            _scheduler_log_write(job_id, job_name, scheduled_at, "pending")

    def before_job(event):
        _job_start_times[event.job_id] = _time.time()
        job = scheduler.get_job(event.job_id)
        job_name = job.name if job else event.job_id
        scheduled_at = getattr(event, "scheduled_run_time", datetime.now(IST))
        _scheduler_log_write(event.job_id, job_name, scheduled_at, "running",
                             started_at=datetime.now(IST))

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
_bg_scheduler: Optional[BackgroundScheduler] = None


def _add_all_jobs(scheduler):
    """Add all scheduled jobs to any scheduler instance (shared between blocking and background)."""
    # DAILY JOBS — after market close (IST)
    # 15:35 — EOD prices (smart date detection, ~17 min for 499 stocks)
    scheduler.add_job(collect_eod_data_job, CronTrigger(hour=15, minute=35, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="eod_data", name="EOD Price Collection", misfire_grace_time=3600, replace_existing=True)
    # 16:00 — Index data (NIFTY50/500, SENSEX, VIX via Angel One)
    scheduler.add_job(collect_index_data_eod_job, CronTrigger(hour=16, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="index_data_eod", name="Index & Market Overview", misfire_grace_time=3600, replace_existing=True)
    # 16:05 — Legacy index collector (price_collector.py fallback)
    scheduler.add_job(collect_index_data_job, CronTrigger(hour=16, minute=5, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="index_data", name="Index Data Collection (legacy)", misfire_grace_time=3600, replace_existing=True)
    # NOTE: indicators + trade signals are now chained inside collect_eod_data_job (step 2 & 3)
    # 18:00 — NSE delivery % (NSE uploads bhavcopy ~5:30 PM IST)
    scheduler.add_job(collect_delivery_job, CronTrigger(hour=18, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="delivery_data", name="NSE Delivery % Collection", misfire_grace_time=3600, replace_existing=True)
    # 18:30 — NSE corporate announcements (covers all 499 stocks, no API key required)
    scheduler.add_job(collect_nse_announcements_job, CronTrigger(hour=18, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="nse_announcements", name="NSE Corporate Announcements", misfire_grace_time=3600, replace_existing=True)
    # 18:45 — Trendlyne bonus/split oracle (7-day lookback, catches any new announcements)
    scheduler.add_job(collect_corporate_actions_job, CronTrigger(hour=18, minute=45, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="corporate_actions", name="Trendlyne Corporate Actions (Splits/Bonus)", misfire_grace_time=3600, replace_existing=True)
    # 16:30 — RSS market-wide news (ET, Moneycontrol, Business Standard)
    scheduler.add_job(collect_rss_news_job, CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="rss_news", name="RSS Market News", misfire_grace_time=3600, replace_existing=True)
    # 16:45 — yfinance per-stock news (~10 articles × 499 stocks)
    scheduler.add_job(collect_yfinance_news_job, CronTrigger(hour=16, minute=45, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="yfinance_news", name="yfinance Per-Stock News", misfire_grace_time=3600, replace_existing=True)
    # Legacy news collector (fallback)
    scheduler.add_job(collect_news_job, CronTrigger(hour=16, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="daily_news", name="Daily News Collection", misfire_grace_time=3600, replace_existing=True)
    # 16:45 — Alpha Vantage news (25 stocks/day free tier)
    scheduler.add_job(collect_av_news_job, CronTrigger(hour=16, minute=45, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="av_news", name="Alpha Vantage News", misfire_grace_time=3600, replace_existing=True)
    # 17:00 — FII/DII Step 1: live cash-market API (net Cr, available right after close)
    scheduler.add_job(
        collect_fii_data_job,
        CronTrigger(hour=17, minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="fii_dii_live", name="FII/DII Data (live cash-market)",
        misfire_grace_time=3600, replace_existing=True,
    )
    # 17:30 — FII/DII Step 2: F&O participant volume (buy/sell breakdown, NSE archives ~17:15)
    scheduler.add_job(
        collect_fii_fo_job,
        CronTrigger(hour=17, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"),
        id="fii_dii_fo", name="FII/DII F&O Participant Volume",
        misfire_grace_time=3600, replace_existing=True,
    )
    # NOTE: trade signal generation is now chained inside collect_eod_data_job (step 3)
    # 17:30 — Notify users of signal changes on watchlisted stocks
    scheduler.add_job(notify_signal_changes_job, CronTrigger(hour=17, minute=30, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="signal_notifications", name="Signal Change Notifications", misfire_grace_time=3600, replace_existing=True)
    # HOURLY JOBS — 9 AM–4 PM IST weekdays
    # Every hour: RSS news refresh
    scheduler.add_job(collect_news_job, CronTrigger(hour="9-16", minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="hourly_news", name="Hourly News Refresh", misfire_grace_time=1800, replace_existing=True)
    # Every hour: score unscored news with FinBERT (batch_limit=2000)
    scheduler.add_job(score_pending_news_job, CronTrigger(hour="9-20", minute=5, timezone="Asia/Kolkata"), id="score_news", name="FinBERT News Scoring", misfire_grace_time=1800, replace_existing=True)
    # 23:00 nightly: high-capacity backlog clearing (batch_limit=5000)
    scheduler.add_job(score_pending_news_nightly_job, CronTrigger(hour=23, minute=0, timezone="Asia/Kolkata"), id="score_news_nightly", name="FinBERT Nightly Scoring", misfire_grace_time=3600, replace_existing=True)
    # Every hour (market hours): check watchlist price alerts
    scheduler.add_job(price_alert_job, CronTrigger(hour="9-15", minute=0, day_of_week="mon-fri", timezone="Asia/Kolkata"), id="price_alerts", name="Price Alert Checker", misfire_grace_time=1800, replace_existing=True)

    # INTRADAY JOBS — market hours (9:15–15:30 IST)
    # Every 30 min: 30m candles for open positions (new dedicated collector)
    scheduler.add_job(collect_intraday_30m_job, CronTrigger(hour="9-15", minute="*/30", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="intraday_30min_new", name="Intraday 30m (positions)", misfire_grace_time=600, replace_existing=True)
    # Every 30 min: legacy 30m candle fetch
    scheduler.add_job(intraday_price_fetch_job, CronTrigger(hour="9-15", minute="15,45", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="intraday_30min", name="Intraday 30-Min Candle Fetch", misfire_grace_time=600, replace_existing=True)
    # Every 5 min: SL/Target price monitor
    scheduler.add_job(price_monitor_job, CronTrigger(hour="9-15", minute="*/5", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="price_monitor", name="Price Monitor (SL/Target)", misfire_grace_time=300, replace_existing=True)
    # Every 5 min: GTT status sync
    scheduler.add_job(sync_gtt_status_job, CronTrigger(hour="9-15", minute="*/5", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="gtt_sync", name="GTT Status Sync (Angel One)", misfire_grace_time=300, replace_existing=True)
    # Every 5 min: Autopilot mandate settlement
    scheduler.add_job(sync_autopilot_job, CronTrigger(hour="9-15", minute="*/5", day_of_week="mon-fri", timezone="Asia/Kolkata"), id="autopilot_sync", name="Autopilot Mandate Sync", misfire_grace_time=300, replace_existing=True)

    # WEEKLY JOBS — Sunday 8 PM IST
    scheduler.add_job(cleanup_old_data_job, CronTrigger(day_of_week="sun", hour=20, minute=0, timezone="Asia/Kolkata"), id="cleanup", name="Weekly Data Cleanup", misfire_grace_time=7200, replace_existing=True)
    scheduler.add_job(verify_data_integrity_job, CronTrigger(day_of_week="sun", hour=20, minute=30, timezone="Asia/Kolkata"), id="integrity", name="Weekly Data Integrity Check", misfire_grace_time=7200, replace_existing=True)
    # Friday 22:00 — Retrain all 502 models → regenerate signals so Monday has fresh predictions
    # misfire_grace_time=28800 (8h) allows a delayed start if server was down at 22:00
    # but still completes before Monday market open (~04:00 IST finish + 8h window = covered)
    # scheduler.add_job(weekly_retrain_job, CronTrigger(day_of_week="fri", hour=22, minute=0, timezone="Asia/Kolkata"), id="weekly_retrain", name="Friday Night Model Retrain + Signals", misfire_grace_time=28800, replace_existing=True)



def get_scheduler_status() -> dict:
    """
    Return scheduler state visible to ANY worker process.

    Strategy (multi-worker safe):
      1. If this process owns the scheduler, return live APScheduler state.
      2. Otherwise, check the lock file: if the owner PID is alive → running.
      3. Check scheduler.log for last activity timestamp.
    """
    import os

    global _bg_scheduler

    # If this worker owns the scheduler, return live state
    if _bg_scheduler is not None:
        jobs = []
        for job in _bg_scheduler.get_jobs():
            nrt = job.next_run_time
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": nrt.isoformat() if nrt else None,
            })
        return {
            "running": _bg_scheduler.running,
            "owner_pid": os.getpid(),
            "jobs": len(jobs),
            "job_list": jobs,
        }

    # Cross-process: inspect lock file
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    lock_path = os.path.join(log_dir, ".scheduler_owner.pid")
    owner_pid = None
    owner_alive = False
    try:
        with open(lock_path) as f:
            owner_pid = int(f.read().strip())
        os.kill(owner_pid, 0)
        owner_alive = True
    except Exception:
        pass

    # Read last log line from scheduler.log for recent-activity timestamp
    last_log = None
    sched_log = os.path.join(log_dir, "scheduler.log")
    try:
        with open(sched_log, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 512))
            tail = f.read().decode(errors="replace")
        for line in reversed(tail.splitlines()):
            if line.strip():
                last_log = line.strip()
                break
    except Exception:
        pass

    return {
        "running": owner_alive,
        "owner_pid": owner_pid,
        "this_pid": os.getpid(),
        "jobs": len(RECOVERABLE_JOBS),
        "last_log": last_log,
        "job_list": [],
    }


def start_background_scheduler() -> Optional[BackgroundScheduler]:
    """
    Start a non-blocking BackgroundScheduler inside the API server process.

    Unlike start_scheduler() (which blocks), this runs in a daemon thread
    and is safe to call from FastAPI's startup event.

    Returns the scheduler instance (or None if already running).
    """
    import os
    import time as _time
    from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR, EVENT_JOB_MISSED, EVENT_JOB_SUBMITTED
    from apscheduler.executors.pool import ThreadPoolExecutor as APSThreadPool

    global _bg_scheduler

    if _bg_scheduler and _bg_scheduler.running:
        logger.warning("Background scheduler already running — skipping")
        return _bg_scheduler

    # ==========================================
    # LOGGING SETUP — attach handler to both
    # 'scheduler' and 'scheduler.jobs' so that
    # messages reach scheduler.log even if the
    # root-logger propagation is disrupted.
    # ==========================================
    log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "scheduler.log")

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    def _add_sched_file_handler(logger_name: str) -> None:
        lg = logging.getLogger(logger_name)
        lg.setLevel(logging.INFO)
        if not any(
            isinstance(h, logging.FileHandler)
            and getattr(h, "baseFilename", "").endswith("scheduler.log")
            for h in lg.handlers
        ):
            fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            lg.addHandler(fh)

    # Add only to scheduler.jobs; it propagates to root for the daily log.
    # Do NOT also add to the parent "scheduler" logger — that doubles every line.
    _add_sched_file_handler("scheduler.jobs")

    logger.info("=" * 60)
    logger.info("🚀 TradeMind AI — Background Scheduler Starting (inside API)")
    logger.info(f"📁 Log file: {log_file}")
    logger.info(f"🕐 Current time (IST): {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # 20 executor threads — prevents exhaustion when several intraday jobs
    # fire simultaneously during market hours.
    # coalesce=True + max_instances=1 ensures a lagging job can't pile up:
    # missed fire times are merged into one catch-up run.
    _bg_scheduler = BackgroundScheduler(
        timezone="Asia/Kolkata",
        executors={"default": APSThreadPool(20)},
        job_defaults={"coalesce": True, "max_instances": 1, "misfire_grace_time": 3600},
    )

    # Event listeners
    _job_start_times: dict = {}

    def job_listener(event):
        job_id = event.job_id
        job = _bg_scheduler.get_job(job_id) if _bg_scheduler else None
        job_name = job.name if job else job_id
        scheduled_at = getattr(event, "scheduled_run_time", datetime.now(IST))

        if event.code == EVENT_JOB_EXECUTED:
            elapsed = _time.time() - _job_start_times.pop(job_id, _time.time())
            logger.info(f"✅ {job_name} completed in {elapsed:.1f}s")
            _scheduler_log_write(job_id, job_name, scheduled_at, "done",
                                 completed_at=datetime.now(IST))
        elif event.code == EVENT_JOB_ERROR:
            logger.error(f"❌ {job_name} FAILED: {event.exception}")
            _scheduler_log_write(job_id, job_name, scheduled_at, "failed",
                                 error_msg=str(event.exception))
        elif event.code == EVENT_JOB_MISSED:
            logger.warning(f"⏭️  {job_name} MISSED (scheduled time passed)")
            _scheduler_log_write(job_id, job_name, scheduled_at, "pending")

    def before_job(event):
        _job_start_times[event.job_id] = _time.time()
        job = _bg_scheduler.get_job(event.job_id) if _bg_scheduler else None
        job_name = job.name if job else event.job_id
        scheduled_at = getattr(event, "scheduled_run_time", datetime.now(IST))
        _scheduler_log_write(event.job_id, job_name, scheduled_at, "running",
                             started_at=datetime.now(IST))

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


# ──────────────────────────────────────────────────────────────────────────────
# Populate RECOVERABLE_JOBS at module level so run_recovery_queue() works even
# when called without starting the scheduler first (e.g. from server.py startup).
# ──────────────────────────────────────────────────────────────────────────────
RECOVERABLE_JOBS.update({
    # (job_name, cron_hour, cron_minute, fn, day_of_week, lookback_hours)
    # Daily weekday jobs — 24h lookback covers today + yesterday
    "eod_data":             ("EOD Price Collection",                        15, 35, collect_eod_data_job,          "mon-fri", 24),
    "index_data_eod":       ("Index & Market Overview",                     16,  0, collect_index_data_eod_job,    "mon-fri", 24),
    "index_data":           ("Index Data Collection (legacy)",              16,  5, collect_index_data_job,        "mon-fri", 24),
    "fii_dii_live":         ("FII/DII Data (live)",                         17,  0, collect_fii_data_job,          "mon-fri", 24),
    "fii_dii_fo":           ("FII/DII F&O Volume",                          17, 30, collect_fii_fo_job,            "mon-fri", 24),
    "daily_news":           ("Daily News Collection",                       16, 30, collect_news_job,              "mon-fri", 24),
    "rss_news":             ("RSS Market News",                             16, 30, collect_rss_news_job,          "mon-fri", 24),
    "delivery_data":        ("NSE Delivery % Collection",                   18,  0, collect_delivery_job,          "mon-fri", 24),
    "nse_announcements":    ("NSE Corporate Announcements",                 18, 30, collect_nse_announcements_job, "mon-fri", 24),
    "corporate_actions":    ("Trendlyne Corporate Actions",                 18, 45, collect_corporate_actions_job, "mon-fri", 24),
    "signal_notifications": ("Signal Change Notifications",                 17, 30, notify_signal_changes_job,     "mon-fri", 24),
    # Weekly Friday retrain — 80h lookback so Monday startup still recovers a missed Friday run
    # (Friday 22:00 → Monday 06:00 = 56h; 80h gives 24h extra margin)
    # "weekly_retrain":       ("Friday Night Model Retrain + Signals",        22,  0, weekly_retrain_job,            "fri",     80),
})
