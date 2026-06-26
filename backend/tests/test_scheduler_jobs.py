"""
Scheduler job tests against the TEST database.

DB-only jobs (calculate_indicators_job, cleanup_old_data_job,
verify_data_integrity_job) are exercised directly against seeded test-DB
data. External-API jobs aren't run end-to-end here (that would mean either
hitting Angel One/yfinance for real, or deep-mocking the Angel login/TOTP
flow) — instead test_external_api_contracts.py verifies the parsing
functions those jobs depend on against fixtures shaped exactly like the
real API responses, which is the part that actually breaks when an
external API changes shape.
"""
from datetime import datetime, timedelta

from database.db import get_connection, release_connection, _execute, insert_prices_batch


def _seed_price_history(symbol: str, days: int = 30):
    """Synthetic but realistic daily OHLCV history — enough for indicators (needs >=14 rows)."""
    rows = []
    base = 100.0
    start = datetime.now() - timedelta(days=days)
    for i in range(days):
        d = (start + timedelta(days=i)).strftime("%Y-%m-%d")
        price = base + i * 0.5
        rows.append((symbol, "NSE", d, None, price, price * 1.01, price * 0.99, price + 0.2, 100000 + i * 1000, "1d"))
    insert_prices_batch(rows)


def test_calculate_indicators_job():
    from scheduler.jobs import calculate_indicators_job

    _seed_price_history("TESTSTOCK.NS", days=30)
    calculate_indicators_job()

    conn = get_connection()
    try:
        row = _execute(
            conn, "SELECT COUNT(*) FROM technical_indicators WHERE symbol = ?", ("TESTSTOCK.NS",)
        ).fetchone()
    finally:
        release_connection(conn)
    assert row[0] >= 1


def test_cleanup_old_data_job():
    from scheduler.jobs import cleanup_old_data_job

    old_date = (datetime.now() - timedelta(days=45)).strftime("%Y-%m-%d")
    recent_date = datetime.now().strftime("%Y-%m-%d")
    insert_prices_batch([
        ("OLDINTRA.NS", "NSE", old_date, "10:00:00", 100, 101, 99, 100.5, 1000, "30m"),
        ("RECENTINTRA.NS", "NSE", recent_date, "10:00:00", 100, 101, 99, 100.5, 1000, "30m"),
    ])

    cleanup_old_data_job()

    conn = get_connection()
    try:
        remaining = _execute(
            conn, "SELECT symbol FROM prices WHERE interval = '30m'"
        ).fetchall()
    finally:
        release_connection(conn)
    remaining_symbols = {r[0] for r in remaining}
    assert "OLDINTRA.NS" not in remaining_symbols
    assert "RECENTINTRA.NS" in remaining_symbols


def test_verify_data_integrity_job_runs_clean():
    from scheduler.jobs import verify_data_integrity_job

    _seed_price_history("TESTSTOCK.NS", days=15)
    # Should just log stats — must not raise even on a near-empty test DB.
    verify_data_integrity_job()
