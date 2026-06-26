"""
Shared pytest fixtures for the TradeMind backend test suite.

Every test in this suite runs against the dedicated Timescale Cloud TEST
instance, never production. APP_ENV=test must be set BEFORE any `database.db`
import happens anywhere in the process, so it's set here at module import
time, before pytest collects any test module.
"""
import json
import os

os.environ["APP_ENV"] = "test"

import pytest

from database.db import get_connection, init_database, release_connection, _execute

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES_DIR, f"{name}.json")) as f:
        return json.load(f)


# Tables mutated by the suite — truncated before every test so tests don't
# leak state into each other. Intentionally excludes anything not touched
# by these tests (users, portfolios, orders, etc.).
_TEST_TABLES = [
    "prices", "technical_indicators", "trade_signals", "news_sentiment",
    "notifications", "notification_preferences", "watchlist", "risk_settings",
    "portfolio_stocks", "portfolio_sectors", "portfolios", "users",
]


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_test_db():
    """Apply the schema to the test instance once per test session."""
    # database.db reads PGHOST eagerly at import time — fail loudly if a
    # prod-shaped host ever ends up here, instead of silently testing prod.
    from database.db import PGHOST
    assert "tsdb.cloud.timescale.com" in PGHOST, (
        f"Refusing to run tests against unexpected host: {PGHOST}. "
        "Check APP_ENV=test and backend/.env.test."
    )
    init_database()
    yield


@pytest.fixture(autouse=True)
def clean_db():
    """Truncate mutable tables before each test for isolation."""
    conn = get_connection()
    try:
        for table in _TEST_TABLES:
            _execute(conn, f"TRUNCATE TABLE {table} CASCADE")
        conn.commit()
    finally:
        release_connection(conn)
    yield


@pytest.fixture
def api_client(monkeypatch):
    """
    TestClient against the real FastAPI app, wired to the test DB.

    Deliberately NOT used as a context manager (`with TestClient(...)`) —
    that would run api/server.py's startup_event, which elects a scheduler
    owner and spawns the real APScheduler. We also belt-and-suspenders
    no-op the scheduler starter in case that ever changes.
    """
    import scheduler.jobs as jobs_module
    monkeypatch.setattr(jobs_module, "start_background_scheduler", lambda: None)

    from fastapi.testclient import TestClient
    from api.server import app
    return TestClient(app)
