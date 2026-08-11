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

# api/server.py builds its CORS allow-list at import time from
# CORS_ALLOWED_ORIGINS, falling back to the local dev origins when unset. A
# developer's real backend/.env sets that variable to the deployed frontend
# domains, so test_cors_preflight_allowed_origin — which asks for
# http://localhost:5173 — got a 400 locally while passing in CI, where no .env
# exists. Pin the allow-list so the CORS tests assert on a known configuration
# instead of on whatever happens to be in the developer's environment.
#
# Set here for the same reason APP_ENV is: it has to land before anything
# imports api.server (the api_client fixture does so lazily). load_dotenv()
# leaves variables already present in the environment alone, so .env cannot
# clobber this. Keep the value out of .env.test too — db.py reloads that file
# with override=True.
os.environ["CORS_ALLOWED_ORIGINS"] = (
    "http://localhost:5173,http://localhost:5174,http://127.0.0.1:5173"
)

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
    "model_training_stats", "market_holidays",
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
    # market_holidays is cached in-process; truncating the table behind the
    # cache's back would leak one test's calendar into the next.
    from database.db import clear_holiday_cache
    clear_holiday_cache()

    # slowapi's Limiter is a module-level singleton keyed by client IP, and
    # TestClient presents the same IP for every request — so its counters are
    # shared by the WHOLE session, not per test. /api/trading/register is capped
    # at 5/hour, and almost every test registers a user to get a token, so from
    # the sixth test onward they failed with
    #   AssertionError: {"error":"Rate limit exceeded: 5 per 1 hour"}
    # That single leak accounted for the bulk of the suite's failures, and it
    # hid behind looking like many unrelated broken features: each test passes
    # on its own and only fails in company.
    #
    # Reset rather than disable, so a test that deliberately exercises a limit
    # still sees real limiter behaviour within its own run.
    from api.rate_limit import limiter
    limiter.reset()
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
