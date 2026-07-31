"""
NSE trading-calendar tests — holiday storage, calendar maths, and the
price-date verification that depends on both.

Layers covered:
  * contract  — the real NSE holiday-master response shape (fixture) fed
                through the collector's parser
  * DB        — upsert/read of `market_holidays`
  * calendar  — is_trading_day / previous / next / last_expected_trading_day
  * verify    — verify_price_dates() against seeded `prices` rows, which is
                the whole point of storing the calendar: telling a missed EOD
                collection apart from an exchange holiday
  * API       — /api/market/holidays, /api/market/data-freshness,
                and /api/market/status going holiday-aware
"""
from datetime import date, datetime, timedelta

import pytz

from database.db import (
    clear_holiday_cache, get_connection, get_market_holidays, release_connection,
    upsert_market_holidays, _execute,
)

from conftest import load_fixture

IST = pytz.timezone("Asia/Kolkata")

# A fixed week used by the calendar tests so they don't drift with the clock.
# 2026-01-26 (Republic Day) is a Monday holiday; 24th/25th are the weekend.
_MON_HOLIDAY = date(2026, 1, 26)
_FRI_BEFORE = date(2026, 1, 23)
_TUE_AFTER = date(2026, 1, 27)


def _seed_calendar(rows=None):
    """Put a small NSE-shaped calendar in the DB and drop the in-process cache."""
    rows = rows or [
        {"date": "2026-01-26", "weekday": "Monday", "description": "Republic Day"},
        {"date": "2026-03-03", "weekday": "Tuesday", "description": "Holi"},
        {"date": "2026-08-15", "weekday": "Saturday", "description": "Independence Day"},
    ]
    upsert_market_holidays(rows)
    clear_holiday_cache()
    return rows


def _seed_prices(dates, symbols=("AAA.NS", "BBB.NS", "CCC.NS", "DDD.NS")):
    """Insert daily bars for every (date, symbol) pair — how EOD data arrives."""
    conn = get_connection()
    try:
        for d in dates:
            for i, sym in enumerate(symbols):
                _execute(conn,
                    "INSERT INTO prices (symbol, date, open, high, low, close, volume, interval) "
                    "VALUES (?,?,?,?,?,?,?,'1d') ON CONFLICT DO NOTHING",
                    (sym, str(d), 100.0 + i, 101.0 + i, 99.0 + i, 100.5 + i, 10000))
        conn.commit()
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# Contract: the real NSE response shape
# ---------------------------------------------------------------------------

def test_parse_holiday_master_reads_only_the_equity_segment():
    from collectors.nse_holidays_collector import parse_holiday_master

    parsed = parse_holiday_master(load_fixture("nse_holiday_master"))
    dates = [h["date"] for h in parsed]

    # CM entries only — the CD-segment-only holiday must not leak in.
    assert "2026-01-26" in dates
    assert "2026-12-25" in dates
    assert "2026-01-02" not in dates, "currency-segment holiday leaked into the equity calendar"
    assert dates == sorted(dates), "parser must return ascending dates"


def test_parse_holiday_master_normalises_descriptions_and_dates():
    from collectors.nse_holidays_collector import parse_holiday_master

    by_date = {h["date"]: h for h in parse_holiday_master(load_fixture("nse_holiday_master"))}

    # '08-Nov-2026' → ISO, and the Muhurat footnote marker '*' is stripped.
    assert by_date["2026-11-08"]["description"] == "Diwali Laxmi Pujan"
    assert by_date["2026-01-26"]["weekday"] == "Monday"


def test_parse_holiday_master_skips_malformed_rows_without_losing_the_rest():
    from collectors.nse_holidays_collector import parse_holiday_master

    payload = {"CM": [
        {"tradingDate": "not-a-date", "weekDay": "?", "description": "Garbage"},
        {"tradingDate": None, "description": "Missing"},
        {"tradingDate": "26-Jan-2026", "weekDay": "Monday", "description": "Republic Day"},
    ]}
    parsed = parse_holiday_master(payload)
    assert [h["date"] for h in parsed] == ["2026-01-26"]


def test_collect_holidays_writes_the_fixture_calendar_to_the_db(monkeypatch):
    import collectors.nse_holidays_collector as collector

    monkeypatch.setattr(collector, "fetch_holiday_master",
                        lambda timeout=30: load_fixture("nse_holiday_master"))
    result = collector.collect_holidays()
    clear_holiday_cache()

    assert result["status"] == "ok"
    assert result["stored"] == 5
    stored = {h["date"]: h["description"] for h in get_market_holidays()}
    assert stored["2026-12-25"] == "Christmas"


def test_collect_holidays_survives_a_dead_nse(monkeypatch):
    """A failed fetch must not raise or wipe the calendar already stored."""
    import collectors.nse_holidays_collector as collector
    _seed_calendar()

    def _boom(timeout=30):
        raise ConnectionError("NSE unreachable")

    monkeypatch.setattr(collector, "fetch_holiday_master", _boom)
    result = collector.collect_holidays()

    assert result["status"] == "error"
    assert len(get_market_holidays()) == 3, "existing calendar must survive a failed refresh"


def test_upsert_market_holidays_is_idempotent_and_refreshes_descriptions():
    _seed_calendar()
    upsert_market_holidays([
        {"date": "2026-01-26", "weekday": "Monday", "description": "Republic Day (revised)"},
    ])
    clear_holiday_cache()

    rows = {h["date"]: h["description"] for h in get_market_holidays()}
    assert len(rows) == 3, "re-upserting an existing date must not duplicate it"
    assert rows["2026-01-26"] == "Republic Day (revised)"


# ---------------------------------------------------------------------------
# Calendar maths
# ---------------------------------------------------------------------------

def test_is_trading_day_excludes_holidays_and_weekends():
    _seed_calendar()
    from analysis.trading_calendar import is_trading_day

    assert is_trading_day(_FRI_BEFORE) is True
    assert is_trading_day(_MON_HOLIDAY) is False, "Republic Day is a weekday but not a trading day"
    assert is_trading_day(date(2026, 1, 25)) is False, "Sunday"


def test_previous_and_next_trading_day_skip_the_holiday_weekend_run():
    _seed_calendar()
    from analysis.trading_calendar import next_trading_day, previous_trading_day

    # Fri 23rd → Sat/Sun → Mon 26th (holiday) → Tue 27th
    assert next_trading_day(_FRI_BEFORE) == _TUE_AFTER
    assert previous_trading_day(_TUE_AFTER) == _FRI_BEFORE


def test_last_expected_trading_day_waits_for_the_eod_window():
    _seed_calendar()
    from analysis.trading_calendar import last_expected_trading_day

    # Tue 27th 10:00 IST — market open, today's EOD bar doesn't exist yet.
    mid_session = IST.localize(datetime(2026, 1, 27, 10, 0))
    assert last_expected_trading_day(mid_session) == _FRI_BEFORE

    # Same day 16:00 IST — after the 15:35 collection, today is expected.
    after_close = IST.localize(datetime(2026, 1, 27, 16, 0))
    assert last_expected_trading_day(after_close) == _TUE_AFTER

    # On the holiday itself, even after close, the answer is the previous Friday.
    on_holiday = IST.localize(datetime(2026, 1, 26, 16, 0))
    assert last_expected_trading_day(on_holiday) == _FRI_BEFORE


def test_upcoming_holidays_flags_weekend_holidays():
    _seed_calendar()
    from analysis.trading_calendar import upcoming_holidays

    ahead = upcoming_holidays(limit=5, from_date=date(2026, 1, 1))
    assert [h["date"] for h in ahead] == ["2026-01-26", "2026-03-03", "2026-08-15"]
    assert ahead[0]["days_away"] == 25
    assert ahead[2]["is_weekend"] is True, "15-Aug-2026 falls on a Saturday"


# ---------------------------------------------------------------------------
# Price-date verification
# ---------------------------------------------------------------------------

def test_verify_price_dates_clean_run_reports_ok():
    _seed_calendar()
    from analysis.trading_calendar import trading_days_between, verify_price_dates

    now = IST.localize(datetime(2026, 2, 6, 18, 0))   # Friday, after close
    expected = trading_days_between(date(2026, 1, 12), date(2026, 2, 6))
    _seed_prices(expected)

    report = verify_price_dates(days=25, now=now)
    assert report["status"] == "ok", report
    assert report["missing_dates"] == []
    assert report["latest_price_date"] == "2026-02-06"
    assert report["stale_by_days"] == 0


def test_verify_price_dates_ignores_the_holiday_but_catches_the_real_gap():
    """The whole point: a holiday is not a gap, a skipped weekday is."""
    _seed_calendar()
    from analysis.trading_calendar import trading_days_between, verify_price_dates

    now = IST.localize(datetime(2026, 2, 6, 18, 0))
    expected = trading_days_between(date(2026, 1, 12), date(2026, 2, 6))
    skipped = date(2026, 1, 28)                       # a Wednesday we "missed"
    _seed_prices([d for d in expected if d != skipped])

    report = verify_price_dates(days=25, now=now)
    assert report["status"] == "gaps"
    assert report["missing_dates"] == ["2026-01-28"]
    assert "2026-01-26" not in report["missing_dates"], "Republic Day must not count as a gap"


def test_verify_price_dates_flags_a_partially_collected_day():
    _seed_calendar()
    from analysis.trading_calendar import trading_days_between, verify_price_dates

    now = IST.localize(datetime(2026, 2, 6, 18, 0))
    expected = trading_days_between(date(2026, 1, 12), date(2026, 2, 6))
    thin_day = date(2026, 1, 29)
    _seed_prices([d for d in expected if d != thin_day])
    _seed_prices([thin_day], symbols=("AAA.NS",))     # 1 of 4 symbols

    report = verify_price_dates(days=25, now=now)
    assert report["status"] == "gaps"
    assert [p["date"] for p in report["partial_dates"]] == ["2026-01-29"]
    assert report["partial_dates"][0]["symbols"] == 1


def test_verify_price_dates_flags_bars_dated_on_a_holiday():
    """A bar dated on a non-trading day means the source misdated a candle."""
    _seed_calendar()
    from analysis.trading_calendar import trading_days_between, verify_price_dates

    now = IST.localize(datetime(2026, 2, 6, 18, 0))
    _seed_prices(trading_days_between(date(2026, 1, 12), date(2026, 2, 6)))
    _seed_prices([_MON_HOLIDAY])

    report = verify_price_dates(days=25, now=now)
    flagged = {u["date"]: u for u in report["unexpected_dates"]}
    assert "2026-01-26" in flagged
    assert flagged["2026-01-26"]["holiday"] == "Republic Day"


def test_verify_price_dates_detects_stale_data():
    _seed_calendar()
    from analysis.trading_calendar import trading_days_between, verify_price_dates

    now = IST.localize(datetime(2026, 2, 6, 18, 0))
    # Stop three trading days short of the expected last day.
    expected = trading_days_between(date(2026, 1, 12), date(2026, 2, 3))
    _seed_prices(expected)

    report = verify_price_dates(days=25, now=now)
    assert report["latest_price_date"] == "2026-02-03"
    assert report["stale_by_days"] == 3
    assert report["status"] in ("stale", "gaps")


def test_verify_price_dates_refuses_to_guess_without_a_calendar():
    """No holiday rows → every real holiday would read as a gap. Say so instead."""
    from analysis.trading_calendar import verify_price_dates

    report = verify_price_dates(days=25, now=IST.localize(datetime(2026, 2, 6, 18, 0)))
    assert report["status"] == "no_calendar"
    assert report["missing_dates"] == []


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

def test_market_holidays_route(api_client):
    _seed_calendar()

    resp = api_client.get("/api/market/holidays?year=2026")
    assert resp.status_code == 200
    body = resp.json()

    assert body["total"] == 3
    assert [h["date"] for h in body["holidays"]] == ["2026-01-26", "2026-03-03", "2026-08-15"]
    assert body["years_covered"] == [2026]
    assert {"date", "is_trading_day", "is_holiday", "next_trading_day"} <= set(body["today"].keys())
    for h in body["holidays"]:
        assert {"date", "weekday", "description", "is_weekend", "is_past"} <= set(h.keys())


def test_market_holidays_route_year_filter_excludes_other_years(api_client):
    _seed_calendar([
        {"date": "2025-12-25", "weekday": "Thursday", "description": "Christmas"},
        {"date": "2026-01-26", "weekday": "Monday", "description": "Republic Day"},
    ])

    body = api_client.get("/api/market/holidays?year=2026").json()
    assert [h["date"] for h in body["holidays"]] == ["2026-01-26"]
    assert body["years_covered"] == [2025, 2026]


def test_data_freshness_route(api_client):
    _seed_calendar()
    from analysis.trading_calendar import last_expected_trading_day, trading_days_between

    end = last_expected_trading_day()
    _seed_prices(trading_days_between(end - timedelta(days=20), end))

    resp = api_client.get("/api/market/data-freshness?days=20")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok", body
    assert body["latest_price_date"] == str(end)
    assert body["missing_dates"] == []


def test_data_freshness_route_without_a_calendar(api_client):
    resp = api_client.get("/api/market/data-freshness?days=20")
    assert resp.status_code == 200
    assert resp.json()["status"] == "no_calendar"


def test_market_status_route_reports_a_holiday(api_client):
    """A weekday that NSE has closed must read as a holiday, never as 'open'.

    The route derives its own date from the clock, so the calendar is seeded
    with *today's* IST date to drive it.
    """
    import analysis.trading_calendar as cal
    today = cal.today_ist()
    _seed_calendar([{"date": str(today), "weekday": today.strftime("%A"),
                     "description": "Test Closure"}])

    body = api_client.get("/api/market/status").json()

    assert set(["is_open", "session", "is_trading_day", "holiday"]) <= set(body.keys())
    assert body["holiday"] == "Test Closure"
    assert body["is_trading_day"] is False
    assert body["is_open"] is False, "market cannot be open on a holiday"
    assert body["session"] == "holiday"
    assert body["next_trading_day"] > str(today)


def test_market_status_route_without_a_calendar_is_weekday_only(api_client):
    """No holiday rows must not break the endpoint — it falls back to weekdays."""
    body = api_client.get("/api/market/status").json()

    assert body["holiday"] is None
    assert body["is_trading_day"] == (datetime.now(IST).weekday() < 5)
    assert "next_trading_day" not in body or body["is_trading_day"] is False
