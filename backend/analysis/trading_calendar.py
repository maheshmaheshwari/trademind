"""
TradeMind AI — NSE Trading Calendar & Price-Date Verification

Single source of truth for "was the market open on date X". A trading day is a
weekday (Mon–Fri) that is not in the `market_holidays` table — the exchange's
own calendar, loaded by collectors/nse_holidays_collector.py.

Two jobs:

1. **Calendar** — is_trading_day / previous_trading_day / next_trading_day /
   last_expected_trading_day. Used by the market-status endpoint and the
   holiday banner so "MARKET CLOSED" can say *why*.

2. **Verification** — verify_price_dates() compares the trading days the
   calendar says should exist against the dates actually present in `prices`,
   so a missed EOD collection is caught instead of silently becoming a hole in
   every model's feature window. Without the holiday calendar this check is
   impossible to do without false positives: ~20 weekdays a year legitimately
   have no bars.

Coverage caveat: NSE publishes only the current year's holidays, so the table
fills in year by year. Verification refuses to report gaps for a year it has no
holiday rows for (they'd be indistinguishable from holidays) and reports those
years under `uncovered_years` instead.
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pytz

from database.db import (get_holiday_map, get_holiday_years, get_price_dates,
                         get_special_sessions)

logger = logging.getLogger(__name__)

IST = pytz.timezone("Asia/Kolkata")

# EOD bars are only complete once the 15:35 IST collection has run (see
# scheduler/jobs.py:collect_eod_data_job). Before that, today is not yet
# *expected* to have data even though it is a trading day.
EOD_READY_HOUR = 15
EOD_READY_MINUTE = 35

# A trading day whose bar count is below this fraction of the recent median is
# a partial collection — some symbols got written, most didn't.
PARTIAL_COVERAGE_RATIO = 0.5


def _as_date(value) -> date:
    """Coerce date | datetime | 'YYYY-MM-DD' to a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def today_ist() -> date:
    """Current date in IST — the market's timezone, not the server's."""
    return datetime.now(IST).date()


def is_weekend(d) -> bool:
    return _as_date(d).weekday() >= 5


def holiday_name(d) -> Optional[str]:
    """Holiday description for `d`, or None if it isn't a listed holiday."""
    return get_holiday_map().get(_as_date(d))


def is_trading_day(d) -> bool:
    """True when the NSE equity segment trades on `d`.

    A weekday that is not a listed holiday — OR a recorded special session.
    The exchange does open outside the ordinary pattern: Muhurat (Diwali)
    evenings, the Feb-1 Union Budget session, and NSE's special live / DR-site
    Saturdays. 19 such dates carry 2,993 real price bars, and without this
    branch every one of them was called a non-trading day, which made
    verify_price_dates() report correct bars as `unexpected_dates`
    ("the source misdated a candle").
    """
    d = _as_date(d)
    if d in get_special_sessions():
        return True
    return d.weekday() < 5 and d not in get_holiday_map()


def previous_trading_day(d) -> date:
    """The most recent trading day strictly before `d`."""
    cur = _as_date(d) - timedelta(days=1)
    # 20 days is far beyond the longest NSE closure; the guard just bounds the
    # loop if the holiday table were ever seeded with nonsense.
    for _ in range(20):
        if is_trading_day(cur):
            return cur
        cur -= timedelta(days=1)
    return cur


def next_trading_day(d) -> date:
    """The next trading day strictly after `d`."""
    cur = _as_date(d) + timedelta(days=1)
    for _ in range(20):
        if is_trading_day(cur):
            return cur
        cur += timedelta(days=1)
    return cur


def trading_days_between(start, end) -> List[date]:
    """Every trading day in [start, end], ascending."""
    start, end = _as_date(start), _as_date(end)
    out, cur = [], start
    while cur <= end:
        if is_trading_day(cur):
            out.append(cur)
        cur += timedelta(days=1)
    return out


def last_expected_trading_day(now: Optional[datetime] = None) -> date:
    """The latest date `prices` should already hold a daily bar for.

    Today only counts once the 15:35 IST EOD window has passed; before that the
    answer is the previous trading day. This is the reference the freshness
    check compares the newest stored price date against.
    """
    now = now.astimezone(IST) if now else datetime.now(IST)
    today = now.date()
    if is_trading_day(today) and (now.hour, now.minute) >= (EOD_READY_HOUR, EOD_READY_MINUTE):
        return today
    return previous_trading_day(today)


def upcoming_holidays(limit: int = 5, from_date=None) -> List[Dict]:
    """Next `limit` holidays on/after `from_date` (default today IST), ascending.

    Weekend holidays are included but flagged — the market was closed anyway,
    so the UI can de-emphasise them.
    """
    start = _as_date(from_date) if from_date else today_ist()
    rows = sorted((d, name) for d, name in get_holiday_map().items() if _as_date(d) >= start)
    return [
        {
            "date": str(d),
            "description": name,
            "weekday": _as_date(d).strftime("%A"),
            "is_weekend": is_weekend(d),
            "days_away": (_as_date(d) - start).days,
        }
        for d, name in rows[:limit]
    ]


def verify_price_dates(days: int = 60, interval: str = "1d",
                       now: Optional[datetime] = None) -> Dict:
    """Check `prices` against the trading calendar over the trailing `days`.

    Returns a report with:
      status           — "ok" | "stale" | "gaps" | "no_calendar" | "no_data"
      last_trading_day — what the calendar says should be the newest bar date
      latest_price_date— the newest date actually stored
      stale_by_days    — trading days between those two (0 when current)
      missing_dates    — trading days with no bars at all
      partial_dates    — trading days whose symbol count is far below the median
      unexpected_dates — dates with bars that the calendar calls non-trading
                         (a holiday NSE added late, or bad source data)
      uncovered_years  — years in the window with no holiday rows, where
                         missing/unexpected cannot be judged

    `unexpected_dates` matters as much as `missing_dates`: a bar dated on a
    holiday means the source returned a stale or misdated candle, which is
    exactly the "wrong date" case this is meant to catch.
    """
    now = now.astimezone(IST) if now else datetime.now(IST)
    end = last_expected_trading_day(now)
    start = end - timedelta(days=max(days, 1))

    holiday_years = set(get_holiday_years())
    window_years = set(range(start.year, end.year + 1))
    uncovered = sorted(window_years - holiday_years)

    if not holiday_years:
        return {
            "status": "no_calendar",
            "message": ("market_holidays is empty — run "
                        "collectors/nse_holidays_collector.py before trusting date checks"),
            "last_trading_day": str(end),
            "latest_price_date": None,
            "stale_by_days": None,
            "missing_dates": [],
            "partial_dates": [],
            "unexpected_dates": [],
            "uncovered_years": sorted(window_years),
        }

    # Judge only the part of the window whose year has a holiday calendar;
    # otherwise every real holiday in an uncovered year reads as a gap.
    judge_start = start
    if uncovered:
        covered = sorted(y for y in window_years if y in holiday_years)
        if not covered:
            return {
                "status": "no_calendar",
                "message": f"no holiday calendar for {sorted(window_years)}",
                "last_trading_day": str(end),
                "latest_price_date": None,
                "stale_by_days": None,
                "missing_dates": [],
                "partial_dates": [],
                "unexpected_dates": [],
                "uncovered_years": uncovered,
            }
        judge_start = max(start, date(min(covered), 1, 1))

    counts = get_price_dates(str(judge_start), str(end), interval)
    expected = trading_days_between(judge_start, end)
    expected_set = set(expected)

    present = {_as_date(d): n for d, n in counts.items()}
    latest = max(present) if present else None

    if not present:
        return {
            "status": "no_data",
            "message": f"no {interval} price rows between {judge_start} and {end}",
            "last_trading_day": str(end),
            "latest_price_date": None,
            "stale_by_days": None,
            "missing_dates": [str(d) for d in expected],
            "partial_dates": [],
            "unexpected_dates": [],
            "uncovered_years": uncovered,
        }

    missing = sorted(d for d in expected_set if d not in present)
    unexpected = sorted(d for d in present if d not in expected_set)

    # Median symbol count across days that did get collected — the yardstick
    # for "this day only half-collected".
    tallies = sorted(present[d] for d in present if d in expected_set)
    partial: List[Dict] = []
    if tallies:
        median = tallies[len(tallies) // 2]
        floor = median * PARTIAL_COVERAGE_RATIO
        partial = [
            {"date": str(d), "symbols": present[d], "expected_symbols": median}
            for d in sorted(expected_set)
            if d in present and present[d] < floor
        ]

    stale_by = len(trading_days_between(latest + timedelta(days=1), end)) if latest else None

    if missing or partial:
        status = "gaps"
    elif stale_by:
        status = "stale"
    else:
        status = "ok"

    return {
        "status": status,
        "interval": interval,
        "window_start": str(judge_start),
        "last_trading_day": str(end),
        "latest_price_date": str(latest),
        "stale_by_days": stale_by,
        "trading_days_expected": len(expected),
        "trading_days_present": len([d for d in expected_set if d in present]),
        "missing_dates": [str(d) for d in missing],
        "partial_dates": partial,
        "unexpected_dates": [
            {"date": str(d), "symbols": present[d],
             "reason": "holiday" if holiday_name(d) else "weekend",
             "holiday": holiday_name(d)}
            for d in unexpected
        ],
        "uncovered_years": uncovered,
    }


def market_day_status(now: Optional[datetime] = None) -> Dict:
    """Calendar view of today: trading day or not, and why not.

    Complements /api/market/status's clock-based session logic — that answers
    "what time is it", this answers "does the exchange trade today at all".
    """
    now = now.astimezone(IST) if now else datetime.now(IST)
    today = now.date()
    name = holiday_name(today)
    weekend = is_weekend(today)
    nxt = next_trading_day(today)
    return {
        "date": str(today),
        "is_trading_day": is_trading_day(today),
        "is_holiday": name is not None,
        "is_weekend": weekend,
        "holiday_name": name,
        "reason": ("holiday" if name else "weekend" if weekend else None),
        "next_trading_day": str(nxt),
        "last_trading_day": str(previous_trading_day(today)) if not is_trading_day(today) else str(today),
        "calendar_loaded": bool(get_holiday_map()),
    }
