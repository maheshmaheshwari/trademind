"""Record the days the NSE traded outside its ordinary pattern.

`market_holidays` answers "when is the market shut" and the NSE feed never says
"the market opens on this Sunday" — so Muhurat (Diwali) sessions, the Feb-1
Union Budget session and NSE's special live / DR-site Saturdays are invisible to
the calendar. `is_trading_day()` then calls them non-trading days, and
`verify_price_dates()` reports their genuine bars as `unexpected_dates`
("the source misdated a candle").

Detection is evidence-based rather than a hardcoded list: any date carrying a
broad set of daily price bars *was* a trading day, whatever the calendar says.
That keeps working for future Muhurat and Budget sessions with no code change.

    MIN_BARS   a real session prints bars for many symbols at once; a stray
               mis-stamped candle prints one or two. 5 separates them across
               the whole range, including the ~7-symbol pre-2016 era.

Usage (from backend/):
    python scripts/seed_special_sessions.py --dry-run
    python scripts/seed_special_sessions.py
    APP_ENV=test python scripts/seed_special_sessions.py
"""

import argparse
import logging
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import (  # noqa: E402
    _execute,
    clear_holiday_cache,
    get_connection,
    init_database,
    release_connection,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("seed_special_sessions")

MIN_BARS = 5
START_YEAR = 2010


def detect(conn, end: date):
    """Dates with >= MIN_BARS daily bars that the calendar calls non-trading."""
    from analysis.trading_calendar import _as_date  # noqa: F401  (kept for parity)
    from database.db import get_holiday_map

    holidays = set(get_holiday_map())
    found = []
    for y in range(START_YEAR, end.year + 1):
        a = f"{y}-01-01"
        b = min(date(y, 12, 31), end).isoformat()
        # Year-bounded: an unbounded aggregate over `prices` is a full scan of a
        # ~200-chunk compressed hypertable and has OOM'd this instance before.
        cur = _execute(conn, """SELECT date, COUNT(*) FROM prices
                                 WHERE interval='1d' AND date>=? AND date<=?
                                 GROUP BY date HAVING COUNT(*) >= ?
                                 ORDER BY date""", (a, b, MIN_BARS))
        for d, n in cur.fetchall():
            if d.weekday() >= 5 or d in holidays:
                found.append((d, n))
    return found


def from_nse_csv(path: str):
    """Muhurat dates from an NSE holiday CSV, identified by the trailing '*'.

    NSE's published calendar marks Diwali-Laxmi Pujan with an asterisk —
    "Diwali-Laxmi Pujan*" — which is its notation for "closed for the normal
    session, open for Muhurat". Those dates are the dangerous case: most fall on
    a WEEKDAY and are absent from market_holidays, so is_trading_day() already
    returns True and detect() cannot see them. The moment someone imports the
    NSE list verbatim they become holidays and four working dates break.

    Recording them as special_session now makes that import safe.
    """
    import csv as _csv
    from datetime import datetime as _dt
    out = []
    with open(path) as f:
        for row in _csv.DictReader(f):
            occ = (row.get("Occasion") or "").strip()
            if not occ.endswith("*"):
                continue
            d = _dt.strptime(row["Date"].strip(), "%d/%m/%Y").date()
            out.append((d, occ.rstrip("*").strip()))
    return out


def label(d: date) -> str:
    """Best-effort description. The month is enough to tell these apart:
    Budget sessions are 1 February; Muhurat falls in Oct/Nov with Diwali."""
    if d.month == 2 and d.day == 1:
        return "Union Budget special session"
    if d.month in (10, 11):
        return "Muhurat trading (Diwali)"
    return "NSE special trading session"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    ap.add_argument("--end", default=None, help="last date to consider (YYYY-MM-DD)")
    ap.add_argument("--nse-csv", default=None,
                    help="NSE holiday CSV; rows whose Occasion ends with '*' are "
                         "Muhurat sessions and are recorded as special_session too")
    args = ap.parse_args(argv)

    end = date(*map(int, args.end.split("-"))) if args.end else date.today()

    # init_database() issues DDL (CREATE TABLE / ALTER TABLE / CREATE INDEX), so
    # a --dry-run that called it was not dry: an earlier "dry run" of this
    # script is what actually added market_holidays.session_type to prod. A dry
    # run must be able to report against an un-migrated database without
    # changing it — detect() and from_nse_csv() touch only prices and the CSV.
    if not args.dry_run:
        init_database()
    conn = get_connection()
    try:
        rows = detect(conn, end)
        detected = {d for d, _ in rows}

        if args.nse_csv:
            extra = [(d, occ) for d, occ in from_nse_csv(args.nse_csv) if d not in detected]
            if extra:
                print(f"\n⭐ Muhurat dates from {os.path.basename(args.nse_csv)} "
                      f"(marked '*'): {len(extra)}")
                counted = []
                for d, occ in extra:
                    cur = _execute(conn, "SELECT COUNT(*) FROM prices "
                                         "WHERE interval='1d' AND date=?", (d.isoformat(),))
                    bars = cur.fetchone()[0]
                    print(f"   {d.isoformat()}  {d.strftime('%a')}  {bars:>5} bars   {occ}")
                    counted.append((d, bars))
                # Carry the real bar count into the main table. Appending a 0
                # placeholder printed "0 bars" for dates that plainly have
                # hundreds, which reads as a defect in a log someone finds later.
                rows = rows + counted

        print(f"\n📅 Special sessions to record: {len(rows)}\n")
        print(f"{'date':<13}{'day':<5}{'bars':>7}   description")
        for d, n in rows:
            print(f"{d.isoformat():<13}{d.strftime('%a'):<5}{n:>7}   {label(d)}")

        if args.dry_run:
            print("\n🔍 Dry run — nothing written.")
            return 0

        written = 0
        for d, _n in rows:
            cur = _execute(conn, """
                INSERT INTO market_holidays
                    (holiday_date, segment, exchange, weekday, description, source, session_type)
                VALUES (?, 'CM', 'NSE', ?, ?, 'derived: price bars present on a non-trading date',
                        'special_session')
                ON CONFLICT (exchange, segment, holiday_date)
                DO UPDATE SET session_type = 'special_session',
                              description  = EXCLUDED.description
            """, (d.isoformat(), d.strftime("%A"), label(d)))
            written += cur.rowcount or 0
        conn.commit()
        clear_holiday_cache()
        print(f"\n✅ Recorded {written} special session(s).")

        from analysis.trading_calendar import is_trading_day
        bad = [d for d, _ in rows if not is_trading_day(d)]
        if bad:
            print(f"❌ still not trading days: {bad}")
            return 1
        print("   is_trading_day() now returns True for every one.")
        return 0
    finally:
        release_connection(conn)


if __name__ == "__main__":
    sys.exit(main())
