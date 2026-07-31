"""
TradeMind AI — NSE Trading Holidays Collector

Populates the `market_holidays` table from the exchange's own calendar, i.e.
the API backing https://www.nseindia.com/resources/exchange-communication-holidays:

    https://www.nseindia.com/api/holiday-master?type=trading

The response is keyed by NSE segment code — 'CM' (capital market / equity) is
the one this platform trades; 'FO', 'CD', 'COM' etc. are other segments and are
ignored. Each entry looks like:

    {"tradingDate": "26-Jan-2026", "weekDay": "Monday",
     "description": "Republic Day", "Sr_no": 2}

Two important properties of this feed:

  * It only ever returns the **current calendar year**. Rows therefore
    accumulate as this collector re-runs year after year — the upsert never
    deletes, and neither should anything else, because the price-date
    verification needs prior years to distinguish a missed EOD collection from
    a genuine exchange holiday.
  * Some listed holidays fall on a Saturday/Sunday (e.g. Independence Day
    2026 is a Saturday). Those are stored as-is; the trading calendar treats
    weekends as non-trading days anyway, so they are harmless duplicates.

Usage:
    PYTHONPATH=. python collectors/nse_holidays_collector.py
    APP_ENV=test PYTHONPATH=. python collectors/nse_holidays_collector.py
"""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import upsert_market_holidays, get_market_holidays  # noqa: E402

logger = logging.getLogger(__name__)

NSE_HOLIDAY_URL = "https://www.nseindia.com/api/holiday-master?type=trading"
NSE_HOLIDAY_PAGE = "https://www.nseindia.com/resources/exchange-communication-holidays"

# NSE rejects requests without a browser-ish UA and a matching Referer.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NSE_HOLIDAY_PAGE,
}

# NSE segment whose calendar governs equity trading (and therefore our prices).
EQUITY_SEGMENT = "CM"


def _parse_nse_date(raw: str) -> str:
    """'26-Jan-2026' → '2026-01-26'. Raises ValueError on an unexpected shape."""
    return datetime.strptime(raw.strip(), "%d-%b-%Y").strftime("%Y-%m-%d")


def parse_holiday_master(payload: Dict, segment: str = EQUITY_SEGMENT) -> List[Dict]:
    """Extract [{date, weekday, description}] for one segment of the NSE response.

    Entries with an unparseable/missing tradingDate are skipped rather than
    failing the whole run — a single malformed row must not cost us the
    calendar. Kept separate from the HTTP call so the response contract is
    testable against a fixture (tests/fixtures/nse_holiday_master.json).
    """
    out: List[Dict] = []
    seen = set()
    for entry in (payload or {}).get(segment, []) or []:
        raw = (entry or {}).get("tradingDate")
        if not raw:
            continue
        try:
            iso = _parse_nse_date(raw)
        except (ValueError, TypeError):
            logger.warning("Skipping holiday with unparseable date: %r", raw)
            continue
        if iso in seen:
            continue
        seen.add(iso)
        out.append({
            "date": iso,
            "weekday": (entry.get("weekDay") or "").strip() or None,
            # Diwali Muhurat rows carry a trailing '*' footnote marker.
            "description": (entry.get("description") or "").strip().rstrip("*").strip() or None,
        })
    return sorted(out, key=lambda h: h["date"])


def fetch_holiday_master(timeout: int = 30) -> Dict:
    """GET the NSE holiday-master JSON, bootstrapping cookies from the page first."""
    session = requests.Session()
    session.headers.update(HEADERS)
    try:
        # NSE hands out the cookies its API gate wants only on a page load.
        session.get(NSE_HOLIDAY_PAGE, timeout=timeout)
    except requests.RequestException as exc:
        logger.debug("NSE cookie bootstrap failed (continuing): %s", exc)
    resp = session.get(NSE_HOLIDAY_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def collect_holidays(segment: str = EQUITY_SEGMENT) -> Dict:
    """Fetch NSE holidays and upsert them. Returns a summary dict.

    On any fetch/parse failure this returns {"status": "error", ...} instead of
    raising, so a scheduler run can log it without killing the job chain — the
    previously stored calendar stays valid regardless.
    """
    try:
        payload = fetch_holiday_master()
    except Exception as exc:
        logger.error("NSE holiday fetch failed: %s", exc)
        return {"status": "error", "error": str(exc), "stored": 0}

    holidays = parse_holiday_master(payload, segment)
    if not holidays:
        logger.warning("NSE holiday response had no '%s' segment entries", segment)
        return {"status": "empty", "stored": 0}

    written = upsert_market_holidays(holidays, segment=segment)
    total = len(get_market_holidays(segment=segment))
    logger.info("NSE holidays: upserted %d (%s → %s) · table now %d rows",
                written, holidays[0]["date"], holidays[-1]["date"], total)
    return {
        "status": "ok",
        "stored": written,
        "total": total,
        "first": holidays[0]["date"],
        "last": holidays[-1]["date"],
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = collect_holidays()
    print(result)
    for h in get_market_holidays():
        print(f"  {h['date']}  {h['weekday'] or '':<10} {h['description'] or ''}")
