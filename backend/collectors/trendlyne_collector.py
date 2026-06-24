"""
TradeMind AI — Trendlyne Corporate Actions Collector

Scrapes the Trendlyne Nifty 500 Bonus/Split calendar and stores new events
in the `corporate_actions` table. Runs daily at EOD to detect the latest
splits and bonus issues for all 499 tracked stocks.

Each event stored:
    nse_symbol  — e.g. "HDFCBANK.NS"
    ticker      — e.g. "HDFCBANK"
    ex_date     — date the event takes effect (price adjusts on this date)
    event_type  — "Split" or "Bonus"
    ratio       — e.g. "5:1", "1:1"
    adj_factor  — price multiplier for pre-event rows:
                    Split N:1  → adj_factor = 1/N  (e.g. 5:1 → 0.2)
                    Bonus N:M  → adj_factor = M/(M+N) (e.g. 1:1 → 0.5)
    notes       — raw text from Trendlyne

Usage:
    PYTHONPATH=. python collectors/trendlyne_collector.py
    PYTHONPATH=. python collectors/trendlyne_collector.py --from-date 2023-01-01
"""

import logging
import os
import re
import sys
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.db import get_connection, release_connection, _execute, _executemany

load_dotenv()
logger = logging.getLogger(__name__)

_BASE_URL = (
    "https://trendlyne.com/equity/calendar-v1/upcoming-bonus-split/"
    "?start_date={start_date}&end_date=&corporate_actions=Bonus%2FSplit"
    "&defaultStockgroup=index%2FNIFTY500%2Fnifty-500%2F"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://trendlyne.com/",
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_split_from_facevalue(notes: str) -> Tuple[Optional[float], str]:
    """Extract adj_factor and ratio from face-value text.

    Handles: 'from Rs. 10/- to Rs. 2/-'  and  'from Rs. 2/- to 1/-'
    """
    m = re.search(
        r"from\s+R[se]?\.\s*([\d.]+)\s*/-\s*to\s+(?:R[se]?\.\s*)?([\d.]+)",
        notes, re.IGNORECASE,
    )
    if m:
        old_fv, new_fv = float(m.group(1)), float(m.group(2))
        if old_fv > new_fv > 0:
            n = int(round(old_fv / new_fv))
            return round(new_fv / old_fv, 6), f"{n}:1"
    return None, "N/A"


def _parse_event(event_type: str, notes: str) -> Tuple[Optional[float], str]:
    """Return (adj_factor, ratio_str) for a row."""
    # Strip Trendlyne's leading score number e.g. "0.5 Bonus issue..."
    notes = re.sub(r"^\d+\.\d+\s+", "", notes).strip()

    explicit = re.search(r"(\d+)\s*:\s*(\d+)", notes)

    if explicit and "Bonus" in event_type:
        a, b = int(explicit.group(1)), int(explicit.group(2))
        return round(b / (a + b), 6), f"{a}:{b}"

    if explicit and "Split" in event_type:
        a, b = int(explicit.group(1)), int(explicit.group(2))
        return round(b / a, 6), f"{a}:{b}"

    if "Split" in event_type:
        return _parse_split_from_facevalue(notes)

    return None, "N/A"


# ---------------------------------------------------------------------------
# Scraper
# ---------------------------------------------------------------------------

def scrape_trendlyne(from_date: str) -> List[Dict]:
    """Scrape the Trendlyne Bonus/Split calendar from `from_date` to today.

    Returns a list of dicts with keys:
        ticker, nse_symbol, ex_date, event_type, ratio, adj_factor, notes
    """
    url = _BASE_URL.format(start_date=from_date)
    try:
        r = requests.get(url, headers=_HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as exc:
        logger.error(f"Trendlyne fetch failed: {exc}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table")
    if not table:
        logger.warning("No table found on Trendlyne calendar page")
        return []

    records = []
    for row in table.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        # Extract NSE ticker from corporate-actions URL
        href_tag = cols[0].find(
            "a", href=lambda h: h and "/corporate-actions/" in h
        )
        if not href_tag:
            continue
        parts = href_tag["href"].rstrip("/").split("/")
        try:
            ticker = parts[parts.index("corporate-actions") + 1]
        except (ValueError, IndexError):
            continue

        ex_date_str = cols[1].get_text(strip=True)
        event_type  = cols[2].get_text(strip=True)   # "Bonus" | "Split"
        notes_raw   = cols[3].get_text(strip=True)
        notes_clean = re.sub(r"^\d+\.\d+\s+", "", notes_raw).strip()

        # Only keep Bonus and Split rows
        if not any(t in event_type for t in ("Bonus", "Split")):
            continue

        adj_factor, ratio_str = _parse_event(event_type, notes_raw)

        records.append({
            "ticker":     ticker,
            "nse_symbol": f"{ticker}.NS",
            "ex_date":    ex_date_str,
            "event_type": event_type,
            "ratio":      ratio_str,
            "adj_factor": adj_factor,
            "notes":      notes_clean[:200],
        })

    logger.info(f"Trendlyne scrape: {len(records)} events from {from_date}")
    return records


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _get_existing_keys(conn) -> set:
    """Return set of (nse_symbol, ex_date_str, event_type) already in DB."""
    cur = _execute(conn,
        "SELECT nse_symbol, ex_date::text, event_type FROM corporate_actions"
    )
    return {(r[0], r[1], r[2]) for r in cur.fetchall()}


def insert_new_actions(records: List[Dict], conn) -> int:
    """Insert records not already in DB. Returns count inserted."""
    existing = _get_existing_keys(conn)
    new_rows = []
    for r in records:
        key = (r["nse_symbol"], r["ex_date"], r["event_type"])
        if key not in existing:
            new_rows.append((
                r["nse_symbol"], r["ticker"], r["ex_date"],
                r["event_type"], r["ratio"], r["adj_factor"],
                r["notes"], "trendlyne",
            ))

    if not new_rows:
        return 0

    _executemany(conn, """
        INSERT INTO corporate_actions
            (nse_symbol, ticker, ex_date, event_type, ratio, adj_factor, notes, source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (nse_symbol, ex_date, event_type) DO NOTHING
    """, new_rows)
    conn.commit()
    return len(new_rows)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def collect_corporate_actions(lookback_days: int = 7) -> Dict:
    """
    Scrape Trendlyne for recent Nifty 500 splits/bonuses and store new ones.

    Args:
        lookback_days: How far back to start the scrape window.
                       Daily job uses 7 days to catch any ex-dates announced
                       during the week. Full backfill uses a large value.

    Returns:
        {"scraped": int, "new": int, "skipped": int}
    """
    from_date = (date.today() - timedelta(days=lookback_days)).isoformat()
    records = scrape_trendlyne(from_date)

    if not records:
        return {"scraped": 0, "new": 0, "skipped": 0}

    conn = get_connection()
    try:
        new_count = insert_new_actions(records, conn)
    finally:
        release_connection(conn)

    skipped = len(records) - new_count
    logger.info(
        f"corporate_actions: {len(records)} scraped, "
        f"{new_count} new, {skipped} already in DB"
    )

    if new_count:
        _log_new_events(records, conn_closed=True)

    return {"scraped": len(records), "new": new_count, "skipped": skipped}


def backfill(from_date: str = "2023-01-01") -> Dict:
    """One-shot backfill from a given date. Idempotent."""
    records = scrape_trendlyne(from_date)
    if not records:
        return {"scraped": 0, "new": 0, "skipped": 0}

    conn = get_connection()
    try:
        new_count = insert_new_actions(records, conn)
    finally:
        release_connection(conn)

    skipped = len(records) - new_count
    logger.info(
        f"backfill: {len(records)} scraped, {new_count} new, {skipped} already in DB"
    )
    return {"scraped": len(records), "new": new_count, "skipped": skipped}


def _log_new_events(records: List[Dict], conn_closed: bool = False) -> None:
    """Log newly detected events at INFO level for visibility."""
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT nse_symbol, ex_date::text, event_type, ratio, adj_factor
            FROM corporate_actions
            ORDER BY created_at DESC LIMIT 20
        """)
        recent = cur.fetchall()
        if recent:
            logger.info("Latest corporate_actions in DB:")
            for row in recent:
                logger.info(
                    f"  {row[0]:15s}  ex={row[1]}  {row[2]:6s}  "
                    f"ratio={row[3]}  adj={row[4]}"
                )
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Trendlyne corporate actions collector"
    )
    parser.add_argument(
        "--from-date", default=None,
        help="Start date for backfill (YYYY-MM-DD). Omit for daily 7-day window.",
    )
    parser.add_argument(
        "--lookback-days", type=int, default=7,
        help="Days back for daily mode (default: 7)",
    )
    args = parser.parse_args()

    if args.from_date:
        result = backfill(from_date=args.from_date)
        print(f"Backfill complete: {result}")
    else:
        result = collect_corporate_actions(lookback_days=args.lookback_days)
        print(f"Daily collect complete: {result}")
