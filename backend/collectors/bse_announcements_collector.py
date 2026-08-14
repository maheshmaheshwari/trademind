"""
TradeMind — BSE Corporate Announcements Backfill

The deep-history counterpart to collectors/nse_announcements_collector.py. BSE
serves the same class of event — official company filings — but its archive
goes back to 2010, where NSE's API stops around 2018 and GDELT's DOC API
refuses any start date before 2017 ("Invalid query start date"). For the
symbols whose price history begins in 2010, this is the only free source that
reaches all the way back.

Two endpoints, neither needing an API key (both want a browser UA + Referer,
the same gate as the other NSE/BSE collectors):

  1. Scrip list — one request for every active equity, used to map our NSE
     symbols onto BSE numeric scrip codes:
       GET api.bseindia.com/BseIndiaAPI/api/ListofScripData/w?segment=Equity&status=Active
       -> [{"SCRIP_CD": "500002", "scrip_id": "ABB", "Scrip_Name": "ABB India Ltd", ...}]
     `scrip_id` is BSE's ticker and matches the NSE symbol for 498 of our 500
     constituents. The two that never match — BSE and CDSL — are NSE-only
     listings with no BSE scrip at all, not a mapping bug.

  2. Announcements — paginated, 50 rows a page, with the range total in
     Table1[0].ROWCNT:
       GET .../AnnSubCategoryGetData/w?strScrip=500180&strPrevDate=20100101
           &strToDate=20101231&strCat=-1&strType=C&subcategory=-1&pageno=1
       -> {"Table": [{NEWSID, NEWS_DT, NEWSSUB, HEADLINE, MORE, CATEGORYNAME,
                      ATTACHMENTNAME, ...}],
           "Table1": [{"ROWCNT": 287}]}

Rows are inserted UNSCORED (sentiment NULL) on purpose. FinBERT scoring is a
separate pass — gdelt_collector.score_pending_news() claims globally-unscored
rows, so concurrent shards would score the same headlines repeatedly. Fetch in
parallel, score once, exactly like the GDELT shards in sync-universe.yml.

Dedupe rides on the uq_news_url_pubdate unique index (url, published_at): each
announcement gets its own URL built from BSE's NEWSID, so a re-run over a range
already collected is a no-op rather than a second copy.

Usage:
    PYTHONPATH=. python collectors/bse_announcements_collector.py --symbol HDFCBANK
    PYTHONPATH=. python collectors/bse_announcements_collector.py --from-date 2010-01-01
    PYTHONPATH=. python collectors/bse_announcements_collector.py --shard 2/8
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import (  # noqa: E402
    get_active_universe, get_connection, release_connection, _execute, _executemany,
    get_backfill_coverage, record_backfill_coverage,
)

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BSE_API           = "https://api.bseindia.com/BseIndiaAPI/api"
SCRIP_LIST_URL    = f"{BSE_API}/ListofScripData/w"
ANNOUNCEMENTS_URL = f"{BSE_API}/AnnSubCategoryGetData/w"

# Public announcement permalink — the browser-facing page for a NEWSID. Not
# fetched by us; it exists so a stored row is traceable back to the filing.
ANN_PERMALINK = "https://www.bseindia.com/corporates/anndet_new.aspx?newsid={newsid}"

# How close to from_date a symbol's oldest stored announcement has to be for
# the window to count as already fetched.
#
# Without this the check was `covered <= from_date` — an exact-date comparison
# against a calendar date nobody files on. The BSE window opens 2010-01-01, a
# holiday; the earliest announcement in the entire archive is 2010-01-04. So
# ZERO symbols ever matched and every re-run re-fetched all 351 of them over
# HTTP to insert nothing, which is what timed out the DB on run 2.
#
# A month of slack: a company that filed anything in the first weeks of the
# window clearly had that window collected. Symbols whose first filing is later
# (late listings, thin filers) still re-fetch — we cannot tell "listed in 2015"
# from "only got fetched back to 2015" without real coverage tracking — but
# those are cheap, returning few rows. --no-skip forces a full re-fetch.
COVERAGE_GRACE_DAYS  = 31

PAGE_SIZE            = 50      # fixed server-side; ROWCNT is the range total
FROM_DATE            = "2010-01-01"
SLEEP_BETWEEN_PAGES  = 0.4     # no published rate limit; stay polite
SLEEP_BETWEEN_STOCKS = 0.8
MAX_PAGES_PER_WINDOW = 60      # 3,000 rows — far beyond any real symbol-year

BSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.bseindia.com/corporates/ann.html",
}


def _make_session() -> requests.Session:
    session = requests.Session()
    # 429 belongs here as much as the 5xx codes: a backfill runs four shards
    # concurrently for hours against an endpoint with no published rate limit,
    # so being throttled is an expected outcome, not an error. Without it a
    # single 429 fails the symbol outright (and now the shard with it).
    # urllib3 honours Retry-After for 429; backoff_factor covers the rest.
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(BSE_HEADERS)
    return session


# ---------------------------------------------------------------------------
# Symbol -> scrip code
# ---------------------------------------------------------------------------

def parse_scrip_list(payload) -> Dict[str, str]:
    """Build {NSE_SYMBOL: scrip_code} from the ListofScripData response.

    Kept separate from the HTTP call so the response contract is testable
    against tests/fixtures/bse_scrip_list.json.
    """
    rows = payload if isinstance(payload, list) else (payload or {}).get("Table") or []
    out: Dict[str, str] = {}
    for row in rows:
        ticker = ((row or {}).get("scrip_id") or "").strip().upper()
        code = str((row or {}).get("SCRIP_CD") or "").strip()
        if ticker and code:
            out.setdefault(ticker, code)
    return out


def fetch_scrip_map(session: requests.Session, timeout: int = 60) -> Dict[str, str]:
    """One request for BSE's whole active-equity list (~5,000 rows)."""
    params = {"Group": "", "Scripcode": "", "industry": "",
              "segment": "Equity", "status": "Active"}
    resp = session.get(SCRIP_LIST_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return parse_scrip_list(resp.json())


# ---------------------------------------------------------------------------
# Announcement fetch + parse
# ---------------------------------------------------------------------------

def _headline(ann: Dict) -> str:
    """Subject plus body — the text FinBERT actually scores.

    NEWSSUB alone is often a bare regulation reference ("Disclosures under
    Reg.13(4)..."); HEADLINE/MORE carries what actually happened. Joining them
    gives the model something to work with while keeping the subject, which is
    the part that is always populated.
    """
    subject = (ann.get("NEWSSUB") or "").strip()
    body = (ann.get("HEADLINE") or "").strip() or (ann.get("MORE") or "").strip()
    if subject and body and body != subject:
        return f"{subject} - {body}"
    return subject or body or "Corporate announcement"


def _published_at(ann: Dict) -> Optional[datetime]:
    """NEWS_DT ('2010-11-03T17:40:35'), falling back to DT_TM."""
    for key in ("NEWS_DT", "DT_TM", "DissemDT"):
        raw = (ann.get(key) or "").strip()
        if not raw:
            continue
        try:
            return datetime.strptime(raw[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            try:
                return datetime.strptime(raw[:10], "%Y-%m-%d")
            except ValueError:
                continue
    return None


def parse_announcements(payload: Dict, symbol_ns: str) -> List[Tuple]:
    """Response page -> news_sentiment insert tuples (sentiment left NULL).

    Announcements with no parseable date or no NEWSID are skipped: without a
    date the row cannot join to a price bar, and without a NEWSID it has no
    unique URL and would defeat the dedupe index.
    """
    rows: List[Tuple] = []
    for ann in (payload or {}).get("Table") or []:
        newsid = ((ann or {}).get("NEWSID") or "").strip()
        pub = _published_at(ann or {})
        if not newsid or not pub:
            continue
        rows.append((
            _headline(ann)[:500],
            "BSE",
            pub.strftime("%Y-%m-%d %H:%M:%S"),
            symbol_ns,
            None,   # sentiment — filled by the scoring pass
            None,   # confidence
            ANN_PERMALINK.format(newsid=newsid),
        ))
    return rows


def total_for_range(payload: Dict) -> int:
    """Table1[0].ROWCNT — how many announcements the range holds in total."""
    table1 = (payload or {}).get("Table1") or []
    if not table1:
        return 0
    try:
        return int(table1[0].get("ROWCNT") or 0)
    except (TypeError, ValueError):
        return 0


def fetch_page(session: requests.Session, scrip_code: str, from_date: date,
               to_date: date, pageno: int, timeout: int = 30) -> Dict:
    """One page of announcements for a scrip over [from_date, to_date]."""
    params = {
        "pageno": pageno,
        "strCat": "-1",
        "strPrevDate": from_date.strftime("%Y%m%d"),
        "strToDate": to_date.strftime("%Y%m%d"),
        "strScrip": scrip_code,
        "strSearch": "P",
        "strType": "C",
        "subcategory": "-1",
    }
    resp = session.get(ANNOUNCEMENTS_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def fetch_symbol_window(session: requests.Session, scrip_code: str, symbol_ns: str,
                        from_date: date, to_date: date) -> List[Tuple]:
    """Every announcement for one scrip over one window, following pagination."""
    rows: List[Tuple] = []
    seen_urls = set()
    total = None
    for pageno in range(1, MAX_PAGES_PER_WINDOW + 1):
        try:
            payload = fetch_page(session, scrip_code, from_date, to_date, pageno)
        except Exception as exc:
            logger.warning("%s: page %d failed (%s) - keeping %d rows so far",
                           symbol_ns, pageno, exc, len(rows))
            break

        page_rows = parse_announcements(payload, symbol_ns)
        if not page_rows:
            break

        # ROWCNT only tells us when to stop early. A page that carries rows but
        # no Table1 still gets kept — the short-page check below ends the loop.
        if total is None:
            total = total_for_range(payload) or 0

        # BSE occasionally repeats a row across page boundaries; the DB index
        # would catch it, but filtering here keeps the reported counts honest.
        fresh = [r for r in page_rows if r[6] not in seen_urls]
        seen_urls.update(r[6] for r in fresh)
        rows.extend(fresh)

        if (total and len(rows) >= total) or len(page_rows) < PAGE_SIZE:
            break
        time.sleep(SLEEP_BETWEEN_PAGES)

    return rows


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

INSERT_SQL = """INSERT INTO news_sentiment
    (headline, source, published_at, symbol, sentiment, confidence, url)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT DO NOTHING"""


def store(rows: List[Tuple]) -> int:
    """Batch-insert announcement rows.

    Returns rows ATTEMPTED, not inserted — duplicates are dropped by
    uq_news_url_pubdate, so treat this as an upper bound and trust the
    end-of-run coverage query for what actually landed.

    Retries once, then RAISES. It used to swallow the error and return 0, which
    meant a symbol's entire 2010-2017 history could vanish while the run still
    counted it as processed and exited green. Raising hands the symbol to
    backfill()'s except clause, which records it in `failed`.

    The retry is not paranoia: a shard runs ~25 minutes against Timescale Cloud
    and a pooled connection can be closed underneath it ("connection already
    closed" — seen in this repo's own test runs). One fresh-connection retry
    turns that from a lost symbol into a blip.
    """
    if not rows:
        return 0
    last_exc = None
    for attempt in (1, 2):
        conn = get_connection()
        try:
            _executemany(conn, INSERT_SQL, rows)
            conn.commit()
            return len(rows)
        except Exception as exc:
            last_exc = exc
            # Guarded: rollback on an already-dead connection raises
            # InterfaceError, which would mask the real error above it.
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning("BSE batch insert attempt %d/2 failed: %s", attempt, exc)
        finally:
            release_connection(conn)
    raise RuntimeError(f"BSE batch insert failed after 2 attempts: {last_exc}")


def earliest_covered(symbol_ns: str) -> Optional[date]:
    """Oldest BSE announcement already stored for this symbol, or None."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT MIN(published_at) FROM news_sentiment WHERE symbol = ? AND source = 'BSE'",
            (symbol_ns,))
        row = cur.fetchone()
        val = row[0] if row else None
        return val.date() if isinstance(val, datetime) else val
    finally:
        release_connection(conn)


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------

def _year_windows(from_date: date, to_date: date) -> List[Tuple[date, date]]:
    """Split a range into calendar-year windows.

    A single request for the whole range would be simpler, but a busy large-cap
    files ~300 announcements a year: a decade in one range is 60+ pages, and any
    failure mid-way costs the whole symbol. Per-year windows keep each unit
    small and independently retryable.
    """
    out = []
    for year in range(from_date.year, to_date.year + 1):
        start = max(from_date, date(year, 1, 1))
        end = min(to_date, date(year, 12, 31))
        if start <= end:
            out.append((start, end))
    return out


def backfill(symbols: List[str], from_date: date, to_date: date,
             skip_covered: bool = True, session: Optional[requests.Session] = None) -> Dict:
    """Fetch BSE announcements for `symbols` over [from_date, to_date].

    `symbols` are bare NSE tickers (HDFCBANK, not HDFCBANK.NS). Rows land
    unscored; run the FinBERT pass afterwards.
    """
    session = session or _make_session()

    try:
        scrip_map = fetch_scrip_map(session)
    except Exception as exc:
        logger.error("BSE scrip list unavailable: %s", exc)
        return {"status": "error", "error": str(exc), "rows": 0,
                "processed": 0, "unmapped": [], "failed": []}

    logger.info("BSE scrip list: %d active equities", len(scrip_map))

    total_rows = 0
    processed = 0
    unmapped: List[str] = []
    empty: List[str] = []
    failed: List[str] = []

    for idx, symbol in enumerate(symbols, 1):
        scrip_code = scrip_map.get(symbol.upper())
        if not scrip_code:
            # BSE/CDSL and any other NSE-only listing — expected, not an error.
            unmapped.append(symbol)
            continue

        symbol_ns = f"{symbol}.NS"
        if skip_covered:
            # backfill_coverage records what was actually fetched. The old probe
            # (earliest stored row vs from_date) could not tell a late listing
            # from a shallow fetch, so it skipped nothing for any symbol that
            # listed after the window opened — which re-fetched 16 years per
            # symbol and exhausted the database in run 31771174759.
            cov = get_backfill_coverage(symbol_ns, "BSE")
            if cov and cov[0] <= from_date + timedelta(days=COVERAGE_GRACE_DAYS) \
                   and cov[1] >= to_date - timedelta(days=COVERAGE_GRACE_DAYS):
                logger.debug("[%d/%d] %s: already covered %s..%s",
                             idx, len(symbols), symbol, cov[0], cov[1])
                continue

        try:
            rows: List[Tuple] = []
            for w_start, w_end in _year_windows(from_date, to_date):
                rows.extend(fetch_symbol_window(session, scrip_code, symbol_ns,
                                                w_start, w_end))
            if not rows:
                # A company listed after the window opened has nothing to give
                # — worth naming, since it is otherwise indistinguishable from
                # a scrip code that quietly stopped matching.
                logger.info("[%d/%d] %s (scrip %s): 0 announcements in window",
                            idx, len(symbols), symbol, scrip_code)
                empty.append(symbol)
                # An empty window is still a fetched window. Without this the
                # symbol re-fetches on every run forever — and "nothing to
                # find" is exactly the case where re-fetching buys nothing.
                record_backfill_coverage(symbol_ns, "BSE", from_date, to_date, 0)
                time.sleep(SLEEP_BETWEEN_STOCKS)
                continue

            stored = store(rows)
            total_rows += stored
            processed += 1
            # Only after store() has committed — recording coverage for a symbol
            # whose rows failed to land would make the next run skip a gap.
            record_backfill_coverage(symbol_ns, "BSE", from_date, to_date, stored)
            logger.info("[%d/%d] %s (scrip %s): %d announcements %s to %s",
                        idx, len(symbols), symbol, scrip_code, stored,
                        from_date, to_date)
        except Exception as exc:
            logger.error("[%d/%d] %s: %s", idx, len(symbols), symbol, exc)
            failed.append(symbol)

        time.sleep(SLEEP_BETWEEN_STOCKS)

    if unmapped:
        logger.info("No BSE scrip for %d symbol(s): %s", len(unmapped), ", ".join(unmapped))
    if empty:
        logger.info("No announcements in window for %d symbol(s): %s",
                    len(empty), ", ".join(empty[:20]))

    return {"status": "ok", "rows": total_rows, "processed": processed,
            "unmapped": unmapped, "empty": empty, "failed": failed}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main():
    parser = argparse.ArgumentParser(description="BSE corporate announcements backfill")
    parser.add_argument("--symbol", type=str, default=None,
                        help="single symbol, e.g. HDFCBANK")
    parser.add_argument("--from-date", type=str, default=FROM_DATE,
                        help=f"start date YYYY-MM-DD (default {FROM_DATE})")
    parser.add_argument("--to-date", type=str, default=None,
                        help="end date YYYY-MM-DD (default today)")
    parser.add_argument("--shard", type=str, default=None, metavar="i/N",
                        help="process only shard i of N (round-robin over the universe)")
    parser.add_argument("--no-skip", action="store_true",
                        help="re-fetch symbols whose stored history already reaches --from-date")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    symbols = [s.replace(".NS", "") for s in get_active_universe()]
    if args.symbol:
        symbols = [s for s in symbols if s.upper() == args.symbol.upper()]
        if not symbols:
            logger.error("Symbol %s is not in the active universe", args.symbol)
            sys.exit(1)
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        symbols = symbols[i - 1::n]
        logger.info("Shard %d/%d - %d symbols", i, n, len(symbols))

    result = backfill(
        symbols,
        from_date=_parse_date(args.from_date),
        to_date=_parse_date(args.to_date) if args.to_date else date.today(),
        skip_covered=not args.no_skip,
    )
    logger.info("BSE backfill done: %d rows over %d symbols (%d unmapped, %d failed)",
                result["rows"], result["processed"],
                len(result["unmapped"]), len(result["failed"]))
    if result["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
