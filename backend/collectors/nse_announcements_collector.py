"""
TradeMind — NSE Corporate Announcements Backfill

Fetches 3 years of official corporate announcements for all 499 Nifty 500
stocks from NSE India's API, runs FinBERT sentiment scoring, and stores
results in news_sentiment. No API key required.

API: GET https://www.nseindia.com/api/corporate-announcements
     Params: index=equities, symbol=HDFCBANK, from_date=01-01-2023, to_date=02-06-2026
     Returns: [{an_dt, desc, attchmntText, symbol}, ...]

Depth: NSE serves roughly 2018 onwards (a 01-01-2018 query returns ~1,500
announcements for a large-cap, against ~1,000 from 2021). Anything older comes
from collectors/bse_announcements_collector.py, whose archive reaches 2010.

Usage:
    PYTHONPATH=. python collectors/nse_announcements_collector.py
    PYTHONPATH=. python collectors/nse_announcements_collector.py --symbol HDFCBANK
    PYTHONPATH=. python collectors/nse_announcements_collector.py --from-date 01-01-2018
    PYTHONPATH=. python collectors/nse_announcements_collector.py \
        --from-date 01-01-2018 --to-date 31-12-2022 --fetch-only
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import init_database, insert_news, get_connection, release_connection, _execute

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROM_DATE          = "01-01-2023"          # dd-mm-yyyy for NSE API
SLEEP_BETWEEN_STOCKS = 1.0                 # seconds — polite, no stated rate limit
# Days of slack when deciding a symbol's window is already collected. Kept in
# step with the BSE collector's constant of the same name, which carries the
# full rationale.
COVERAGE_GRACE_DAYS  = 31
BATCH_SIZE         = 128                   # FinBERT batch size (larger = faster on MPS)

NSE_URL = "https://www.nseindia.com/api/corporate-announcements"
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.nseindia.com",
    "Accept-Language": "en-US,en;q=0.9",
}

_TOKENS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "angel_tokens.json"
)

# ---------------------------------------------------------------------------
# Session with retry
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    session = requests.Session()
    # 429 included deliberately — NSE throttles harder than BSE, and a backfill
    # runs three shards concurrently for the better part of an hour. Without it
    # a single throttled response fails the symbol (and now the shard).
    retry = Retry(total=3, backoff_factor=2,
                  status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry, pool_connections=5, pool_maxsize=5)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Seed cookies by hitting NSE homepage first
    try:
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
    except Exception:
        pass
    return session


# ---------------------------------------------------------------------------
# FinBERT sentiment (singleton, lazy-loaded)
# ---------------------------------------------------------------------------

class _FinBERT:
    _pipeline = None

    @classmethod
    def score(cls, texts: List[str]) -> List[Dict]:
        if cls._pipeline is None:
            logger.info("Loading FinBERT model (first run only)…")
            from transformers import pipeline
            cls._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=1,
            )
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            truncated = [t[:512] for t in batch]
            preds = cls._pipeline(truncated)
            for pred in preds:
                top = pred[0] if isinstance(pred, list) else pred
                label = top["label"].lower()       # positive / negative / neutral
                score = float(top["score"])
                if label == "positive":
                    sentiment = score
                elif label == "negative":
                    sentiment = -score
                else:
                    sentiment = 0.0
                results.append({"sentiment": sentiment, "confidence": score, "label": label})
        return results


# ---------------------------------------------------------------------------
# NSE fetch
# ---------------------------------------------------------------------------

def fetch_announcements(session: requests.Session, symbol: str, from_date: str,
                        to_date: Optional[str] = None) -> List[Dict]:
    """
    Fetch corporate announcements for one symbol from NSE.
    symbol: bare symbol without .NS, e.g. "HDFCBANK"
    from_date / to_date: dd-mm-yyyy (to_date defaults to today)
    Returns list of {an_dt, desc, attchmntText} or [].
    """
    to_date = to_date or datetime.now().strftime("%d-%m-%Y")
    params = {
        "index":      "equities",
        "symbol":     symbol,
        "from_date":  from_date,
        "to_date":    to_date,
    }
    for attempt in range(3):
        try:
            r = session.get(NSE_URL, params=params, headers=NSE_HEADERS, timeout=15)
            if r.ok:
                data = r.json()
                if isinstance(data, list):
                    return data
                return []
            # 403 / session expired — refresh cookies and retry
            if r.status_code in (403, 401):
                logger.warning(f"{symbol}: session expired (HTTP {r.status_code}), refreshing…")
                session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
                time.sleep(2)
                continue
            logger.warning(f"{symbol}: HTTP {r.status_code}")
            return []
        except Exception as e:
            wait = 3 * (attempt + 1)
            logger.warning(f"{symbol}: attempt {attempt+1}/3 error — {e} — retrying in {wait}s")
            time.sleep(wait)
    return []


# ---------------------------------------------------------------------------
# Main backfill
# ---------------------------------------------------------------------------

def already_has_news(symbol_ns: str) -> bool:
    """Return True if we already have NSE announcement rows for this symbol."""
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT COUNT(*) FROM news_sentiment WHERE symbol = ? AND source = 'NSE'",
            (symbol_ns,)
        )
        return (cur.fetchone()[0] or 0) > 0
    finally:
        release_connection(conn)


def earliest_covered(symbol_ns: str) -> Optional[datetime]:
    """Oldest NSE announcement already stored for this symbol, or None.

    What "already done" means for a backfill that can be re-run with an earlier
    --from-date. already_has_news() only answers "any rows at all", which would
    skip every symbol the moment the first 2023 pass landed and make a deeper
    backfill a no-op.
    """
    conn = get_connection()
    try:
        cur = _execute(conn,
            "SELECT MIN(published_at) FROM news_sentiment WHERE symbol = ? AND source = 'NSE'",
            (symbol_ns,)
        )
        row = cur.fetchone()
        return row[0] if row else None
    finally:
        release_connection(conn)


def _ann_url(symbol: str, pub_dt: datetime, headline: str) -> str:
    """A URL unique to one announcement.

    NSE's API returns no per-announcement permalink, but dedupe rides on the
    uq_news_url_pubdate unique index (url, published_at) — so a URL that is
    just the symbol endpoint collapses every announcement a company filed on
    the same day into a single stored row. A company routinely files several
    (results + board meeting + disclosure), and the ones being dropped were the
    substantive ones. The headline digest makes each row distinct while staying
    stable across re-runs, so re-fetching a window is still idempotent.
    """
    digest = hashlib.sha1(headline.encode("utf-8", "replace")).hexdigest()[:12]
    return (f"https://www.nseindia.com/api/corporate-announcements"
            f"?symbol={symbol}&an_dt={pub_dt.date()}&ann={digest}")


def collect_daily(lookback_days: int = 2) -> dict:
    """
    Incremental daily job: fetch announcements from the last `lookback_days` days
    for all 499 stocks. Designed for the daily scheduler (runs after market close).

    Returns dict with total_rows, processed, failed counts.
    """
    init_database()

    from_dt = (datetime.now() - timedelta(days=lookback_days)).strftime("%d-%m-%Y")

    if not os.path.exists(_TOKENS_FILE):
        # Active constituents only — no point fetching announcements for names
        # that left the index or for the index tickers themselves.
        from database.db import get_active_universe
        all_symbols = [s.replace(".NS", "") for s in get_active_universe()]
    else:
        with open(_TOKENS_FILE) as f:
            all_symbols = list(json.load(f).keys())

    session    = _make_session()
    total_rows = 0
    processed  = 0
    failed     = 0

    for symbol in all_symbols:
        symbol_ns = f"{symbol}.NS"
        try:
            announcements = fetch_announcements(session, symbol, from_dt)
            if not announcements:
                time.sleep(0.3)
                continue

            headlines = []
            for ann in announcements:
                desc = ann.get("desc", "") or ""
                text = ann.get("attchmntText", "") or ""
                headline = (text[:400] if text.strip() else desc).strip() or "Corporate announcement"
                headlines.append(headline)

            scores = _FinBERT.score(headlines)

            rows = []
            for ann, headline, score in zip(announcements, headlines, scores):
                raw_dt = ann.get("an_dt", "") or ann.get("sort_date", "")
                if not raw_dt:
                    continue
                try:
                    pub_dt = (datetime.strptime(raw_dt[:10], "%Y-%m-%d")
                              if "T" in raw_dt
                              else datetime.strptime(raw_dt[:11], "%d-%b-%Y"))
                except ValueError:
                    continue
                rows.append((
                    headline[:500], "NSE",
                    pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    symbol_ns,
                    str(score["sentiment"]), score["confidence"],
                    _ann_url(symbol, pub_dt, headline),
                ))

            if rows:
                conn = get_connection()
                try:
                    from database.db import _executemany as _em
                    _em(conn,
                        """INSERT INTO news_sentiment
                           (headline, source, published_at, symbol, sentiment, confidence, url)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT DO NOTHING""",
                        rows,
                    )
                    conn.commit()
                    total_rows += len(rows)
                    processed  += 1
                except Exception as e:
                    conn.rollback()
                    logger.warning(f"{symbol} insert error: {e}")
                    failed += 1
                finally:
                    release_connection(conn)

        except Exception as e:
            logger.warning(f"{symbol}: {e}")
            failed += 1

        time.sleep(0.3)

    logger.info(f"NSE daily done: {total_rows} rows, {processed} stocks, {failed} errors")
    return {"total_rows": total_rows, "processed": processed, "failed": failed}


def backfill_all(from_date: str = FROM_DATE, symbol_filter: Optional[str] = None,
                 skip_existing: bool = True, start_idx: int = 0, end_idx: int = None,
                 to_date: Optional[str] = None, symbols: Optional[List[str]] = None,
                 score: bool = True) -> Dict:
    """Backfill NSE announcements over [from_date, to_date] (both dd-mm-yyyy).

    score=False stores rows with sentiment NULL for a later batch FinBERT pass
    (gdelt_collector.score_pending_news). Parallel shards must use it: scoring
    claims globally-unscored rows, so concurrent scorers duplicate each other's
    work — and it keeps torch off the critical path in every shard.

    skip_existing compares against what is already stored: a symbol is skipped
    only when its oldest stored announcement is already at or before from_date,
    so re-running with an earlier from_date genuinely deepens coverage.
    """
    init_database()

    # Load all symbols
    if symbols is not None:
        all_symbols = [s.replace(".NS", "") for s in symbols]
    elif not os.path.exists(_TOKENS_FILE):
        # Fall back: active constituents from the DB (not every symbol in
        # prices, which includes de-indexed names and index tickers).
        from database.db import get_active_universe
        all_symbols = [s.replace(".NS", "") for s in get_active_universe()]
    else:
        with open(_TOKENS_FILE) as f:
            all_symbols = list(json.load(f).keys())

    if symbol_filter:
        all_symbols = [s for s in all_symbols if s.upper() == symbol_filter.upper()]
        if not all_symbols:
            logger.error(f"Symbol {symbol_filter} not found")
            sys.exit(1)

    # Slice for parallel workers
    if end_idx is None:
        end_idx = len(all_symbols)
    all_symbols = all_symbols[start_idx:end_idx]

    total = len(all_symbols)
    window = f"{from_date} to {to_date or 'today'}"
    logger.info(f"Backfilling NSE announcements for {total} stocks, {window}"
                f"{'' if score else ' (unscored)'}…")

    from_dt = datetime.strptime(from_date, "%d-%m-%Y")
    session      = _make_session()
    total_rows   = 0
    failed       = []
    empty        = []
    skipped      = 0

    for idx, symbol in enumerate(all_symbols, 1):
        symbol_ns = f"{symbol}.NS"

        if skip_existing:
            covered = earliest_covered(symbol_ns)
            # Grace window, not an exact-date match — see COVERAGE_GRACE_DAYS in
            # the BSE collector for why the strict form skipped nothing and made
            # every re-run a full re-fetch.
            if covered and covered.replace(tzinfo=None) <= from_dt + timedelta(days=COVERAGE_GRACE_DAYS):
                skipped += 1
                continue

        try:
            announcements = fetch_announcements(session, symbol, from_date, to_date)
            if not announcements:
                # Not always transient: a symbol whose listed entity is younger
                # than the window has no history under this ticker at all
                # (TATAMOTORS returns nothing for 2018-2022 while TATASTEEL
                # returns 1,254). Surfaced rather than logged at debug so an
                # empty result is a fact in the summary, not a silent skip.
                logger.info(f"[{idx}/{total}] {symbol}: 0 announcements in window")
                empty.append(symbol)
                time.sleep(SLEEP_BETWEEN_STOCKS)
                continue

            # Build headlines for FinBERT
            headlines = []
            for ann in announcements:
                desc  = ann.get("desc", "") or ""
                text  = ann.get("attchmntText", "") or ""
                # Use full text if available, else just desc
                headline = (text[:400] if text.strip() else desc).strip()
                if not headline:
                    headline = desc or "Corporate announcement"
                headlines.append(headline)

            # Score all headlines in one FinBERT batch call — unless this is a
            # fetch-only shard, where a later single pass does the scoring.
            scores = (_FinBERT.score(headlines) if score
                      else [{"sentiment": None, "confidence": None}] * len(headlines))

            # Build rows for batch insert — one DB round-trip per stock
            rows = []
            for ann, headline, sc in zip(announcements, headlines, scores):
                raw_dt = ann.get("an_dt", "") or ann.get("sort_date", "")
                if not raw_dt:
                    continue
                try:
                    if "T" in raw_dt:
                        pub_dt = datetime.strptime(raw_dt[:10], "%Y-%m-%d")
                    else:
                        pub_dt = datetime.strptime(raw_dt[:11], "%d-%b-%Y")
                except ValueError:
                    continue
                rows.append((
                    headline[:500],
                    "NSE",
                    pub_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    symbol_ns,
                    None if sc["sentiment"] is None else str(sc["sentiment"]),
                    sc["confidence"],
                    _ann_url(symbol, pub_dt, headline),
                ))

            # Single batch insert for entire stock — 1 DB connection instead of N
            inserted = 0
            if rows:
                from database.db import get_connection as _gc, _executemany as _em
                # Retry once, then raise so the outer handler records this
                # symbol in `failed`. Swallowing the error here used to drop a
                # symbol's whole 2018-2022 window while the run still reported
                # success — see the same fix in the BSE collector's store().
                last_exc = None
                for attempt in (1, 2):
                    conn = _gc()
                    try:
                        _em(conn,
                            """INSERT INTO news_sentiment
                               (headline, source, published_at, symbol, sentiment, confidence, url)
                               VALUES (?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT DO NOTHING""",
                            rows,
                        )
                        conn.commit()
                        inserted = len(rows)
                        last_exc = None
                        break
                    except Exception as e:
                        last_exc = e
                        # rollback on a dead connection raises and would mask e
                        try:
                            conn.rollback()
                        except Exception:
                            pass
                        logger.warning(f"{symbol} batch insert attempt {attempt}/2 failed: {e}")
                    finally:
                        release_connection(conn)
                if last_exc is not None:
                    raise RuntimeError(
                        f"{symbol} batch insert failed after 2 attempts: {last_exc}")

            total_rows += inserted
            logger.info(f"[{idx}/{total}] {symbol}: {len(announcements)} announcements → {inserted} stored")

        except Exception as e:
            logger.error(f"[{idx}/{total}] {symbol}: {e}")
            failed.append(symbol)

        time.sleep(SLEEP_BETWEEN_STOCKS)

    print(f"\n{'='*60}")
    print(f"✅ NSE backfill complete")
    print(f"   Stocks processed : {total - len(failed) - skipped - len(empty)}/{total}")
    print(f"   Skipped (already had data): {skipped}")
    print(f"   Total rows inserted: {total_rows:,}")
    if empty:
        print(f"   ⚠️  No announcements in window ({len(empty)}): {', '.join(empty[:20])}")
    if failed:
        print(f"   ❌ Failed ({len(failed)}): {', '.join(failed[:20])}")
    print(f"{'='*60}\n")

    return {"rows": total_rows, "processed": total - len(failed) - skipped - len(empty),
            "skipped": skipped, "failed": failed, "empty": empty}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE announcements sentiment backfill")
    parser.add_argument("--symbol",       type=str, default=None,      help="Single symbol e.g. HDFCBANK")
    parser.add_argument("--from-date",    type=str, default=FROM_DATE, help="Start date dd-mm-yyyy (default 01-01-2023)")
    parser.add_argument("--to-date",      type=str, default=None,      help="End date dd-mm-yyyy (default today)")
    parser.add_argument("--no-skip",    action="store_true",       help="Re-process symbols that already have data")
    parser.add_argument("--fetch-only", action="store_true",       help="Store unscored; score later with a single FinBERT pass")
    parser.add_argument("--start-idx",  type=int, default=0,       help="Start from this symbol index (for parallel runs)")
    parser.add_argument("--end-idx",    type=int, default=None,    help="Stop at this symbol index (exclusive)")
    args = parser.parse_args()

    backfill_all(
        from_date     = args.from_date,
        to_date       = args.to_date,
        symbol_filter = args.symbol,
        skip_existing = not args.no_skip,
        start_idx     = args.start_idx,
        end_idx       = args.end_idx,
        score         = not args.fetch_only,
    )
