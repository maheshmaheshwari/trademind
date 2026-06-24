"""
TradeMind — NSE Corporate Announcements Backfill

Fetches 3 years of official corporate announcements for all 499 Nifty 500
stocks from NSE India's API, runs FinBERT sentiment scoring, and stores
results in news_sentiment. No API key required.

API: GET https://www.nseindia.com/api/corporate-announcements
     Params: index=equities, symbol=HDFCBANK, from_date=01-01-2023, to_date=02-06-2026
     Returns: [{an_dt, desc, attchmntText, symbol}, ...]

Usage:
    PYTHONPATH=. python collectors/nse_announcements_collector.py
    PYTHONPATH=. python collectors/nse_announcements_collector.py --symbol HDFCBANK
    PYTHONPATH=. python collectors/nse_announcements_collector.py --from-date 2024-01-01
"""

import argparse
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
    retry = Retry(total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
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

def fetch_announcements(session: requests.Session, symbol: str, from_date: str) -> List[Dict]:
    """
    Fetch corporate announcements for one symbol from NSE.
    symbol: bare symbol without .NS, e.g. "HDFCBANK"
    from_date: dd-mm-yyyy
    Returns list of {an_dt, desc, attchmntText} or [].
    """
    to_date = datetime.now().strftime("%d-%m-%Y")
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


def collect_daily(lookback_days: int = 2) -> dict:
    """
    Incremental daily job: fetch announcements from the last `lookback_days` days
    for all 499 stocks. Designed for the daily scheduler (runs after market close).

    Returns dict with total_rows, processed, failed counts.
    """
    init_database()

    from_dt = (datetime.now() - timedelta(days=lookback_days)).strftime("%d-%m-%Y")

    if not os.path.exists(_TOKENS_FILE):
        conn = get_connection()
        try:
            cur = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d' ORDER BY symbol")
            all_symbols = [r[0].replace(".NS", "") for r in cur.fetchall()]
        finally:
            release_connection(conn)
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
            url = f"https://www.nseindia.com/api/corporate-announcements?symbol={symbol}"

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
                    f"{url}&an_dt={pub_dt.date()}",  # unique URL per announcement
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
                 skip_existing: bool = True, start_idx: int = 0, end_idx: int = None):
    init_database()

    # Load all symbols
    if not os.path.exists(_TOKENS_FILE):
        # Fall back: get symbols from prices table
        conn = get_connection()
        try:
            cur = _execute(conn, "SELECT DISTINCT symbol FROM prices WHERE interval='1d' ORDER BY symbol")
            all_symbols = [r[0].replace(".NS", "") for r in cur.fetchall()]
        finally:
            release_connection(conn)
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
    logger.info(f"Backfilling NSE announcements for {total} stocks from {from_date}…")

    session      = _make_session()
    total_rows   = 0
    failed       = []
    skipped      = 0

    for idx, symbol in enumerate(all_symbols, 1):
        symbol_ns = f"{symbol}.NS"

        if skip_existing and already_has_news(symbol_ns):
            skipped += 1
            continue

        try:
            announcements = fetch_announcements(session, symbol, from_date)
            if not announcements:
                logger.debug(f"[{idx}/{total}] {symbol}: 0 announcements")
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

            # Score all headlines in one FinBERT batch call
            scores = _FinBERT.score(headlines)

            # Build rows for batch insert — one DB round-trip per stock
            rows = []
            url = f"https://www.nseindia.com/api/corporate-announcements?symbol={symbol}"
            for ann, headline, score in zip(announcements, headlines, scores):
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
                    str(score["sentiment"]),
                    score["confidence"],
                    url,
                ))

            # Single batch insert for entire stock — 1 DB connection instead of N
            inserted = 0
            if rows:
                from database.db import get_connection as _gc, _executemany as _em
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
                except Exception as e:
                    conn.rollback()
                    logger.error(f"{symbol} batch insert failed: {e}")
                finally:
                    release_connection(conn)

            total_rows += inserted
            logger.info(f"[{idx}/{total}] {symbol}: {len(announcements)} announcements → {inserted} stored")

        except Exception as e:
            logger.error(f"[{idx}/{total}] {symbol}: {e}")
            failed.append(symbol)

        time.sleep(SLEEP_BETWEEN_STOCKS)

    print(f"\n{'='*60}")
    print(f"✅ NSE backfill complete")
    print(f"   Stocks processed : {total - len(failed) - skipped}/{total}")
    print(f"   Skipped (already had data): {skipped}")
    print(f"   Total rows inserted: {total_rows:,}")
    if failed:
        print(f"   ❌ Failed ({len(failed)}): {', '.join(failed[:20])}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE announcements sentiment backfill")
    parser.add_argument("--symbol",       type=str, default=None,      help="Single symbol e.g. HDFCBANK")
    parser.add_argument("--from-date",    type=str, default=FROM_DATE, help="Start date dd-mm-yyyy (default 01-01-2023)")
    parser.add_argument("--no-skip",    action="store_true",       help="Re-process symbols that already have data")
    parser.add_argument("--start-idx",  type=int, default=0,       help="Start from this symbol index (for parallel runs)")
    parser.add_argument("--end-idx",    type=int, default=None,    help="Stop at this symbol index (exclusive)")
    args = parser.parse_args()

    backfill_all(
        from_date     = args.from_date,
        symbol_filter = args.symbol,
        skip_existing = not args.no_skip,
        start_idx     = args.start_idx,
        end_idx       = args.end_idx,
    )
