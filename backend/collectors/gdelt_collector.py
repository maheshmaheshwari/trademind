"""
TradeMind AI — GDELT News Collector

Bootstraps 5 years of news articles from the GDELT Project v2 API for all
499 Nifty 500 stocks, stores headlines in the news_sentiment table, and
provides a batch FinBERT scoring pass for articles that have no sentiment yet.

Functions:
    fetch_gdelt_month(company_name, symbol, year, month)
        — fetch one month of articles from GDELT for a single company

    bootstrap_gdelt(from_year, from_month, only_missing)
        — iterate all stocks × all months and persist via insert_news()

    score_pending_news(batch_limit)
        — run FinBERT on rows where sentiment IS NULL, update DB

CLI:
    python gdelt_collector.py
        — full bootstrap from 2021-01-01 for all stocks

    python gdelt_collector.py --from-year 2024 --from-month 6 --symbol TCS
        — partial fill for a single stock from a specific month
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import requests
from dotenv import load_dotenv

# ---- path bootstrap so this file can be run directly ----
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import insert_news, get_connection, release_connection, init_database, _execute

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_SLEEP = 12.0         # seconds between requests (conservative — GDELT enforces 1 req/5s but bans on bursts)
_TOKENS_FILE = os.path.join(_BACKEND_DIR, "data", "angel_tokens.json")
_MAX_RECORDS = 250

# Circuit breaker. A 429 from GDELT is indistinguishable, per-request, from
# "busy right now" — but the API also refuses traffic wholesale for long
# stretches (observed 2026-08-03: every request 429'd, from CI and from a
# residential IP, via requests and curl, with any User-Agent and any query,
# while gdeltproject.org itself served 200). Its own 429 body says as much:
# "All high-traffic users should switch to our ngrams dataset."
#
# Without a breaker, that state costs 12 + 60 + 120 + 180 = 6.2 min PER MONTH
# fetched and never terminates early: one shard ground through 19 months in 2
# hours, wrote nothing, and had to be cancelled by hand. Retrying harder against
# a wholesale block is pure waste, so give up quickly and fail loudly instead.
_MAX_CONSECUTIVE_429_MONTHS = 5

# Reset by any successful (non-429) response; see fetch_gdelt_month.
_consecutive_429_months = 0


class GdeltUnavailable(RuntimeError):
    """GDELT refused enough consecutive requests that the run should abort."""


# ---------------------------------------------------------------------------
# Token map loader
# ---------------------------------------------------------------------------
def _load_token_map() -> Dict[str, Dict[str, Any]]:
    """Load the angel_tokens.json symbol→metadata map."""
    try:
        with open(_TOKENS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"angel_tokens.json not found at {_TOKENS_FILE}")
        return {}


# ---------------------------------------------------------------------------
# Public function: fetch_gdelt_month
# ---------------------------------------------------------------------------
def fetch_gdelt_month(
    company_name: str,
    symbol: str,
    year: int,
    month: int,
) -> List[Dict[str, Any]]:
    """
    Fetch one calendar month of news articles from GDELT for a company.

    Args:
        company_name: Full company name, e.g. "Tata Consultancy Services Ltd."
        symbol:       NSE ticker, e.g. "TCS"
        year:         4-digit year, e.g. 2023
        month:        Month number 1–12

    Returns:
        List of dicts ready for insert_news(), each containing:
            headline, source, published_at, symbol, url
        sentiment and confidence are None (filled later by score_pending_news).
    """
    # Build start / end datetimes
    start_dt = datetime(year, month, 1)
    if month == 12:
        end_dt = datetime(year + 1, 1, 1)
    else:
        end_dt = datetime(year, month + 1, 1)

    # GDELT datetime format: YYYYMMDDHHMMSS
    start_str = start_dt.strftime("%Y%m%d%H%M%S")
    end_str   = end_dt.strftime("%Y%m%d%H%M%S")

    params = {
        "query":         company_name,
        "mode":          "artlist",
        "maxrecords":    _MAX_RECORDS,
        "startdatetime": start_str,
        "enddatetime":   end_str,
        "format":        "json",
        "sort":          "DateDesc",
    }

    global _consecutive_429_months

    for attempt in range(3):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=20)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning(f"[{symbol}] {year}-{month:02d}: 429 rate limit — sleeping {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            _consecutive_429_months = 0     # a live response clears the breaker
            break
        except requests.exceptions.JSONDecodeError:
            # GDELT sometimes returns an empty body when there are no results
            logger.debug(f"[{symbol}] {year}-{month:02d}: empty/invalid JSON — skipping")
            return []
        except Exception as exc:
            logger.warning(f"[{symbol}] {year}-{month:02d} GDELT error: {exc}")
            return []
    else:
        _consecutive_429_months += 1
        logger.warning(f"[{symbol}] {year}-{month:02d}: exhausted retries "
                       f"({_consecutive_429_months} consecutive)")
        if _consecutive_429_months >= _MAX_CONSECUTIVE_429_MONTHS:
            raise GdeltUnavailable(
                f"GDELT returned 429 for {_consecutive_429_months} consecutive "
                f"month-fetches, exhausting all retries each time. The API is "
                f"refusing traffic; continuing would burn "
                f"~{(12 + 60 + 120 + 180) / 60:.0f} min per month for no data."
            )
        return []

    try:
        payload = resp.json()
    except requests.exceptions.JSONDecodeError:
        logger.debug(f"[{symbol}] {year}-{month:02d}: empty/invalid JSON — skipping")
        return []

    articles = payload.get("articles")
    if not articles:
        logger.debug(f"[{symbol}] {year}-{month:02d}: no articles")
        return []

    results: List[Dict[str, Any]] = []
    for art in articles:
        raw_date = art.get("seendate", "")
        # seendate format: "20240115T143000Z"
        try:
            published_at = datetime.strptime(raw_date, "%Y%m%dT%H%M%SZ").strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        except ValueError:
            logger.debug("[%s] Skipping article with unparseable date: %r", symbol, raw_date)
            continue  # discard rather than persist raw API string

        headline = (art.get("title") or "").strip()
        if not headline:
            continue

        results.append({
            "headline":     headline,
            "source":       art.get("domain") or None,
            "published_at": published_at,
            "symbol":       symbol,
            "url":          art.get("url") or None,
        })

    logger.debug(f"[{symbol}] {year}-{month:02d}: {len(results)} articles")
    return results


# ---------------------------------------------------------------------------
# Helper: get symbols already present in news_sentiment
# ---------------------------------------------------------------------------
def _get_symbols_with_news() -> Set[str]:
    """Return the set of symbols that already have at least one news row."""
    try:
        conn = get_connection()
        cur = _execute(conn, "SELECT DISTINCT symbol FROM news_sentiment WHERE symbol IS NOT NULL")
        symbols = {row[0] for row in cur.fetchall()}
        release_connection(conn)
        return symbols
    except Exception as exc:
        logger.warning(f"Could not query existing news symbols: {exc}")
        return set()


# ---------------------------------------------------------------------------
# Helper: month iterator
# ---------------------------------------------------------------------------
def _iter_months(from_year: int, from_month: int):
    """Yield (year, month) tuples from (from_year, from_month) up to today."""
    now = datetime.now()
    y, m = from_year, from_month
    while (y, m) <= (now.year, now.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ---------------------------------------------------------------------------
# Public function: bootstrap_gdelt
# ---------------------------------------------------------------------------
def bootstrap_gdelt(
    from_year: int = 2021,
    from_month: int = 1,
    only_missing: bool = False,
    only_symbol: Optional[str] = None,
) -> None:
    """
    Fetch GDELT news for all Nifty 500 stocks from a given start date.

    Args:
        from_year:    Starting year (default 2021).
        from_month:   Starting month (default 1 = January).
        only_missing: If True, skip symbols that already have any news rows.
        only_symbol:  If provided, process only this NSE ticker (for partial fills).
    """
    init_database()
    token_map = _load_token_map()

    if not token_map:
        logger.error("Token map empty — aborting bootstrap_gdelt")
        return

    # Filter to single symbol if requested
    if only_symbol:
        upper = only_symbol.upper()
        if upper not in token_map:
            print(f"Symbol {upper} not found in angel_tokens.json")
            return
        token_map = {upper: token_map[upper]}

    # Optionally skip symbols already in DB
    existing: Set[str] = set()
    if only_missing:
        existing = _get_symbols_with_news()
        logger.info(f"only_missing: {len(existing)} symbols already have news — will skip")

    stock_list = [
        (sym, info) for sym, info in token_map.items()
        if sym not in existing
    ]

    total_stocks = len(stock_list)
    months_list = list(_iter_months(from_year, from_month))
    total_months = len(months_list)

    print(
        f"GDELT bootstrap: {total_stocks} stocks × {total_months} months "
        f"(from {from_year}-{from_month:02d})"
    )

    for stock_idx, (symbol, info) in enumerate(stock_list, 1):
        company_name: str = info.get("name") or symbol  # fallback to ticker
        if not company_name:
            logger.warning(f"[{symbol}] no name field — using symbol as query")
            company_name = symbol

        stock_total = 0
        for year, month in months_list:
            articles = fetch_gdelt_month(company_name, symbol, year, month)

            for art in articles:
                try:
                    insert_news(
                        headline=art["headline"],
                        source=art.get("source"),
                        published_at=art.get("published_at"),
                        symbol=art["symbol"],
                        sentiment=None,
                        confidence=None,
                        url=art.get("url"),
                    )
                    stock_total += 1
                except Exception as exc:
                    logger.warning(f"[{symbol}] insert_news error: {exc}")

            time.sleep(_GDELT_SLEEP)

        if stock_idx % 10 == 0 or stock_idx == total_stocks:
            pct = stock_idx / total_stocks * 100
            print(
                f"  [{stock_idx:>4}/{total_stocks}] ({pct:5.1f}%) "
                f"{symbol:15s} — {stock_total} articles inserted so far"
            )
        else:
            logger.info(f"[{symbol}] {stock_total} articles inserted")

    print("GDELT bootstrap complete.")


# ---------------------------------------------------------------------------
# Public function: score_pending_news
# ---------------------------------------------------------------------------
def score_pending_news(batch_limit: int = 2000, shard: Optional[str] = None) -> int:
    """
    Score news headlines that have no sentiment yet using FinBERT batch inference.

    Fetches up to batch_limit rows where sentiment IS NULL, processes all
    headlines in batched forward passes (32 per pass), then writes results
    back to DB in a single executemany — ~20× faster than one-at-a-time.

    Args:
        batch_limit: Maximum number of headlines to process per call.
        shard:       "i/N" — claim only rows where MOD(id, N) = i-1. Without it
                     this claims EVERY globally unscored row, so two concurrent
                     callers process the same headlines twice. Sharding on the
                     primary key partitions the backlog with no coordination
                     and no shared cursor.

                     This exists because FinBERT on a CPU runner measures ~2.7
                     rows/sec: a 315k-row backfill is ~32 hours, which no single
                     6-hour CI job can finish. Scoring has to fan out the same
                     way fetching does.

    Returns:
        Number of headlines successfully scored.
    """
    try:
        from analysis.sentiment import analyze_sentiment_batch
    except ImportError:
        logger.error("analysis.sentiment not found — cannot score pending news")
        return 0

    # MOD(), not the % operator: db.py's _execute rewrites ? to %s, so a literal
    # % in the SQL collides with psycopg2's own placeholder parsing.
    # published_at comes along for the UPDATE's WHERE clause, not for scoring.
    # news_sentiment is a hypertable partitioned on published_at with 203 chunks
    # after the 2010-2022 backfill, and there is NO index on id. So
    # "WHERE id = ?" alone cannot exclude chunks and sequential-scans all 203 to
    # find one row: EXPLAIN shows 204 scan nodes at cost 27,176, versus 3 nodes
    # at cost 126 once published_at is included — 216x cheaper.
    #
    # This, not FinBERT, was the real cost of scoring. The measured ~2.7 rows/sec
    # that the shard count was sized around was the write path scanning the whole
    # hypertable per row, which is also why it was identical on a CPU runner and
    # on a local GPU.
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        sql = ("SELECT id, headline, published_at FROM news_sentiment "
               "WHERE sentiment IS NULL AND MOD(id, ?) = ? LIMIT ?")
        params = (n, i - 1, batch_limit)
    else:
        sql = ("SELECT id, headline, published_at FROM news_sentiment "
               "WHERE sentiment IS NULL LIMIT ?")
        params = (batch_limit,)

    conn = get_connection()
    try:
        rows = _execute(conn, sql, params).fetchall()
    except Exception as exc:
        logger.error(f"score_pending_news: DB read error: {exc}")
        release_connection(conn)
        return 0

    if not rows:
        logger.info("score_pending_news: no pending rows")
        release_connection(conn)
        return 0

    logger.info(f"score_pending_news: scoring {len(rows)} headlines (batch mode)")

    ids       = [r[0] for r in rows]
    headlines = [r[1] or "" for r in rows]
    pubs      = [r[2] for r in rows]

    try:
        scored_pairs = analyze_sentiment_batch(headlines)
    except Exception as exc:
        logger.error(f"score_pending_news: batch inference failed: {exc}")
        release_connection(conn)
        return 0

    updates = [
        (score_str, float(conf), row_id, pub)
        for row_id, pub, (score_str, conf) in zip(ids, pubs, scored_pairs)
    ]

    # Retry once, then RAISE. Returning 0 here was actively harmful: run_scoring
    # loops until a call returns fewer than batch_limit, so a single failed
    # write ended the whole shard's loop and the driver then reported "scoring
    # complete" and exited 0 — a green check over an abandoned shard. It also
    # discarded a batch of finished FinBERT inference (~12 min of compute at the
    # measured 2.7 rows/sec), since the headlines are scored before the write.
    #
    # Callers are safe: both scheduler jobs wrap this in try/except and log.
    from database.db import _executemany
    last_exc = None
    for attempt in (1, 2):
        try:
            _executemany(conn,
                "UPDATE news_sentiment SET sentiment=?, confidence=? "
                "WHERE id=? AND published_at=?",
                updates,
            )
            conn.commit()
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
            # Guarded — rollback on a dead connection raises and would mask exc.
            try:
                conn.rollback()
            except Exception:
                pass
            logger.warning(
                f"score_pending_news: write attempt {attempt}/2 failed: {exc}")
            release_connection(conn)
            conn = get_connection() if attempt == 1 else None

    if last_exc is not None:
        raise RuntimeError(
            f"score_pending_news: write failed after 2 attempts "
            f"({len(updates)} scored headlines lost): {last_exc}")

    release_connection(conn)
    logger.info(f"score_pending_news: {len(updates)} headlines scored")
    return len(updates)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="TradeMind GDELT news collector")
    parser.add_argument(
        "--from-year",
        type=int,
        default=2021,
        help="Start year for history (default: 2021)",
    )
    parser.add_argument(
        "--from-month",
        type=int,
        default=1,
        help="Start month for history (default: 1)",
    )
    parser.add_argument(
        "--symbol",
        default=None,
        help="Fetch only this NSE ticker (e.g. TCS). Omit for all 499 stocks.",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Skip symbols that already have news in the DB",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run FinBERT scoring pass on pending headlines instead of fetching",
    )
    parser.add_argument(
        "--batch-limit",
        type=int,
        default=500,
        help="Max headlines to score per --score run (default: 500)",
    )
    args = parser.parse_args()

    if args.score:
        print(f"Scoring up to {args.batch_limit} pending headlines with FinBERT ...")
        n = score_pending_news(batch_limit=args.batch_limit)
        print(f"Done — {n} headlines scored.")
    else:
        bootstrap_gdelt(
            from_year=args.from_year,
            from_month=args.from_month,
            only_missing=args.only_missing,
            only_symbol=args.symbol,
        )
