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

from database.db import insert_news, get_local_connection, init_database

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
_GDELT_SLEEP = 12.0         # seconds between requests (conservative — GDELT enforces 1 req/5s but bans on bursts)
_TOKENS_FILE = os.path.join(_BACKEND_DIR, "data", "angel_tokens.json")
_MAX_RECORDS = 250


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

    for attempt in range(3):
        try:
            resp = requests.get(_GDELT_URL, params=params, timeout=20)
            if resp.status_code == 429:
                wait = 60 * (attempt + 1)
                logger.warning(f"[{symbol}] {year}-{month:02d}: 429 rate limit — sleeping {wait}s (attempt {attempt+1}/3)")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.JSONDecodeError:
            # GDELT sometimes returns an empty body when there are no results
            logger.debug(f"[{symbol}] {year}-{month:02d}: empty/invalid JSON — skipping")
            return []
        except Exception as exc:
            logger.warning(f"[{symbol}] {year}-{month:02d} GDELT error: {exc}")
            return []
    else:
        logger.warning(f"[{symbol}] {year}-{month:02d}: exhausted retries — skipping")
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
            published_at = raw_date  # keep as-is if unparseable

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
        conn = get_local_connection()
        cur = conn.execute(
            "SELECT DISTINCT symbol FROM news_sentiment WHERE symbol IS NOT NULL"
        )
        symbols = {row[0] for row in cur.fetchall()}
        conn.close()
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
def score_pending_news(batch_limit: int = 500) -> int:
    """
    Score news headlines that have no sentiment yet using FinBERT.

    Reads up to batch_limit rows where sentiment IS NULL, calls
    analysis.sentiment.analyze_sentiment() on each headline, and
    writes sentiment + confidence back to the DB.

    Args:
        batch_limit: Maximum number of headlines to process per call.

    Returns:
        Number of headlines successfully scored.
    """
    # Dynamically import FinBERT scorer — may not be installed in all envs
    try:
        from analysis.sentiment import analyze_sentiment  # type: ignore
    except ImportError:
        logger.error(
            "analysis.sentiment module not found — install FinBERT or implement "
            "analysis/sentiment.py with analyze_sentiment(text) -> (label, score)"
        )
        return 0

    conn = get_local_connection()
    try:
        rows = conn.execute(
            "SELECT id, headline FROM news_sentiment WHERE sentiment IS NULL LIMIT ?",
            (batch_limit,),
        ).fetchall()
    except Exception as exc:
        logger.error(f"score_pending_news: DB read error: {exc}")
        conn.close()
        return 0

    if not rows:
        logger.info("score_pending_news: no pending rows")
        conn.close()
        return 0

    logger.info(f"score_pending_news: scoring {len(rows)} headlines")
    scored = 0

    for row_id, headline in rows:
        try:
            sentiment, confidence = analyze_sentiment(headline)
            conn.execute(
                "UPDATE news_sentiment SET sentiment=?, confidence=? WHERE id=?",
                (sentiment, float(confidence), row_id),
            )
            scored += 1
        except Exception as exc:
            logger.warning(f"Failed to score row {row_id}: {exc}")

    try:
        conn.commit()
    except Exception as exc:
        logger.error(f"score_pending_news: commit error: {exc}")

    conn.close()
    logger.info(f"score_pending_news: {scored}/{len(rows)} headlines scored")
    print(f"Scored {scored} headlines.")
    return scored


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
