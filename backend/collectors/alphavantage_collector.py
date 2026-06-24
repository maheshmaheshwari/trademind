"""
Alpha Vantage News & Sentiment Collector for Nifty 500 stocks.

Fetches news articles with sentiment scores from the Alpha Vantage
NEWS_SENTIMENT endpoint (free tier: 25 requests/day).

Coverage note: Alpha Vantage only indexes US-listed equities. Indian NSE-only
stocks return 0 articles. We maintain a mapping of NSE symbols → US ADR tickers
(e.g. HDFCBANK.NS → HDB) for the ~7 Indian large-caps with NYSE/NASDAQ ADR
listings. The av_coverage_tracker table tracks rotation so each stock is
refreshed every ≤7 days.

To add more ADR mappings, insert into av_coverage_tracker:
    INSERT INTO av_coverage_tracker (nse_symbol, adr_ticker) VALUES ('XYZ.NS', 'XYZ');

Usage:
    cd backend && source venv/bin/activate
    python collectors/alphavantage_collector.py              # batch of 7 (all ADRs)
    python collectors/alphavantage_collector.py --symbol HDFCBANK  # single stock
"""

import argparse
import logging
import os
import sys
import time
from datetime import date, datetime
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from database.db import get_connection, release_connection, insert_news, _execute

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

AV_BASE_URL = "https://www.alphavantage.co/query"
RATE_LIMIT_SECS = 3


_LABEL_MAP = {
    "Bullish": "positive",
    "Somewhat-Bullish": "positive",
    "Bearish": "negative",
    "Somewhat-Bearish": "negative",
    "Neutral": "neutral",
}


def _map_sentiment(label: str) -> str:
    return _LABEL_MAP.get(label, "neutral")


def _parse_av_timestamp(ts: str) -> str:
    try:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return ts


def _get_rotation_batch(max_requests: int) -> List[Dict]:
    """
    Return the least-recently-covered ADR stocks from av_coverage_tracker,
    up to max_requests entries.

    Returns list of dicts: {nse_symbol, adr_ticker, last_covered}
    """
    conn = get_connection()
    try:
        cur = _execute(conn, """
            SELECT nse_symbol, adr_ticker, last_covered
            FROM av_coverage_tracker
            ORDER BY last_covered ASC NULLS FIRST
            LIMIT %s
        """, (max_requests,))
        rows = cur.fetchall()
        return [{"nse_symbol": r[0], "adr_ticker": r[1], "last_covered": r[2]} for r in rows]
    finally:
        release_connection(conn)


def _update_coverage(nse_symbol: str, articles_added: int) -> None:
    """Mark a symbol as covered today in av_coverage_tracker."""
    conn = get_connection()
    try:
        _execute(conn, """
            UPDATE av_coverage_tracker
            SET last_covered    = %s,
                articles_total  = articles_total + %s,
                attempt_count   = attempt_count + 1
            WHERE nse_symbol = %s
        """, (date.today(), articles_added, nse_symbol))
        conn.commit()
    finally:
        release_connection(conn)


def fetch_av_news(adr_ticker: str, nse_symbol: str) -> List[Dict]:
    """
    Fetch news + sentiment from Alpha Vantage for one stock.

    Uses the US ADR ticker (e.g. 'HDB') to query AV, then tags articles
    with the NSE symbol (e.g. 'HDFCBANK.NS') for storage.

    Returns list of article dicts ready for insert_news().
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": adr_ticker,
        "limit": 200,
        "apikey": api_key,
    }

    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error(f"HTTP error fetching AV news for {adr_ticker}: {e}")
        return []

    if "Information" in payload:
        logger.warning(f"Alpha Vantage rate limit: {payload['Information']}")
        return []

    feed = payload.get("feed", [])
    articles = []

    for item in feed:
        title = item.get("title", "").strip()
        source = item.get("source", "").strip()
        published_raw = item.get("time_published", "")
        url = item.get("url", "").strip()
        published_at = _parse_av_timestamp(published_raw)

        # Find sentiment entry for this ADR ticker
        ticker_sentiments = item.get("ticker_sentiment", [])
        matched = next(
            (ts for ts in ticker_sentiments
             if ts.get("ticker", "").upper() == adr_ticker.upper()),
            None,
        )
        if matched is None:
            continue

        label = matched.get("ticker_sentiment_label", "Neutral")
        sentiment = _map_sentiment(label)
        try:
            confidence = float(matched.get("relevance_score", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        articles.append({
            "headline": title,
            "source": "alphavantage",
            "published_at": published_at,
            "symbol": nse_symbol,  # store as NSE symbol for model features
            "sentiment": sentiment,
            "confidence": confidence,
            "url": url,
        })

    logger.info(f"  {adr_ticker} ({nse_symbol}): {len(articles)} articles fetched")
    return articles


def collect_av_batch(
    symbols: Optional[List[str]] = None,
    max_requests: int = 25,
) -> Dict:
    """
    Fetch Alpha Vantage news using the rotation queue.

    Picks the least-recently-covered ADR stocks from av_coverage_tracker,
    processes up to max_requests, and updates last_covered after each call.

    Args:
        symbols: Optional explicit NSE symbol list (e.g. ['HDFCBANK.NS']).
                 If None, uses rotation queue order.
        max_requests: Max API calls (default 25, free-tier limit).

    Returns:
        Dict: processed, total_articles, remaining
    """
    if symbols is not None:
        # Manual override: look up ADR tickers for the given NSE symbols
        conn = get_connection()
        try:
            batch = []
            for nse_sym in symbols[:max_requests]:
                cur = _execute(conn,
                    "SELECT nse_symbol, adr_ticker FROM av_coverage_tracker WHERE nse_symbol = %s",
                    (nse_sym,))
                row = cur.fetchone()
                if row:
                    batch.append({"nse_symbol": row[0], "adr_ticker": row[1]})
                else:
                    logger.warning(f"{nse_sym}: no ADR mapping in av_coverage_tracker — skipping")
        finally:
            release_connection(conn)
        remaining_after = max(0, len(symbols) - len(batch))
    else:
        batch = _get_rotation_batch(max_requests)
        total_tracked = _execute(
            get_connection(), "SELECT COUNT(*) FROM av_coverage_tracker", ()
        ).fetchone()[0]
        remaining_after = max(0, total_tracked - len(batch))
        logger.info(
            f"Rotation queue: {len(batch)} stocks this run, {remaining_after} remaining"
        )

    processed = 0
    total_articles = 0

    for entry in batch:
        nse_sym = entry["nse_symbol"]
        adr = entry["adr_ticker"]
        try:
            articles = fetch_av_news(adr, nse_sym)
            inserted = 0
            for art in articles:
                if insert_news(
                    headline=art["headline"],
                    source=art["source"],
                    published_at=art["published_at"],
                    symbol=art["symbol"],
                    sentiment=art["sentiment"],
                    confidence=art["confidence"],
                    url=art["url"],
                ):
                    inserted += 1
            _update_coverage(nse_sym, inserted)
            total_articles += inserted
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process {nse_sym} ({adr}): {e}")

        time.sleep(RATE_LIMIT_SECS)

    result = {
        "processed": processed,
        "total_articles": total_articles,
        "remaining": remaining_after,
    }
    logger.info(f"AV batch complete: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Fetch Alpha Vantage news & sentiment for Indian ADR stocks"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="NSE symbol to fetch (e.g. HDFCBANK). Omit to run full rotation batch.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=25,
        help="Max API requests (default: 25, free tier limit).",
    )
    args = parser.parse_args()

    if args.symbol:
        nse_sym = args.symbol.upper()
        if not nse_sym.endswith(".NS"):
            nse_sym += ".NS"
        print(f"\nFetching AV news for: {nse_sym}")
        result = collect_av_batch(symbols=[nse_sym], max_requests=1)
        print(f"  Articles inserted: {result['total_articles']}")
    else:
        print(f"\nRunning Alpha Vantage rotation batch (max {args.max} requests)...")
        result = collect_av_batch(max_requests=args.max)
        print(f"\nDone:")
        print(f"  Symbols processed : {result['processed']}")
        print(f"  Articles inserted : {result['total_articles']}")
        print(f"  Symbols remaining : {result['remaining']}")


if __name__ == "__main__":
    main()
