"""
Alpha Vantage News & Sentiment Collector for Nifty 500 stocks.

Fetches news articles with sentiment scores from the Alpha Vantage
NEWS_SENTIMENT endpoint (free tier: 25 requests/day).

Usage:
    cd backend && source venv/bin/activate
    python collectors/alphavantage_collector.py              # batch of 25
    python collectors/alphavantage_collector.py --symbol TCS # single stock
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
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

# ==========================================
# Config
# ==========================================
AV_BASE_URL = "https://www.alphavantage.co/query"
TOKENS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "angel_tokens.json")
RATE_LIMIT_SECS = 3  # 25 req/day free tier — be conservative


# ==========================================
# Sentiment label mapping
# ==========================================
_LABEL_MAP = {
    "Bullish": "positive",
    "Somewhat-Bullish": "positive",
    "Bearish": "negative",
    "Somewhat-Bearish": "negative",
    "Neutral": "neutral",
}


def _map_sentiment(label: str) -> str:
    """Convert Alpha Vantage sentiment label to positive/negative/neutral."""
    return _LABEL_MAP.get(label, "neutral")


def _parse_av_timestamp(ts: str) -> str:
    """
    Convert Alpha Vantage timestamp '20240115T143000' to
    ISO-8601 string '2024-01-15T14:30:00'.
    """
    try:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return ts


def fetch_av_news(symbol: str, from_date: str = "20210101T0000") -> List[Dict]:
    """
    Fetch news + sentiment from Alpha Vantage for one NSE symbol.

    Args:
        symbol: NSE ticker without exchange prefix, e.g. "TCS".
        from_date: Earliest article date in 'YYYYMMDDTHHMM' format.

    Returns:
        List of dicts, each containing:
            headline, source, published_at, symbol, sentiment,
            confidence (float), url
    """
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set in environment.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": f"NSE:{symbol}",
        "time_from": from_date,
        "limit": 200,
        "apikey": api_key,
    }

    try:
        resp = requests.get(AV_BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as e:
        logger.error(f"HTTP error fetching AV news for {symbol}: {e}")
        return []

    # Alpha Vantage returns {"Information": "..."} when rate-limited
    if "Information" in payload:
        logger.warning(f"Alpha Vantage rate limit / info message: {payload['Information']}")
        return []

    feed = payload.get("feed", [])
    articles = []

    for item in feed:
        title = item.get("title", "").strip()
        source = item.get("source", "").strip()
        published_raw = item.get("time_published", "")
        url = item.get("url", "").strip()
        published_at = _parse_av_timestamp(published_raw)

        # Find the ticker_sentiment entry for this specific symbol
        ticker_sentiments = item.get("ticker_sentiment", [])
        matched = None
        for ts in ticker_sentiments:
            if ts.get("ticker", "").upper() == f"NSE:{symbol}".upper():
                matched = ts
                break

        if matched is None:
            # Article mentions the ticker but no direct sentiment entry — skip
            continue

        label = matched.get("ticker_sentiment_label", "Neutral")
        sentiment = _map_sentiment(label)

        # relevance_score as confidence proxy (0.0–1.0 string from AV)
        try:
            confidence = float(matched.get("relevance_score", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0

        articles.append({
            "headline": title,
            "source": "alphavantage",
            "published_at": published_at,
            "symbol": symbol,
            "sentiment": sentiment,
            "confidence": confidence,
            "url": url,
        })

    logger.info(f"  {symbol}: {len(articles)} articles fetched from Alpha Vantage")
    return articles


def _get_covered_symbols() -> set:
    """Return set of symbols already in news_sentiment from alphavantage."""
    try:
        conn = get_connection()
        rows = _execute(conn,
            "SELECT DISTINCT symbol FROM news_sentiment WHERE source='alphavantage'"
        ).fetchall()
        release_connection(conn)
        return {row[0] for row in rows}
    except Exception as e:
        logger.warning(f"Could not query covered symbols: {e}")
        return set()


def _load_all_symbols() -> List[str]:
    """Load all symbol names from angel_tokens.json."""
    with open(TOKENS_FILE) as f:
        token_map = json.load(f)
    return list(token_map.keys())


def collect_av_batch(
    symbols: Optional[List[str]] = None,
    max_requests: int = 25,
) -> Dict:
    """
    Fetch Alpha Vantage news for a batch of symbols and insert into DB.

    Prioritises symbols that have no AV news yet. Processes up to
    max_requests symbols (free tier = 25/day).

    Args:
        symbols: Explicit list of symbols to process. If None, loads all
                 499 from data/angel_tokens.json and picks uncovered ones.
        max_requests: Maximum API calls to make (default: 25 for free tier).

    Returns:
        Dict with keys: processed, total_articles, remaining
    """
    if symbols is None:
        all_symbols = _load_all_symbols()
        covered = _get_covered_symbols()
        # Prioritise uncovered symbols
        pending = [s for s in all_symbols if s not in covered]
        remaining_after = max(0, len(pending) - max_requests)
        batch = pending[:max_requests]
        logger.info(
            f"Symbols total={len(all_symbols)}, covered={len(covered)}, "
            f"pending={len(pending)}, processing={len(batch)}"
        )
    else:
        batch = symbols[:max_requests]
        remaining_after = max(0, len(symbols) - max_requests)

    processed = 0
    total_articles = 0

    for symbol in batch:
        try:
            articles = fetch_av_news(symbol)
            for art in articles:
                insert_news(
                    headline=art["headline"],
                    source=art["source"],
                    published_at=art["published_at"],
                    symbol=art["symbol"],
                    sentiment=art["sentiment"],
                    confidence=art["confidence"],
                    url=art["url"],
                )
            total_articles += len(articles)
            processed += 1
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")

        time.sleep(RATE_LIMIT_SECS)

    result = {
        "processed": processed,
        "total_articles": total_articles,
        "remaining": remaining_after,
    }
    logger.info(f"AV batch complete: {result}")
    return result


# ==========================================
# CLI entry point
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Fetch Alpha Vantage news & sentiment for Nifty 500 stocks"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Fetch a single stock (e.g. TCS). Omit to run today's batch of 25.",
    )
    parser.add_argument(
        "--max",
        type=int,
        default=25,
        help="Max API requests (default: 25, free tier limit).",
    )
    args = parser.parse_args()

    if args.symbol:
        print(f"\nFetching AV news for single symbol: {args.symbol}")
        articles = fetch_av_news(args.symbol.upper())
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
        print(f"  Fetched {len(articles)} articles, inserted {inserted}")
    else:
        print(f"\nRunning Alpha Vantage batch (max {args.max} requests)...")
        result = collect_av_batch(max_requests=args.max)
        print(f"\nDone:")
        print(f"  Symbols processed : {result['processed']}")
        print(f"  Articles inserted : {result['total_articles']}")
        print(f"  Symbols remaining : {result['remaining']}")


if __name__ == "__main__":
    main()
