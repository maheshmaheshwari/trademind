"""
TradeMind AI — yfinance Per-Stock News Collector

Fetches ~10 recent news articles per stock from Yahoo Finance for all 499
Nifty 500 stocks, runs FinBERT sentiment scoring, and stores in news_sentiment.

Deduplicates by URL — safe to run daily.
Sources: Reuters, Economic Times, Mint, Business Standard (via Yahoo).

Usage:
    PYTHONPATH=. python collectors/yfinance_news_collector.py
    PYTHONPATH=. python collectors/yfinance_news_collector.py --symbol HDFCBANK.NS
"""

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection, release_connection, _execute, _executemany, insert_news

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SLEEP_BETWEEN_STOCKS = 0.5
BATCH_SIZE = 32
_PER_SYMBOL_TIMEOUT = 12   # seconds before giving up on a single ticker.news call
_MAX_JOB_SECONDS = 3600    # hard cap: abort after 60 min to protect downstream jobs


class _FinBERT:
    _pipeline = None

    @classmethod
    def score(cls, texts: List[str]) -> List[dict]:
        if not texts:
            return []
        if cls._pipeline is None:
            logger.info("Loading FinBERT model…")
            from transformers import pipeline
            cls._pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                top_k=1,
            )
        results = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch = [t[:512] for t in texts[i:i + BATCH_SIZE]]
            for pred in cls._pipeline(batch):
                top = pred[0] if isinstance(pred, list) else pred
                label = top["label"].lower()
                score = float(top["score"])
                sentiment = score if label == "positive" else (-score if label == "negative" else 0.0)
                results.append({"sentiment": sentiment, "confidence": score, "label": label})
        return results


def _url_exists(conn, url: str) -> bool:
    """Check if a URL is already in news_sentiment."""
    cur = _execute(conn, "SELECT 1 FROM news_sentiment WHERE url = ? LIMIT 1", (url,))
    return cur.fetchone() is not None


def _fetch_news_with_timeout(symbol: str):
    """Fetch ticker.news with a hard timeout to prevent indefinite hangs."""
    import yfinance as yf
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: yf.Ticker(symbol).news)
        try:
            return future.result(timeout=_PER_SYMBOL_TIMEOUT)
        except FuturesTimeoutError:
            logger.warning(f"{symbol}: yfinance news timed out after {_PER_SYMBOL_TIMEOUT}s — skipping")
            return []


def collect_stock(symbol: str) -> int:
    """Fetch and store yfinance news for one stock. Returns rows inserted."""
    try:
        articles = _fetch_news_with_timeout(symbol)
        if not articles:
            return 0

        conn = get_connection()
        try:
            rows = []
            headlines = []
            metas = []

            for art in articles:
                # Handle both yfinance v1 and v2 response formats
                title = (art.get("title")
                         or art.get("content", {}).get("title", ""))
                url   = (art.get("link")
                         or art.get("content", {}).get("canonicalUrl", {}).get("url", "")
                         or art.get("url", ""))
                pub_ts = (art.get("providerPublishTime")
                          or art.get("content", {}).get("pubDate"))

                if not title or not url:
                    continue
                if _url_exists(conn, url):
                    continue

                if isinstance(pub_ts, (int, float)):
                    pub_dt = datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M:%S")
                elif isinstance(pub_ts, str):
                    try:
                        pub_dt = datetime.strptime(pub_ts[:19], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        pub_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    pub_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                headlines.append(title[:500])
                metas.append((url, pub_dt))

            if not headlines:
                return 0

            scores = _FinBERT.score(headlines)
            for headline, (url, pub_dt), score in zip(headlines, metas, scores):
                rows.append((
                    headline,
                    "yfinance",
                    pub_dt,
                    symbol,
                    str(score["sentiment"]),
                    score["confidence"],
                    url,
                ))

            if rows:
                _executemany(conn,
                    """INSERT INTO news_sentiment
                       (headline, source, published_at, symbol, sentiment, confidence, url)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT DO NOTHING""",
                    rows,
                )
                conn.commit()
            return len(rows)

        finally:
            release_connection(conn)

    except Exception as e:
        logger.error(f"{symbol}: {e}")
        return 0


def collect_all(symbol_filter: Optional[str] = None) -> dict:
    from database.db import get_all_symbols
    symbols = get_all_symbols()

    if symbol_filter:
        symbols = [s for s in symbols if s == symbol_filter]

    total_inserted = 0
    job_start = time.time()
    logger.info(f"yfinance news collection — {len(symbols)} stocks (max {_MAX_JOB_SECONDS}s)")

    for idx, symbol in enumerate(symbols, 1):
        if time.time() - job_start > _MAX_JOB_SECONDS:
            remaining = len(symbols) - idx + 1
            logger.warning(f"yfinance news: hit {_MAX_JOB_SECONDS}s cap — skipping {remaining} remaining stocks")
            break

        inserted = collect_stock(symbol)
        total_inserted += inserted
        if inserted:
            logger.info(f"[{idx}/{len(symbols)}] {symbol}: {inserted} new articles")
        time.sleep(SLEEP_BETWEEN_STOCKS)

    logger.info(f"Done. Total new articles: {total_inserted}")
    return {"total": total_inserted, "symbols": len(symbols)}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default=None)
    args = parser.parse_args()
    collect_all(symbol_filter=args.symbol)
