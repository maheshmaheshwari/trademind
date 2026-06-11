"""
TradeMind AI — RSS Market News Collector

Scrapes 3 Indian financial RSS feeds for market-wide news.
Articles stored with symbol=NULL → feed `mkt_sentiment` ML feature.
Stock-specific articles tagged by matching company names in headlines.

Sources:
  - Economic Times Markets
  - Moneycontrol Latest News
  - Business Standard Markets

Usage:
    PYTHONPATH=. python collectors/rss_collector.py
"""

import json
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db import get_connection, release_connection, _execute, _executemany

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RSS_FEEDS = [
    ("Economic Times Markets",  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Moneycontrol",            "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("Business Standard",       "https://www.business-standard.com/rss/markets-106.rss"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml",
}

BATCH_SIZE = 32
_TOKENS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "angel_tokens.json")


def _load_name_map() -> Dict[str, str]:
    """Build {company_name_lower: SYMBOL.NS} from angel_tokens.json."""
    name_map = {}
    if not os.path.exists(_TOKENS_FILE):
        return name_map
    with open(_TOKENS_FILE) as f:
        tokens = json.load(f)
    for sym, info in tokens.items():
        name = info.get("name", "")
        if name:
            name_map[name.lower()] = f"{sym}.NS"
            # Also index short versions (e.g. "HDFC" matches "HDFC Bank")
            words = name.lower().split()
            if words:
                name_map[words[0]] = f"{sym}.NS"
    return name_map


def _tag_symbol(headline: str, name_map: Dict[str, str]) -> Optional[str]:
    """Return symbol if headline mentions a known company, else None."""
    h = headline.lower()
    for name, symbol in name_map.items():
        if len(name) >= 4 and name in h:
            return symbol
    return None


def _fetch_rss(url: str, source: str) -> List[Dict]:
    """Parse one RSS feed. Returns list of {title, url, published_at}."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        items = []
        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or item.findtext("{http://www.w3.org/2005/Atom}link") or "").strip()
            pub   = item.findtext("pubDate") or item.findtext("published") or ""
            if not title or not link:
                continue
            try:
                pub_dt = parsedate_to_datetime(pub).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pub_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            items.append({"title": title, "url": link, "published_at": pub_dt, "source": source})
        return items
    except Exception as e:
        logger.warning(f"{source}: {e}")
        return []


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
                results.append({"sentiment": sentiment, "confidence": score})
        return results


def _url_exists(conn, url: str) -> bool:
    cur = _execute(conn, "SELECT 1 FROM news_sentiment WHERE url = ? LIMIT 1", (url,))
    return cur.fetchone() is not None


def collect_all_rss() -> dict:
    name_map = _load_name_map()
    all_articles = []

    for source, url in RSS_FEEDS:
        articles = _fetch_rss(url, source)
        logger.info(f"{source}: {len(articles)} articles fetched")
        all_articles.extend(articles)

    conn = get_connection()
    try:
        # Deduplicate by URL
        new_articles = [a for a in all_articles if not _url_exists(conn, a["url"])]
        logger.info(f"New articles (not in DB): {len(new_articles)}")

        if not new_articles:
            return {"total": 0}

        # Score with FinBERT
        headlines = [a["title"][:500] for a in new_articles]
        scores = _FinBERT.score(headlines)

        rows = []
        for art, score in zip(new_articles, scores):
            symbol = _tag_symbol(art["title"], name_map)  # None = market-wide
            rows.append((
                art["title"][:500],
                art["source"],
                art["published_at"],
                symbol,
                str(score["sentiment"]),
                score["confidence"],
                art["url"],
            ))

        _executemany(conn,
            """INSERT INTO news_sentiment
               (headline, source, published_at, symbol, sentiment, confidence, url)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT DO NOTHING""",
            rows,
        )
        conn.commit()
        logger.info(f"RSS collection done: {len(rows)} articles stored")
        return {"total": len(rows)}

    finally:
        release_connection(conn)


if __name__ == "__main__":
    result = collect_all_rss()
    print(f"Done: {result['total']} new articles stored")
