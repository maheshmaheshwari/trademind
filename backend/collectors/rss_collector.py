"""
TradeMind AI — RSS Market News Collector

Scrapes 6 Indian financial RSS feeds. Every article is classified:
  • Stock-specific: NSE symbol (e.g. "HDFCBANK.NS") via two-stage matching:
      1. ALL-CAPS word → known NSE ticker  (e.g. "HDFCBANK rally")
      2. Company name substring → longest match wins
  • Market-wide: one of five category tags:
      MARKET:INDEX   — Nifty / Sensex / broad market moves
      MARKET:RBI     — RBI, repo rate, monetary policy, rupee
      MARKET:MACRO   — GDP, budget, FII/DII flows, trade, GST
      MARKET:GLOBAL  — US Fed, crude oil, global markets, dollar
      MARKET:SEBI    — SEBI regulations, IPO rules, disclosures
      MARKET:GENERAL — catch-all for unclassified market news

Sources:
  - Economic Times Markets
  - Moneycontrol Latest News
  - Business Standard Markets
  - Livemint Markets
  - NDTV Profit
  - Hindu Business Line

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
    ("Livemint Markets",        "https://www.livemint.com/rss/markets"),
    ("NDTV Profit",             "https://feeds.feedburner.com/ndtvprofit-latest"),
    ("Hindu Business Line",     "https://www.thehindubusinessline.com/markets/feeder/default.rss"),
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml",
}

BATCH_SIZE = 32
_TOKENS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "angel_tokens.json")


_GENERIC_TOKENS = frozenset({
    # Financial/regulatory acronyms that are NOT NSE stock tickers
    "NSE", "RBI", "SEBI", "FII", "DII", "IPO", "AGM", "EPS",
    "GDP", "CPI", "WPI", "IMF", "MPC", "GST", "ETF", "AIF", "NCD",
    "SIP", "NAV", "AUM", "RERA", "NPA",
    # Note: BSE and PNB are listed stocks on NSE — intentionally NOT blacklisted
})


def _load_name_map() -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Returns:
        name_to_sym : {full_clean_name_lower: SYMBOL.NS}
        ticker_to_sym: {NSE_TICKER_UPPER: SYMBOL.NS}
    """
    name_to_sym: Dict[str, str] = {}
    ticker_to_sym: Dict[str, str] = {}
    if not os.path.exists(_TOKENS_FILE):
        return name_to_sym, ticker_to_sym
    with open(_TOKENS_FILE) as f:
        tokens = json.load(f)
    for sym, info in tokens.items():
        nse_sym = f"{sym}.NS"
        if sym not in _GENERIC_TOKENS:
            ticker_to_sym[sym.upper()] = nse_sym
        name = (info.get("name") or "").strip()
        if not name:
            continue
        # Full name (lower)
        if len(name) >= 5:
            name_to_sym[name.lower()] = nse_sym
        # Without trailing legal suffix: "Ltd.", "Limited", "Corp.", etc.
        clean = re.sub(
            r'\s*(Ltd\.?|Limited|Corp\.?|Corporation|Inc\.?|Pvt\.?)\s*$',
            '', name, flags=re.IGNORECASE,
        ).strip()
        if len(clean) >= 5 and clean.lower() != name.lower():
            name_to_sym[clean.lower()] = nse_sym
        # Without leading "The " (e.g. "The New India Assurance Co." → "New India Assurance Co.")
        for variant in (name, clean):
            stripped = re.sub(r'^The\s+', '', variant, flags=re.IGNORECASE).strip()
            if len(stripped) >= 5 and stripped.lower() not in name_to_sym:
                name_to_sym[stripped.lower()] = nse_sym
            # Also strip trailing "Company" / "Corporation" that weren't caught above
            for sec_pat in (r'\s*Company\s*$', r'\s*Co\.?\s*$', r'\s*Corporation\s*$'):
                shorter = re.sub(sec_pat, '', stripped, flags=re.IGNORECASE).strip()
                if len(shorter) >= 5 and shorter.lower() not in name_to_sym:
                    name_to_sym[shorter.lower()] = nse_sym
    return name_to_sym, ticker_to_sym


# Company names that appear as substrings inside common non-stock phrases.
# If the full phrase is present, the embedded company name should NOT match a stock.
_NON_STOCK_PHRASES: Dict[str, str] = {
    "reserve bank of india": "bank of india",   # RBI headline ≠ BANKINDIA.NS
    "world bank":            "bank",
    "central bank":          "bank",
    "national stock exchange": "national",
    "bombay stock exchange":   "bombay",
}


def _tag_symbol(
    headline: str,
    name_to_sym: Dict[str, str],
    ticker_to_sym: Dict[str, str],
) -> Optional[str]:
    """
    Return NSE symbol if headline mentions a known stock, else None.

    Priority:
      1. Word-boundary match on NSE ticker (e.g. "HDFCBANK rally" → HDFCBANK.NS)
      2. Longest substring match on cleaned company name (false-positive phrases excluded)
    """
    # 1. Ticker: extract ALL-CAPS words and look up against known tickers
    for word in re.findall(r'\b[A-Z]{2,20}\b', headline):
        if word in ticker_to_sym:
            return ticker_to_sym[word]

    # 2. Company name: longest match wins
    h = headline.lower()

    # Build set of company-name substrings that are disqualified in this headline
    disqualified: set = set()
    for full_phrase, embedded in _NON_STOCK_PHRASES.items():
        if full_phrase in h:
            disqualified.add(embedded)

    best_sym: Optional[str] = None
    best_len = 0
    for name, sym in name_to_sym.items():
        if len(name) > best_len and name in h and name not in disqualified:
            best_sym = sym
            best_len = len(name)
    return best_sym


_MARKET_CATEGORIES = [
    # Checked in priority order — first match wins.
    # SEBI first: very specific regulatory language
    ("MARKET:SEBI", [
        "sebi", "securities board", "market regulator", "insider trading",
        "listing rules", "disclosure norms", "takeover code", "delisting",
        "ipo allotment", "grey market",
    ]),
    # INDEX second: Nifty/Sensex are unambiguous even when FII/DII appears alongside
    ("MARKET:INDEX", [
        "nifty 50", "nifty50", "nifty 500", "nifty500", "sensex",
        "nifty bank", "nifty it", "nifty midcap", "nifty smallcap",
        "broader market", "benchmark index", "market rally",
        "market sell-off", "market crash", "market decline", "market gains",
        "indices ", "midcap index", "smallcap index",
        "market breadth", "advance decline",
    ]),
    # GLOBAL third: must come before RBI so "Federal Reserve rate cut" → GLOBAL not RBI
    ("MARKET:GLOBAL", [
        "federal reserve", "us fed", " fed ", "fomc", "jerome powell",
        "wall street", "nasdaq", "dow jones", "s&p 500",
        "global market", "us market", "us economy",
        "crude oil", "brent crude", "wti crude", "dollar index",
        "china market", "asian market", "european market",
        "opec", "oil prices", "us treasury",
    ]),
    # RBI: Indian-specific only — generic "rate cut/hike" removed to avoid cross-matching
    ("MARKET:RBI", [
        "rbi", "reserve bank of india", "monetary policy committee",
        "repo rate", "reverse repo", " mpc ", "crr ", "slr ",
        "rupee ", "indian rupee", "forex reserve", "currency intervention",
    ]),
    # MACRO: broad economic — FII/DII flows, budget, trade
    ("MARKET:MACRO", [
        " gdp ", "budget ", "fiscal deficit", "trade deficit", "current account",
        " gst ", "fii ", "dii ", " fpi ", "foreign portfolio",
        "finance minister", "ministry of finance", "government policy",
        "economic growth", "industrial output", "iip ", " pmi ",
        "wholesale price", "consumer price", "core inflation",
    ]),
]


def _classify_market_news(headline: str) -> str:
    """
    Classify a non-stock headline into a market category tag.

    Checks keyword lists in priority order (SEBI → RBI → GLOBAL → MACRO → INDEX).
    Falls back to MARKET:GENERAL if nothing matches.
    """
    h = f" {headline.lower()} "   # pad with spaces so word-boundary checks are simple
    for category, keywords in _MARKET_CATEGORIES:
        if any(kw in h for kw in keywords):
            return category
    return "MARKET:GENERAL"


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
    name_to_sym, ticker_to_sym = _load_name_map()
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
            symbol = _tag_symbol(art["title"], name_to_sym, ticker_to_sym) \
                     or _classify_market_news(art["title"])
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
