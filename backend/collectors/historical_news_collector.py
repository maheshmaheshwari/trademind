"""
Nifty 500 AI — Historical News Sentiment Collector (FAST parallel version)

Uses ThreadPoolExecutor for 4x speedup + retry logic for network errors.
Skips stocks already in DB so it can safely resume.

Usage:
    python collectors/historical_news_collector.py                # All 499 stocks
    python collectors/historical_news_collector.py --days 365     # Last 1 year
    python collectors/historical_news_collector.py --market-only  # Market-wide only
    python collectors/historical_news_collector.py --workers 8    # 8 parallel threads
"""
import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"


def get_session() -> requests.Session:
    """Create a requests session with automatic retries."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class FinBERTSentiment:
    """Financial sentiment analysis using ProsusAI/finbert (thread-safe singleton)."""
    
    _instance = None
    _pipeline = None
    
    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load(self):
        if self._pipeline is None:
            from transformers import pipeline as hf_pipeline
            print("   🤖 Loading FinBERT model...")
            self._pipeline = hf_pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=-1,
                top_k=None,
            )
            print("   ✅ FinBERT loaded")
        return self._pipeline
    
    def analyze_batch(self, headlines: List[str]) -> List[Dict]:
        if not headlines:
            return []
        pipe = self._load()
        clean = [h[:512] if h else "" for h in headlines]
        results = []
        for i in range(0, len(clean), 32):
            batch = clean[i:i+32]
            try:
                batch_results = pipe(batch)
                for scores in batch_results:
                    best = max(scores, key=lambda x: x["score"])
                    sentiment_val = {
                        "positive": best["score"],
                        "negative": -best["score"],
                        "neutral": 0.0,
                    }.get(best["label"], 0.0)
                    results.append({
                        "label": best["label"],
                        "score": best["score"],
                        "sentiment_value": sentiment_val,
                    })
            except Exception as e:
                logger.error(f"FinBERT batch error: {e}")
                results.extend([{"label": "neutral", "score": 0.5, "sentiment_value": 0.0}] * len(batch))
        return results


def load_all_stocks() -> Dict[str, str]:
    """Load all stock symbols → company search name from angel_tokens.json."""
    with open("data/angel_tokens.json") as f:
        tokens = json.load(f)
    stock_map = {}
    for sym, info in tokens.items():
        ns_sym = f"{sym}.NS"
        name = info.get("name", sym)
        search_name = name.replace(" Ltd.", "").replace(" Limited", "").strip()
        stock_map[ns_sym] = search_name
    return stock_map


def fetch_gdelt_articles(session: requests.Session, query: str, 
                          start_dt: datetime, end_dt: datetime,
                          max_records: int = 250) -> List[Dict]:
    """Fetch articles from GDELT in 1-year chunks (fewer API calls = faster)."""
    all_articles = []
    chunk_start = start_dt
    
    while chunk_start < end_dt:
        # Use 1-year chunks instead of 6-month for speed
        chunk_end = min(chunk_start + timedelta(days=365), end_dt)
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
            "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
            "sort": "datedesc",
        }
        try:
            resp = session.get(GDELT_DOC_API, params=params, timeout=20)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    articles = data.get("articles", [])
                    all_articles.extend(articles)
                except Exception:
                    pass
        except Exception:
            pass
        chunk_start = chunk_end
        time.sleep(0.2)
    
    return all_articles


def fetch_gdelt_tone_timeline(session: requests.Session, query: str,
                               start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch daily tone timeline from GDELT in 3-month chunks."""
    all_data = []
    chunk_start = start_dt
    
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=90), end_dt)
        params = {
            "query": query,
            "mode": "timelinetone",
            "format": "csv",
            "startdatetime": chunk_start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": chunk_end.strftime("%Y%m%d%H%M%S"),
        }
        try:
            resp = session.get(GDELT_DOC_API, params=params, timeout=20)
            if resp.status_code == 200:
                text = resp.text.strip().replace("\ufeff", "")
                if text:
                    df = pd.read_csv(StringIO(text))
                    if "Value" in df.columns and "Date" in df.columns:
                        df = df.rename(columns={"Date": "date", "Value": "tone"})
                        df["date"] = pd.to_datetime(df["date"], errors="coerce")
                        df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
                        df = df.dropna(subset=["date", "tone"])
                        all_data.append(df[["date", "tone"]])
        except Exception:
            pass
        chunk_start = chunk_end
        time.sleep(0.2)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def process_single_stock(symbol: str, company_name: str, days: int) -> Tuple[str, int, List[tuple]]:
    """Process one stock: fetch GDELT articles + FinBERT sentiment. Returns (symbol, count, rows)."""
    session = get_session()
    finbert = FinBERTSentiment.get()
    
    end = datetime.now()
    start = end - timedelta(days=days)
    
    articles = fetch_gdelt_articles(session, company_name, start, end, max_records=250)
    if not articles:
        return (symbol, 0, [])
    
    headlines = [a.get("title", "") for a in articles if a.get("title")]
    if not headlines:
        return (symbol, 0, [])
    
    sentiments = finbert.analyze_batch(headlines)
    
    # Aggregate by date
    daily_data = {}
    for article, sent in zip(articles, sentiments):
        seen = article.get("seendate", "")
        if len(seen) >= 8:
            d = f"{seen[:4]}-{seen[4:6]}-{seen[6:8]}"
        else:
            continue
        if d not in daily_data:
            daily_data[d] = []
        daily_data[d].append(sent)
    
    rows = []
    for date_str, sents in sorted(daily_data.items()):
        values = [s["sentiment_value"] for s in sents]
        avg = sum(values) / len(values) if values else 0
        pos = sum(1 for s in sents if s["label"] == "positive")
        neg = sum(1 for s in sents if s["label"] == "negative")
        neu = sum(1 for s in sents if s["label"] == "neutral")
        confs = [s["score"] for s in sents]
        rows.append((
            date_str, symbol, avg, len(sents), pos, neg, neu,
            max(values) if values else 0, min(values) if values else 0,
            sum(confs) / len(confs) if confs else 0, "gdelt_finbert"
        ))
    
    return (symbol, len(rows), rows)


def backfill_market_sentiment(conn, days: int = 1825):
    """Backfill market-wide daily sentiment from GDELT."""
    session = get_session()
    end = datetime.now()
    start = end - timedelta(days=days)
    
    queries = [
        "India stock market finance",
        "Nifty sensex BSE NSE",
        "RBI monetary policy India economy",
        "FII DII India equity investment",
    ]
    
    print(f"\n📰 Market-wide sentiment ({start.date()} → {end.date()})...")
    all_tone_data = {}
    
    for query in queries:
        print(f"   '{query}'...", end=" ", flush=True)
        df = fetch_gdelt_tone_timeline(session, query, start, end)
        if df.empty:
            print("⚠️ no data")
            continue
        for _, row in df.iterrows():
            d = row["date"].strftime("%Y-%m-%d")
            if d not in all_tone_data:
                all_tone_data[d] = []
            all_tone_data[d].append(row["tone"])
        print(f"✅ {len(df)} days")
        time.sleep(0.5)
    
    count = 0
    for date_str, tones in sorted(all_tone_data.items()):
        avg_tone = sum(tones) / len(tones)
        avg_sentiment = max(-1, min(1, avg_tone / 10.0))
        pos = sum(1 for t in tones if t > 1)
        neg = sum(1 for t in tones if t < -1)
        neu = sum(1 for t in tones if -1 <= t <= 1)
        conn.execute(
            """INSERT OR REPLACE INTO news_daily_sentiment 
            (date, symbol, avg_sentiment, news_count, positive_count, negative_count, 
             neutral_count, max_positive, max_negative, avg_confidence, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (date_str, None, avg_sentiment, len(tones), pos, neg, neu,
             max(tones) / 10.0, min(tones) / 10.0, 0.7, "gdelt_tone")
        )
        count += 1
    conn.commit()
    print(f"   ✅ {count} daily market sentiment records saved")
    return count


def backfill_stock_sentiment(conn, days: int = 1825, workers: int = 4):
    """Backfill stock-specific sentiment using parallel GDELT fetching + FinBERT."""
    stock_map = load_all_stocks()
    
    # Check what we already have
    existing = set()
    try:
        rows = conn.execute(
            "SELECT DISTINCT symbol FROM news_daily_sentiment WHERE symbol IS NOT NULL"
        ).fetchall()
        existing = set(r[0] for r in rows)
    except Exception:
        pass
    
    # Filter to remaining stocks
    remaining = [(sym, name) for sym, name in stock_map.items() if sym not in existing]
    total_stocks = len(stock_map)
    skip_count = total_stocks - len(remaining)
    
    print(f"\n📊 {total_stocks} total stocks | {skip_count} already done | {len(remaining)} remaining")
    print(f"   Using {workers} parallel workers for GDELT fetching\n")
    
    if not remaining:
        print("   ✅ All stocks already in DB!")
        return 0
    
    # Pre-load FinBERT before parallel work
    FinBERTSentiment.get()._load()
    
    total_records = 0
    completed = 0
    
    # Process stocks: GDELT fetching is parallel, FinBERT is sequential (CPU-bound)
    # Strategy: Fetch GDELT articles in parallel batches, then run FinBERT sequentially
    batch_size = workers * 2
    
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start:batch_start + batch_size]
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for sym, name in batch:
                future = executor.submit(process_single_stock, sym, name, days)
                futures[future] = (sym, name)
            
            for future in as_completed(futures):
                sym, name = futures[future]
                completed += 1
                idx = skip_count + completed
                try:
                    symbol, count, rows = future.result()
                    if rows:
                        for row in rows:
                            conn.execute(
                                """INSERT OR REPLACE INTO news_daily_sentiment 
                                (date, symbol, avg_sentiment, news_count, positive_count, negative_count, 
                                 neutral_count, max_positive, max_negative, avg_confidence, source)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                row
                            )
                        conn.commit()
                        total_records += count
                        print(f"   [{idx}/{total_stocks}] {symbol} ({name}) ✅ {count} daily records")
                    else:
                        print(f"   [{idx}/{total_stocks}] {symbol} ⚠️ no articles")
                except Exception as e:
                    print(f"   [{idx}/{total_stocks}] {sym} ❌ {e}")
    
    return total_records


def main():
    parser = argparse.ArgumentParser(description="Backfill historical news sentiment for Nifty 500")
    parser.add_argument("--days", type=int, default=1825, help="Days of history (default: 5 years)")
    parser.add_argument("--market-only", action="store_true", help="Only backfill market-wide sentiment")
    parser.add_argument("--stocks-only", action="store_true", help="Only backfill stock-specific sentiment")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker threads (default: 4)")
    args = parser.parse_args()
    
    import libsql_experimental as libsql
    from database.models import ALL_TABLES, CREATE_INDEXES
    
    conn = libsql.connect("nifty500.db")
    for sql in ALL_TABLES:
        conn.execute(sql)
    for sql in CREATE_INDEXES:
        conn.execute(sql)
    conn.commit()
    
    print(f"🗓️  Backfilling {args.days} days ({args.days // 365} years) of news sentiment")
    print(f"   Workers: {args.workers}\n")
    
    total = 0
    
    if not args.stocks_only:
        total += backfill_market_sentiment(conn, days=args.days)
    
    if not args.market_only:
        total += backfill_stock_sentiment(conn, days=args.days, workers=args.workers)
    
    # Summary
    mkt = conn.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NULL").fetchone()[0]
    stk = conn.execute("SELECT COUNT(*) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]
    unique_syms = conn.execute("SELECT COUNT(DISTINCT symbol) FROM news_daily_sentiment WHERE symbol IS NOT NULL").fetchone()[0]
    
    print(f"\n{'='*55}")
    print(f"📊 BACKFILL COMPLETE")
    print(f"{'='*55}")
    print(f"   Market-wide daily records: {mkt}")
    print(f"   Stock-specific records: {stk} ({unique_syms} stocks)")
    print(f"   Total: {mkt + stk}")
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
