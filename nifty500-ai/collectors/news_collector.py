"""
Nifty 500 AI — News Collector

Fetches market and stock-specific news from multiple sources:
1. NewsAPI.org (requires free API key)
2. Economic Times web scraper (no API key needed)

Usage:
    from collectors.news_collector import fetch_market_news, scrape_economic_times
    
    # NewsAPI (requires NEWSAPI_KEY in .env)
    news = fetch_market_news()
    
    # Economic Times scraper (no key needed)
    news = scrape_economic_times()
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from database.db import insert_news

load_dotenv()
logger = logging.getLogger(__name__)

# NewsAPI configuration
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_URL = "https://newsapi.org/v2/everything"

# Browser-like headers for web scraping
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_market_news(max_articles: int = 30, save_to_db: bool = True) -> List[Dict]:
    """
    Fetch Indian stock market news from NewsAPI.

    Queries for broad market news covering Nifty, NSE, RBI, FII/DII.

    Args:
        max_articles: Maximum number of articles to fetch
        save_to_db: Whether to save results to the database

    Returns:
        List of article dicts with keys: headline, source, published_at, url, symbol

    Note:
        Requires NEWSAPI_KEY environment variable. Get a free key at https://newsapi.org/
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here":
        logger.warning("NewsAPI key not configured. Set NEWSAPI_KEY in .env file.")
        return []

    all_articles = []

    # Multiple search queries to cover different aspects of Indian markets
    queries = [
        "Nifty 500 OR NSE India stocks",
        "Indian stock market today",
        "RBI monetary policy India",
        "FII DII India equity",
    ]

    for query in queries:
        try:
            response = requests.get(
                NEWSAPI_URL,
                params={
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": min(max_articles, 100),
                    "apiKey": NEWSAPI_KEY,
                },
                timeout=10,
            )

            if response.status_code != 200:
                logger.warning(f"NewsAPI returned {response.status_code} for query: {query}")
                continue

            data = response.json()
            articles = data.get("articles", [])

            for article in articles:
                parsed = {
                    "headline": article.get("title", ""),
                    "source": article.get("source", {}).get("name", "Unknown"),
                    "published_at": article.get("publishedAt", ""),
                    "url": article.get("url", ""),
                    "symbol": None,  # Market-wide news
                }

                # Skip articles with no title
                if not parsed["headline"]:
                    continue

                all_articles.append(parsed)

                # Save to database
                if save_to_db:
                    insert_news(
                        headline=parsed["headline"],
                        source=parsed["source"],
                        published_at=parsed["published_at"],
                        url=parsed["url"],
                        symbol=None,
                    )

            logger.info(f"Fetched {len(articles)} articles for query: {query}")
            time.sleep(0.5)  # Rate limit

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching news for '{query}': {e}")
        except Exception as e:
            logger.error(f"Error processing news for '{query}': {e}")

    logger.info(f"Total market news fetched: {len(all_articles)}")
    return all_articles


def fetch_stock_news(symbol: str, max_articles: int = 10, save_to_db: bool = True) -> List[Dict]:
    """
    Fetch news specific to a single stock.

    Args:
        symbol: Stock symbol, e.g. "TCS.NS"
        max_articles: Maximum articles to fetch
        save_to_db: Whether to save to database

    Returns:
        List of article dicts.
    """
    if not NEWSAPI_KEY or NEWSAPI_KEY == "your_newsapi_key_here":
        logger.warning("NewsAPI key not configured.")
        return []

    # Clean symbol for search (remove .NS suffix)
    company_name = symbol.replace(".NS", "").replace(".", " ")

    try:
        response = requests.get(
            NEWSAPI_URL,
            params={
                "q": f"{company_name} stock NSE",
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_articles,
                "apiKey": NEWSAPI_KEY,
            },
            timeout=10,
        )

        if response.status_code != 200:
            logger.warning(f"NewsAPI returned {response.status_code} for {symbol}")
            return []

        data = response.json()
        articles = []

        for article in data.get("articles", []):
            parsed = {
                "headline": article.get("title", ""),
                "source": article.get("source", {}).get("name", "Unknown"),
                "published_at": article.get("publishedAt", ""),
                "url": article.get("url", ""),
                "symbol": symbol,
            }

            if not parsed["headline"]:
                continue

            articles.append(parsed)

            if save_to_db:
                insert_news(
                    headline=parsed["headline"],
                    source=parsed["source"],
                    published_at=parsed["published_at"],
                    url=parsed["url"],
                    symbol=symbol,
                )

        logger.info(f"Fetched {len(articles)} news for {symbol}")
        return articles

    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return []


def scrape_economic_times(max_articles: int = 20, save_to_db: bool = True) -> List[Dict]:
    """
    Scrape latest stock market news from Economic Times.

    No API key needed — uses web scraping with BeautifulSoup.

    Args:
        max_articles: Maximum articles to scrape
        save_to_db: Whether to save to database

    Returns:
        List of article dicts.

    Note:
        Website structure may change. This scraper is best-effort.
    """
    url = "https://economictimes.indiatimes.com/markets/stocks/news"
    articles = []

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)

        if response.status_code != 200:
            logger.warning(f"Economic Times returned status {response.status_code}")
            return []

        soup = BeautifulSoup(response.text, "lxml")

        # Find news article links — ET uses various CSS classes
        # Try multiple selectors for resilience
        story_links = soup.find_all("a", class_="wrapLines")
        if not story_links:
            story_links = soup.find_all("a", {"data-orefid": True})
        if not story_links:
            # Fallback: find all links in the main content area
            main_content = soup.find("div", class_="eachStory") or soup.find("div", id="pageContent")
            if main_content:
                story_links = main_content.find_all("a", href=True)

        for link in story_links[:max_articles]:
            headline = link.get_text(strip=True)
            href = link.get("href", "")

            if not headline or len(headline) < 10:
                continue

            # Build full URL if relative
            if href.startswith("/"):
                href = f"https://economictimes.indiatimes.com{href}"

            article = {
                "headline": headline,
                "source": "Economic Times",
                "published_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                "url": href,
                "symbol": None,  # Market-wide
            }
            articles.append(article)

            if save_to_db:
                insert_news(
                    headline=article["headline"],
                    source=article["source"],
                    published_at=article["published_at"],
                    url=article["url"],
                    symbol=None,
                )

        logger.info(f"Scraped {len(articles)} articles from Economic Times")

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error scraping Economic Times: {e}")
    except Exception as e:
        logger.error(f"Error scraping Economic Times: {e}")

    return articles


def collect_all_news(save_to_db: bool = True) -> Dict:
    """
    Collect news from all sources: NewsAPI + Economic Times.

    Returns:
        Summary dict with total articles and per-source counts.
    """
    all_articles = []

    # Source 1: NewsAPI
    print("📰 Fetching news from NewsAPI...")
    newsapi_articles = fetch_market_news(save_to_db=save_to_db)
    all_articles.extend(newsapi_articles)

    # Source 2: Economic Times scraper
    print("📰 Scraping Economic Times...")
    et_articles = scrape_economic_times(save_to_db=save_to_db)
    all_articles.extend(et_articles)

    print(f"\n✅ Total news collected: {len(all_articles)}")
    print(f"   NewsAPI: {len(newsapi_articles)}")
    print(f"   Economic Times: {len(et_articles)}")

    return {
        "total": len(all_articles),
        "newsapi": len(newsapi_articles),
        "economic_times": len(et_articles),
        "articles": all_articles,
    }


if __name__ == "__main__":
    """Quick test: scrape Economic Times (doesn't need API key)."""
    logging.basicConfig(level=logging.INFO)

    print("Testing Economic Times scraper...")
    articles = scrape_economic_times(max_articles=5, save_to_db=False)
    for a in articles:
        print(f"  📌 {a['headline'][:80]}...")
