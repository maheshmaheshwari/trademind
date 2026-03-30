# 03 — News & Sentiment Data

## Current State

| Table | Rows | Date range | Issue |
|-------|------|-----------|-------|
| `news_sentiment` | 234 | 2026-02-05 → 2026-03-05 | **Only 1 month** |
| `news_daily_sentiment` | 0 | — | **Empty** |

The ML models use 12 sentiment features per stock per day. With only 1 month of news,
any signal generated for dates before Feb 2026 has `sentiment=NULL`, degrading model
accuracy for stocks where sentiment is a top driver.

**Target:** 5 years of news (2021-01-01 → today) for all 499 Nifty 500 stocks.
Estimated volume: **~2–2.5 million articles**.

---

## Data Sources

### 1. GDELT Project (primary bootstrap source)

| Property | Value |
|----------|-------|
| Cost | Free, no API key |
| Coverage | 2013 → present (15-min delayed) |
| Rate | No stated limit; use 1 req/sec to be safe |
| Indian stock coverage | Moderate — international news outlets only |
| Sentiment | Raw `tone` score provided (-100 to +100) |

**API endpoint:**
```
GET https://api.gdeltproject.org/api/v2/doc/doc
  ?query=<company name>
  &mode=artlist
  &maxrecords=250
  &startdatetime=YYYYMMDDHHMMSS
  &enddatetime=YYYYMMDDHHMMSS
  &format=json
  &sort=DateDesc
```

**Example query for Reliance:**
```
query = "Reliance Industries"
startdatetime = "20210101000000"
enddatetime   = "20210201000000"
```

**Response fields used:**
```json
{
  "articles": [
    {
      "url":        "https://...",
      "title":      "Reliance posts record Q3 profit",
      "seendate":   "20210115T143000Z",
      "domain":     "economictimes.com",
      "tone":       -1.23
    }
  ]
}
```

**Chunking strategy:**
GDELT returns max 250 articles per request. For busy stocks, chunk by month:
```
499 stocks × 60 months × 1 req/sec = ~8.3 hours
```

**File to create:** `backend/collectors/gdelt_collector.py`

```python
GDELT_API = "https://api.gdeltproject.org/api/v2/doc/doc"
COMPANY_NAMES = {
    "RELIANCE":  "Reliance Industries",
    "TCS":       "Tata Consultancy Services",
    "HDFCBANK":  "HDFC Bank",
    # ... load from angel_tokens.json 'name' field
}

def fetch_gdelt_month(company_name: str, year: int, month: int) -> list:
    start = f"{year}{month:02d}01000000"
    end   = last_day_of_month(year, month)

    params = {
        "query":         company_name,
        "mode":          "artlist",
        "maxrecords":    "250",
        "startdatetime": start,
        "enddatetime":   end,
        "format":        "json",
        "sort":          "DateDesc",
    }
    resp = requests.get(GDELT_API, params=params, timeout=30)
    articles = resp.json().get("articles", [])

    rows = []
    for a in articles:
        rows.append({
            "headline":     a["title"],
            "source":       a.get("domain"),
            "published_at": parse_gdelt_date(a["seendate"]),
            "symbol":       None,           # matched later via company name
            "gdelt_tone":   a.get("tone"),
            "url":          a.get("url"),
        })
    return rows


def bootstrap_gdelt(from_year=2021, from_month=1):
    token_map = load_token_map()            # 499 stocks with 'name' field
    conn = get_db_connection()

    for symbol, info in token_map.items():
        company_name = info.get("name", symbol)
        for year, month in month_range(from_year, from_month):
            try:
                rows = fetch_gdelt_month(company_name, year, month)
                # Attach symbol before inserting
                for r in rows:
                    r["symbol"] = symbol
                insert_news_batch(conn, rows)
                time.sleep(1.0)             # polite rate
            except Exception as e:
                log_error(symbol, year, month, e)
```

**Note:** GDELT `tone` score is not the same as FinBERT sentiment.
After inserting raw GDELT rows, a second pass runs FinBERT on the headlines
to populate `sentiment` + `confidence` columns.

---

### 2. Alpha Vantage (secondary source, pre-scored sentiment)

| Property | Value |
|----------|-------|
| Cost | Free (25 req/day) |
| Coverage | ~2 years historical on free tier |
| Indian stocks | Partial — uses NSE tickers |
| Sentiment | Pre-scored (relevance + sentiment score) |

**API endpoint:**
```
GET https://www.alphavantage.co/query
  ?function=NEWS_SENTIMENT
  &tickers=NSE:<symbol>
  &time_from=YYYYMMDDTHHMM
  &limit=200
  &apikey=<YOUR_FREE_KEY>
```

**Free tier math:**
- 25 req/day → 499 stocks takes 20 days to bootstrap
- Run the bootstrap over 20 days, 25 stocks per day

**File to create:** `backend/collectors/alphavantage_collector.py`

```python
AV_API = "https://www.alphavantage.co/query"
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY")   # free key from alphavantage.co

def fetch_av_news(symbol: str, from_date: str) -> list:
    """
    symbol: NSE short code like "TCS"
    from_date: "20240101T0000"
    """
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   f"NSE:{symbol}",
        "time_from": from_date,
        "limit":     "200",
        "apikey":    AV_KEY,
    }
    resp = requests.get(AV_API, params=params, timeout=30)
    data = resp.json()

    rows = []
    for article in data.get("feed", []):
        # Find the relevance/sentiment for our specific ticker
        ticker_data = next(
            (t for t in article.get("ticker_sentiment", [])
             if t["ticker"] == f"NSE:{symbol}"),
            None
        )
        if not ticker_data:
            continue

        rows.append({
            "headline":     article["title"],
            "source":       article.get("source"),
            "published_at": parse_av_date(article["time_published"]),
            "symbol":       symbol,
            "sentiment":    ticker_data["ticker_sentiment_label"].lower(),  # bullish→positive
            "confidence":   float(ticker_data["relevance_score"]),
            "url":          article.get("url"),
        })
    return rows
```

**Environment variable needed:**
```env
ALPHAVANTAGE_API_KEY=your_free_key    # Register at alphavantage.co, free
```

---

### 3. RSS Feeds (live updates only)

| Source | Feed URL | Frequency |
|--------|----------|-----------|
| Economic Times Markets | `https://economictimes.indiatimes.com/markets/rss.cms` | Real-time |
| Mint Markets | `https://www.livemint.com/rss/markets` | Real-time |
| Business Standard | `https://www.business-standard.com/rss/markets-106.rss` | Real-time |

Use for live updates only (no historical data beyond ~2 weeks in RSS).
Run every 15 minutes during market hours via scheduler.

**Existing collector:** `collectors/news_collector.py` — already handles RSS.
Wire into scheduler at 15-min intervals during market hours.

---

## FinBERT Sentiment Scoring

After inserting raw articles (from GDELT or RSS), run FinBERT to populate
`sentiment` and `confidence` columns in `news_sentiment`.

**Model:** `ProsusAI/finbert` (loaded from Hugging Face, ~400MB, cached after first download)

**Current code:** `backend/analysis/sentiment.py` — already implemented.

**Batch scoring script to create:** `backend/sync_sentiment.py` (already exists — verify it handles bulk scoring)

```python
# Expected flow in sync_sentiment.py:
# 1. SELECT id, headline FROM news_sentiment WHERE sentiment IS NULL LIMIT 10000
# 2. Run FinBERT on batch of 64 headlines at a time
# 3. UPDATE news_sentiment SET sentiment=?, confidence=? WHERE id=?
# 4. Repeat until no NULL rows remain

# GPU estimate:  ~2.5M headlines / 64 per batch / ~0.1s per batch ≈ ~1 hour on GPU
# CPU estimate:  ~2.5M headlines → ~10–12 hours on CPU
```

---

## Pipeline Flow

```
[GDELT bootstrap]         (one-time, ~8 hours)
  ↓ raw articles in news_sentiment (sentiment=NULL)

[Alpha Vantage bootstrap] (20 days, 25 stocks/day)
  ↓ pre-scored articles appended to news_sentiment

[FinBERT batch scoring]   (one-time, ~1–12 hours depending on hardware)
  ↓ fills sentiment + confidence for all NULL rows

[TimescaleDB cagg]        (auto, every hour)
  ↓ news_daily_sentiment auto-populated

[Scheduler: RSS + AV]     (every 15 min, live)
  ↓ new articles → immediate FinBERT score → cagg refreshed within 1 hour
```

---

## Scheduler Jobs (add to `scheduler/jobs.py`)

```python
# Every 15 min during market hours — RSS live news
@scheduler.scheduled_job('cron', day_of_week='mon-fri',
                          hour='9-16', minute='*/15', timezone='Asia/Kolkata')
def job_live_news():
    from collectors.news_collector import collect_rss_news
    collect_rss_news()

# Every hour — score any unscored articles with FinBERT
@scheduler.scheduled_job('cron', hour='*', minute=5)
def job_score_sentiment():
    from analysis.sentiment import score_pending_articles
    score_pending_articles(batch_limit=500)
```

---

## Files Summary

| File | Status | Purpose |
|------|--------|---------|
| `collectors/news_collector.py` | ✅ exists | RSS + NewsAPI live feed |
| `collectors/historical_news_collector.py` | ✅ exists | GDELT historical (verify) |
| `collectors/gdelt_collector.py` | ❌ to create | Clean GDELT bootstrap for 499 stocks |
| `collectors/alphavantage_collector.py` | ❌ to create | Alpha Vantage 2-year fill |
| `analysis/sentiment.py` | ✅ exists | FinBERT scoring |
| `sync_sentiment.py` | ✅ exists | Bulk FinBERT batch scoring |

---

## Environment Variables

```env
ALPHAVANTAGE_API_KEY=your_free_key    # alphavantage.co — free, no credit card
# GDELT requires no API key
# FinBERT model downloaded automatically on first run (~400MB cached to ~/.cache/huggingface)
```
