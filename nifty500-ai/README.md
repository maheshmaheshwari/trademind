# 📊 Nifty 500 AI Trading Data Pipeline

AI-powered stock market data pipeline for Indian equities (NSE/BSE).
Collects price data, calculates technical indicators, scores news sentiment,
and generates BUY/SELL signals for Nifty 500 stocks.

> ⚠️ **Disclaimer**: This is not financial advice. Always do your own research before trading.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip (Python package manager)

### 1. Clone & Install

```bash
cd nifty500-ai
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (see "Getting API Keys" below)
```

### 3. First Run (Setup)

```bash
python main.py setup
```

This will:

- Create the SQLite database (`nifty500.db`)
- Download 2 years of price data for 50 stocks (~15 min)
- Calculate all technical indicators
- Generate BUY/SELL signals

### 4. Start the API Server

```bash
python main.py server
```

Open http://localhost:8000/docs to see all API endpoints.

### 5. Start Automated Collection

```bash
python main.py schedule
```

This runs data collection daily at 4 PM IST after market close.

---

## 🔑 Getting Free API Keys

### NewsAPI (for news data)

1. Go to https://newsapi.org/register
2. Sign up for a free account
3. Copy your API key to `.env` → `NEWSAPI_KEY`
4. Free tier: 100 requests/day

### Angel One SmartAPI (optional, for live data)

1. Open an Angel One account: https://www.angelone.in/
2. Enable SmartAPI from your account settings
3. Copy credentials to `.env`

---

## 💻 CLI Commands

| Command                   | Description                                       |
| ------------------------- | ------------------------------------------------- |
| `python main.py setup`    | First-time setup — creates DB, downloads 2yr data |
| `python main.py collect`  | Run one manual collection cycle                   |
| `python main.py server`   | Start FastAPI server on port 8000                 |
| `python main.py schedule` | Start automated daily scheduler                   |
| `python main.py status`   | Show database statistics                          |
| `python main.py signals`  | Print today's top BUY/SELL signals                |

---

## 📡 API Endpoints

Base URL: `http://localhost:8000`

| Endpoint                           | Description                   |
| ---------------------------------- | ----------------------------- |
| `GET /api/health`                  | Health check + market status  |
| `GET /api/market/overview`         | Today's market overview       |
| `GET /api/prices/{symbol}?days=90` | Price history (OHLCV)         |
| `GET /api/indicators/{symbol}`     | Technical indicators + signal |
| `GET /api/sentiment/market`        | Market Fear & Greed score     |
| `GET /api/sentiment/{symbol}`      | Stock-specific sentiment      |
| `GET /api/signals/top-buys`        | Top BUY signals today         |
| `GET /api/signals/top-sells`       | Top SELL signals today        |
| `GET /api/watchlist/{symbol}`      | Combined stock data           |
| `GET /api/heatmap/sectors`         | Sector performance heatmap    |

Interactive docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
nifty500-ai/
├── main.py                 ← CLI entry point
├── requirements.txt        ← All Python packages
├── .env                    ← Your API keys (local only)
├── nifty500.db             ← SQLite database (created on setup)
├── database/
│   ├── db.py               ← Database connection + CRUD helpers
│   └── models.py           ← SQL table definitions
├── collectors/
│   ├── price_collector.py  ← yfinance data downloader
│   ├── news_collector.py   ← NewsAPI + Economic Times scraper
│   └── fii_collector.py    ← FII/DII institutional flows
├── analysis/
│   ├── indicators.py       ← Technical indicators (RSI, MACD, etc)
│   ├── signals.py          ← BUY/SELL signal generator
│   └── sentiment.py        ← FinBERT news sentiment scorer
├── api/
│   ├── server.py           ← FastAPI application
│   └── routes/             ← API route handlers
├── scheduler/
│   └── jobs.py             ← APScheduler automated tasks
└── data/
    ├── stocks_list.py      ← Nifty 50 stock symbols
    └── backups/            ← CSV data backups
```

---

## 🔧 Common Errors

| Error                                       | Fix                                                                     |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'ta'` | `pip install ta`                                                        |
| `No price data returned for XYZ.NS`         | Stock may be delisted or yfinance issue. Will be skipped.               |
| `NEWSAPI_KEY not configured`                | Add your NewsAPI key to `.env`                                          |
| `Address already in use (port 8000)`        | Change PORT in `.env` or kill the other process                         |
| `FinBERT model download slow`               | First run downloads ~400MB. Use `score_sentiment_simple()` as fallback. |

---

## 🗂️ Database

- **Development**: SQLite (`nifty500.db` — zero setup)
- **Production**: [Turso](https://turso.tech/) (cloud SQLite, 5GB free)
  - Just change `DATABASE_URL` in `.env` to `libsql://your-db.turso.io`

---

## 📊 Tech Stack

| Component          | Technology                        |
| ------------------ | --------------------------------- |
| Language           | Python 3.11+                      |
| Data Collection    | yfinance, requests, BeautifulSoup |
| Technical Analysis | ta library                        |
| AI Sentiment       | ProsusAI/FinBERT (transformers)   |
| Database           | SQLite → Turso (production)       |
| API Server         | FastAPI + Uvicorn                 |
| Scheduler          | APScheduler                       |

---

_Built with ❤️ for Indian stock market analysis_
