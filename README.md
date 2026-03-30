# 🧠 TradeMind AI

AI-powered stock trading platform for Indian equities (NSE). Generates BUY/SELL signals using ML models, executes paper & live trades with auto bracket orders (SL + Target), and monitors positions in real-time via Angel One SmartAPI.

> ⚠️ **Disclaimer**: This is not financial advice. Use live trading at your own risk.

---

## 🚀 Quick Start (After Cloning)

### Prerequisites

- **Python 3.11+** (backend)
- **Node.js 18+** & npm (frontend)
- Angel One SmartAPI account (optional, for live trading)

---

### 1. Backend Setup

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your keys
```

**Required `.env` variables:**

```env
# News API (free: https://newsapi.org/register)
NEWSAPI_KEY=your_newsapi_key

# Angel One SmartAPI (https://smartapi.angelone.in)
ANGEL_API_KEY=your_api_key
ANGEL_SECRET_KEY=your_secret_key
ANGEL_CLIENT_ID=your_client_id
ANGEL_PASSWORD=your_pin
ANGEL_TOTP_SECRET=your_totp_secret

# Database (local SQLite by default, Turso for production)
# TURSO_DATABASE_URL=libsql://your-db.turso.io
# TURSO_AUTH_TOKEN=your_token
ENV=local

# App Config
PORT=8000
LOG_LEVEL=INFO
JWT_SECRET=your-secret-key-change-this
```

### 3. Initialize Database

```bash
python main.py setup
```

This creates the SQLite DB, downloads 2 years of price data, calculates indicators, and generates signals (~15 min).

### 4. Start Backend Server

```bash
source venv/bin/activate
uvicorn api.server:app --reload --port 8000
```

API docs: http://localhost:8000/docs

---

### 5. Frontend Setup

Open a **new terminal**:

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

App: http://localhost:5173

---

## 🔧 Run Commands Cheatsheet

| What          | Command                                                                                 | Terminal   |
| ------------- | --------------------------------------------------------------------------------------- | ---------- |
| **Backend**   | `cd backend && source venv/bin/activate && uvicorn api.server:app --reload --port 8000` | Terminal 1 |
| **Frontend**  | `cd frontend && npm run dev`                                                            | Terminal 2 |
| **Scheduler** | `cd backend && source venv/bin/activate && python main.py schedule`                     | Terminal 3 |
| **DB Status** | `cd backend && source venv/bin/activate && python main.py status`                       | Any        |
| **Signals**   | `cd backend && source venv/bin/activate && python main.py signals`                      | Any        |

---

## 📡 API Endpoints

| Endpoint                                          | Description                   |
| ------------------------------------------------- | ----------------------------- |
| `GET /api/health`                                 | Health check + market status  |
| `GET /api/stocks?page=0&size=25`                  | Paginated stock list          |
| `GET /api/prices/{symbol}?days=90`                | Price history (OHLCV)         |
| `GET /api/indicators/{symbol}`                    | Technical indicators + signal |
| `GET /api/signals/latest?page=0&size=25`          | AI trade signals (paginated)  |
| `GET /api/watchlist/{symbol}`                     | Combined stock data           |
| `POST /api/auth/register`                         | Create account                |
| `POST /api/auth/login`                            | Login → JWT token             |
| `POST /api/trading/execute-signal`                | Execute trade (Paper or Live) |
| `GET /api/trading/positions/{user_id}`            | Open positions                |
| `GET /api/trading/orders/{user_id}`               | Order history                 |
| `POST /api/trading/square-off/{user_id}/{symbol}` | Sell position                 |

---

## 🏗️ Project Structure

```
trademind/
├── frontend/                  ← React + Vite + TypeScript
│   ├── src/
│   │   ├── pages/             ← Dashboard, Market, Signals, Trade, Portfolio, Orders
│   │   ├── components/        ← Navbar, Pagination, Layout
│   │   ├── api.ts             ← API client with auth
│   │   └── AuthContext.tsx     ← JWT auth context
│   └── package.json
│
└── backend/                   ← Python FastAPI
    ├── main.py                ← CLI (setup/collect/server/schedule)
    ├── requirements.txt
    ├── .env                   ← API keys (gitignored)
    ├── nifty500.db            ← SQLite DB (created on setup)
    ├── api/
    │   ├── server.py          ← FastAPI app
    │   ├── auth.py            ← JWT authentication
    │   └── routes/            ← API route handlers
    ├── trading/
    │   ├── trading_engine.py  ← Paper & Live trade execution
    │   ├── gtt_manager.py     ← Angel One GTT orders (SL/Target)
    │   ├── price_monitor.py   ← Intraday SL/Target checker
    │   └── risk_manager.py    ← Risk checks before trade
    ├── collectors/
    │   ├── price_collector.py ← yfinance daily prices
    │   ├── angel_collector.py ← Angel One historical data
    │   ├── ltp_fetcher.py     ← Live LTP from Angel One
    │   └── news_collector.py  ← NewsAPI headlines
    ├── analysis/
    │   ├── indicators.py      ← RSI, MACD, Bollinger, etc.
    │   ├── signals.py         ← ML signal generator
    │   └── sentiment.py       ← FinBERT news sentiment
    ├── scheduler/
    │   └── jobs.py            ← APScheduler (daily/hourly/intraday)
    └── database/
        ├── db.py              ← DB connection + helpers
        └── models.py          ← SQL schema
```

---

## 📊 Key Features

- **AI Trade Signals** — ML models generate BUY/SELL with confidence scores, target, and SL
- **Paper Trading** — Virtual ₹10L account, auto bracket orders
- **Live Trading** — Real orders on Angel One with GTT (Good Till Triggered) for SL/Target
- **Risk Management** — 6 checks before every trade (balance, concentration, daily limits)
- **Server-Side Pagination** — All list pages paginated via backend APIs
- **Dark Theme UI** — Premium React dashboard with charting

---

## 📊 Tech Stack

| Component | Technology                              |
| --------- | --------------------------------------- |
| Frontend  | React 19, Vite, TypeScript, TailwindCSS |
| Backend   | Python 3.11+, FastAPI, Uvicorn          |
| Database  | SQLite (dev) → Turso (prod)             |
| Data      | yfinance, Angel One SmartAPI            |
| AI/ML     | scikit-learn, FinBERT (sentiment)       |
| Scheduler | APScheduler                             |
| Auth      | JWT (Bearer tokens)                     |

---

_Built with ❤️ for Indian stock market analysis_
