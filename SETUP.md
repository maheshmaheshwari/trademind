# TradeMind AI — Complete Setup Guide

Everything needed to run the full application from scratch.

---

## Table of Contents

1. [Accounts & Credentials](#1-accounts--credentials)
2. [System Prerequisites](#2-system-prerequisites)
3. [Database Setup](#3-database-setup)
4. [Backend Setup](#4-backend-setup)
5. [Environment Variables](#5-environment-variables)
6. [Data Import](#6-data-import)
7. [Frontend Setup](#7-frontend-setup)
8. [Running the App](#8-running-the-app)
9. [Ports & Services](#9-ports--services)
10. [Scheduler Jobs](#10-scheduler-jobs)
11. [Troubleshooting](#11-troubleshooting)
12. [Current Setup Status](#12-current-setup-status)

---

## 1. Accounts & Credentials

You need accounts with these services before running the app. All have free tiers.

### Required

| Service | Purpose | Sign Up | Free Tier |
|---|---|---|---|
| **Timescale Cloud** | Hosted TimescaleDB (main database) | console.cloud.timescale.com | 90-day trial, then ~$50/mo |
| **Angel One** | Live stock price data (Nifty 500 EOD candles) | angelone.in | Free with demat account |

### Optional (improves ML accuracy)

| Service | Purpose | Sign Up | Free Tier |
|---|---|---|---|
| **NewsAPI** | Daily news headlines | newsapi.org | 100 req/day |
| **Alpha Vantage** | News with pre-scored sentiment | alphavantage.co | 25 req/day |
| **GDELT Project** | 5-year historical news (no sign up) | gdeltproject.org | Completely free |
| **NSE India** | Delivery %, Option chain, Corporate actions (no sign up) | nseindia.com | Completely free |

### What you get from each

**Angel One SmartAPI** — you need:
- `ANGEL_API_KEY` — from Angel One developer portal (myapi.angelone.in)
- `ANGEL_CLIENT_ID` — your Angel One login ID
- `ANGEL_PASSWORD` — your Angel One login password
- `ANGEL_MPIN` — your 4-digit Angel One MPIN
- `ANGEL_TOTP_SECRET` — generated when you enable TOTP in Angel One app settings

**Timescale Cloud** — you need:
- `PGHOST` — service hostname (from Timescale Cloud dashboard)
- `PGPORT` — service port (usually 35986 or similar)
- `PGDATABASE` — database name (default: `tsdb`)
- `PGUSER` — username (default: `tsdbadmin`)
- `PGPASSWORD` — password (set during service creation)

**NewsAPI** — you need:
- `NEWSAPI_KEY` — API key from newsapi.org dashboard

**Alpha Vantage** — you need:
- `ALPHAVANTAGE_API_KEY` — API key from alphavantage.co

---

## 2. System Prerequisites

### macOS
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.9

# Install Node.js 18+
brew install node

# Verify versions
python3 --version   # 3.9+
node --version      # 18+
npm --version       # 9+
```

### Windows
- Python 3.9+: python.org/downloads
- Node.js 18+: nodejs.org
- Git: git-scm.com

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip nodejs npm git
```

---

## 3. Database Setup

The app uses **TimescaleDB** (PostgreSQL extension for time-series data).

### Option A — Timescale Cloud (Recommended, already set up)
1. Go to **console.cloud.timescale.com**
2. Create a new service → note Host, Port, Database, User, Password
3. Add credentials to `backend/.env` (see Section 5)
4. Schema is created automatically on first backend start

### Option B — Local Docker
```bash
# First-time setup
docker run -d --name trademind-db --restart unless-stopped \
  -e POSTGRES_PASSWORD=trademind \
  -e POSTGRES_DB=trademind \
  -e POSTGRES_USER=trademind \
  -p 5433:5432 \
  -v ~/trademind-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

# Subsequent starts
docker start trademind-db

# Verify
docker ps --filter name=trademind-db
```

For local Docker, set these in `.env`:
```
PGHOST=localhost
PGPORT=5433
PGDATABASE=trademind
PGUSER=trademind
PGPASSWORD=trademind
```

### Connect with DBeaver (GUI)
| Field | Value |
|---|---|
| Driver | PostgreSQL |
| Host | your Timescale Cloud host |
| Port | your Timescale Cloud port |
| Database | tsdb |
| Username | tsdbadmin |
| Password | your password |
| SSL mode | require |

---

## 4. Backend Setup

```bash
cd backend

# Create virtual environment (Python 3.9+)
python3 -m venv venv

# Activate
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Install all dependencies
pip install -r requirements.txt

# Install additional packages (may be missing from requirements.txt)
pip install bcrypt PyJWT "python-jose[cryptography]" pyotp smartapi-python \
            logzero websocket-client apscheduler psycopg2-binary \
            transformers torch catboost

# Initialise database schema (run once — safe to re-run)
python -c "from database.db import init_database; init_database()"

# Verify DB connection
python -c "from database.db import get_db_stats; [print(f'{t}: {n:,}') for t,n in get_db_stats().items()]"
```

---

## 5. Environment Variables

Create `backend/.env` with the following (replace values with your own):

```env
# ============================================================
# TradeMind AI — Environment Variables
# ============================================================

# ------------------------------------------------------------
# TimescaleDB
# Option A: Timescale Cloud
# ------------------------------------------------------------
PGHOST=your-service.timescaledb.io
PGPORT=35986
PGDATABASE=tsdb
PGUSER=tsdbadmin
PGPASSWORD=your-password

# Option B: Local Docker (uncomment if using local)
# PGHOST=localhost
# PGPORT=5433
# PGDATABASE=trademind
# PGUSER=trademind
# PGPASSWORD=trademind

# ------------------------------------------------------------
# Angel One SmartAPI (required for live stock data)
# Get from: myapi.angelone.in
# ------------------------------------------------------------
ANGEL_API_KEY=your-api-key
ANGEL_SECRET_KEY=your-secret-key
ANGEL_CLIENT_ID=your-client-id      # e.g. M123456
ANGEL_PASSWORD=your-login-password
ANGEL_MPIN=your-4-digit-mpin
ANGEL_TOTP_SECRET=your-totp-secret  # from Angel One app → Settings → TOTP

# ------------------------------------------------------------
# JWT Auth (generate any strong random string)
# Run: python -c "import secrets; print(secrets.token_hex(32))"
# ------------------------------------------------------------
JWT_SECRET=your-random-64-char-hex-string

# ------------------------------------------------------------
# News APIs (optional — improves ML sentiment features)
# ------------------------------------------------------------
NEWSAPI_KEY=your-newsapi-key              # newsapi.org
ALPHAVANTAGE_API_KEY=your-av-key          # alphavantage.co

# ------------------------------------------------------------
# App Config
# ------------------------------------------------------------
PORT=8000
LOG_LEVEL=INFO
```

### How to get Angel One TOTP Secret
1. Open Angel One app → Profile → Settings → Security
2. Enable TOTP (Google Authenticator)
3. During setup, Angel One shows a **secret key** (32-character string)
4. Copy that string — it's your `ANGEL_TOTP_SECRET`
5. Also scan the QR code with Google Authenticator for your phone

---

## 6. Data Import

Run these in order after backend setup. All commands from `backend/` directory.

### Step 1 — Import 498 stock price data (Angel One)
Fetches ~400 days of daily EOD candles for all Nifty 500 stocks.
~40 minutes to complete.

```bash
source venv/bin/activate
python update_stocks_angel.py --days 400
```

### Step 2 — Calculate technical indicators
Computes RSI, MACD, Bollinger Bands, ATR, ADX, etc. for all stocks.

```bash
python -c "from analysis.signals import process_all_stocks; print(process_all_stocks())"
```

### Step 3 — Import historical news (GDELT)
Free, no API key. Fetches 2+ years of news for all stocks.
**Important:** Run after midnight UTC (5:30 AM IST) to avoid rate limits.
Takes 2–3 days to complete in background.

```bash
python collectors/gdelt_collector.py --from-year 2023 --from-month 1 --only-missing > /tmp/gdelt.log 2>&1 &
```

Monitor progress:
```bash
tail -f /tmp/gdelt.log
```

### Step 4 — Score news sentiment (FinBERT)
Run after GDELT has imported articles. Downloads ProsusAI/finbert model (~500MB first run).

```bash
python collectors/gdelt_collector.py --score --batch-limit 5000
```

### Step 5 — Generate trade signals
Runs all 493 ML models against current data and writes signals to DB.

```bash
python generate_trades.py
```

### Step 6 — Verify data
```bash
python -c "
from database.db import get_db_stats
for t, n in get_db_stats().items():
    print(f'{t}: {n:,}')
"
```

Expected output after full setup:
```
prices:                 ~135,000+ rows
technical_indicators:   ~130,000+ rows
news_sentiment:         ~500,000+ rows (after GDELT)
trade_signals:          ~3,000+ rows
```

---

## 7. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Verify
npm run build   # should complete without errors
```

---

## 8. Running the App

### Development (two terminals)

**Terminal 1 — Backend**
```bash
cd backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — Frontend**
```bash
cd frontend
npm run dev
```

### Production

**Backend**
```bash
cd backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Frontend**
```bash
cd frontend
npm run build
# Serve dist/ with nginx or any static file server
```

### Check everything is working
```bash
# Health check
curl http://localhost:8000/api/health
# Expected: {"status":"ok","market_open":true,...}

# Signals
curl "http://localhost:8000/api/signals?limit=5"
```

---

## 9. Ports & Services

| Service | Port | URL |
|---|---|---|
| Frontend (Vite dev) | 5173 | http://localhost:5173 |
| Backend (FastAPI) | 8000 | http://localhost:8000 |
| API Docs (Swagger) | 8000 | http://localhost:8000/docs |
| TimescaleDB (Cloud) | 35986 | Timescale Cloud console |
| TimescaleDB (Local Docker) | 5433 | localhost:5433 |

---

## 10. Scheduler Jobs

The backend runs an APScheduler in the background automatically when the server starts. No extra setup needed.

| Job | Time (IST) | What it does |
|---|---|---|
| EOD Price Collection | 3:35 PM Mon–Fri | Fetches today's candles for all 498 stocks via Angel One |
| Index Data | 4:00 PM Mon–Fri | Updates Nifty 50/500, Sensex, India VIX |
| Technical Indicators | 4:15 PM Mon–Fri | Recalculates all indicators |
| News Collection | 4:30 PM Mon–Fri | Fetches latest headlines |
| Trade Signal Generation | 5:15 PM Mon–Fri | Runs all ML models, updates signals |
| FinBERT Scoring | Every hour (9AM–8PM) | Scores unscored news articles |
| Price Monitor | Every 5 min (9–3:30 PM) | Checks SL/Target triggers |
| Data Cleanup | Sunday 8:00 PM | Removes intraday data older than 30 days |

---

## 11. Troubleshooting

### Backend won't start — `ModuleNotFoundError`
```bash
cd backend && source venv/bin/activate
pip install bcrypt PyJWT apscheduler pyotp smartapi-python logzero websocket-client
```

### Angel One login fails — `Please enter 4 digit mpin`
Make sure `ANGEL_MPIN` is set in `.env` (not just `ANGEL_PASSWORD`).

### GDELT rate limited (429 errors)
GDELT enforces per-IP rate limits. Wait until after midnight UTC (5:30 AM IST) and restart:
```bash
pkill -f gdelt_collector
python collectors/gdelt_collector.py --from-year 2023 --from-month 1 --only-missing > /tmp/gdelt.log 2>&1 &
```

### DB connection error
```bash
# Test connection
python -c "from database.db import get_connection; conn = get_connection(); print('Connected'); conn.close()"
```
- Check `PGHOST`, `PGPORT`, `PGPASSWORD` in `.env`
- Timescale Cloud: ensure your IP is allowlisted in the service settings

### Frontend blank / CORS errors
- Confirm backend is running: `curl http://localhost:8000/api/health`
- Check browser console for the exact error
- Ensure `frontend/src/api.ts` base URL is `http://localhost:8000`

### Scheduler failed to start
```bash
pip install apscheduler
# Restart backend
```

### `BackgroundScheduler | None` type error
Python 3.9 doesn't support `X | Y` union syntax. Already fixed in `scheduler/jobs.py` — ensure you have the latest code.

---

## 12. Current Setup Status

As of last session:

| Component | Status | Notes |
|---|---|---|
| Timescale Cloud DB | ✅ Connected | Cloud-hosted, no Docker needed |
| Backend dependencies | ✅ Installed | venv at `backend/venv/` |
| DB schema | ✅ Initialised | All tables created |
| Price data | ✅ 498 symbols | ~134,747 rows, latest 2026-06-01 |
| Technical indicators | ❌ Empty | Run Step 2 above |
| News / GDELT | ❌ Empty | GDELT rate-limited — retry tomorrow 5:30 AM IST |
| Trade signals | ❌ Empty | Run after indicators are calculated |
| ML models | ✅ 493 models | `backend/final_models/*.pkl` |
| Angel One | ✅ Configured | MPIN stored in `.env` |
| NewsAPI | ✅ Configured | Key in `.env` |
| Alpha Vantage | ✅ Configured | Key in `.env` (limited NSE coverage) |
| Scheduler | ✅ Fixed | `apscheduler` installed, Python 3.9 type fix applied |
| Backend server | ⏹ Stopped | Run `uvicorn api.server:app --port 8000 --reload` |
| Frontend | ⏹ Not started | Run `npm run dev` from `frontend/` |

### Next steps (in order)
1. Calculate technical indicators
2. Restart GDELT tomorrow after 5:30 AM IST
3. Start backend + frontend
4. After GDELT: score sentiment → retrain models → regenerate signals
