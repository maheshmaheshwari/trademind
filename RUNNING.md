# Running TradeMind AI

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.13 | `brew install python@3.13` |
| Node.js | 18+ | `brew install node` |
| Docker Desktop | any | https://docker.com |

> All database credentials are read from `backend/.env` automatically.
> You never need to pass `PGHOST`, `PGPORT`, etc. in the command line.

---

## 1. Start TimescaleDB (Docker)

```bash
# First-time setup (run once)
docker run -d --name trademind-db --restart unless-stopped \
  -e POSTGRES_PASSWORD=trademind \
  -e POSTGRES_DB=trademind \
  -e POSTGRES_USER=trademind \
  -p 5433:5432 \
  -v ~/trademind-pgdata:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg16

# Subsequent starts
docker start trademind-db

# Verify it's running
docker ps --filter name=trademind-db
```

---

## 2. Backend

### First-time setup

```bash
cd backend

# Create virtual environment
python3.13 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install psycopg2-binary lightgbm xgboost

# Initialise database schema (reads .env automatically)
python -c "from database.db import init_database; init_database()"
```

### Start the API server

```bash
cd backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

API is available at: **http://localhost:8000**
Interactive docs: **http://localhost:8000/docs**

---

## 3. Frontend

### First-time setup

```bash
cd frontend
npm install
```

### Start dev server

```bash
cd frontend
npm run dev
```

App is available at: **http://localhost:5173**

---

## Quick Start (both at once)

Open two terminal tabs:

**Tab 1 — Backend**
```bash
cd /Users/mahesh/Desktop/personal/trademind/backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Tab 2 — Frontend**
```bash
cd /Users/mahesh/Desktop/personal/trademind/frontend
npm run dev
```

---

## Build for Production

```bash
# Frontend
cd frontend
npm run build
# Output in frontend/dist/

# Backend (no build needed — run directly)
cd backend
source venv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## Useful Commands

```bash
cd backend
source venv/bin/activate

# Row counts in TimescaleDB
python -c "
from database.db import get_db_stats
for t, n in get_db_stats().items():
    print(f'{t}: {n:,}')
"

# Regenerate trade signals (after retraining models)
python generate_trades.py

# Direct DB access
docker exec -it trademind-db psql -U trademind -d trademind

# Backend logs
tail -f server.log

# Check TimescaleDB hypertables
docker exec -it trademind-db psql -U trademind -d trademind \
  -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"
```

---

## Ports

| Service | Port |
|---------|------|
| Frontend (Vite dev) | 5173 |
| Backend (FastAPI) | 8000 |
| TimescaleDB | 5433 |

---

## Troubleshooting

**Backend can't connect to DB**
```bash
docker start trademind-db
docker ps --filter name=trademind-db   # should show "Up"
```

**`ModuleNotFoundError: No module named 'psycopg2'`**
```bash
cd backend && source venv/bin/activate && pip install psycopg2-binary
```

**`ModuleNotFoundError: No module named 'lightgbm'`**
```bash
cd backend && source venv/bin/activate && pip install lightgbm xgboost
```

**Frontend shows blank / API errors**
- Confirm backend is running on port 8000
- Check browser console for CORS errors
- Verify `src/api.ts` base URL matches backend port

**Port 5432 already in use**
- TimescaleDB uses port **5433** (not 5432) to avoid conflicts
- The `backend/.env` file already has `PGPORT=5433`
