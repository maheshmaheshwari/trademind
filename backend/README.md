# TradeMind Backend

AI-powered stock analysis platform for Indian equity markets (NIFTY 500).

## Features

- Daily OHLC data ingestion for NIFTY 500 stocks
- Technical indicators computation (RSI, EMA, volatility, etc.)
- XGBoost-based signal generation (BUY/HOLD/AVOID)
- REST API with authentication and rate limiting
- Redis caching for latest signals
- Celery workers for background jobs

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis
- **ML**: XGBoost, scikit-learn, pandas, numpy
- **Task Queue**: Celery
- **Data Source**: yfinance

## Project Structure

```
backend/
├── app/
│   ├── api/           # API routes and middleware
│   ├── models/        # SQLAlchemy database models
│   ├── schemas/       # Pydantic schemas
│   ├── services/      # Business logic services
│   ├── ml/            # Machine learning pipeline
│   └── workers/       # Celery tasks
├── migrations/        # Alembic migrations
├── models/            # Saved ML models
├── data/              # Static data files
└── tests/             # Unit and integration tests
```

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 14+
- Redis 7+

### Installation

1. Create virtual environment:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment:

```bash
cp .env.example .env
# Edit .env with your database and Redis credentials
```

4. Run database migrations:

```bash
alembic upgrade head
```

5. Start the API server:

```bash
uvicorn app.main:app --reload
```

### Data Setup (First Time)

1. Seed the database with NIFTY 500 stocks:

```bash
python scripts/seed_database.py --stocks-only
```

2. Fetch historical data (1 year):

```bash
python scripts/seed_database.py --days 365
```

3. Train the initial model:

```bash
python scripts/train_model.py --version v1.0.0
```

4. Run the daily pipeline manually:

```bash
python scripts/run_pipeline.py
```

### Background Workers

6. Start Celery worker (in another terminal):

```bash
celery -A app.workers.celery_app worker --loglevel=info
```

7. Start Celery beat scheduler:

```bash
celery -A app.workers.celery_app beat --loglevel=info
```

## API Endpoints

| Endpoint            | Method | Description               |
| ------------------- | ------ | ------------------------- |
| `/signals/today`    | GET    | Get today's signals       |
| `/signals/{symbol}` | GET    | Get signals for a stock   |
| `/market/overview`  | GET    | Market summary and regime |
| `/stocks/search`    | GET    | Search stocks             |
| `/model/health`     | GET    | Model status and metrics  |

## Data Flow

1. **Daily Ingestion** (6:30 PM IST): Fetch OHLC data for all NIFTY 500 stocks
2. **Feature Engineering**: Compute technical indicators
3. **Model Inference**: Run XGBoost predictions
4. **Signal Generation**: Convert predictions to actionable signals
5. **Cache Update**: Store latest signals in Redis

## Disclaimer

This platform generates analytical signals for educational and research purposes only.
It does not constitute financial advice. Always conduct your own research and consult
with a qualified financial advisor before making investment decisions.

## License

MIT
