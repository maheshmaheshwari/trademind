"""
Nifty 500 AI — Database Table Definitions

Pure SQL CREATE TABLE statements for all 5 tables.
These match the exact schema from the project specification.
"""

# ==========================================
# Table: prices
# Stores OHLCV data for stocks (daily + intraday)
# ==========================================
CREATE_PRICES_TABLE = """
CREATE TABLE IF NOT EXISTS prices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange TEXT DEFAULT 'NSE',
    date TEXT NOT NULL,
    time TEXT,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume INTEGER,
    interval TEXT DEFAULT '1d',
    UNIQUE(symbol, date, time, interval)
);
"""

# ==========================================
# Table: technical_indicators
# Stores computed technical indicators per stock per date
# ==========================================
CREATE_TECHNICAL_INDICATORS_TABLE = """
CREATE TABLE IF NOT EXISTS technical_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    rsi_14 REAL,
    macd REAL,
    macd_signal REAL,
    macd_hist REAL,
    bb_upper REAL,
    bb_middle REAL,
    bb_lower REAL,
    sma_20 REAL,
    sma_50 REAL,
    sma_200 REAL,
    ema_9 REAL,
    ema_21 REAL,
    atr_14 REAL,
    adx_14 REAL,
    stoch_k REAL,
    stoch_d REAL,
    obv REAL,
    support_1 REAL,
    support_2 REAL,
    support_3 REAL,
    resistance_1 REAL,
    resistance_2 REAL,
    resistance_3 REAL,
    signal TEXT,
    signal_strength REAL,
    UNIQUE(symbol, date)
);
"""

# ==========================================
# Table: news_sentiment
# Stores news headlines with AI sentiment scores
# ==========================================
CREATE_NEWS_SENTIMENT_TABLE = """
CREATE TABLE IF NOT EXISTS news_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT NOT NULL,
    source TEXT,
    published_at TEXT,
    symbol TEXT,
    sentiment TEXT,
    confidence REAL,
    url TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

# ==========================================
# Table: market_overview
# Daily snapshot of broad market metrics
# ==========================================
CREATE_MARKET_OVERVIEW_TABLE = """
CREATE TABLE IF NOT EXISTS market_overview (
    date TEXT PRIMARY KEY,
    nifty500_close REAL,
    nifty500_change_pct REAL,
    nifty50_close REAL,
    nifty50_change_pct REAL,
    sensex_close REAL,
    india_vix REAL,
    advances INTEGER,
    declines INTEGER,
    unchanged INTEGER,
    total_volume REAL,
    fii_net REAL,
    dii_net REAL,
    overall_sentiment_score REAL,
    fear_greed_label TEXT
);
"""

# ==========================================
# Table: ai_signals
# Stores AI-generated trading signals with reasoning
# ==========================================
CREATE_AI_SIGNALS_TABLE = """
CREATE TABLE IF NOT EXISTS ai_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    generated_at TEXT DEFAULT (datetime('now')),
    signal TEXT NOT NULL,
    confidence REAL,
    model_version TEXT,
    target_price REAL,
    stop_loss REAL,
    reasoning TEXT,
    features_used TEXT,
    UNIQUE(symbol, generated_at, model_version)
);
"""

# ==========================================
# Table: news_daily_sentiment
# Pre-aggregated daily sentiment scores for ML features
# ==========================================
CREATE_NEWS_DAILY_SENTIMENT_TABLE = """
CREATE TABLE IF NOT EXISTS news_daily_sentiment (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT,
    avg_sentiment REAL DEFAULT 0,
    news_count INTEGER DEFAULT 0,
    positive_count INTEGER DEFAULT 0,
    negative_count INTEGER DEFAULT 0,
    neutral_count INTEGER DEFAULT 0,
    max_positive REAL DEFAULT 0,
    max_negative REAL DEFAULT 0,
    avg_confidence REAL DEFAULT 0,
    source TEXT,
    UNIQUE(date, symbol)
);
"""

# ==========================================
# Indexes for fast queries
# ==========================================
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices(symbol, date);",
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol ON prices(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_prices_date ON prices(date);",
    "CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON technical_indicators(symbol, date);",
    "CREATE INDEX IF NOT EXISTS idx_indicators_symbol ON technical_indicators(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_indicators_signal ON technical_indicators(signal);",
    "CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_sentiment(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_news_published ON news_sentiment(published_at);",
    "CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_sentiment(sentiment);",
    "CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol ON ai_signals(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_ai_signals_signal ON ai_signals(signal);",
    "CREATE INDEX IF NOT EXISTS idx_ai_signals_confidence ON ai_signals(confidence);",
    "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date ON news_daily_sentiment(date);",
    "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_symbol ON news_daily_sentiment(symbol);",
    "CREATE INDEX IF NOT EXISTS idx_daily_sentiment_date_symbol ON news_daily_sentiment(date, symbol);",
]

# All table creation statements in order
ALL_TABLES = [
    CREATE_PRICES_TABLE,
    CREATE_TECHNICAL_INDICATORS_TABLE,
    CREATE_NEWS_SENTIMENT_TABLE,
    CREATE_MARKET_OVERVIEW_TABLE,
    CREATE_AI_SIGNALS_TABLE,
    CREATE_NEWS_DAILY_SENTIMENT_TABLE,
]
