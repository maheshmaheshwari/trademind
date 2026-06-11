"""
TradeMind AI — TimescaleDB (PostgreSQL) Schema

Replaces models.py for the TimescaleDB setup.
Run once via init_timescale() to create tables, hypertables,
compression policies, and the news_daily_sentiment continuous aggregate.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regular tables (not time-series partitioned)
# ---------------------------------------------------------------------------

SQL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id               BIGSERIAL PRIMARY KEY,
    username         TEXT NOT NULL UNIQUE,
    email            TEXT UNIQUE,
    password_hash    TEXT NOT NULL,
    display_name     TEXT,
    virtual_balance  DOUBLE PRECISION DEFAULT 1000000,
    virtual_invested DOUBLE PRECISION DEFAULT 0,
    total_pnl        DOUBLE PRECISION DEFAULT 0,
    win_count        INTEGER DEFAULT 0,
    loss_count       INTEGER DEFAULT 0,
    mode             TEXT DEFAULT 'PAPER',
    angel_client_id  TEXT,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_PORTFOLIOS = """
CREATE TABLE IF NOT EXISTS portfolios (
    id                BIGSERIAL PRIMARY KEY,
    name              TEXT NOT NULL,
    investment_amount DOUBLE PRECISION NOT NULL,
    time_horizon      TEXT NOT NULL,
    risk_profile      TEXT DEFAULT 'moderate',
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_PORTFOLIO_SECTORS = """
CREATE TABLE IF NOT EXISTS portfolio_sectors (
    id               BIGSERIAL PRIMARY KEY,
    portfolio_id     BIGINT NOT NULL REFERENCES portfolios(id),
    sector           TEXT NOT NULL,
    allocation_pct   DOUBLE PRECISION NOT NULL,
    ai_suggested_pct DOUBLE PRECISION,
    num_stocks       INTEGER DEFAULT 0
);
"""

SQL_PORTFOLIO_STOCKS = """
CREATE TABLE IF NOT EXISTS portfolio_stocks (
    id               BIGSERIAL PRIMARY KEY,
    portfolio_id     BIGINT NOT NULL REFERENCES portfolios(id),
    symbol           TEXT NOT NULL,
    sector           TEXT,
    signal           TEXT,
    confidence       DOUBLE PRECISION,
    buy_price        DOUBLE PRECISION,
    target_price     DOUBLE PRECISION,
    stop_loss        DOUBLE PRECISION,
    allocated_amount DOUBLE PRECISION,
    quantity         INTEGER,
    status           TEXT DEFAULT 'recommended',
    added_at         TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_RISK_SETTINGS = """
CREATE TABLE IF NOT EXISTS risk_settings (
    id               BIGSERIAL PRIMARY KEY,
    user_id          BIGINT NOT NULL UNIQUE REFERENCES users(id),
    max_daily_loss   DOUBLE PRECISION DEFAULT 10000,
    max_daily_trades INTEGER DEFAULT 10,
    max_position_pct DOUBLE PRECISION DEFAULT 20,
    max_position_size DOUBLE PRECISION DEFAULT 50000,
    stop_loss_pct    DOUBLE PRECISION DEFAULT 7,
    target_pct       DOUBLE PRECISION DEFAULT 15,
    auto_stop_loss   INTEGER DEFAULT 1,
    auto_target      INTEGER DEFAULT 1,
    mode             TEXT DEFAULT 'PAPER'
);

CREATE TABLE IF NOT EXISTS user_signal_volume (
    id                BIGSERIAL PRIMARY KEY,
    user_id           BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trade_signal_id   BIGINT NOT NULL REFERENCES trade_signals(id) ON DELETE CASCADE,
    symbol            TEXT NOT NULL,
    quantity_consumed INTEGER NOT NULL DEFAULT 0,
    investment_amount DOUBLE PRECISION DEFAULT 0,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, trade_signal_id)
);
"""

SQL_TRADE_SIGNALS = """
CREATE TABLE IF NOT EXISTS trade_signals (
    id                      BIGSERIAL PRIMARY KEY,
    symbol                  TEXT NOT NULL,
    name                    TEXT,
    signal                  TEXT NOT NULL,
    confidence              DOUBLE PRECISION,
    trade_type              TEXT,
    buy_price               DOUBLE PRECISION,
    target_price            DOUBLE PRECISION,
    stop_loss               DOUBLE PRECISION,
    risk_reward             DOUBLE PRECISION,
    expected_return_pct     DOUBLE PRECISION,
    current_price           DOUBLE PRECISION,
    atr_14                  DOUBLE PRECISION,
    atr_pct                 DOUBLE PRECISION,
    avg_daily_volume        BIGINT,
    daily_turnover_cr       DOUBLE PRECISION,
    liquidity               TEXT,
    max_safe_qty            INTEGER,
    max_qty_per_user        INTEGER,
    max_investment_per_user DOUBLE PRECISION,
    min_qty                 INTEGER,
    recommended_volume      INTEGER,
    consumed_volume         INTEGER DEFAULT 0,
    model_name              TEXT,
    model_horizon           TEXT,
    model_accuracy          DOUBLE PRECISION,
    model_precision         DOUBLE PRECISION,
    top_drivers             TEXT,
    sentiment               TEXT,
    generated_date          DATE NOT NULL,
    generated_at            TIMESTAMPTZ NOT NULL,
    is_active               BOOLEAN DEFAULT TRUE,
    UNIQUE(symbol, generated_date)
);
"""

SQL_ORDERS = """
CREATE TABLE IF NOT EXISTS orders (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES users(id),
    bracket_id    TEXT,
    order_id      TEXT,
    symbol        TEXT NOT NULL,
    name          TEXT,
    exchange      TEXT DEFAULT 'NSE',
    order_type    TEXT NOT NULL,
    order_purpose TEXT DEFAULT 'ENTRY',
    quantity      INTEGER NOT NULL,
    price         DOUBLE PRECISION NOT NULL,
    trigger_price DOUBLE PRECISION,
    status        TEXT DEFAULT 'PENDING',
    mode          TEXT DEFAULT 'PAPER',
    signal        TEXT,
    confidence    DOUBLE PRECISION,
    horizon       TEXT,
    fill_price    DOUBLE PRECISION,
    fees          DOUBLE PRECISION DEFAULT 0,
    pnl           DOUBLE PRECISION,
    gtt_rule_id      TEXT,
    gtt_status       TEXT,
    trade_signal_id  BIGINT REFERENCES trade_signals(id) ON DELETE SET NULL,
    created_at       TIMESTAMPTZ DEFAULT NOW(),
    updated_at       TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_POSITIONS = """
CREATE TABLE IF NOT EXISTS positions (
    id                 BIGSERIAL PRIMARY KEY,
    user_id            BIGINT NOT NULL REFERENCES users(id),
    symbol             TEXT NOT NULL,
    name               TEXT,
    quantity           INTEGER NOT NULL,
    avg_buy_price      DOUBLE PRECISION NOT NULL,
    current_price      DOUBLE PRECISION,
    target_price       DOUBLE PRECISION,
    stop_loss          DOUBLE PRECISION,
    unrealized_pnl     DOUBLE PRECISION,
    unrealized_pnl_pct DOUBLE PRECISION,
    invested_amount    DOUBLE PRECISION,
    current_value      DOUBLE PRECISION,
    mode               TEXT DEFAULT 'PAPER',
    bracket_id         TEXT,
    updated_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, symbol)
);
"""

SQL_WATCHLIST = """
CREATE TABLE IF NOT EXISTS watchlist (
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol      TEXT   NOT NULL,
    alert_above DOUBLE PRECISION,
    alert_below DOUBLE PRECISION,
    added_at    TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, symbol)
);
"""

SQL_NOTIFICATIONS = """
CREATE TABLE IF NOT EXISTS notifications (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type       TEXT NOT NULL CHECK (type IN ('trade','signal','price','news','system')),
    title      TEXT NOT NULL,
    message    TEXT,
    icon       TEXT,
    color      TEXT,
    is_read    BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_AUTHORIZED_TRADES = """
CREATE TABLE IF NOT EXISTS authorized_trades (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol      TEXT NOT NULL,
    name        TEXT,
    sector      TEXT,
    signal      TEXT NOT NULL DEFAULT 'BUY',
    mode        TEXT DEFAULT 'PAPER',
    qty         INTEGER NOT NULL DEFAULT 0,
    amount      DOUBLE PRECISION NOT NULL DEFAULT 0,
    entry       DOUBLE PRECISION,
    target      DOUBLE PRECISION,
    sl          DOUBLE PRECISION,
    exp_profit  DOUBLE PRECISION DEFAULT 0,
    max_loss    DOUBLE PRECISION DEFAULT 0,
    cmp         DOUBLE PRECISION,
    actual_pnl  DOUBLE PRECISION,
    status      TEXT DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING','EXECUTED','COMPLETED','STOPPED')),
    bracket_id  TEXT,
    sl_gtt_id   TEXT,
    target_gtt_id TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_AUTOPILOT_SETTINGS = """
CREATE TABLE IF NOT EXISTS autopilot_settings (
    id         BIGSERIAL PRIMARY KEY,
    user_id    BIGINT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    enabled    BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_FII_DII_DAILY = """
CREATE TABLE IF NOT EXISTS fii_dii_daily (
    date        DATE        PRIMARY KEY,
    fii_net     DOUBLE PRECISION,   -- net buy/sell in crores (positive = net buy)
    dii_net     DOUBLE PRECISION,
    fii_buy     DOUBLE PRECISION,
    fii_sell    DOUBLE PRECISION,
    dii_buy     DOUBLE PRECISION,
    dii_sell    DOUBLE PRECISION,
    source      TEXT DEFAULT 'nse',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_fii_dii_daily_date ON fii_dii_daily (date DESC);
"""

SQL_SCHEDULER_LOG = """
CREATE TABLE IF NOT EXISTS scheduler_log (
    id              BIGSERIAL PRIMARY KEY,
    job_id          TEXT NOT NULL,
    job_name        TEXT NOT NULL,
    scheduled_at    TIMESTAMPTZ NOT NULL,
    started_at      TIMESTAMPTZ,
    completed_at    TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'pending',
    attempt         INTEGER NOT NULL DEFAULT 0,
    error_msg       TEXT,
    UNIQUE (job_id, scheduled_at)
);
CREATE INDEX IF NOT EXISTS idx_sched_log_status ON scheduler_log (status, scheduled_at DESC);
CREATE INDEX IF NOT EXISTS idx_sched_log_job    ON scheduler_log (job_id, scheduled_at DESC);
"""

SQL_PASSWORD_RESET_OTPS = """
CREATE TABLE IF NOT EXISTS password_reset_otps (
  id         SERIAL PRIMARY KEY,
  email      TEXT NOT NULL,
  otp_hash   TEXT NOT NULL,
  expires_at TIMESTAMPTZ NOT NULL,
  used       BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_USER_SESSIONS = """
CREATE TABLE IF NOT EXISTS user_sessions (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     INTEGER REFERENCES users(id) ON DELETE CASCADE,
  token_hash  TEXT NOT NULL,
  device      TEXT,
  ip_address  TEXT,
  location    TEXT,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  last_seen   TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_NOTIFICATION_PREFERENCES = """
CREATE TABLE IF NOT EXISTS notification_preferences (
  user_id        INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
  signal_change  BOOLEAN DEFAULT TRUE,
  price_alert    BOOLEAN DEFAULT TRUE,
  trade_executed BOOLEAN DEFAULT TRUE,
  news_sentiment BOOLEAN DEFAULT FALSE,
  eod_summary    BOOLEAN DEFAULT TRUE,
  weekly_report  BOOLEAN DEFAULT FALSE,
  ch_email       BOOLEAN DEFAULT TRUE,
  ch_push        BOOLEAN DEFAULT TRUE,
  ch_sms         BOOLEAN DEFAULT FALSE,
  updated_at     TIMESTAMPTZ DEFAULT NOW()
);
"""

SQL_BROKER_CONNECTIONS = """
CREATE TABLE IF NOT EXISTS broker_connections (
  id             SERIAL PRIMARY KEY,
  user_id        INTEGER REFERENCES users(id) ON DELETE CASCADE,
  broker         TEXT NOT NULL,
  access_token   TEXT,
  refresh_token  TEXT,
  client_id      TEXT,
  expires_at     TIMESTAMPTZ,
  connected      BOOLEAN DEFAULT FALSE,
  created_at     TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, broker)
);
"""

# ALTER TABLE statements to add new columns to the existing users table (idempotent)
SQL_USER_ALTER = """
ALTER TABLE users ADD COLUMN IF NOT EXISTS phone TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_secret TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS totp_enabled BOOLEAN DEFAULT FALSE;
ALTER TABLE users ADD COLUMN IF NOT EXISTS google_sub TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
ALTER TABLE users ADD COLUMN IF NOT EXISTS default_account TEXT DEFAULT 'PAPER';
ALTER TABLE users ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'INR';
"""

SQL_MARKET_OVERVIEW = """
CREATE TABLE IF NOT EXISTS market_overview (
    date                    DATE PRIMARY KEY,
    nifty500_close          DOUBLE PRECISION,
    nifty500_change_pct     DOUBLE PRECISION,
    nifty50_close           DOUBLE PRECISION,
    nifty50_change_pct      DOUBLE PRECISION,
    sensex_close            DOUBLE PRECISION,
    india_vix               DOUBLE PRECISION,
    advances                INTEGER,
    declines                INTEGER,
    unchanged               INTEGER,
    total_volume            DOUBLE PRECISION,
    fii_net                 DOUBLE PRECISION,
    dii_net                 DOUBLE PRECISION,
    overall_sentiment_score DOUBLE PRECISION,
    fear_greed_label        TEXT
);
"""

# ---------------------------------------------------------------------------
# Hypertables (time-series partitioned by TimescaleDB)
# ---------------------------------------------------------------------------

SQL_PRICES = """
CREATE TABLE IF NOT EXISTS prices (
    id       BIGSERIAL,
    symbol   TEXT NOT NULL,
    exchange TEXT DEFAULT 'NSE',
    date     DATE NOT NULL,
    time     TIME,
    open     DOUBLE PRECISION,
    high     DOUBLE PRECISION,
    low      DOUBLE PRECISION,
    close    DOUBLE PRECISION,
    volume   BIGINT,
    interval TEXT DEFAULT '1d',
    UNIQUE(symbol, date, time, interval)
);
"""

SQL_TECHNICAL_INDICATORS = """
CREATE TABLE IF NOT EXISTS technical_indicators (
    id           BIGSERIAL,
    symbol       TEXT NOT NULL,
    date         DATE NOT NULL,
    rsi_14       DOUBLE PRECISION,
    macd         DOUBLE PRECISION,
    macd_signal  DOUBLE PRECISION,
    macd_hist    DOUBLE PRECISION,
    bb_upper     DOUBLE PRECISION,
    bb_middle    DOUBLE PRECISION,
    bb_lower     DOUBLE PRECISION,
    sma_20       DOUBLE PRECISION,
    sma_50       DOUBLE PRECISION,
    sma_200      DOUBLE PRECISION,
    ema_9        DOUBLE PRECISION,
    ema_21       DOUBLE PRECISION,
    atr_14       DOUBLE PRECISION,
    adx_14       DOUBLE PRECISION,
    stoch_k      DOUBLE PRECISION,
    stoch_d      DOUBLE PRECISION,
    obv          DOUBLE PRECISION,
    support_1    DOUBLE PRECISION,
    support_2    DOUBLE PRECISION,
    support_3    DOUBLE PRECISION,
    resistance_1 DOUBLE PRECISION,
    resistance_2 DOUBLE PRECISION,
    resistance_3 DOUBLE PRECISION,
    signal       TEXT,
    signal_strength DOUBLE PRECISION,
    UNIQUE(symbol, date)
);
"""

SQL_NEWS_SENTIMENT = """
CREATE TABLE IF NOT EXISTS news_sentiment (
    id           BIGSERIAL,
    headline     TEXT NOT NULL,
    source       TEXT,
    published_at TIMESTAMPTZ DEFAULT NOW(),
    symbol       TEXT,
    sentiment    TEXT,
    confidence   DOUBLE PRECISION,
    url          TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);
"""

# ---------------------------------------------------------------------------
# Hypertable conversion + compression
# ---------------------------------------------------------------------------

SQL_HYPERTABLES = [
    # prices: partition monthly by date
    """
    SELECT create_hypertable('prices', 'date',
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );
    """,
    # technical_indicators: partition monthly by date
    """
    SELECT create_hypertable('technical_indicators', 'date',
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );
    """,
    # news_sentiment: partition monthly by published_at
    """
    SELECT create_hypertable('news_sentiment', 'published_at',
        chunk_time_interval => INTERVAL '1 month',
        if_not_exists => TRUE
    );
    """,
]

SQL_COMPRESSION = [
    # Compress prices chunks older than 7 days
    """
    ALTER TABLE prices SET (
        timescaledb.compress,
        timescaledb.compress_orderby   = 'date DESC',
        timescaledb.compress_segmentby = 'symbol, interval'
    );
    """,
    "SELECT add_compression_policy('prices', INTERVAL '7 days', if_not_exists => TRUE);",

    # Compress indicators
    """
    ALTER TABLE technical_indicators SET (
        timescaledb.compress,
        timescaledb.compress_orderby   = 'date DESC',
        timescaledb.compress_segmentby = 'symbol'
    );
    """,
    "SELECT add_compression_policy('technical_indicators', INTERVAL '7 days', if_not_exists => TRUE);",

    # Compress news
    """
    ALTER TABLE news_sentiment SET (
        timescaledb.compress,
        timescaledb.compress_orderby   = 'published_at DESC',
        timescaledb.compress_segmentby = 'symbol'
    );
    """,
    "SELECT add_compression_policy('news_sentiment', INTERVAL '7 days', if_not_exists => TRUE);",
]

# Continuous aggregate — auto-computed from news_sentiment every hour
SQL_NEWS_DAILY_CAGG = """
CREATE MATERIALIZED VIEW IF NOT EXISTS news_daily_sentiment
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', published_at)                          AS date,
    symbol,
    COUNT(*)                                                    AS news_count,
    AVG(confidence)                                             AS avg_confidence,
    AVG(CASE sentiment
        WHEN 'positive' THEN  1.0
        WHEN 'negative' THEN -1.0
        ELSE 0.0 END)                                           AS avg_sentiment,
    SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END)     AS positive_count,
    SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END)     AS negative_count,
    SUM(CASE WHEN sentiment = 'neutral'  THEN 1 ELSE 0 END)     AS neutral_count,
    MAX(CASE WHEN sentiment = 'positive' THEN confidence END)   AS max_positive,
    MAX(CASE WHEN sentiment = 'negative' THEN confidence END)   AS max_negative
FROM news_sentiment
GROUP BY 1, 2
WITH NO DATA;
"""

SQL_NEWS_DAILY_CAGG_POLICY = """
SELECT add_continuous_aggregate_policy('news_daily_sentiment',
    start_offset      => INTERVAL '3 days',
    end_offset        => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists     => TRUE
);
"""

# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------

SQL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol_date ON prices (symbol, date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_prices_symbol      ON prices (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_prices_interval    ON prices (interval);",
    "CREATE INDEX IF NOT EXISTS idx_ti_symbol_date     ON technical_indicators (symbol, date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ti_symbol          ON technical_indicators (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_news_symbol_pub    ON news_sentiment (symbol, published_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_news_sentiment_col ON news_sentiment (sentiment);",
    "CREATE INDEX IF NOT EXISTS idx_ts_date_signal     ON trade_signals (generated_date DESC, signal);",
    "CREATE INDEX IF NOT EXISTS idx_ts_confidence      ON trade_signals (confidence DESC);",
    "CREATE INDEX IF NOT EXISTS idx_ts_symbol          ON trade_signals (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_ts_is_active        ON trade_signals (is_active);",
    "CREATE INDEX IF NOT EXISTS idx_orders_user        ON orders (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_orders_symbol      ON orders (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_positions_user     ON positions (user_id);",
    "CREATE INDEX IF NOT EXISTS idx_positions_symbol   ON positions (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_port_stocks_pid    ON portfolio_stocks (portfolio_id);",
    "CREATE INDEX IF NOT EXISTS idx_port_sectors_pid   ON portfolio_sectors (portfolio_id);",
    "CREATE INDEX IF NOT EXISTS idx_watchlist_user      ON watchlist(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_notifications_user_unread ON notifications(user_id, is_read) WHERE is_read = FALSE;",
]


def init_timescale(conn) -> None:
    """
    Create all tables, hypertables, compression policies,
    continuous aggregate, and indexes.

    Args:
        conn: psycopg2 connection to the TimescaleDB database.
    """
    cur = conn.cursor()

    # Enable extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

    # Regular tables first (no time-series partitioning)
    for sql in [
        SQL_USERS, SQL_PORTFOLIOS, SQL_PORTFOLIO_SECTORS, SQL_PORTFOLIO_STOCKS,
        SQL_RISK_SETTINGS, SQL_TRADE_SIGNALS,
        SQL_ORDERS, SQL_POSITIONS, SQL_WATCHLIST, SQL_NOTIFICATIONS,
        SQL_AUTHORIZED_TRADES, SQL_AUTOPILOT_SETTINGS, SQL_MARKET_OVERVIEW,
        SQL_FII_DII_DAILY, SQL_SCHEDULER_LOG,
        SQL_PASSWORD_RESET_OTPS, SQL_USER_SESSIONS,
        SQL_NOTIFICATION_PREFERENCES, SQL_BROKER_CONNECTIONS,
    ]:
        cur.execute(sql)

    # Add new columns to users table (idempotent ALTER TABLE)
    for stmt in SQL_USER_ALTER.strip().split("\n"):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.rollback()
                logger.warning(f"ALTER TABLE users: {e}")

    # Hypertable candidates
    for sql in [SQL_PRICES, SQL_TECHNICAL_INDICATORS, SQL_NEWS_SENTIMENT]:
        cur.execute(sql)

    conn.commit()

    # Convert to hypertables (must be after commit)
    for sql in SQL_HYPERTABLES:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Hypertable already exists or error: {e}")

    # Compression policies
    for sql in SQL_COMPRESSION:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Compression policy: {e}")

    # Continuous aggregate for news_daily_sentiment
    try:
        cur.execute(SQL_NEWS_DAILY_CAGG)
        conn.commit()
        cur.execute(SQL_NEWS_DAILY_CAGG_POLICY)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Continuous aggregate (may already exist): {e}")

    # Indexes
    for sql in SQL_INDEXES:
        try:
            cur.execute(sql)
        except Exception as e:
            logger.warning(f"Index: {e}")
    conn.commit()
    cur.close()

    print("✅ TimescaleDB schema ready")
    logger.info("TimescaleDB schema initialised")
