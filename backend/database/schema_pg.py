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
    created_at         TIMESTAMPTZ DEFAULT NOW(),
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

SQL_DELIVERY_DATA = """
CREATE TABLE IF NOT EXISTS delivery_data (
    symbol       TEXT NOT NULL,
    date         DATE NOT NULL,
    delivery_pct DOUBLE PRECISION,
    total_volume BIGINT,
    PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_delivery_symbol ON delivery_data (symbol, date DESC);
"""

# NSE trading holidays, scraped from the exchange-communication-holidays page's
# backing API (https://www.nseindia.com/api/holiday-master?type=trading) by
# collectors/nse_holidays_collector.py. This is the trading calendar the whole
# app reasons about: "is the market open today", "what was the last trading
# day", and — most importantly — which dates the `prices` table is *expected*
# to have rows for, so a genuinely missed EOD collection can be told apart from
# a legitimate exchange holiday.
#
# NSE's API only ever returns the current calendar year, so rows accumulate
# year by year as the collector re-runs (upsert, never truncate) — dropping the
# table loses the history that the price-date verification needs. `segment` is
# the NSE segment code ('CM' = capital market / equity, the one we trade);
# holidays are stored per-segment so a future F&O calendar can coexist.
SQL_MARKET_HOLIDAYS = """
CREATE TABLE IF NOT EXISTS market_holidays (
    holiday_date DATE NOT NULL,
    segment      TEXT NOT NULL DEFAULT 'CM',
    exchange     TEXT NOT NULL DEFAULT 'NSE',
    weekday      TEXT,
    description  TEXT,
    source       TEXT,
    updated_at   TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (exchange, segment, holiday_date)
);
CREATE INDEX IF NOT EXISTS idx_market_holidays_date ON market_holidays (holiday_date);
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

# Strategy backtest results — realized out-of-sample equity curves + metrics
# produced by scripts/backtest_signals.py. One row per run (history kept); the
# API serves the newest. Curves/configs/benchmarks are a document-shaped blob,
# stored as JSON text in `payload` (in the DB, not a file — see CLAUDE.md). The
# headline metrics are also flat columns so they're queryable without parsing.
SQL_STRATEGY_BACKTEST = """
CREATE TABLE IF NOT EXISTS strategy_backtest_results (
    id                BIGSERIAL PRIMARY KEY,
    generated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_start      TEXT,
    universe_symbols  INTEGER,
    cost_pct          DOUBLE PRECISION,
    min_rr            DOUBLE PRECISION,
    capital           DOUBLE PRECISION,
    risk_frac         DOUBLE PRECISION,
    headline_label    TEXT,
    headline_cagr     DOUBLE PRECISION,
    headline_maxdd    DOUBLE PRECISION,
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_strategy_backtest_gen ON strategy_backtest_results (generated_at DESC);
"""

# Nifty 500 constituent list (symbol → name → sector). Was a Python module in
# data/ (data.nifty500_full), but data/ is gitignored and excluded from the
# Space deploy, so /api/market/sectors crashed in production with
# ModuleNotFoundError. Reference data belongs in the DB (see CLAUDE.md); seed
# it with scripts/seed_nifty_constituents.py.
SQL_NIFTY_CONSTITUENTS = """
CREATE TABLE IF NOT EXISTS nifty_constituents (
    symbol      TEXT PRIMARY KEY,
    name        TEXT,
    sector      TEXT,
    updated_at  TIMESTAMPTZ DEFAULT NOW(),
    -- Dropped constituents are deactivated, never deleted; is_active = TRUE is
    -- the authoritative tradable universe. See SQL_NIFTY_CONSTITUENTS_ALTER.
    is_active   BOOLEAN NOT NULL DEFAULT TRUE,
    removed_at  TIMESTAMPTZ
);
-- NB: the matching index lives in SQL_NIFTY_CONSTITUENTS_ALTER, not here. On a
-- pre-existing table CREATE TABLE IF NOT EXISTS is a no-op, so an index over
-- is_active in this block would run before the ALTER adds the column and fail
-- with "column is_active does not exist".
"""

# Per-symbol model training metrics from scripts/retrain_walk_forward.py. Was
# logged only to data/retrain_results.csv — a file, and worse, /api/backtest/summary
# read EVERY row ever appended (all historical runs blended together, inflating
# total_models and averaging stale metrics). Now one row per (run_id, symbol);
# the API reads only the latest run_id. Reference/stats data belongs in the DB
# (see CLAUDE.md). `precision` is a non-reserved keyword in PG — safe as a column.
SQL_MODEL_TRAINING_STATS = """
CREATE TABLE IF NOT EXISTS model_training_stats (
    run_id        TEXT NOT NULL,
    symbol        TEXT NOT NULL,
    status        TEXT,
    best_model    TEXT,
    horizon       TEXT,
    accuracy      DOUBLE PRECISION,
    precision     DOUBLE PRECISION,
    recall        DOUBLE PRECISION,
    f1            DOUBLE PRECISION,
    quality_tier  TEXT,
    corp_adj      BOOLEAN,
    error         TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_model_training_stats_run ON model_training_stats (run_id, created_at DESC);
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

# Security fix (audit C1): portfolios had no ownership column at all, so
# every /api/portfolio/* route was a full-CRUD IDOR — any caller could
# read/rebalance/delete any portfolio by guessing an integer id. Nullable
# because existing rows predate the ownership concept.
SQL_PORTFOLIOS_ALTER = """
ALTER TABLE portfolios ADD COLUMN IF NOT EXISTS user_id BIGINT REFERENCES users(id);
CREATE INDEX IF NOT EXISTS idx_portfolios_user_id ON portfolios (user_id);
"""

# Migration: created_at was added to SQL_POSITIONS later, but CREATE TABLE
# IF NOT EXISTS never adds columns to a table that already exists — any DB
# created before that change (e.g. the test instance) is missing the column
# and every INSERT INTO positions fails.
SQL_POSITIONS_ALTER = """
ALTER TABLE positions ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW();
"""

# Migration: an index review drops names, and step_constituents used to DELETE
# those rows outright — leaving no record a symbol was ever a constituent, and
# no way to tell "dropped from the index" from "never seen". So every job that
# needed the tradable universe fell back to `SELECT DISTINCT symbol FROM prices`,
# which is 537, not 500: the current names, plus 33 de-indexed ones whose price
# history is deliberately kept for backtests, plus the 4 index tickers (^NSEI,
# ^BSESN, ^INDIAVIX, ^CRSLDX) that are benchmarks rather than stocks. The weekly
# retrain trained all of them.
#
# Deactivating instead of deleting makes nifty_constituents the authoritative
# tradable universe — `WHERE is_active = TRUE` is exactly the 500, and a name
# that left the index stays on record with the date it went.
SQL_NIFTY_CONSTITUENTS_ALTER = """
ALTER TABLE nifty_constituents ADD COLUMN IF NOT EXISTS is_active BOOLEAN NOT NULL DEFAULT TRUE;
ALTER TABLE nifty_constituents ADD COLUMN IF NOT EXISTS removed_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS idx_nifty_constituents_active ON nifty_constituents (is_active);
"""

# Migration: widen trade_signals UNIQUE from (symbol, date) to (symbol, date, horizon)
# so that 6 per-horizon signals can coexist for the same stock on the same day.
SQL_TRADE_SIGNALS_MIGRATE = [
    "ALTER TABLE trade_signals DROP CONSTRAINT IF EXISTS trade_signals_symbol_generated_date_key",
    """DO $$ BEGIN
        IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conname = 'trade_signals_symbol_date_horizon_key'
        ) THEN
            ALTER TABLE trade_signals
              ADD CONSTRAINT trade_signals_symbol_date_horizon_key
              UNIQUE (symbol, generated_date, model_horizon);
        END IF;
    END $$""",
]

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

SQL_AV_COVERAGE_TRACKER = """
CREATE TABLE IF NOT EXISTS av_coverage_tracker (
    nse_symbol      TEXT PRIMARY KEY,
    adr_ticker      TEXT NOT NULL,
    last_covered    DATE,
    articles_total  INT DEFAULT 0,
    attempt_count   INT DEFAULT 0
);
"""

SQL_CORPORATE_ACTIONS = """
CREATE TABLE IF NOT EXISTS corporate_actions (
    id              BIGSERIAL PRIMARY KEY,
    nse_symbol      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    ex_date         DATE NOT NULL,
    event_type      TEXT NOT NULL,
    ratio           TEXT,
    adj_factor      DOUBLE PRECISION,
    notes           TEXT,
    source          TEXT DEFAULT 'trendlyne',
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (nse_symbol, ex_date, event_type)
);
CREATE INDEX IF NOT EXISTS idx_corp_actions_symbol  ON corporate_actions (nse_symbol);
CREATE INDEX IF NOT EXISTS idx_corp_actions_ex_date ON corporate_actions (ex_date DESC);
"""

# Persistent application logs (the HF Space container filesystem is ephemeral,
# so file logs vanish on every restart — this table is the durable copy).
# Hypertable + 30-day retention policy; fed by api/logging_setup.DBLogHandler.
SQL_APP_LOGS = """
CREATE TABLE IF NOT EXISTS app_logs (
    time     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    level    TEXT NOT NULL,
    logger   TEXT NOT NULL,
    message  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_app_logs_time  ON app_logs (time DESC);
CREATE INDEX IF NOT EXISTS idx_app_logs_level ON app_logs (level, time DESC);
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
    # app_logs: partition weekly by time
    """
    SELECT create_hypertable('app_logs', 'time',
        chunk_time_interval => INTERVAL '1 week',
        if_not_exists => TRUE
    );
    """,
]

# Drop old chunks automatically — logs only need a rolling window.
SQL_RETENTION = [
    "SELECT add_retention_policy('app_logs', INTERVAL '30 days', if_not_exists => TRUE);",
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

# Continuous aggregate — auto-computed from news_sentiment every hour.
#
# The sentiment column's canonical format is a SIGNED NUMERIC STRING
# ("-0.83", "0.0") — sign = direction, magnitude = strength — written by the
# FinBERT scorer and all collectors. The CASE below also tolerates legacy
# text labels ('positive'/'negative'/'neutral') so a stray label row can
# never break a cagg refresh; anything unparseable becomes NULL (excluded
# from the aggregates). The old definition treated ONLY labels as signal,
# silently zeroing every numeric row — see _NEWS_CAGG_OLD_MARKER migration.
_SENTIMENT_VAL = """CASE
        WHEN sentiment = 'positive' THEN  COALESCE(confidence, 1.0)
        WHEN sentiment = 'negative' THEN -COALESCE(confidence, 1.0)
        WHEN sentiment = 'neutral'  THEN 0.0
        WHEN sentiment ~ '^-?[0-9]+\\.?[0-9]*([eE]-?[0-9]+)?$'
            THEN sentiment::double precision
        ELSE NULL END"""

SQL_NEWS_DAILY_CAGG = f"""
CREATE MATERIALIZED VIEW IF NOT EXISTS news_daily_sentiment
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', published_at)                          AS date,
    symbol,
    COUNT(*)                                                    AS news_count,
    AVG(confidence)                                             AS avg_confidence,
    AVG({_SENTIMENT_VAL})                                       AS avg_sentiment,
    SUM(CASE WHEN ({_SENTIMENT_VAL}) > 0 THEN 1 ELSE 0 END)     AS positive_count,
    SUM(CASE WHEN ({_SENTIMENT_VAL}) < 0 THEN 1 ELSE 0 END)     AS negative_count,
    SUM(CASE WHEN ({_SENTIMENT_VAL}) = 0 THEN 1 ELSE 0 END)     AS neutral_count,
    MAX(CASE WHEN ({_SENTIMENT_VAL}) > 0 THEN confidence END)   AS max_positive,
    MAX(CASE WHEN ({_SENTIMENT_VAL}) < 0 THEN confidence END)   AS max_negative
FROM news_sentiment
GROUP BY 1, 2
WITH NO DATA;
"""

# The numeric-format definition is recognisable by its regex literal. If the
# live view's definition lacks it, the cagg predates the numeric-format fix
# (it zeroed all numeric rows) and must be dropped, recreated, and refreshed.
_NEWS_CAGG_NEW_MARKER = "[eE]-?[0-9]+"

# One-time data normalisation: convert legacy label rows (alphavantage
# collector wrote these until 2026-07) to the canonical signed-numeric
# format. Idempotent — matches nothing once converted. The rows live in
# compressed chunks, so lift the DML decompression cap for this session
# (0 = unlimited) or the UPDATE aborts with "tuple decompression limit
# exceeded by operation".
SQL_SENTIMENT_LABEL_MIGRATE = """
SET LOCAL timescaledb.max_tuples_decompressed_per_dml_transaction TO 0;
UPDATE news_sentiment SET sentiment = (CASE
    WHEN sentiment = 'positive' THEN  COALESCE(confidence, 1.0)
    WHEN sentiment = 'negative' THEN -COALESCE(confidence, 1.0)
    ELSE 0.0 END)::text
WHERE sentiment IN ('positive', 'negative', 'neutral');
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
    # Required by insert_price()/insert_prices_batch()'s
    # "ON CONFLICT (symbol, date, interval) WHERE time IS NULL" upsert —
    # without this exact partial unique index, daily-row upserts fail with
    # "no unique or exclusion constraint matching the ON CONFLICT specification".
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_prices_daily_unique ON prices (symbol, date, interval) WHERE (time IS NULL);",
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
        SQL_TRADE_SIGNALS, SQL_RISK_SETTINGS,
        SQL_ORDERS, SQL_POSITIONS, SQL_WATCHLIST, SQL_NOTIFICATIONS,
        SQL_AUTHORIZED_TRADES, SQL_AUTOPILOT_SETTINGS, SQL_MARKET_OVERVIEW,
        SQL_FII_DII_DAILY, SQL_SCHEDULER_LOG, SQL_STRATEGY_BACKTEST,
        SQL_NIFTY_CONSTITUENTS, SQL_MODEL_TRAINING_STATS,
        SQL_PASSWORD_RESET_OTPS, SQL_USER_SESSIONS,
        SQL_NOTIFICATION_PREFERENCES, SQL_BROKER_CONNECTIONS,
        SQL_AV_COVERAGE_TRACKER,
        SQL_CORPORATE_ACTIONS, SQL_DELIVERY_DATA, SQL_MARKET_HOLIDAYS,
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

    # Add ownership column to portfolios (idempotent ALTER TABLE) — see C1 fix
    for stmt in SQL_PORTFOLIOS_ALTER.strip().split("\n"):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.rollback()
                logger.warning(f"ALTER TABLE portfolios: {e}")

    # Add created_at to positions (idempotent ALTER TABLE)
    for stmt in SQL_POSITIONS_ALTER.strip().split("\n"):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.rollback()
                logger.warning(f"ALTER TABLE positions: {e}")

    # Add is_active/removed_at to nifty_constituents (idempotent ALTER TABLE)
    for stmt in SQL_NIFTY_CONSTITUENTS_ALTER.strip().split("\n"):
        stmt = stmt.strip()
        if stmt:
            try:
                cur.execute(stmt)
            except Exception as e:
                conn.rollback()
                logger.warning(f"ALTER TABLE nifty_constituents: {e}")

    # Widen trade_signals UNIQUE constraint to include model_horizon (per-horizon signals)
    for stmt in SQL_TRADE_SIGNALS_MIGRATE:
        try:
            cur.execute(stmt)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"trade_signals migration: {e}")

    # Hypertable candidates
    for sql in [SQL_PRICES, SQL_TECHNICAL_INDICATORS, SQL_NEWS_SENTIMENT,
                SQL_APP_LOGS]:
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

    # Retention policies
    for sql in SQL_RETENTION:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Retention policy: {e}")

    # Compression policies
    for sql in SQL_COMPRESSION:
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Compression policy: {e}")

    # Normalise legacy label-format sentiment rows to signed numeric (one-time;
    # idempotent). Must happen before any cagg refresh so corrected values land.
    try:
        cur.execute(SQL_SENTIMENT_LABEL_MIGRATE)
        if cur.rowcount:
            logger.info(f"sentiment format migration: {cur.rowcount} label rows → numeric")
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"sentiment label migration: {e}")

    # If the live cagg still has the old label-based definition (which zeroed
    # every numeric-format row), drop it so the numeric-aware one replaces it.
    cagg_dropped = False
    try:
        cur.execute("""SELECT view_definition FROM timescaledb_information.continuous_aggregates
                       WHERE view_name = 'news_daily_sentiment'""")
        row = cur.fetchone()
        if row and _NEWS_CAGG_NEW_MARKER not in (row[0] or ""):
            logger.info("Dropping outdated news_daily_sentiment cagg (label-based definition)...")
            cur.execute("DROP MATERIALIZED VIEW news_daily_sentiment")
            conn.commit()
            cagg_dropped = True
    except Exception as e:
        conn.rollback()
        logger.warning(f"cagg definition check: {e}")

    # Continuous aggregate for news_daily_sentiment
    try:
        cur.execute(SQL_NEWS_DAILY_CAGG)
        conn.commit()
        cur.execute(SQL_NEWS_DAILY_CAGG_POLICY)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.warning(f"Continuous aggregate (may already exist): {e}")

    # After a definition migration, rebuild the full history (the hourly policy
    # only refreshes the trailing 3 days). CALL cannot run in a transaction.
    if cagg_dropped:
        try:
            old_autocommit = conn.autocommit
            conn.autocommit = True
            cur.execute("CALL refresh_continuous_aggregate('news_daily_sentiment', NULL, NULL)")
            conn.autocommit = old_autocommit
            logger.info("news_daily_sentiment fully refreshed with numeric-format definition")
        except Exception as e:
            logger.warning(f"cagg full refresh: {e}")

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
