-- ============================================================
-- SQLite Schema for Quant Database
-- Adapted from PostgreSQL schema for SQLite compatibility
-- ============================================================

-- ============================================================
-- 1. Ticker master table — one row per instrument
-- ============================================================
CREATE TABLE IF NOT EXISTS tickers (
    ticker_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL UNIQUE,        -- e.g. 'TCS.NS'
    company_name    TEXT,
    sector          TEXT NOT NULL,
    exchange        TEXT NOT NULL,               -- 'NSE' or 'BSE'
    is_active       INTEGER DEFAULT 1,           -- 0 if delisted/replaced
    replaces_symbol TEXT,                        -- e.g. EICHERMOT replaces TATAMOTORS
    listed_date     TEXT,                        -- ISO date string
    notes           TEXT,                        -- free-text: "BSE fallback", "demerger", etc.
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- 2. Daily prices — long format, one row per (ticker, date)
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_prices (
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),
    trade_date      TEXT NOT NULL,               -- ISO date string
    open            REAL NOT NULL,
    high            REAL NOT NULL,
    low             REAL NOT NULL,
    close           REAL NOT NULL,
    volume          INTEGER,                     -- NULL = unknown, not zero
    is_adjusted     INTEGER DEFAULT 1,           -- confirms auto_adjust source
    data_quality    TEXT DEFAULT 'clean',         -- 'clean' | 'forward_filled' | 'vol_missing'
    source          TEXT DEFAULT 'yfinance',
    PRIMARY KEY (ticker_id, trade_date),
    CHECK (high >= low),
    CHECK (open >= 0 AND close >= 0)
);

-- Index for the query pattern: date-range per ticker
CREATE INDEX IF NOT EXISTS idx_prices_ticker_date
    ON daily_prices (ticker_id, trade_date);

-- ============================================================
-- 3. Corporate actions — explicit audit trail
-- ============================================================
CREATE TABLE IF NOT EXISTS corporate_actions (
    action_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker_id       INTEGER NOT NULL REFERENCES tickers(ticker_id),
    action_date     TEXT NOT NULL,
    action_type     TEXT NOT NULL,                -- 'split' | 'bonus' | 'dividend' | 'demerger'
    ratio_or_amount TEXT,                         -- '10:1', '1:1', '75.00 INR', etc.
    notes           TEXT
);

-- ============================================================
-- 4. Trading calendar — reconciled consensus dates
-- ============================================================
CREATE TABLE IF NOT EXISTS trading_calendar (
    trade_date      TEXT PRIMARY KEY,
    is_full_session INTEGER DEFAULT 1,           -- 0 for special/Muhurat sessions
    notes           TEXT
);

-- ============================================================
-- 5. Data quality log — errata notes, structured
-- ============================================================
CREATE TABLE IF NOT EXISTS data_quality_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker_id       INTEGER REFERENCES tickers(ticker_id),
    trade_date      TEXT,
    issue_type      TEXT,                        -- 'stale_row', 'missing_volume', 'exchange_fallback'
    description     TEXT,
    resolution      TEXT,
    logged_at       TEXT DEFAULT (datetime('now'))
);
