"""
================================================================================
Data Access Layer — Query Functions for quant.db
================================================================================
Provides clean DataFrame-returning functions for the most common query patterns.
All functions handle connection lifecycle internally.
================================================================================
"""
import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "quant.db")


def _get_conn():
    """Get a connection to the quant database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def get_prices(symbol, start, end, exclude_quality_issues=False):
    """
    Get daily OHLCV prices for a ticker within a date range.

    Args:
        symbol: Ticker symbol (e.g. 'TCS.NS')
        start: Start date string (e.g. '2020-01-01')
        end: End date string (e.g. '2020-12-31')
        exclude_quality_issues: If True, only return rows with data_quality='clean'

    Returns:
        pd.DataFrame indexed by trade_date with columns:
        open, high, low, close, volume, data_quality
    """
    conn = _get_conn()
    query = """
        SELECT dp.trade_date, dp.open, dp.high, dp.low, dp.close,
               dp.volume, dp.data_quality
        FROM daily_prices dp
        JOIN tickers t ON dp.ticker_id = t.ticker_id
        WHERE t.symbol = ? AND dp.trade_date BETWEEN ? AND ?
    """
    params = [symbol, start, end]

    if exclude_quality_issues:
        query += " AND dp.data_quality = 'clean'"

    query += " ORDER BY dp.trade_date"

    df = pd.read_sql(query, conn, params=params, parse_dates=["trade_date"])
    conn.close()
    return df.set_index("trade_date")


def get_ticker_info(symbol=None):
    """
    Get ticker metadata. If symbol is None, returns all tickers.

    Args:
        symbol: Optional ticker symbol (e.g. 'TCS.NS')

    Returns:
        pd.DataFrame with ticker metadata
    """
    conn = _get_conn()

    if symbol:
        query = "SELECT * FROM tickers WHERE symbol = ?"
        df = pd.read_sql(query, conn, params=[symbol])
    else:
        query = "SELECT * FROM tickers ORDER BY sector, symbol"
        df = pd.read_sql(query, conn)

    conn.close()
    return df


def get_calendar(start=None, end=None):
    """
    Get trading calendar dates.

    Args:
        start: Optional start date string
        end: Optional end date string

    Returns:
        pd.DataFrame with trade_date column
    """
    conn = _get_conn()
    query = "SELECT * FROM trading_calendar"
    params = []

    conditions = []
    if start:
        conditions.append("trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("trade_date <= ?")
        params.append(end)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY trade_date"

    df = pd.read_sql(query, conn, params=params, parse_dates=["trade_date"])
    conn.close()
    return df


def get_quality_issues(symbol=None):
    """
    Get data quality log entries.

    Args:
        symbol: Optional ticker symbol to filter by

    Returns:
        pd.DataFrame with quality issue details
    """
    conn = _get_conn()

    if symbol:
        query = """
            SELECT dql.*, t.symbol
            FROM data_quality_log dql
            JOIN tickers t ON dql.ticker_id = t.ticker_id
            WHERE t.symbol = ?
            ORDER BY dql.trade_date
        """
        df = pd.read_sql(query, conn, params=[symbol])
    else:
        query = """
            SELECT dql.*, t.symbol
            FROM data_quality_log dql
            JOIN tickers t ON dql.ticker_id = t.ticker_id
            ORDER BY dql.trade_date, t.symbol
        """
        df = pd.read_sql(query, conn)

    conn.close()
    return df


def get_close_matrix(start=None, end=None):
    """
    Get a close-price matrix (dates x tickers) — the shape most strategies need.

    Args:
        start: Optional start date string
        end: Optional end date string

    Returns:
        pd.DataFrame with DatetimeIndex and one column per ticker
    """
    conn = _get_conn()
    query = """
        SELECT dp.trade_date, t.symbol, dp.close
        FROM daily_prices dp
        JOIN tickers t ON dp.ticker_id = t.ticker_id
    """
    params = []

    conditions = []
    if start:
        conditions.append("dp.trade_date >= ?")
        params.append(start)
    if end:
        conditions.append("dp.trade_date <= ?")
        params.append(end)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY dp.trade_date, t.symbol"

    df = pd.read_sql(query, conn, params=params, parse_dates=["trade_date"])
    conn.close()

    # Pivot to matrix form
    matrix = df.pivot(index="trade_date", columns="symbol", values="close")
    return matrix
