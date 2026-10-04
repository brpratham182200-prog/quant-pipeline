"""
Benchmark

Builds an equal-weight buy-and-hold equity curve over the SAME universe
and date range as the backtest, for comparison against strategy
performance (see quiz Q5 -- absolute returns are meaningless without a
reference point).

Owns: constructing ONE static allocation, bought once at the first bar
and held flat with no rebalancing. Does NOT own: any trading logic.
"""

import pandas as pd
from data_feed import DataFeed
from config import per_leg_cost_fraction


def build_equal_weight_benchmark(feed: DataFeed, initial_cash: float) -> pd.Series:
    """
    Buys an equal dollar amount of every ticker in `feed.tickers` at the
    first bar's close, applies a ONE-TIME entry cost (same per-leg bps
    as the strategy's ExecutionHandler -- so the comparison isn't
    artificially unfair to the benchmark, which trades once vs. the
    strategy's many trades), then holds flat. Whole shares only, same
    as the strategy -- leftover cash from rounding sits idle, uninvested.

    Returns a pd.Series of total portfolio value, indexed by date,
    covering the same dates the feed iterates.

    NOTE: re-iterates `feed` from the start (DataFeed.__iter__ resets on
    each `for` loop) -- safe to call after the main engine has already
    run through the same feed.
    """
    entry_cost_frac = per_leg_cost_fraction()
    n = len(feed.tickers)
    per_ticker_cash = initial_cash / n

    shares: dict[str, int] = {}
    leftover_cash = 0.0
    dates, values = [], []

    for i, date in enumerate(feed):
        dates.append(date)
        if i == 0:
            for ticker in feed.tickers:
                price = feed.current_bar(ticker)["Close"]
                qty = int((per_ticker_cash / (1 + entry_cost_frac)) // price)
                shares[ticker] = qty
                spent = qty * price * (1 + entry_cost_frac)
                leftover_cash += per_ticker_cash - spent

        total_value = leftover_cash + sum(
            shares[t] * feed.current_bar(t)["Close"] for t in feed.tickers
        )
        values.append(total_value)

    return pd.Series(values, index=pd.DatetimeIndex(dates), name="benchmark_equity")
