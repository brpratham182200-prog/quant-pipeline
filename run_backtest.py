"""
run_backtest.py

Runs the full Stage 2 pipeline (MA crossover + RSI mean reversion,
shared capital pool, vol-targeted sizing, 30bps cost model) against the
REAL quant.db over the full validated date range, and prints a
benchmark comparison report.

Run this from the same directory as data_access.py and db/quant.db.
"""

import pandas as pd
from data_access import get_ticker_info
from data_feed import DataFeed
from strategy import MovingAverageCrossoverStrategy, RSIMeanReversionStrategy
from portfolio import Portfolio
from execution_handler import ExecutionHandler
from engine import BacktestEngine

# --- Assumptions to confirm before trusting the output ---
INITIAL_CASH = 1_000_000       # placeholder -- see conversation, this is your call to set
START_DATE = "2010-01-04"      # full validated Stage 1 range
END_DATE = "2026-08-28"
TOP_K = 5                      # see config.py -- also a placeholder, not derived
HISTORY_LOOKBACK = 50          # must be >= MA long_window (50)

def main():
    tickers = get_ticker_info()["symbol"].tolist()
    print(f"Loaded {len(tickers)} tickers from DB: {tickers}")
    assert len(tickers) == 23, (
        f"Expected 23 tickers, got {len(tickers)} -- STOP and check the DB "
        f"before trusting anything downstream."
    )

    feed = DataFeed(tickers, START_DATE, END_DATE)
    ma_strategy = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
    rsi_strategy = RSIMeanReversionStrategy(period=14)
    portfolio = Portfolio(initial_cash=INITIAL_CASH, top_k=TOP_K)
    execution = ExecutionHandler()

    engine = BacktestEngine(feed, ma_strategy, rsi_strategy, portfolio, execution,
                             history_lookback=HISTORY_LOOKBACK)

    print("Running backtest...")
    analyzer = engine.run()

    print(f"\nBars processed: {len(portfolio.equity_curve)}")
    print(f"Final cash: {portfolio.cash:,.2f}")
    print(f"Open positions at end: {len(portfolio.positions)}")
    for ticker, pos in portfolio.positions.items():
        print(f"  {ticker}: {pos.quantity} shares, owned by {pos.strategy_id}, "
              f"avg_price={pos.avg_price:.2f}")

    print("\n--- Performance vs. Equal-Weight Buy-and-Hold Benchmark ---")
    comparison = analyzer.benchmark_comparison(risk_free_rate=0.0)  # TODO: real T-bill rate
    for k, v in comparison.items():
        label = "return/drawdown" if ("return" in k or "drawdown" in k) else "ratio"
        print(f"  {k}: {v:.4f}" if label == "return/drawdown" else f"  {k}: {v:.3f}")

    # Save equity curves for independent spot-checking -- don't just trust the printout.
    eq = analyzer.equity_series()
    eq.to_csv("stage2_strategy_equity_curve.csv")
    analyzer.benchmark.to_csv("stage2_benchmark_equity_curve.csv")
    print("\nSaved: stage2_strategy_equity_curve.csv, stage2_benchmark_equity_curve.csv")
    print("Spot-check a few rows of these by hand before trusting the summary above.")


if __name__ == "__main__":
    main()
