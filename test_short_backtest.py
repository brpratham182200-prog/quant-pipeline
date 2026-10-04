"""Quick diagnostic: run a short backtest (Q1 2020) to verify the loop works."""
from data_access import get_ticker_info
from data_feed import DataFeed
from strategy import MovingAverageCrossoverStrategy, RSIMeanReversionStrategy
from portfolio import Portfolio
from execution_handler import ExecutionHandler
from engine import BacktestEngine

tickers = get_ticker_info()["symbol"].tolist()
feed = DataFeed(tickers, "2020-01-01", "2020-03-31")
ma = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
rsi = RSIMeanReversionStrategy(period=14)
portfolio = Portfolio(initial_cash=1_000_000, top_k=5)
execution = ExecutionHandler()
engine = BacktestEngine(feed, ma, rsi, portfolio, execution, history_lookback=50)

print("Running short backtest (Q1 2020)...")
analyzer = engine.run()
eq = portfolio.equity_curve
print("Bars processed: %d" % len(eq))
print("Final equity: %.2f" % eq[-1]["total_equity"])
print("Open positions: %d" % len(portfolio.positions))
for t, p in portfolio.positions.items():
    print("  %s: %d shares, %s, avg=%.2f" % (t, p.quantity, p.strategy_id, p.avg_price))
print("DONE")
