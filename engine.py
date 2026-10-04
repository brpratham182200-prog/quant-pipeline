"""
BacktestEngine

Responsibility: the event loop. Ties DataFeed, both Strategies,
Portfolio, and ExecutionHandler together bar by bar. This is the only
file that should know about ALL the other components -- everything else
stays decoupled (Strategy doesn't know Portfolio exists, Portfolio
doesn't know ExecutionHandler's cost model, etc).

Owns: control flow and the signal -> fill lag (bar t -> bar t+1). Runs
BOTH strategies every bar (not one at a time), since Portfolio needs
both full signal sets simultaneously to resolve cross-strategy conflicts.

Does NOT own: any trading, sizing, ranking, or cost logic itself.
"""

from data_feed import DataFeed
from strategy import MovingAverageCrossoverStrategy, RSIMeanReversionStrategy
from portfolio import Portfolio
from execution_handler import ExecutionHandler
from performance_analyzer import PerformanceAnalyzer
from benchmark import build_equal_weight_benchmark
from config import VOLATILITY_LOOKBACK


class BacktestEngine:
    def __init__(self, feed: DataFeed, ma_strategy: MovingAverageCrossoverStrategy,
                 rsi_strategy: RSIMeanReversionStrategy, portfolio: Portfolio,
                 execution: ExecutionHandler, history_lookback: int = 50):
        self.feed = feed
        self.ma_strategy = ma_strategy
        self.rsi_strategy = rsi_strategy
        self.portfolio = portfolio
        self.execution = execution
        self.history_lookback = history_lookback  # must be >= max(ma.long_window, rsi.period+1)
        self._pending_orders: list = []  # orders sized on bar t, awaiting bar t+1 fill

    def run(self) -> PerformanceAnalyzer:
        for date in self.feed:
            # 1. Fill any orders queued from the PREVIOUS bar, using
            #    TODAY's bar as the t+1 fill bar.
            for order in self._pending_orders:
                next_bar = self.feed.current_bar(order.ticker)
                fill = self.execution.execute_order(order, next_bar)
                self.portfolio.update_on_fill(
                    fill.ticker, fill.direction, fill.quantity,
                    fill.fill_price, fill.cost, strategy_id=order.strategy_id
                )
            self._pending_orders.clear()

            # 2. Generate BOTH strategies' signals for ALL tickers off
            #    today's bar. Portfolio.process_signals needs the full
            #    cross-section from both strategies at once to resolve
            #    same-ticker conflicts and rank globally.
            ma_signals, rsi_signals = [], []
            current_prices, volatilities = {}, {}
            for ticker in self.feed.tickers:
                bar = self.feed.current_bar(ticker)
                history = self.feed.history(ticker, lookback=self.history_lookback)
                ma_signals.append(self.ma_strategy.generate_signals(ticker, bar, history))
                rsi_signals.append(self.rsi_strategy.generate_signals(ticker, bar, history))
                current_prices[ticker] = bar["Close"]

                returns = history["Close"].pct_change().dropna()
                volatilities[ticker] = (returns.tail(VOLATILITY_LOOKBACK).std()
                                         if len(returns) >= 2 else float("nan"))

            # 3. Resolve signals into orders (exits + ranked new entries,
            #    vol-targeted sizing) -- this becomes tomorrow's fills.
            new_orders = self.portfolio.process_signals(
                ma_signals, rsi_signals, current_prices, volatilities
            )
            self._pending_orders.extend(new_orders)

            # 4. Mark to market at the end of the bar.
            self.portfolio.mark_to_market(date, current_prices)

        # 5. Build the equal-weight buy-and-hold benchmark over the same
        #    universe/date range, for a comparison that actually means
        #    something (see quiz Q5).
        benchmark_curve = build_equal_weight_benchmark(self.feed, self.portfolio.initial_cash)
        return PerformanceAnalyzer(self.portfolio, benchmark_equity_curve=benchmark_curve)
