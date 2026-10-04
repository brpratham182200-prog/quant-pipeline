"""
PerformanceAnalyzer

Responsibility: consume Portfolio's equity_curve and produce standard
performance metrics, plus a benchmark comparison (per quiz Q5 -- absolute
returns mean nothing without a reference point).

Owns: metric calculations. Does NOT own: any trading logic.
"""

import numpy as np
import pandas as pd
from portfolio import Portfolio

TRADING_DAYS_PER_YEAR = 252


class PerformanceAnalyzer:
    def __init__(self, portfolio: Portfolio, benchmark_equity_curve: pd.Series | None = None):
        self.portfolio = portfolio
        self.benchmark = benchmark_equity_curve  # equal-weight buy-and-hold, see benchmark.py

    def equity_series(self) -> pd.Series:
        if not self.portfolio.equity_curve:
            raise ValueError("Portfolio has no equity curve -- did you run the engine?")
        df = pd.DataFrame(self.portfolio.equity_curve).set_index("date")
        return df["total_equity"]

    def total_return(self) -> float:
        eq = self.equity_series()
        return eq.iloc[-1] / eq.iloc[0] - 1

    def annualized_return(self) -> float:
        eq = self.equity_series()
        years = len(eq) / TRADING_DAYS_PER_YEAR
        if years <= 0:
            return float("nan")
        return (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        risk_free_rate is an ANNUAL rate (e.g. 0.07 for 7%), converted
        to daily internally. Defaults to 0.0 as a placeholder -- this
        technically overstates Sharpe. TODO: replace with a real 91-day
        T-bill yield before trusting this number in front of anyone.
        """
        eq = self.equity_series()
        daily_returns = eq.pct_change().dropna()
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        excess = daily_returns - daily_rf
        if excess.std() == 0 or len(excess) < 2:
            return float("nan")
        return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)

    def max_drawdown(self) -> float:
        eq = self.equity_series()
        running_max = eq.cummax()
        return ((eq - running_max) / running_max).min()

    def benchmark_comparison(self, risk_free_rate: float = 0.0) -> dict:
        """
        Returns strategy vs. benchmark side by side -- this is the table
        to actually show, not the strategy's numbers alone (per quiz Q5).
        Requires `benchmark_equity_curve` to have been provided at
        construction (see benchmark.build_equal_weight_benchmark).
        """
        if self.benchmark is None:
            raise ValueError(
                "No benchmark_equity_curve was provided to PerformanceAnalyzer -- "
                "pass one at construction (see benchmark.py)."
            )

        bm_returns = self.benchmark.pct_change().dropna()
        daily_rf = (1 + risk_free_rate) ** (1 / TRADING_DAYS_PER_YEAR) - 1
        bm_excess = bm_returns - daily_rf
        bm_sharpe = ((bm_excess.mean() / bm_excess.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
                     if bm_excess.std() != 0 else float("nan"))
        bm_running_max = self.benchmark.cummax()
        bm_drawdown = ((self.benchmark - bm_running_max) / bm_running_max).min()
        bm_total_return = self.benchmark.iloc[-1] / self.benchmark.iloc[0] - 1

        return {
            "strategy_total_return": self.total_return(),
            "strategy_sharpe": self.sharpe_ratio(risk_free_rate),
            "strategy_max_drawdown": self.max_drawdown(),
            "benchmark_total_return": bm_total_return,
            "benchmark_sharpe": bm_sharpe,
            "benchmark_max_drawdown": bm_drawdown,
        }
