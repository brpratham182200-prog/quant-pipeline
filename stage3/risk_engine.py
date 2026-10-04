"""
risk_engine.py — Rolling covariance wrapper + risk decomposition (MCR/CTR)

Responsibility:
  - Call covariance.estimate_covariance() at each monthly rebalance date
    across the full 16-year history.
  - Cache the sequence so optimizers can look up "give me the Σ that was
    live as of rebalance date X" without recomputing.
  - Provide MCR/CTR risk decomposition functions that consume a Σ + weights.

Imports only from data_access.py and covariance.py — never touches the
DB or pickles directly.

Rebalance convention: first trading day of each calendar month (deliberate
choice, documented here, not a silent default).
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_calendar
from stage3.covariance import estimate_covariance, CovarianceEstimate


# ======================================================================
# Part 2: Rolling wrapper
# ======================================================================

class RollingCovarianceEngine:
    """
    Produces a Σ at each monthly rebalance date across the full history.

    Usage:
        engine = RollingCovarianceEngine(start_date="2010-01-04", end_date="2026-08-28")
        engine.run()
        sigma = engine.get_estimate("2020-03-02")  # lookup by rebalance date
    """

    def __init__(self, start_date: str = "2010-01-04", end_date: str = "2026-08-28"):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.estimates: dict[pd.Timestamp, CovarianceEstimate] = {}
        self.skipped_dates: list[tuple[pd.Timestamp, str]] = []  # (date, reason)
        self._rebalance_dates: list[pd.Timestamp] = []

    def _generate_rebalance_schedule(self) -> list[pd.Timestamp]:
        """
        First trading day of each calendar month within [start_date, end_date].

        Convention: first trading day, not last. This is a deliberate choice
        documented in the module docstring.
        """
        calendar = get_calendar(
            start=self.start_date.strftime("%Y-%m-%d"),
            end=self.end_date.strftime("%Y-%m-%d"),
        )
        trading_days = calendar["trade_date"].sort_values().reset_index(drop=True)

        # Group by (year, month), take the first trading day of each group
        monthly_first = (
            trading_days
            .to_frame(name="trade_date")
            .assign(year_month=lambda df: df["trade_date"].dt.to_period("M"))
            .groupby("year_month")["trade_date"]
            .first()
            .sort_values()
            .tolist()
        )
        return monthly_first

    def run(self, verbose: bool = True) -> None:
        """
        Compute Σ at each monthly rebalance date. Skips early months
        where 252 days of history aren't yet available, and logs which
        dates were skipped and why.
        """
        self._rebalance_dates = self._generate_rebalance_schedule()

        if verbose:
            print(f"Rebalance schedule: {len(self._rebalance_dates)} monthly dates "
                  f"from {self._rebalance_dates[0].date()} to {self._rebalance_dates[-1].date()}")

        for i, date in enumerate(self._rebalance_dates):
            try:
                est = estimate_covariance(date)
                self.estimates[date] = est
                if verbose and (i + 1) % 12 == 0:
                    print(f"  {i + 1}/{len(self._rebalance_dates)}: "
                          f"{date.date()} shrinkage={est.shrinkage_intensity:.4f}")
            except ValueError as e:
                self.skipped_dates.append((date, str(e)))
                if verbose:
                    print(f"  SKIPPED {date.date()}: {e}")

        if verbose:
            print(f"\nDone: {len(self.estimates)} estimates computed, "
                  f"{len(self.skipped_dates)} skipped (insufficient history)")

    def get_estimate(self, rebalance_date: str | pd.Timestamp) -> CovarianceEstimate:
        """Look up the Σ for a specific rebalance date."""
        date = pd.Timestamp(rebalance_date)
        if date not in self.estimates:
            available = sorted(self.estimates.keys())
            raise KeyError(
                f"No covariance estimate for {date.date()}. "
                f"Available: {len(available)} dates from "
                f"{available[0].date() if available else 'none'} to "
                f"{available[-1].date() if available else 'none'}"
            )
        return self.estimates[date]

    def get_nearest_estimate(self, date: str | pd.Timestamp) -> CovarianceEstimate:
        """
        Return the most recent Σ estimate on or before `date`.
        Useful for looking up "what covariance matrix was live on date X?"
        """
        date = pd.Timestamp(date)
        available = sorted(self.estimates.keys())
        candidates = [d for d in available if d <= date]
        if not candidates:
            raise KeyError(
                f"No covariance estimate on or before {date.date()}. "
                f"First available: {available[0].date() if available else 'none'}"
            )
        return self.estimates[candidates[-1]]

    @property
    def rebalance_dates(self) -> list[pd.Timestamp]:
        """All rebalance dates that have computed estimates (no skips)."""
        return sorted(self.estimates.keys())


# ======================================================================
# Part 3: Risk decomposition (MCR / CTR)
# ======================================================================

@dataclass
class RiskDecomposition:
    """Output of decompose_risk() — portfolio risk broken down by ticker."""
    portfolio_variance: float           # w' Σ w
    portfolio_volatility: float         # sqrt(portfolio_variance)
    marginal_contribution: np.ndarray   # MCR per ticker: (Σw)_i / σ_p
    contribution_to_risk: np.ndarray    # CTR per ticker: w_i * MCR_i
    tickers: list[str]                  # ticker order matching the arrays
    ctr_sum_check: float                # sum(CTR) — should == portfolio_volatility


def decompose_risk(
    weights: np.ndarray | dict[str, float],
    cov_estimate: CovarianceEstimate,
) -> RiskDecomposition:
    """
    Given portfolio weights and a covariance estimate, compute:
      1. Portfolio variance: w' Σ w
      2. Marginal Contribution to Risk (MCR): (Σw)_i / σ_p
      3. Contribution to Risk (CTR): w_i × MCR_i
      4. Verification: sum(CTR) must equal σ_p (hard identity)

    Args:
        weights: either a 1D array matching cov_estimate.tickers order,
                 or a dict {ticker: weight}.
        cov_estimate: the CovarianceEstimate to use.

    Raises:
        ValueError: if weights don't match tickers, or if the CTR
                    identity fails beyond floating-point tolerance.
    """
    tickers = cov_estimate.tickers
    sigma = cov_estimate.covariance

    # Convert dict weights to array if needed
    if isinstance(weights, dict):
        missing = set(tickers) - set(weights.keys())
        if missing:
            raise ValueError(f"Weights missing for tickers: {missing}")
        w = np.array([weights[t] for t in tickers])
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(tickers):
            raise ValueError(
                f"Weight vector length ({len(w)}) doesn't match "
                f"ticker count ({len(tickers)})"
            )

    # Step 1: portfolio variance = w' Σ w
    sigma_w = sigma @ w                         # (N,) vector
    portfolio_variance = float(w @ sigma_w)      # scalar

    if portfolio_variance < 0:
        raise ValueError(
            f"Negative portfolio variance ({portfolio_variance:.6e}) — "
            f"this means the covariance matrix is not positive semi-definite. "
            f"This should never happen with Ledoit-Wolf output."
        )

    # Step 2: portfolio volatility = sqrt(variance)
    portfolio_vol = np.sqrt(portfolio_variance)

    if portfolio_vol == 0:
        # Edge case: zero-weight portfolio or perfectly hedged
        return RiskDecomposition(
            portfolio_variance=0.0,
            portfolio_volatility=0.0,
            marginal_contribution=np.zeros(len(tickers)),
            contribution_to_risk=np.zeros(len(tickers)),
            tickers=tickers,
            ctr_sum_check=0.0,
        )

    # Step 3: MCR = (Σw)_i / σ_p
    mcr = sigma_w / portfolio_vol

    # Step 4: CTR = w_i × MCR_i
    ctr = w * mcr

    # Step 5: verification — sum(CTR) must equal σ_p exactly (up to float rounding)
    ctr_sum = float(np.sum(ctr))
    tolerance = 1e-10
    if abs(ctr_sum - portfolio_vol) > tolerance:
        raise ValueError(
            f"CTR identity violation: sum(CTR)={ctr_sum:.12e} vs "
            f"σ_p={portfolio_vol:.12e}, diff={abs(ctr_sum - portfolio_vol):.2e}. "
            f"This indicates a bug in the MCR/CTR calculation, not a data issue."
        )

    return RiskDecomposition(
        portfolio_variance=portfolio_variance,
        portfolio_volatility=portfolio_vol,
        marginal_contribution=mcr,
        contribution_to_risk=ctr,
        tickers=tickers,
        ctr_sum_check=ctr_sum,
    )
