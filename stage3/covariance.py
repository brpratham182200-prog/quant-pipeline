"""
covariance.py — Point-in-time Ledoit-Wolf shrinkage covariance estimator

Responsibility: given an as-of date, return the shrunk covariance matrix
using the trailing 252 trading days of simple percentage returns.

Imports only from data_access.py — never touches the DB or pickles directly.

Return convention: simple percentage returns (p_t / p_{t-1} - 1), matching
Stage 2's performance_analyzer.py (pct_change()) so metrics are comparable.

Forward-filled rows: included as-is. Only 28 stale rows exist across the
entire dataset (0.03% of ~94,500 rows) — completely immaterial to a
252-day window. This is a deliberate, documented decision.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from dataclasses import dataclass

# data_access.py lives in the parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_calendar, get_close_matrix


TRAILING_WINDOW = 252  # trading days, ~1 year


@dataclass
class CovarianceEstimate:
    """Output of the point-in-time estimator — the shrunk Σ plus metadata."""
    covariance: np.ndarray          # (N, N) shrunk covariance matrix
    tickers: list[str]              # column order matching covariance indices
    as_of_date: pd.Timestamp        # the date this estimate is anchored to
    window_start: pd.Timestamp      # first trading day in the lookback window
    window_end: pd.Timestamp        # last trading day (== as_of_date)
    shrinkage_intensity: float      # Ledoit-Wolf estimated shrinkage coefficient
    n_returns: int                  # number of return observations used


def estimate_covariance(as_of_date: str | pd.Timestamp) -> CovarianceEstimate:
    """
    Compute the Ledoit-Wolf shrunk covariance matrix using the trailing
    252 trading days of simple percentage returns ending on `as_of_date`.

    Steps:
      1. Resolve the trailing 252 trading days from the calendar.
      2. Fail loudly if insufficient history.
      3. Pull the close-price matrix for that window.
      4. Compute simple percentage returns.
      5. Compute raw sample covariance -> apply Ledoit-Wolf shrinkage.
      6. Return the shrunk Σ plus metadata.

    Raises:
        ValueError: if fewer than 252 trading days exist up to as_of_date,
                    or if the price matrix has missing values.
    """
    as_of_date = pd.Timestamp(as_of_date)

    # --- Step 1: resolve the trailing window ---
    # We need 252 return observations, which requires 253 price observations
    # (returns = prices.pct_change() drops the first row).
    calendar = get_calendar(end=as_of_date.strftime("%Y-%m-%d"))
    trading_days = calendar["trade_date"].sort_values().reset_index(drop=True)

    # Find the position of as_of_date in the calendar
    mask = trading_days <= as_of_date
    available_days = trading_days[mask]

    # --- Step 2: fail loudly if not enough history ---
    # Need 253 price days to get 252 return days
    required_price_days = TRAILING_WINDOW + 1  # 253
    if len(available_days) < required_price_days:
        raise ValueError(
            f"Insufficient history for covariance estimation as of {as_of_date.date()}: "
            f"need {required_price_days} trading days, only {len(available_days)} available. "
            f"First usable date is approximately 1 year after data start."
        )

    # Take the last 253 trading days (to produce 252 returns)
    window_days = available_days.iloc[-required_price_days:]
    window_start = window_days.iloc[0]
    window_end = window_days.iloc[-1]

    # --- Step 3: pull close prices ---
    prices = get_close_matrix(
        start=window_start.strftime("%Y-%m-%d"),
        end=window_end.strftime("%Y-%m-%d"),
    )

    # Validate: no NaNs allowed — the rectangular matrix should be complete
    # per Stage 1's data integrity guarantees.
    if prices.isnull().any().any():
        nan_counts = prices.isnull().sum()
        bad_tickers = nan_counts[nan_counts > 0].to_dict()
        raise ValueError(
            f"Close-price matrix has NaN values in window "
            f"{window_start.date()} to {window_end.date()}: {bad_tickers}. "
            f"This violates Stage 1's data integrity — investigate before proceeding."
        )

    if len(prices) != required_price_days:
        raise ValueError(
            f"Expected {required_price_days} price rows in window "
            f"{window_start.date()} to {window_end.date()}, got {len(prices)}. "
            f"Calendar/price mismatch — investigate."
        )

    # --- Step 4: compute simple percentage returns ---
    # Matches Stage 2's convention (performance_analyzer.py uses pct_change())
    returns = prices.pct_change().dropna()

    if len(returns) != TRAILING_WINDOW:
        raise ValueError(
            f"Expected {TRAILING_WINDOW} return observations, got {len(returns)}. "
            f"This shouldn't happen if the price matrix has the right number of rows."
        )

    tickers = list(returns.columns)

    # --- Step 5: Ledoit-Wolf shrinkage ---
    # sklearn's LedoitWolf estimates the optimal shrinkage intensity from
    # the data itself (not a hand-picked number) and returns a properly
    # formed covariance matrix (symmetric, positive semi-definite).
    lw = LedoitWolf()
    lw.fit(returns.values)

    shrunk_cov = lw.covariance_
    shrinkage_intensity = lw.shrinkage_

    # --- Step 6: return result with metadata ---
    return CovarianceEstimate(
        covariance=shrunk_cov,
        tickers=tickers,
        as_of_date=as_of_date,
        window_start=window_start,
        window_end=window_end,
        shrinkage_intensity=shrinkage_intensity,
        n_returns=len(returns),
    )
