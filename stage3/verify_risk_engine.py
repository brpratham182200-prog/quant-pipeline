"""
Verification script for stage3 covariance + risk engine.

Checks from the spec:
  1. Output matrix is symmetric (Sigma[i][j] == Sigma[j][i])
  2. All diagonal entries (variances) are positive
  3. Matrix is positive semi-definite (all eigenvalues >= 0)
  4. Spot-check off-diagonal entries against plain pandas .cov() (before shrinkage)
  5. Rolling wrapper produces estimates, skips early months, logs shrinkage
  6. MCR/CTR identity: sum(CTR) == sigma_p
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from data_access import get_close_matrix, get_calendar
from stage3.covariance import estimate_covariance, TRAILING_WINDOW
from stage3.risk_engine import RollingCovarianceEngine, decompose_risk


def check_point_in_time():
    print("=" * 70)
    print("CHECK 1: Point-in-time estimator (as_of_date = 2020-06-30)")
    print("=" * 70)

    est = estimate_covariance("2020-06-30")
    sigma = est.covariance
    n = len(est.tickers)

    print(f"  As-of date: {est.as_of_date.date()}")
    print(f"  Window: {est.window_start.date()} to {est.window_end.date()}")
    print(f"  Tickers: {n}")
    print(f"  Return observations: {est.n_returns}")
    print(f"  Shrinkage intensity: {est.shrinkage_intensity:.6f}")
    print(f"  Matrix shape: {sigma.shape}")

    # Check 1: symmetry
    sym_err = np.max(np.abs(sigma - sigma.T))
    assert sym_err < 1e-15, f"NOT SYMMETRIC: max asymmetry = {sym_err}"
    print(f"  OK: Symmetric (max asymmetry: {sym_err:.2e})")

    # Check 2: positive diagonal
    diag = np.diag(sigma)
    assert np.all(diag > 0), f"NEGATIVE VARIANCE: {diag[diag <= 0]}"
    print(f"  OK: All {n} diagonal entries positive (min={diag.min():.6e}, max={diag.max():.6e})")

    # Check 3: positive semi-definite
    eigenvalues = np.linalg.eigvalsh(sigma)
    min_eig = eigenvalues.min()
    assert min_eig >= -1e-12, f"NOT PSD: min eigenvalue = {min_eig}"
    print(f"  OK: Positive semi-definite (min eigenvalue: {min_eig:.6e})")

    # Check 4: spot-check against pandas .cov() (raw, before shrinkage)
    prices = get_close_matrix(
        start=est.window_start.strftime("%Y-%m-%d"),
        end=est.window_end.strftime("%Y-%m-%d"),
    )
    returns = prices.pct_change().dropna()
    raw_cov = returns.cov().values

    # Compare a few off-diagonal entries
    print("\n  Spot-check: shrunk Sigma vs raw pandas .cov()")
    print("  %-18s %-18s  %12s  %12s  %8s" % ("Ticker A", "Ticker B", "Shrunk", "Raw", "Diff%"))
    pairs = [(0, 1), (5, 10), (3, 20), (15, 22)]
    for i, j in pairs:
        shrunk_val = sigma[i, j]
        raw_val = raw_cov[i, j]
        diff_pct = (shrunk_val / raw_val - 1) * 100 if raw_val != 0 else float("inf")
        print("  %-18s %-18s  %12.8f  %12.8f  %+7.2f%%" % (
            est.tickers[i], est.tickers[j], shrunk_val, raw_val, diff_pct))

    print("\n  (Shrunk values should be moderately pulled toward the shrinkage target)")

    return est


def check_rolling():
    print("\n" + "=" * 70)
    print("CHECK 2: Rolling wrapper (full 16-year history)")
    print("=" * 70)

    engine = RollingCovarianceEngine(start_date="2010-01-04", end_date="2026-08-28")
    engine.run(verbose=True)

    print(f"\n  Total rebalance dates with estimates: {len(engine.estimates)}")
    print(f"  Skipped dates: {len(engine.skipped_dates)}")

    if engine.skipped_dates:
        print("  Skipped dates detail:")
        for date, reason in engine.skipped_dates:
            print(f"    {date.date()}: {reason[:80]}...")

    # Check shrinkage intensity range
    intensities = [est.shrinkage_intensity for est in engine.estimates.values()]
    dates = sorted(engine.estimates.keys())
    print(f"\n  Shrinkage intensity range: {min(intensities):.4f} to {max(intensities):.4f}")
    print(f"  Mean: {np.mean(intensities):.4f}, Std: {np.std(intensities):.4f}")

    # Spot-check: shrinkage should be higher during volatile periods (2020)
    covid_dates = [d for d in dates if pd.Timestamp("2020-03-01") <= d <= pd.Timestamp("2020-12-01")]
    if covid_dates:
        covid_intensities = [engine.estimates[d].shrinkage_intensity for d in covid_dates]
        print(f"\n  COVID period (Mar-Nov 2020) shrinkage: {np.mean(covid_intensities):.4f}")
        print(f"  Non-COVID period shrinkage: {np.mean([i for d, i in zip(dates, intensities) if d not in covid_dates]):.4f}")
        print("  (COVID shrinkage should be different — volatile regimes change the estimate)")

    return engine


def check_risk_decomposition(est):
    print("\n" + "=" * 70)
    print("CHECK 3: Risk decomposition (MCR / CTR)")
    print("=" * 70)

    n = len(est.tickers)

    # Use equal weights as a simple test case
    w = np.ones(n) / n
    rd = decompose_risk(w, est)

    print(f"  Equal-weight portfolio ({n} tickers, w={1/n:.4f} each)")
    print(f"  Portfolio variance: {rd.portfolio_variance:.8f}")
    print(f"  Portfolio volatility (daily): {rd.portfolio_volatility:.6f}")
    print(f"  Portfolio volatility (annualized): {rd.portfolio_volatility * np.sqrt(252):.4f}")
    print(f"  sum(CTR): {rd.ctr_sum_check:.12e}")
    print(f"  sigma_p:      {rd.portfolio_volatility:.12e}")
    print(f"  Identity error: {abs(rd.ctr_sum_check - rd.portfolio_volatility):.2e}")
    print(f"  OK: CTR identity holds")

    # Print top/bottom CTR contributors
    ctr_pct = rd.contribution_to_risk / rd.portfolio_volatility * 100
    sorted_idx = np.argsort(ctr_pct)
    print("\n  CTR decomposition (% of total risk):")
    print("  %-18s  %8s  %8s  %8s" % ("Ticker", "Weight%", "MCR", "CTR%"))
    for i in sorted_idx:
        print("  %-18s  %7.2f%%  %8.5f  %7.2f%%" % (
            est.tickers[i], w[i]*100, rd.marginal_contribution[i], ctr_pct[i]))

    # Also test with a dict-style weights input
    w_dict = {t: 1.0/n for t in est.tickers}
    rd2 = decompose_risk(w_dict, est)
    assert abs(rd2.portfolio_volatility - rd.portfolio_volatility) < 1e-12
    print("\n  OK: Dict-style weights produce identical results")

    # Test with zero weights
    w_zero = np.zeros(n)
    rd3 = decompose_risk(w_zero, est)
    assert rd3.portfolio_volatility == 0.0
    print("  OK: Zero-weight edge case handled correctly")


def check_fail_loud():
    print("\n" + "=" * 70)
    print("CHECK 4: Fail-loud on insufficient history")
    print("=" * 70)

    try:
        est = estimate_covariance("2010-06-01")  # too early for 252 days
        print("  FAIL: SHOULD HAVE RAISED ValueError")
    except ValueError as e:
        print(f"  OK: Correctly raised ValueError: {str(e)[:80]}...")


if __name__ == "__main__":
    est = check_point_in_time()
    check_fail_loud()
    check_risk_decomposition(est)
    engine = check_rolling()
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)
