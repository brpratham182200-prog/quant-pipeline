"""
Verify risk parity optimizer:
  1. Does it converge on a single date?
  2. Are risk shares actually equal (within tolerance)?
  3. Do weights obey constraints (sum to 1, all >= 0, none > 10%)?
  4. Run across several months -- does it always converge?
  5. Compare risk shares: equal-weight vs risk-parity
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stage3.covariance import estimate_covariance
from stage3.risk_engine import decompose_risk
from stage3.risk_parity import optimize_risk_parity, MAX_WEIGHT


def check_single_date():
    print("=" * 70)
    print("CHECK 1: Single-date optimization (2020-06-30)")
    print("=" * 70)

    est = estimate_covariance("2020-06-30")
    result = optimize_risk_parity(est)

    print(f"  Converged: {result.converged}")
    print(f"  Iterations: {result.n_iterations}")
    print(f"  Objective value: {result.objective_value:.2e}")
    print(f"  Max risk share gap: {result.max_risk_share_gap:.6f}")
    print(f"  Portfolio vol (daily): {result.portfolio_volatility:.6f}")
    print(f"  Portfolio vol (annual): {result.portfolio_volatility * np.sqrt(252):.4f}")

    # Constraint checks
    print(f"\n  Weights sum: {result.weights.sum():.10f} (should be 1.0)")
    print(f"  Min weight: {result.weights.min():.6f} (should be >= 0)")
    print(f"  Max weight: {result.weights.max():.6f} (should be <= {MAX_WEIGHT})")

    assert abs(result.weights.sum() - 1.0) < 1e-8, "Weights don't sum to 1"
    assert result.weights.min() >= -1e-10, "Negative weight found"
    assert result.weights.max() <= MAX_WEIGHT + 1e-8, f"Weight exceeds {MAX_WEIGHT}"
    print("  OK: All constraints satisfied")

    # Print weights and risk shares side by side
    print(f"\n  {'Ticker':<18s}  {'Weight':>8s}  {'Risk Share':>11s}  {'Target':>8s}  {'Gap':>8s}")
    target = result.target_risk_share
    for i, t in enumerate(result.tickers):
        w = result.weights[i]
        rs = result.risk_shares[i]
        gap = rs - target
        print(f"  {t:<18s}  {w:>7.4f}  {rs:>10.4f}  {target:>7.4f}  {gap:>+7.4f}")

    return est, result


def check_vs_equal_weight(est, rp_result):
    print("\n" + "=" * 70)
    print("CHECK 2: Equal-weight vs risk-parity risk shares")
    print("=" * 70)

    n = len(est.tickers)
    ew = np.ones(n) / n
    rd_ew = decompose_risk(ew, est)
    ew_shares = rd_ew.contribution_to_risk / rd_ew.portfolio_volatility

    print(f"\n  {'Ticker':<18s}  {'EW Risk%':>9s}  {'RP Risk%':>9s}  {'EW Weight':>10s}  {'RP Weight':>10s}")
    for i, t in enumerate(est.tickers):
        print(f"  {t:<18s}  {ew_shares[i]*100:>8.2f}%  {rp_result.risk_shares[i]*100:>8.2f}%  "
              f"{ew[i]*100:>9.2f}%  {rp_result.weights[i]*100:>9.2f}%")

    ew_max_gap = np.max(np.abs(ew_shares - 1/n))
    rp_max_gap = rp_result.max_risk_share_gap

    print(f"\n  Equal-weight max risk share gap: {ew_max_gap:.4f}")
    print(f"  Risk-parity max risk share gap:  {rp_max_gap:.6f}")
    print(f"  Improvement: {ew_max_gap / rp_max_gap:.0f}x more equal")

    ew_vol = rd_ew.portfolio_volatility * np.sqrt(252)
    rp_vol = rp_result.portfolio_volatility * np.sqrt(252)
    print(f"\n  Equal-weight annualized vol: {ew_vol:.4f} ({ew_vol*100:.1f}%)")
    print(f"  Risk-parity annualized vol:  {rp_vol:.4f} ({rp_vol*100:.1f}%)")


def check_multiple_dates():
    print("\n" + "=" * 70)
    print("CHECK 3: Convergence across multiple dates")
    print("=" * 70)

    test_dates = [
        "2012-01-02", "2014-06-02", "2016-01-04",
        "2018-01-01", "2020-03-02", "2022-06-01",
        "2024-01-01", "2025-06-02",
    ]

    prev_weights = None
    all_converged = True

    print(f"\n  {'Date':<14s}  {'Conv':>5s}  {'Iters':>6s}  {'MaxGap':>8s}  {'Obj':>10s}  {'Vol(ann)':>9s}")
    for date_str in test_dates:
        est = estimate_covariance(date_str)
        result = optimize_risk_parity(est, fallback_weights=prev_weights)

        status = "YES" if result.converged else "NO"
        vol_ann = result.portfolio_volatility * np.sqrt(252)
        print(f"  {date_str:<14s}  {status:>5s}  {result.n_iterations:>6d}  "
              f"{result.max_risk_share_gap:>8.6f}  {result.objective_value:>10.2e}  {vol_ann:>8.4f}")

        if not result.converged:
            all_converged = False

        prev_weights = result.weights

    if all_converged:
        print("\n  OK: All dates converged")
    else:
        print("\n  WARNING: Some dates did not converge")


if __name__ == "__main__":
    est, result = check_single_date()
    check_vs_equal_weight(est, result)
    check_multiple_dates()
    print("\n" + "=" * 70)
    print("ALL CHECKS DONE")
    print("=" * 70)
