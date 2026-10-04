"""
risk_parity.py -- Risk-parity optimizer

Job: given one covariance matrix (Sigma), return 23 weights where every
stock contributes equal risk (equal CTR).

How it works, step by step:
  1. Start with inverse-vol weights (closer to the answer than equal weights)
  2. Measure error: how far each stock's risk share is from 1/N
  3. Use scipy optimizer to adjust weights until risk shares equalize
  4. Enforce constraints: long-only, weights sum to 1, no weight > MAX_WEIGHT
  5. If solver fails: fall back to previous month's weights, log loudly

Does NOT use return forecasts. Risk parity is a "risk-only" allocation --
it doesn't try to predict which stocks will go up, only how to spread
risk equally.
"""

import sys
import os
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stage3.covariance import CovarianceEstimate
from stage3.risk_engine import decompose_risk


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
MAX_WEIGHT = 0.10          # 10% cap per stock (user's choice)
CONVERGENCE_TOL = 1e-8     # objective function tolerance for scipy
MAX_ITER = 1000            # maximum solver iterations
RISK_SHARE_TOL = 0.005     # max acceptable gap between any stock's
                           # risk share and 1/N (0.5 percentage points)


# ------------------------------------------------------------------
# Output dataclass
# ------------------------------------------------------------------
@dataclass
class RiskParityResult:
    """Everything the optimizer returns -- not just weights."""
    weights: np.ndarray            # (N,) optimal weights
    tickers: list[str]             # ticker order matching weights
    risk_shares: np.ndarray        # (N,) each stock's fraction of total risk
    target_risk_share: float       # 1/N -- what each share should be
    max_risk_share_gap: float      # largest |risk_share_i - 1/N|
    objective_value: float         # final value of the objective function
    n_iterations: int              # how many iterations the solver used
    converged: bool                # did it meet the tolerance?
    portfolio_volatility: float    # annualized daily vol of the optimized portfolio


# ------------------------------------------------------------------
# The objective function (what the solver minimizes)
# ------------------------------------------------------------------
def _risk_parity_objective(w: np.ndarray, sigma: np.ndarray) -> float:
    """
    Minimize the sum of squared deviations of each stock's risk share
    from the equal target (1/N).

    Risk share of stock i = (w_i * (Sigma @ w)_i) / (w' Sigma w)
                          = CTR_i / portfolio_variance

    We work in variance space (not vol space) to avoid the sqrt,
    which makes the gradient smoother for the solver.
    """
    n = len(w)
    target = 1.0 / n

    sigma_w = sigma @ w
    port_var = w @ sigma_w

    if port_var <= 0:
        return 1e10  # penalty for degenerate portfolios

    # Risk share in variance space: w_i * (Sigma w)_i / port_var
    risk_shares = (w * sigma_w) / port_var

    # Sum of squared deviations from equal
    return float(np.sum((risk_shares - target) ** 2))


# ------------------------------------------------------------------
# The optimizer
# ------------------------------------------------------------------
def optimize_risk_parity(
    cov_estimate: CovarianceEstimate,
    fallback_weights: np.ndarray | None = None,
) -> RiskParityResult:
    """
    Find weights that equalize risk contributions across all stocks.

    Args:
        cov_estimate: the Sigma to optimize against.
        fallback_weights: weights to use if the solver fails to converge.
            If None and solver fails, uses inverse-vol weights as fallback.

    Returns:
        RiskParityResult with optimal weights and diagnostics.
    """
    sigma = cov_estimate.covariance
    tickers = cov_estimate.tickers
    n = len(tickers)
    target_share = 1.0 / n

    # ----------------------------------------------------------
    # Step 1: Initial guess -- inverse-volatility weights
    # ----------------------------------------------------------
    # Diagonal of Sigma = variances. sqrt(diagonal) = volatilities.
    vols = np.sqrt(np.diag(sigma))
    inv_vol = 1.0 / vols
    w0 = inv_vol / inv_vol.sum()

    # Clip initial guess to respect the cap (shouldn't be needed usually,
    # but prevents edge cases with very low-vol stocks)
    w0 = np.clip(w0, 0.0, MAX_WEIGHT)
    w0 = w0 / w0.sum()  # re-normalize after clipping

    # ----------------------------------------------------------
    # Step 2: Define constraints
    # ----------------------------------------------------------
    # Constraint 1: all weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Constraint 2: each weight between 0 and MAX_WEIGHT
    bounds = [(0.0, MAX_WEIGHT)] * n

    # ----------------------------------------------------------
    # Step 3: Run the solver
    # ----------------------------------------------------------
    result = minimize(
        _risk_parity_objective,
        w0,
        args=(sigma,),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": MAX_ITER, "ftol": CONVERGENCE_TOL},
    )

    w_opt = result.x

    # ----------------------------------------------------------
    # Step 4: Compute risk shares using the EXISTING decompose_risk
    # ----------------------------------------------------------
    rd = decompose_risk(w_opt, cov_estimate)
    risk_shares = rd.contribution_to_risk / rd.portfolio_volatility
    max_gap = float(np.max(np.abs(risk_shares - target_share)))

    converged = result.success and max_gap < RISK_SHARE_TOL

    # ----------------------------------------------------------
    # Step 5: Handle failure -- fall back, don't crash
    # ----------------------------------------------------------
    if not converged:
        if fallback_weights is not None:
            fb_weights = fallback_weights
        else:
            # Fall back to the initial inverse-vol guess
            fb_weights = w0

        # Log loudly
        print(
            f"  WARNING: Risk parity solver did not converge.\n"
            f"    scipy success: {result.success}\n"
            f"    scipy message: {result.message}\n"
            f"    max risk share gap: {max_gap:.6f} (tolerance: {RISK_SHARE_TOL})\n"
            f"    objective value: {result.fun:.2e}\n"
            f"    iterations: {result.nit}\n"
            f"    Falling back to {'previous month' if fallback_weights is not None else 'inverse-vol'} weights."
        )

        # Recompute diagnostics with fallback weights
        rd_fb = decompose_risk(fb_weights, cov_estimate)
        risk_shares_fb = rd_fb.contribution_to_risk / rd_fb.portfolio_volatility
        max_gap_fb = float(np.max(np.abs(risk_shares_fb - target_share)))

        return RiskParityResult(
            weights=fb_weights,
            tickers=tickers,
            risk_shares=risk_shares_fb,
            target_risk_share=target_share,
            max_risk_share_gap=max_gap_fb,
            objective_value=float(result.fun),
            n_iterations=result.nit,
            converged=False,
            portfolio_volatility=rd_fb.portfolio_volatility,
        )

    # ----------------------------------------------------------
    # Step 6: Return everything
    # ----------------------------------------------------------
    return RiskParityResult(
        weights=w_opt,
        tickers=tickers,
        risk_shares=risk_shares,
        target_risk_share=target_share,
        max_risk_share_gap=max_gap,
        objective_value=float(result.fun),
        n_iterations=result.nit,
        converged=True,
        portfolio_volatility=rd.portfolio_volatility,
    )
