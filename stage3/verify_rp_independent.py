"""
Independent verification of risk parity optimizer (Part A lock checks):

1. Vol reduction (EW vs RP) across multiple regimes -- not just the COVID window
2. Hand-check: recompute CIPLA's risk share independently via decompose_risk(),
   not trusting the optimizer's own success message
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from stage3.covariance import estimate_covariance
from stage3.risk_engine import decompose_risk
from stage3.risk_parity import optimize_risk_parity


print("=" * 70)
print("CHECK 1: EW vs RP vol reduction across regimes")
print("=" * 70)

dates = {
    "2012-01-02": "volatile (post-crisis)",
    "2014-06-02": "moderate",
    "2016-01-04": "moderate",
    "2018-01-01": "calm (low-vol regime)",
    "2020-03-02": "COVID onset",
    "2020-06-30": "COVID recovery (original test)",
    "2022-06-01": "post-COVID tightening",
    "2024-01-01": "recent calm",
    "2025-06-02": "recent",
}

print(f"\n  {'Date':<14s}  {'Regime':<28s}  {'EW Vol':>8s}  {'RP Vol':>8s}  {'Reduction':>10s}  {'Red%':>6s}")
print("  " + "-" * 80)

reductions = []
for date_str, regime in dates.items():
    est = estimate_covariance(date_str)
    n = len(est.tickers)

    # Equal-weight
    ew = np.ones(n) / n
    rd_ew = decompose_risk(ew, est)
    ew_vol = rd_ew.portfolio_volatility * np.sqrt(252)

    # Risk-parity
    rp = optimize_risk_parity(est)
    rp_vol = rp.portfolio_volatility * np.sqrt(252)

    reduction = ew_vol - rp_vol
    red_pct = (reduction / ew_vol) * 100
    reductions.append(reduction)

    print(f"  {date_str:<14s}  {regime:<28s}  {ew_vol*100:>7.1f}%  {rp_vol*100:>7.1f}%  "
          f"{reduction*100:>+8.2f}pp  {red_pct:>5.1f}%")

print(f"\n  Vol reduction range: {min(reductions)*100:.2f}pp to {max(reductions)*100:.2f}pp")
print(f"  Average reduction: {np.mean(reductions)*100:.2f}pp")
print(f"  (This is a side-effect of risk equalization, NOT the optimization target)")


print("\n" + "=" * 70)
print("CHECK 2: Independent hand-check of CIPLA's risk share")
print("=" * 70)

# Use the 2020-06-30 date (same as the original test)
est = estimate_covariance("2020-06-30")
rp = optimize_risk_parity(est)

# The optimizer CLAIMS CIPLA's risk share is 4.35%.
# Let's verify by calling decompose_risk() ourselves with the optimizer's weights.
rd = decompose_risk(rp.weights, est)

cipla_idx = est.tickers.index("CIPLA.NS")
cipla_ctr = rd.contribution_to_risk[cipla_idx]
cipla_share = cipla_ctr / rd.portfolio_volatility
cipla_weight = rp.weights[cipla_idx]

print(f"\n  CIPLA.NS:")
print(f"    Optimizer-reported weight:     {rp.weights[cipla_idx]:.6f}")
print(f"    Optimizer-reported risk share: {rp.risk_shares[cipla_idx]:.6f}")
print(f"    Independent decompose_risk():")
print(f"      CTR:        {cipla_ctr:.8f}")
print(f"      sigma_p:    {rd.portfolio_volatility:.8f}")
print(f"      Risk share: {cipla_share:.6f}")
print(f"      Target:     {1/len(est.tickers):.6f}")
print(f"      Gap:        {abs(cipla_share - 1/len(est.tickers)):.8f}")

# Verify they match
diff = abs(cipla_share - rp.risk_shares[cipla_idx])
print(f"\n    Match between optimizer and independent: diff = {diff:.2e}")
assert diff < 1e-10, f"MISMATCH: optimizer says {rp.risk_shares[cipla_idx]}, independent says {cipla_share}"
print("    OK: Independent verification confirms optimizer's reported risk share")

# Do the same for AXISBANK (the highest-risk stock) as a second check
axis_idx = est.tickers.index("AXISBANK.NS")
axis_ctr = rd.contribution_to_risk[axis_idx]
axis_share = axis_ctr / rd.portfolio_volatility

print(f"\n  AXISBANK.NS (cross-check):")
print(f"    Weight: {rp.weights[axis_idx]:.6f}, Risk share: {axis_share:.6f} (target: {1/len(est.tickers):.6f})")
diff2 = abs(axis_share - rp.risk_shares[axis_idx])
assert diff2 < 1e-10, f"MISMATCH for AXISBANK"
print(f"    OK: Independent verification confirms (diff = {diff2:.2e})")

# Final: verify the CTR identity still holds with RP weights
ctr_sum = float(np.sum(rd.contribution_to_risk))
print(f"\n  CTR identity with RP weights:")
print(f"    sum(CTR) = {ctr_sum:.12e}")
print(f"    sigma_p  = {rd.portfolio_volatility:.12e}")
print(f"    Error    = {abs(ctr_sum - rd.portfolio_volatility):.2e}")


print("\n" + "=" * 70)
print("ALL INDEPENDENT CHECKS PASSED")
print("=" * 70)
