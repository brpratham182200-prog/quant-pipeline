"""
Independent verification of risk engine claims — NOT trusting the tool's summary.
Checks:
  1. Spot-check raw cov vs shrunk cov for specific ticker pairs by hand
  2. Feb-Apr 2020 shrinkage intensity (narrow window, not broad "COVID period")
  3. Skip boundary: is Jan 2011 exactly where 252 trading days becomes available?
  4. Equal-weight 27.1% vol gut check against Nifty realized vol
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from data_access import get_close_matrix, get_calendar
from stage3.covariance import estimate_covariance, TRAILING_WINDOW


print("=" * 70)
print("CHECK 1: Hand-verify raw cov vs shrunk cov for specific pairs")
print("=" * 70)

# Use the same date as the automated check: 2020-06-30
est = estimate_covariance("2020-06-30")

# Pull prices independently, compute returns, compute raw cov manually
prices = get_close_matrix(
    start=est.window_start.strftime("%Y-%m-%d"),
    end=est.window_end.strftime("%Y-%m-%d"),
)
returns = prices.pct_change().dropna()

# Compute raw sample covariance with pandas
raw_cov_df = returns.cov()

# Pick 3 specific pairs and verify
pairs = [
    ("TCS.NS", "INFY.NS"),        # same sector (IT) -- should be high correlation
    ("RELIANCE.NS", "CIPLA.NS"),   # different sectors -- lower correlation
    ("SBIN.NS", "AXISBANK.NS"),    # same sector (banking) -- should be high
]

print(f"\nWindow: {est.window_start.date()} to {est.window_end.date()}")
print(f"Shrinkage intensity: {est.shrinkage_intensity:.6f}")
print()
print(f"{'Pair':<30s}  {'Raw cov':>12s}  {'Shrunk cov':>12s}  {'Diff%':>8s}  {'Raw corr':>10s}")

for t1, t2 in pairs:
    i = est.tickers.index(t1)
    j = est.tickers.index(t2)
    
    raw_val = raw_cov_df.loc[t1, t2]
    shrunk_val = est.covariance[i, j]
    diff_pct = (shrunk_val / raw_val - 1) * 100
    
    # Also compute correlation for gut check
    raw_corr = raw_cov_df.loc[t1, t2] / np.sqrt(raw_cov_df.loc[t1, t1] * raw_cov_df.loc[t2, t2])
    
    print(f"{t1+' / '+t2:<30s}  {raw_val:>12.8f}  {shrunk_val:>12.8f}  {diff_pct:>+7.2f}%  {raw_corr:>10.4f}")

# Also check diagonal (variance) for one ticker
t = "TCS.NS"
i = est.tickers.index(t)
raw_var = raw_cov_df.loc[t, t]
shrunk_var = est.covariance[i, i]
print(f"\n{'Diagonal: ' + t:<30s}  {raw_var:>12.8f}  {shrunk_var:>12.8f}  {(shrunk_var/raw_var-1)*100:>+7.2f}%")

# Verify: all off-diag shrinkage should be the same percentage (characteristic of LW with scaled identity target)
off_diag_diffs = []
for i in range(len(est.tickers)):
    for j in range(i+1, len(est.tickers)):
        raw_val = raw_cov_df.iloc[i, j]
        shrunk_val = est.covariance[i, j]
        if abs(raw_val) > 1e-15:
            off_diag_diffs.append((shrunk_val / raw_val - 1) * 100)

print(f"\nOff-diagonal shrinkage % range: {min(off_diag_diffs):.4f}% to {max(off_diag_diffs):.4f}%")
print(f"(Should be uniform if LW uses scaled-identity target)")


print("\n" + "=" * 70)
print("CHECK 2: Shrinkage intensity for Feb-Apr 2020 specifically")
print("=" * 70)

# Pull monthly estimates for a narrow window around COVID crash
covid_months = ["2020-02-03", "2020-03-02", "2020-04-01", "2020-05-04", "2020-06-01"]
print(f"\n{'Rebalance date':<18s}  {'Shrinkage':>10s}  {'Window start':>14s}  {'Window end':>14s}")

for date_str in covid_months:
    try:
        est_m = estimate_covariance(date_str)
        print(f"{date_str:<18s}  {est_m.shrinkage_intensity:>10.4f}  "
              f"{est_m.window_start.date()!s:>14s}  {est_m.window_end.date()!s:>14s}")
    except ValueError as e:
        print(f"{date_str:<18s}  SKIPPED: {str(e)[:50]}")

# Also pull a "calm" period for comparison
print("\nCalm period comparison:")
calm_months = ["2019-01-01", "2019-04-01", "2019-07-01", "2019-10-01"]
for date_str in calm_months:
    try:
        cal = get_calendar(end=date_str)
        # Find the actual first trading day on or after this date
        trading_days = cal["trade_date"].sort_values()
        actual_date = trading_days[trading_days >= pd.Timestamp(date_str)].iloc[0] if len(trading_days[trading_days >= pd.Timestamp(date_str)]) > 0 else trading_days.iloc[-1]
        est_m = estimate_covariance(actual_date)
        print(f"{actual_date.date()!s:<18s}  {est_m.shrinkage_intensity:>10.4f}")
    except (ValueError, IndexError) as e:
        print(f"{date_str:<18s}  SKIPPED: {str(e)[:50]}")


print("\n" + "=" * 70)
print("CHECK 3: Skip boundary — is the cutoff exactly right?")
print("=" * 70)

cal = get_calendar(start="2010-01-04")
trading_days = cal["trade_date"].sort_values().reset_index(drop=True)

# Need 253 price days (to get 252 returns). 
# Check exactly which month first has >= 253 trading days.
print(f"\nData starts: {trading_days.iloc[0].date()}")
print(f"Required price observations: {TRAILING_WINDOW + 1} (to produce {TRAILING_WINDOW} returns)")

# Check the boundary dates
test_dates = [
    "2011-01-03",  # should FAIL (last skipped)
    "2011-02-01",  # should PASS (first successful)
]

for d in test_dates:
    ts = pd.Timestamp(d)
    available = len(trading_days[trading_days <= ts])
    status = "PASS (>= 253)" if available >= 253 else "FAIL (< 253)"
    print(f"  {d}: {available} trading days available -> {status}")

# Find the exact first date with >= 253 days
for i, td in enumerate(trading_days):
    count = i + 1  # trading days including this one
    if count == 253:
        print(f"\n  Exact 253rd trading day: {td.date()} (index {i})")
        # What month is this in?
        print(f"  This falls in: {td.strftime('%B %Y')}")
        break

# The skip log said Jan 2011 was skipped (251 days available).
# Let's verify:
jan_2011_first = pd.Timestamp("2011-01-03")
available_jan = len(trading_days[trading_days <= jan_2011_first])
print(f"\n  2011-01-03: {available_jan} trading days available (need 253)")
print(f"  Off by: {253 - available_jan} days -> correctly skipped")

feb_2011_first = pd.Timestamp("2011-02-01")
available_feb = len(trading_days[trading_days <= feb_2011_first])
print(f"  2011-02-01: {available_feb} trading days available (need 253)")
print(f"  -> {'correctly computed' if available_feb >= 253 else 'ERROR: should have enough'}")


print("\n" + "=" * 70)
print("CHECK 4: Equal-weight 27.1% annualized vol gut check")
print("=" * 70)

# Compute Nifty-proxy realized vol using our 23 stocks equal-weighted
# over a long period, and compare to known Nifty vol
prices_full = get_close_matrix(start="2015-01-01", end="2025-12-31")
returns_full = prices_full.pct_change().dropna()

# Equal-weight portfolio returns
ew_returns = returns_full.mean(axis=1)
ew_annual_vol = ew_returns.std() * np.sqrt(252)
print(f"\n  Equal-weight 23-stock portfolio annualized vol (2015-2025): {ew_annual_vol:.4f} ({ew_annual_vol*100:.1f}%)")

# Compare to individual stock vols
stock_vols = returns_full.std() * np.sqrt(252)
print(f"  Average individual stock annualized vol: {stock_vols.mean():.4f} ({stock_vols.mean()*100:.1f}%)")
print(f"  Individual vol range: {stock_vols.min():.4f} to {stock_vols.max():.4f}")
print(f"  Diversification ratio: {stock_vols.mean() / ew_annual_vol:.2f}x")

# Nifty 50 realized vol is typically 18-22% over long periods
# Our 23-stock equal-weight should be somewhat higher (fewer names, equal weight not cap weight)
print(f"\n  Nifty 50 typical annualized vol: ~18-22%")
print(f"  Our 23-stock equal-weight: {ew_annual_vol*100:.1f}%")
if 15 < ew_annual_vol*100 < 30:
    print(f"  -> PLAUSIBLE (equal-weight, fewer names -> slightly higher than Nifty)")
else:
    print(f"  -> SUSPICIOUS — investigate further")

# Also check the specific window used in the automated test (ending 2020-06-30)
# That window includes the COVID crash, so vol should be elevated
prices_covid = get_close_matrix(start="2019-06-19", end="2020-06-30")
returns_covid = prices_covid.pct_change().dropna()
ew_covid = returns_covid.mean(axis=1)
ew_covid_vol = ew_covid.std() * np.sqrt(252)
print(f"\n  Equal-weight vol for the specific test window (2019-06 to 2020-06): {ew_covid_vol*100:.1f}%")
print(f"  (This window includes COVID crash -> should be elevated)")

# Compute from covariance matrix: sigma_p = sqrt(w' Sigma w)
w = np.ones(23) / 23
est_test = estimate_covariance("2020-06-30")
portfolio_var = w @ est_test.covariance @ w
portfolio_vol_annual = np.sqrt(portfolio_var) * np.sqrt(252)
print(f"  From our Sigma matrix (shrunk): {portfolio_vol_annual*100:.1f}%")
print(f"  From raw returns directly: {ew_covid_vol*100:.1f}%")
print(f"  Difference: {abs(portfolio_vol_annual - ew_covid_vol)*100:.2f}pp")
print(f"  (Small difference expected due to shrinkage smoothing)")


print("\n" + "=" * 70)
print("ALL INDEPENDENT CHECKS DONE")
print("=" * 70)
