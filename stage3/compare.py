"""
compare.py -- Part D: Three-way comparison

Purpose: isolate exactly what risk parity adds by comparing:
  1. Risk parity, monthly rebalanced (1% threshold, from Part C)
  2. Equal-weight, monthly rebalanced (1% threshold, same simulator)
  3. Equal-weight, buy-and-hold (no rebalancing, true passive baseline)

All three share: same start date (Feb 2011), same 23-ticker universe,
same 30bps cost model, same Rs 10L starting capital.

The key output is two isolated effects:
  - Rebalancing effect = equal-weight-rebalanced minus buy-and-hold
  - Risk-weighting effect = risk parity minus equal-weight-rebalanced
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_ticker_info, get_calendar, get_close_matrix
from stage3.risk_engine import RollingCovarianceEngine, decompose_risk
from stage3.portfolio_sim import (
    run_simulation, PriceData, SimulationResult,
    compute_metrics, compute_risk_share_drift,
    verify_reconciliation, INITIAL_CAPITAL,
    START_DATE, END_DATE, DailySnapshot,
)
from stage3.rebalancer import COST_PER_LEG
from stage3.covariance import CovarianceEstimate


# ------------------------------------------------------------------
# Buy-and-hold simulator
# ------------------------------------------------------------------
def run_buy_and_hold(
    price_data: PriceData,
    engine: RollingCovarianceEngine,
    tickers: list[str],
    verbose: bool = True,
) -> SimulationResult:
    """
    Equal-weight buy-and-hold: buy all 23 stocks equally at close on
    the first rebalance date, pay one-time entry cost, then hold forever.

    Uses the same start date as the rebalanced runs (first rebalance date
    from the covariance engine) so the comparison window is identical.

    Returns a SimulationResult for consistent metric computation.
    """
    n_stocks = len(tickers)
    sim_start = engine.rebalance_dates[0]
    sim_days = price_data.trading_days[price_data.trading_days >= sim_start]

    if verbose:
        print(f"Simulation (buy-and-hold): "
              f"{len(sim_days)} trading days from {sim_days[0].date()} "
              f"to {sim_days[-1].date()}")

    cash = float(INITIAL_CAPITAL)
    holdings: dict[str, int] = {t: 0 for t in tickers}
    entry_cost_total = 0.0
    daily_snapshots: list[DailySnapshot] = []

    for day_idx, today in enumerate(sim_days):
        close_prices = price_data.get_close(today)

        # Day 0: buy equal-weight allocation
        if day_idx == 0:
            per_ticker_cash = cash / n_stocks
            for t in tickers:
                price = close_prices[t]
                # Match Stage 2 benchmark: buy at close, deduct entry cost
                affordable = per_ticker_cash / (1 + COST_PER_LEG)
                qty = int(affordable // price)
                holdings[t] = qty
                spent = qty * price
                cost = spent * COST_PER_LEG
                cash -= (spent + cost)
                entry_cost_total += cost

            if verbose:
                holdings_val = sum(holdings[t] * close_prices[t] for t in tickers)
                print(f"  Entry: bought {n_stocks} stocks, "
                      f"cost=Rs{entry_cost_total:,.0f}, "
                      f"cash remaining=Rs{cash:,.0f}, "
                      f"holdings=Rs{holdings_val:,.0f}")

        # Mark-to-market
        holdings_value = {t: holdings[t] * close_prices[t] for t in tickers}
        total_holdings_value = sum(holdings_value.values())
        portfolio_value = total_holdings_value + cash

        # Compute daily risk shares (uses nearest available Sigma)
        risk_shares = None
        if total_holdings_value > 0:
            current_w = np.array([holdings_value[t] / portfolio_value
                                  for t in tickers])
            try:
                cov_est = engine.get_nearest_estimate(today)
                rd = decompose_risk(current_w, cov_est)
                risk_shares = rd.contribution_to_risk / rd.portfolio_volatility
            except (KeyError, ValueError):
                pass

        daily_snapshots.append(DailySnapshot(
            date=today,
            portfolio_value=portfolio_value,
            cash=cash,
            holdings_value=total_holdings_value,
            is_rebalance_date=False,
            risk_shares=risk_shares,
        ))

    final_value = daily_snapshots[-1].portfolio_value
    total_return = (final_value / INITIAL_CAPITAL) - 1.0

    if verbose:
        print(f"\n  DONE: buy-and-hold")
        print(f"    Final value: Rs{final_value:,.0f}")
        print(f"    Total return: {total_return:.2%}")
        print(f"    Entry cost: Rs{entry_cost_total:,.0f}")

    return SimulationResult(
        threshold=0.0,
        daily_snapshots=daily_snapshots,
        rebalance_logs=[],
        tickers=tickers,
        initial_capital=INITIAL_CAPITAL,
        final_value=final_value,
        total_return=total_return,
        cumulative_cost=entry_cost_total,
        cumulative_turnover=0.0,  # entry-only, tracked separately
        n_rebalances=0,
        n_optimizer_failures=0,
    )


# ------------------------------------------------------------------
# Enhanced metrics (adds Sortino)
# ------------------------------------------------------------------
def compute_full_metrics(result: SimulationResult, label: str) -> dict:
    """
    Compute Sharpe, Sortino, max drawdown, and other metrics.
    Adds Sortino on top of Part C's compute_metrics().
    """
    base = compute_metrics(result)

    # Sortino ratio (downside deviation only)
    values = pd.Series(
        [s.portfolio_value for s in result.daily_snapshots],
        index=pd.DatetimeIndex([s.date for s in result.daily_snapshots]),
    )
    daily_returns = values.pct_change().dropna()
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.0
    sortino = (daily_returns.mean() / downside_std * np.sqrt(252)
               if downside_std > 0 else 0.0)

    base["sortino"] = float(sortino)
    base["label"] = label
    return base


# ------------------------------------------------------------------
# Risk-share drift comparison
# ------------------------------------------------------------------
def risk_share_summary(result: SimulationResult, label: str) -> dict:
    """Compute risk-share drift statistics for a simulation result."""
    drift_df = compute_risk_share_drift(result)
    if drift_df.empty:
        return {"label": label, "mean_dev": None, "p95_dev": None, "max_dev": None}

    target = 1.0 / len(result.tickers)
    deviations = (drift_df - target).abs()
    return {
        "label": label,
        "mean_dev": float(deviations.values.mean()),
        "p95_dev": float(np.percentile(deviations.values, 95)),
        "max_dev": float(deviations.values.max()),
    }


# ------------------------------------------------------------------
# Verification helpers
# ------------------------------------------------------------------
def verify_dates_match(
    rp_result: SimulationResult,
    ew_result: SimulationResult,
    bh_result: SimulationResult,
) -> list[str]:
    """Verify all three runs cover the exact same dates."""
    errors = []
    rp_dates = [s.date for s in rp_result.daily_snapshots]
    ew_dates = [s.date for s in ew_result.daily_snapshots]
    bh_dates = [s.date for s in bh_result.daily_snapshots]

    if rp_dates != ew_dates:
        errors.append(f"RP vs EW: {len(rp_dates)} vs {len(ew_dates)} dates")
    if rp_dates != bh_dates:
        errors.append(f"RP vs BH: {len(rp_dates)} vs {len(bh_dates)} dates")
    if rp_dates and ew_dates and rp_dates[0] != ew_dates[0]:
        errors.append(f"Start mismatch: RP={rp_dates[0].date()}, EW={ew_dates[0].date()}")
    if rp_dates and bh_dates and rp_dates[0] != bh_dates[0]:
        errors.append(f"Start mismatch: RP={rp_dates[0].date()}, BH={bh_dates[0].date()}")
    return errors


def verify_ew_targets_flat(ew_result: SimulationResult, n_checks: int = 5) -> list[str]:
    """
    Spot-check that equal-weight-rebalanced actually used 1/N targets,
    not risk-parity weights from a leftover code path.
    """
    errors = []
    n = len(ew_result.tickers)
    expected = 1.0 / n
    logs = ew_result.rebalance_logs

    if not logs:
        errors.append("No rebalance logs found for equal-weight run")
        return errors

    indices = np.linspace(0, len(logs) - 1, min(n_checks, len(logs)), dtype=int)
    for idx in indices:
        log = logs[idx]
        for t, w in log.target_weights.items():
            if abs(w - expected) > 1e-10:
                errors.append(
                    f"Rebalance {log.date.date()}: {t} target={w:.6f}, "
                    f"expected={expected:.6f}"
                )
    return errors


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    tickers = sorted(get_ticker_info()["symbol"].tolist())
    print(f"Loaded {len(tickers)} tickers")
    assert len(tickers) == 23, f"Expected 23 tickers, got {len(tickers)}"

    THRESHOLD = 0.01  # 1.0% — the chosen threshold from Part C

    # ==============================================================
    # Step 1: Shared infrastructure
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 1: Computing shared infrastructure")
    print("=" * 70)

    engine = RollingCovarianceEngine(start_date=START_DATE, end_date=END_DATE)
    engine.run(verbose=True)

    price_data = PriceData(tickers, START_DATE, END_DATE)
    print(f"  Price data: {price_data.close_matrix.shape}")

    # ==============================================================
    # Step 2: Run risk parity (1% threshold)
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 2: Risk parity (1% threshold)")
    print("=" * 70)
    rp_result = run_simulation(
        engine, price_data, threshold=THRESHOLD,
        weight_source="risk_parity", verbose=True,
    )

    # ==============================================================
    # Step 3: Run equal-weight rebalanced (1% threshold)
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 3: Equal-weight, monthly rebalanced (1% threshold)")
    print("=" * 70)
    ew_result = run_simulation(
        engine, price_data, threshold=THRESHOLD,
        weight_source="equal_weight", verbose=True,
    )

    # ==============================================================
    # Step 4: Run buy-and-hold
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 4: Equal-weight, buy-and-hold")
    print("=" * 70)
    bh_result = run_buy_and_hold(price_data, engine, tickers, verbose=True)

    # ==============================================================
    # Step 5: Verification
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 5: Verification")
    print("=" * 70)

    # 5a: Dates match
    date_errors = verify_dates_match(rp_result, ew_result, bh_result)
    if date_errors:
        print(f"  FAIL: date mismatches: {date_errors}")
    else:
        print(f"  PASS: all three runs cover the same "
              f"{len(rp_result.daily_snapshots)} trading days")

    # 5b: Reconciliation
    for label, result in [("Risk parity", rp_result),
                           ("Equal-weight rebal", ew_result),
                           ("Buy-and-hold", bh_result)]:
        errors = verify_reconciliation(result)
        if errors:
            print(f"  FAIL ({label}): {len(errors)} reconciliation errors")
            for e in errors[:3]:
                print(f"    {e}")
        else:
            print(f"  PASS ({label}): reconciliation OK")

    # 5c: EW targets actually flat
    ew_target_errors = verify_ew_targets_flat(ew_result)
    if ew_target_errors:
        print(f"  FAIL: equal-weight targets not flat:")
        for e in ew_target_errors[:5]:
            print(f"    {e}")
    else:
        print(f"  PASS: equal-weight targets are genuinely 1/23 (spot-checked 5 dates)")

    # ==============================================================
    # Step 6: Metrics comparison
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 6: Three-way comparison")
    print("=" * 70)

    rp_metrics = compute_full_metrics(rp_result, "Risk Parity")
    ew_metrics = compute_full_metrics(ew_result, "EW Rebalanced")
    bh_metrics = compute_full_metrics(bh_result, "Buy & Hold")

    all_metrics = [rp_metrics, ew_metrics, bh_metrics]

    header = (f"{'Strategy':<20} {'Total Ret':>10} {'CAGR':>8} "
              f"{'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} "
              f"{'Cum Cost':>12} {'Cum Turn':>10}")
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        turn_str = f"{m['cumulative_turnover']:.4f}" if m['cumulative_turnover'] > 0 else "entry only"
        print(f"{m['label']:<20} {m['total_return']:>10.2%} "
              f"{m['cagr']:>8.2%} {m['sharpe']:>8.3f} "
              f"{m['sortino']:>8.3f} {m['max_drawdown']:>8.2%} "
              f"Rs{m['cumulative_cost']:>10,.0f} {turn_str:>10}")

    # ==============================================================
    # Step 7: Isolate the two effects
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 7: Isolated effects")
    print("=" * 70)

    rebal_effect_cagr = ew_metrics["cagr"] - bh_metrics["cagr"]
    rebal_effect_sharpe = ew_metrics["sharpe"] - bh_metrics["sharpe"]
    rebal_effect_dd = ew_metrics["max_drawdown"] - bh_metrics["max_drawdown"]
    rebal_effect_sortino = ew_metrics["sortino"] - bh_metrics["sortino"]

    rw_effect_cagr = rp_metrics["cagr"] - ew_metrics["cagr"]
    rw_effect_sharpe = rp_metrics["sharpe"] - ew_metrics["sharpe"]
    rw_effect_dd = rp_metrics["max_drawdown"] - ew_metrics["max_drawdown"]
    rw_effect_sortino = rp_metrics["sortino"] - ew_metrics["sortino"]

    total_effect_cagr = rp_metrics["cagr"] - bh_metrics["cagr"]
    total_effect_sharpe = rp_metrics["sharpe"] - bh_metrics["sharpe"]

    print(f"\n  Rebalancing effect (EW rebal - buy&hold):")
    print(f"    CAGR:    {rebal_effect_cagr:+.2%}")
    print(f"    Sharpe:  {rebal_effect_sharpe:+.3f}")
    print(f"    Sortino: {rebal_effect_sortino:+.3f}")
    print(f"    MaxDD:   {rebal_effect_dd:+.2%} (positive = shallower)")

    print(f"\n  Risk-weighting effect (risk parity - EW rebal):")
    print(f"    CAGR:    {rw_effect_cagr:+.2%}")
    print(f"    Sharpe:  {rw_effect_sharpe:+.3f}")
    print(f"    Sortino: {rw_effect_sortino:+.3f}")
    print(f"    MaxDD:   {rw_effect_dd:+.2%} (positive = shallower)")

    print(f"\n  Sanity check: effects sum to total?")
    print(f"    CAGR:   rebal({rebal_effect_cagr:+.4%}) + riskwt({rw_effect_cagr:+.4%}) "
          f"= {rebal_effect_cagr + rw_effect_cagr:+.4%}, "
          f"actual total = {total_effect_cagr:+.4%}")
    print(f"    Sharpe: rebal({rebal_effect_sharpe:+.4f}) + riskwt({rw_effect_sharpe:+.4f}) "
          f"= {rebal_effect_sharpe + rw_effect_sharpe:+.4f}, "
          f"actual total = {total_effect_sharpe:+.4f}")

    # ==============================================================
    # Step 8: Risk-share drift comparison
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 8: Risk-share drift comparison")
    print("=" * 70)

    rp_drift = risk_share_summary(rp_result, "Risk Parity")
    ew_drift = risk_share_summary(ew_result, "EW Rebalanced")
    bh_drift = risk_share_summary(bh_result, "Buy & Hold")

    print(f"\n  {'Strategy':<20} {'Mean Dev':>10} {'95th Pct':>10} {'Max Dev':>10}")
    print(f"  {'-'*50}")
    for d in [rp_drift, ew_drift, bh_drift]:
        if d["mean_dev"] is not None:
            print(f"  {d['label']:<20} {d['mean_dev']:>10.4%} "
                  f"{d['p95_dev']:>10.4%} {d['max_dev']:>10.4%}")
        else:
            print(f"  {d['label']:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")

    target = 1.0 / 23
    print(f"\n  Target risk share: {target:.4%} (1/23)")
    print(f"  Risk parity keeps risk shares {rp_drift['mean_dev']:.4%} from equal on average"
          if rp_drift['mean_dev'] else "")
    print(f"  Equal-weight lets risk shares drift {ew_drift['mean_dev']:.4%} from equal on average"
          if ew_drift['mean_dev'] else "")
    if bh_drift['mean_dev'] and ew_drift['mean_dev'] and rp_drift['mean_dev']:
        ratio = ew_drift['mean_dev'] / rp_drift['mean_dev']
        print(f"  Equal-weight's risk imbalance is {ratio:.1f}x that of risk parity")

    # ==============================================================
    # Step 9: Save outputs
    # ==============================================================
    print("\n" + "=" * 70)
    print("STEP 9: Saving outputs")
    print("=" * 70)

    out_dir = os.path.join(os.path.dirname(__file__), "..")

    # Save equity curves
    for label, result in [("risk_parity", rp_result),
                           ("ew_rebalanced", ew_result),
                           ("buy_and_hold", bh_result)]:
        eq = pd.Series(
            [s.portfolio_value for s in result.daily_snapshots],
            index=pd.DatetimeIndex([s.date for s in result.daily_snapshots]),
            name="portfolio_value",
        )
        fname = f"stage3_compare_{label}.csv"
        eq.to_csv(os.path.join(out_dir, fname))
        print(f"  Saved: {fname}")

    # Save comparison metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(
        os.path.join(out_dir, "stage3_comparison_metrics.csv"), index=False,
    )
    print(f"  Saved: stage3_comparison_metrics.csv")

    # Save EW risk-share drift
    ew_drift_df = compute_risk_share_drift(ew_result)
    if not ew_drift_df.empty:
        ew_drift_df.to_csv(os.path.join(out_dir, "stage3_ew_risk_share_drift.csv"))
        print(f"  Saved: stage3_ew_risk_share_drift.csv")

    print("\n" + "=" * 70)
    print("DONE -- all comparisons complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
