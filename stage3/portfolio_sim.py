"""
portfolio_sim.py — Risk-Parity Portfolio Simulator (Part C)

Job: chain the risk engine, optimizer (Part A), and rebalancer (Part B)
across all ~187 monthly rebalance dates over 16 years, with real prices,
producing a daily equity curve that can be judged on Sharpe, drawdown, etc.

Step-by-step logic:
  1. Start with ₹10L. Buy the first set of target weights at entry cost.
  2. Walk forward day by day. If not a rebalance date: mark-to-market only.
  3. On rebalance dates: get fresh Σ → optimize → compare drifted weights
     → rebalance → apply trades at NEXT DAY'S OPEN (no look-ahead).
  4. Track daily equity curve, daily risk shares, and full trade log.

Fill convention: signal at today's close, fill at tomorrow's open —
same as Stage 2 (config.py: SIGNAL_TO_FILL_LAG_BARS = 1,
FILL_PRICE_FIELD = "Open"). Non-negotiable.

Trading cost: 15bps per leg, matching Stage 2's execution_handler.py
and rebalancer.py's COST_PER_LEG = 0.0015.
"""

import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_access import get_prices, get_calendar, get_close_matrix
from stage3.risk_engine import RollingCovarianceEngine, decompose_risk
from stage3.risk_parity import optimize_risk_parity, RiskParityResult
from stage3.rebalancer import rebalance, RebalanceResult, COST_PER_LEG
from stage3.covariance import CovarianceEstimate


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
INITIAL_CAPITAL = 1_000_000     # ₹10L, same as Stage 2
START_DATE = "2010-01-04"       # full validated Stage 1 range
END_DATE = "2026-08-28"
THRESHOLD_SWEEP = [0.0, 0.0025, 0.005, 0.01]  # 0%, 0.25%, 0.5%, 1.0%


# ------------------------------------------------------------------
# Output dataclasses
# ------------------------------------------------------------------
@dataclass
class DailySnapshot:
    """One day's portfolio state."""
    date: pd.Timestamp
    portfolio_value: float          # cash + holdings at close
    cash: float
    holdings_value: float           # sum of (shares * close_price) across all stocks
    is_rebalance_date: bool
    risk_shares: np.ndarray | None  # (23,) risk shares if computed, else None


@dataclass
class RebalanceLog:
    """Everything that happened on one rebalance date."""
    date: pd.Timestamp
    fill_date: pd.Timestamp         # the next trading day (where fills happen)
    target_weights: dict[str, float]
    actual_weights_before: dict[str, float]
    actual_weights_after: dict[str, float]
    rebalance_result: RebalanceResult
    optimizer_result: RiskParityResult
    optimizer_converged: bool
    used_fallback: bool


@dataclass
class SimulationResult:
    """Complete output of one simulation run."""
    threshold: float
    daily_snapshots: list[DailySnapshot]
    rebalance_logs: list[RebalanceLog]
    tickers: list[str]
    initial_capital: float
    final_value: float
    total_return: float
    cumulative_cost: float
    cumulative_turnover: float
    n_rebalances: int
    n_optimizer_failures: int


# ------------------------------------------------------------------
# Price data loader
# ------------------------------------------------------------------
class PriceData:
    """
    Pre-loads all price data needed for the simulation, once.

    Provides:
      - close_matrix: (dates × tickers) DataFrame of close prices
      - open_matrix: (dates × tickers) DataFrame of open prices
      - trading_days: sorted DatetimeIndex of all trading days
    """

    def __init__(self, tickers: list[str], start_date: str, end_date: str):
        self.tickers = sorted(tickers)

        # Close matrix — already available from data_access
        self.close_matrix = get_close_matrix(start=start_date, end=end_date)
        # Reorder columns to sorted ticker order
        self.close_matrix = self.close_matrix[self.tickers]

        # Open matrix — build from per-ticker get_prices() calls
        # (no get_open_matrix in data_access.py)
        open_frames = {}
        for t in self.tickers:
            df = get_prices(t, start_date, end_date)
            open_frames[t] = df["open"]
        self.open_matrix = pd.DataFrame(open_frames)
        self.open_matrix.index.name = "trade_date"

        # Trading days — from the close matrix index (canonical calendar)
        self.trading_days = self.close_matrix.index.sort_values()

        # Validate: both matrices must have the same dates
        close_dates = set(self.close_matrix.index)
        open_dates = set(self.open_matrix.index)
        mismatch = close_dates.symmetric_difference(open_dates)
        if mismatch:
            raise ValueError(
                f"Close/open date mismatch: {len(mismatch)} dates differ. "
                f"First few: {sorted(mismatch)[:5]}"
            )

        # Validate: no NaNs
        for name, mat in [("close", self.close_matrix), ("open", self.open_matrix)]:
            if mat.isnull().any().any():
                bad = mat.isnull().sum()
                bad = bad[bad > 0].to_dict()
                raise ValueError(f"{name} matrix has NaN values: {bad}")

    def get_close(self, date: pd.Timestamp) -> dict[str, float]:
        """Close prices on a specific date, as {ticker: price}."""
        row = self.close_matrix.loc[date]
        return row.to_dict()

    def get_open(self, date: pd.Timestamp) -> dict[str, float]:
        """Open prices on a specific date, as {ticker: price}."""
        row = self.open_matrix.loc[date]
        return row.to_dict()

    def get_next_trading_day(self, date: pd.Timestamp) -> pd.Timestamp | None:
        """Return the trading day after `date`, or None if at the end."""
        idx = self.trading_days.get_loc(date)
        if idx + 1 >= len(self.trading_days):
            return None
        return self.trading_days[idx + 1]


# ------------------------------------------------------------------
# The simulator
# ------------------------------------------------------------------
def run_simulation(
    engine: RollingCovarianceEngine,
    price_data: PriceData,
    threshold: float,
    weight_source: str = "risk_parity",
    verbose: bool = True,
) -> SimulationResult:
    """
    Run the full simulation across all trading days.

    Args:
        engine: pre-computed RollingCovarianceEngine with all Σ estimates.
        price_data: pre-loaded close and open price matrices.
        threshold: rebalance threshold (weight gap below which trades are skipped).
        weight_source: 'risk_parity' (use optimizer) or 'equal_weight' (use 1/N).
        verbose: print progress updates.

    Returns:
        SimulationResult with daily equity curve, trade logs, and summary stats.
    """
    if weight_source not in ("risk_parity", "equal_weight"):
        raise ValueError(f"weight_source must be 'risk_parity' or 'equal_weight', got '{weight_source}'")
    first_est = engine.get_estimate(engine.rebalance_dates[0])
    tickers = sorted(first_est.tickers)
    n_stocks = len(tickers)
    rebalance_dates_set = set(engine.rebalance_dates)

    # The simulation starts at the first rebalance date (first date with a Σ),
    # not at the data start. Earlier dates don't have enough history for Σ.
    sim_start = engine.rebalance_dates[0]
    sim_days = price_data.trading_days[price_data.trading_days >= sim_start]

    if verbose:
        label = "risk-parity" if weight_source == "risk_parity" else "equal-weight"
        print(f"Simulation ({label}): threshold={threshold:.4f}, "
              f"{len(sim_days)} trading days from {sim_days[0].date()} "
              f"to {sim_days[-1].date()}")
        print(f"  Rebalance dates: {len(engine.rebalance_dates)}")

    # ----------------------------------------------------------
    # State variables
    # ----------------------------------------------------------
    cash = float(INITIAL_CAPITAL)
    holdings: dict[str, int] = {t: 0 for t in tickers}    # shares held
    last_weights: np.ndarray | None = None                 # fallback for optimizer
    pending_trades: list | None = None     # trades decided today, filled tomorrow
    pending_log: dict | None = None        # log data for pending fill

    daily_snapshots: list[DailySnapshot] = []
    rebalance_logs: list[RebalanceLog] = []
    cumulative_cost = 0.0
    cumulative_turnover = 0.0
    n_optimizer_failures = 0

    is_first_rebalance = True

    # ----------------------------------------------------------
    # Main loop: walk forward day by day
    # ----------------------------------------------------------
    for day_idx, today in enumerate(sim_days):

        # ==============================================================
        # Phase 1: Apply pending trades from yesterday's decision
        #          at today's OPEN price
        # ==============================================================
        if pending_trades is not None:
            open_prices = price_data.get_open(today)

            for trade in pending_trades:
                t = trade.ticker
                fill_price = open_prices[t]
                notional = trade.shares * fill_price
                cost = notional * COST_PER_LEG

                if trade.direction == "BUY":
                    holdings[t] += trade.shares
                    cash -= (notional + cost)
                elif trade.direction == "SELL":
                    holdings[t] -= trade.shares
                    cash += (notional - cost)

                cumulative_cost += cost

            # Compute actual weights after fills (at today's close for logging)
            close_prices = price_data.get_close(today)
            holdings_val_after = {t: holdings[t] * close_prices[t] for t in tickers}
            total_after = sum(holdings_val_after.values()) + cash
            weights_after = {t: holdings_val_after[t] / total_after
                             for t in tickers} if total_after > 0 else {t: 0.0 for t in tickers}

            # Complete the pending log entry
            log_entry = pending_log
            log_entry["fill_date"] = today
            log_entry["actual_weights_after"] = weights_after

            rebalance_logs.append(RebalanceLog(
                date=log_entry["date"],
                fill_date=today,
                target_weights=log_entry["target_weights"],
                actual_weights_before=log_entry["actual_weights_before"],
                actual_weights_after=weights_after,
                rebalance_result=log_entry["rebalance_result"],
                optimizer_result=log_entry["optimizer_result"],
                optimizer_converged=log_entry["optimizer_converged"],
                used_fallback=log_entry["used_fallback"],
            ))

            cumulative_turnover += log_entry["rebalance_result"].turnover
            pending_trades = None
            pending_log = None

        # ==============================================================
        # Phase 2: Mark-to-market at today's close
        # ==============================================================
        close_prices = price_data.get_close(today)
        holdings_value = {t: holdings[t] * close_prices[t] for t in tickers}
        total_holdings_value = sum(holdings_value.values())
        portfolio_value = total_holdings_value + cash

        # ==============================================================
        # Phase 3: Compute daily risk shares
        # ==============================================================
        risk_shares = None
        if total_holdings_value > 0 and last_weights is not None:
            # Compute current drifted weights from actual holdings
            current_w = np.array([holdings_value[t] / portfolio_value
                                  for t in tickers])
            # Only compute risk shares if we have a live Σ
            try:
                cov_est = engine.get_nearest_estimate(today)
                rd = decompose_risk(current_w, cov_est)
                risk_shares = rd.contribution_to_risk / rd.portfolio_volatility
            except (KeyError, ValueError):
                pass  # no estimate available yet or degenerate weights

        # Record daily snapshot
        daily_snapshots.append(DailySnapshot(
            date=today,
            portfolio_value=portfolio_value,
            cash=cash,
            holdings_value=total_holdings_value,
            is_rebalance_date=(today in rebalance_dates_set),
            risk_shares=risk_shares,
        ))

        # ==============================================================
        # Phase 4: If today is a rebalance date, decide trades
        #          (to be filled tomorrow at open)
        # ==============================================================
        if today not in rebalance_dates_set:
            continue  # not a rebalance date, nothing to decide

        # --- Step 1: Get Σ for today ---
        cov_est = engine.get_estimate(today)

        # --- Step 2: Get target weights ---
        if weight_source == "risk_parity":
            # Run the full optimizer
            fallback_w = last_weights  # None on first call, previous weights after
            rp_result = optimize_risk_parity(cov_est, fallback_weights=fallback_w)

            used_fallback = not rp_result.converged
            if used_fallback:
                n_optimizer_failures += 1

            target_weights_dict = {t: float(w) for t, w in
                                   zip(rp_result.tickers, rp_result.weights)}
        else:
            # Equal weight: 1/N for every stock, no optimizer needed
            equal_w = 1.0 / n_stocks
            target_weights_dict = {t: equal_w for t in tickers}
            ew_array = np.full(n_stocks, equal_w)
            # Build a dummy RiskParityResult for logging consistency
            rd_ew = decompose_risk(ew_array, cov_est)
            rs_ew = rd_ew.contribution_to_risk / rd_ew.portfolio_volatility
            rp_result = RiskParityResult(
                weights=ew_array,
                tickers=tickers,
                risk_shares=rs_ew,
                target_risk_share=equal_w,
                max_risk_share_gap=float(np.max(np.abs(rs_ew - equal_w))),
                objective_value=0.0,
                n_iterations=0,
                converged=True,
                portfolio_volatility=rd_ew.portfolio_volatility,
            )
            used_fallback = False

        # --- Step 3: Compute current (drifted) weights ---
        current_weights_dict = {}
        for t in tickers:
            current_weights_dict[t] = holdings_value[t] / portfolio_value if portfolio_value > 0 else 0.0

        # --- Step 4: First rebalance = buy from scratch ---
        if is_first_rebalance:
            # Initial purchase: buy target weights with the full capital,
            # paying entry cost. Use the rebalancer with threshold=0 to
            # ensure all stocks get bought.
            rb_result = rebalance(
                holdings=holdings,
                prices=close_prices,
                target_weights=target_weights_dict,
                cash=cash,
                threshold=0.0,  # force all trades on first entry
            )
            is_first_rebalance = False
        else:
            # --- Normal rebalance with the configured threshold ---
            rb_result = rebalance(
                holdings=holdings,
                prices=close_prices,
                target_weights=target_weights_dict,
                cash=cash,
                threshold=threshold,
            )

        # Update last_weights for next month's fallback
        last_weights = rp_result.weights.copy()

        # --- Step 5: Queue trades for tomorrow's open ---
        # Check if there's a next trading day to fill on
        next_day = price_data.get_next_trading_day(today)
        if next_day is None:
            # Last day of data — can't fill, just log
            if verbose:
                print(f"  {today.date()}: rebalance decided but no next day to fill")
            continue

        if len(rb_result.trades) > 0:
            pending_trades = rb_result.trades
        else:
            pending_trades = None  # no trades needed

        # Prepare log (will be completed when trades fill)
        pending_log = {
            "date": today,
            "target_weights": target_weights_dict,
            "actual_weights_before": current_weights_dict.copy(),
            "rebalance_result": rb_result,
            "optimizer_result": rp_result,
            "optimizer_converged": rp_result.converged,
            "used_fallback": used_fallback,
        }

        # If no trades, complete the log immediately (no fill needed)
        if pending_trades is None:
            pending_log["fill_date"] = today  # no actual fill
            pending_log["actual_weights_after"] = current_weights_dict.copy()

            rebalance_logs.append(RebalanceLog(
                date=today,
                fill_date=today,
                target_weights=target_weights_dict,
                actual_weights_before=current_weights_dict.copy(),
                actual_weights_after=current_weights_dict.copy(),
                rebalance_result=rb_result,
                optimizer_result=rp_result,
                optimizer_converged=rp_result.converged,
                used_fallback=used_fallback,
            ))

        if verbose and (len(rebalance_logs) % 12 == 0 or is_first_rebalance):
            n_reb = len(rebalance_logs)
            print(f"  Rebalance #{n_reb}: {today.date()}, "
                  f"value=Rs{portfolio_value:,.0f}, "
                  f"trades={rb_result.n_traded}, "
                  f"cost=Rs{rb_result.total_cost:,.0f}, "
                  f"turnover={rb_result.turnover:.4f}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    final_value = daily_snapshots[-1].portfolio_value if daily_snapshots else INITIAL_CAPITAL
    total_return = (final_value / INITIAL_CAPITAL) - 1.0

    if verbose:
        print(f"\n  DONE: threshold={threshold:.4f}")
        print(f"    Final value: Rs{final_value:,.0f}")
        print(f"    Total return: {total_return:.2%}")
        print(f"    Cumulative cost: Rs{cumulative_cost:,.0f}")
        print(f"    Cumulative turnover: {cumulative_turnover:.4f}")
        print(f"    Rebalances: {len(rebalance_logs)}")
        print(f"    Optimizer failures: {n_optimizer_failures}")

    return SimulationResult(
        threshold=threshold,
        daily_snapshots=daily_snapshots,
        rebalance_logs=rebalance_logs,
        tickers=tickers,
        initial_capital=INITIAL_CAPITAL,
        final_value=final_value,
        total_return=total_return,
        cumulative_cost=cumulative_cost,
        cumulative_turnover=cumulative_turnover,
        n_rebalances=len(rebalance_logs),
        n_optimizer_failures=n_optimizer_failures,
    )


# ------------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------------
def compute_metrics(result: SimulationResult) -> dict:
    """
    Compute Sharpe, max drawdown, and other metrics from daily snapshots.

    Returns a dict with:
      - total_return, cagr, sharpe, max_drawdown, cumulative_cost,
        cumulative_turnover, n_rebalances, n_optimizer_failures
    """
    values = pd.Series(
        [s.portfolio_value for s in result.daily_snapshots],
        index=pd.DatetimeIndex([s.date for s in result.daily_snapshots]),
    )

    # Daily returns
    daily_returns = values.pct_change().dropna()

    # CAGR
    n_years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1.0 / n_years) - 1.0 if n_years > 0 else 0.0

    # Sharpe (annualized, 0% risk-free for now — matching Stage 2)
    sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)
              if daily_returns.std() > 0 else 0.0)

    # Max drawdown
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax
    max_drawdown = float(drawdown.min())

    return {
        "threshold": result.threshold,
        "total_return": result.total_return,
        "cagr": cagr,
        "sharpe": float(sharpe),
        "max_drawdown": max_drawdown,
        "final_value": result.final_value,
        "cumulative_cost": result.cumulative_cost,
        "cumulative_turnover": result.cumulative_turnover,
        "n_rebalances": result.n_rebalances,
        "n_optimizer_failures": result.n_optimizer_failures,
    }


# ------------------------------------------------------------------
# Risk-share drift analysis
# ------------------------------------------------------------------
def compute_risk_share_drift(result: SimulationResult) -> pd.DataFrame:
    """
    Build a (dates × 23) DataFrame of daily risk shares.

    Columns are tickers, index is dates. NaN for days before the first
    rebalance or when risk shares couldn't be computed.
    """
    tickers = result.tickers
    records = []
    for snap in result.daily_snapshots:
        if snap.risk_shares is not None:
            row = {t: float(rs) for t, rs in zip(tickers, snap.risk_shares)}
            row["date"] = snap.date
            records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    return df[tickers]  # enforce column order


# ------------------------------------------------------------------
# Verification: cash + holdings = portfolio value
# ------------------------------------------------------------------
def verify_reconciliation(result: SimulationResult, tolerance: float = 0.01) -> list[str]:
    """
    Check that cash + holdings = portfolio_value on every single day.
    Returns a list of error strings (empty = all good).
    """
    errors = []
    for snap in result.daily_snapshots:
        recon = snap.cash + snap.holdings_value
        diff = abs(recon - snap.portfolio_value)
        if diff > tolerance:
            errors.append(
                f"{snap.date.date()}: cash({snap.cash:.2f}) + "
                f"holdings({snap.holdings_value:.2f}) = {recon:.2f}, "
                f"but portfolio_value = {snap.portfolio_value:.2f}, "
                f"diff = {diff:.2f}"
            )
    return errors


# ------------------------------------------------------------------
# Verification: no look-ahead spot check
# ------------------------------------------------------------------
def verify_no_lookahead(result: SimulationResult, n_checks: int = 3) -> list[dict]:
    """
    Pick n_checks rebalance dates and return fill details for spot-checking.
    Returns a list of dicts with decision_date, fill_date, and sample trades.
    """
    checks = []
    logs = result.rebalance_logs
    if not logs:
        return checks

    # Pick evenly spaced rebalance dates
    indices = np.linspace(0, len(logs) - 1, min(n_checks, len(logs)), dtype=int)
    for idx in indices:
        log = logs[idx]
        sample_trades = []
        for trade in log.rebalance_result.trades[:3]:  # first 3 trades
            sample_trades.append({
                "ticker": trade.ticker,
                "direction": trade.direction,
                "shares": trade.shares,
                "decision_price": trade.price,  # close price on decision date
            })
        checks.append({
            "decision_date": str(log.date.date()),
            "fill_date": str(log.fill_date.date()),
            "n_trades": log.rebalance_result.n_traded,
            "sample_trades": sample_trades,
        })
    return checks


# ------------------------------------------------------------------
# Main: run the full threshold sweep
# ------------------------------------------------------------------
def main():
    """Run the full simulation across all threshold settings."""
    from data_access import get_ticker_info

    # Load tickers
    tickers = sorted(get_ticker_info()["symbol"].tolist())
    print(f"Loaded {len(tickers)} tickers: {tickers}")
    assert len(tickers) == 23, f"Expected 23 tickers, got {len(tickers)}"

    # ----------------------------------------------------------
    # Step 1: Pre-compute all covariance estimates (shared across runs)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1: Computing rolling covariance estimates")
    print("=" * 70)
    engine = RollingCovarianceEngine(start_date=START_DATE, end_date=END_DATE)
    engine.run(verbose=True)

    # ----------------------------------------------------------
    # Step 2: Pre-load all price data (shared across runs)
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: Loading price data")
    print("=" * 70)
    price_data = PriceData(tickers, START_DATE, END_DATE)
    print(f"  Close matrix: {price_data.close_matrix.shape}")
    print(f"  Open matrix: {price_data.open_matrix.shape}")
    print(f"  Trading days: {len(price_data.trading_days)}")

    # ----------------------------------------------------------
    # Step 3: Run simulation for each threshold
    # ----------------------------------------------------------
    results: dict[float, SimulationResult] = {}

    for threshold in THRESHOLD_SWEEP:
        print("\n" + "=" * 70)
        print(f"STEP 3: Running simulation — threshold = {threshold:.4f} "
              f"({threshold * 100:.2f}%)")
        print("=" * 70)
        result = run_simulation(engine, price_data, threshold, verbose=True)
        results[threshold] = result

    # ----------------------------------------------------------
    # Step 4: Verification
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: Verification")
    print("=" * 70)

    for threshold, result in results.items():
        print(f"\n--- Threshold {threshold:.4f} ---")

        # 4a: Cash + holdings reconciliation
        recon_errors = verify_reconciliation(result)
        if recon_errors:
            print(f"  FAIL: {len(recon_errors)} reconciliation errors!")
            for e in recon_errors[:5]:
                print(f"    {e}")
        else:
            print(f"  PASS: cash + holdings = portfolio_value on all "
                  f"{len(result.daily_snapshots)} days")

        # 4b: No look-ahead spot check
        checks = verify_no_lookahead(result)
        print(f"  No-lookahead spot checks ({len(checks)}):")
        for c in checks:
            print(f"    Decision: {c['decision_date']}, "
                  f"Fill: {c['fill_date']}, "
                  f"Trades: {c['n_trades']}")
            for t in c["sample_trades"]:
                print(f"      {t['ticker']}: {t['direction']} {t['shares']} shares "
                      f"(decision price={t['decision_price']:.2f})")

    # ----------------------------------------------------------
    # Step 5: Comparison table
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: Threshold comparison")
    print("=" * 70)

    header = (f"{'Threshold':>10} {'Total Ret':>10} {'CAGR':>8} "
              f"{'Sharpe':>8} {'MaxDD':>8} {'Cum Cost':>12} "
              f"{'Cum Turn':>10} {'Rebal':>6} {'Fails':>6}")
    print(header)
    print("-" * len(header))

    all_metrics = []
    for threshold in THRESHOLD_SWEEP:
        m = compute_metrics(results[threshold])
        all_metrics.append(m)
        print(f"{m['threshold']:>10.4f} {m['total_return']:>10.2%} "
              f"{m['cagr']:>8.2%} {m['sharpe']:>8.3f} "
              f"{m['max_drawdown']:>8.2%} Rs{m['cumulative_cost']:>10,.0f} "
              f"{m['cumulative_turnover']:>10.4f} {m['n_rebalances']:>6d} "
              f"{m['n_optimizer_failures']:>6d}")

    # ----------------------------------------------------------
    # Step 6: Save equity curves
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 6: Saving outputs")
    print("=" * 70)

    for threshold, result in results.items():
        # Daily equity curve
        eq = pd.Series(
            [s.portfolio_value for s in result.daily_snapshots],
            index=pd.DatetimeIndex([s.date for s in result.daily_snapshots]),
            name="portfolio_value",
        )
        fname = f"stage3_equity_curve_t{threshold:.4f}.csv"
        eq.to_csv(os.path.join(os.path.dirname(__file__), "..", fname))
        print(f"  Saved: {fname}")

    # Risk-share drift for the 0.5% threshold (the default)
    default_result = results[0.005]
    drift_df = compute_risk_share_drift(default_result)
    if not drift_df.empty:
        drift_fname = "stage3_risk_share_drift.csv"
        drift_df.to_csv(os.path.join(os.path.dirname(__file__), "..", drift_fname))
        print(f"  Saved: {drift_fname}")

        # Summary stats on drift
        target = 1.0 / 23
        deviations = (drift_df - target).abs()
        print(f"\n  Risk-share drift summary (threshold=0.50%):")
        print(f"    Mean absolute deviation from 1/23: "
              f"{deviations.values.mean():.6f}")
        print(f"    Max absolute deviation from 1/23: "
              f"{deviations.values.max():.6f}")
        print(f"    95th percentile deviation: "
              f"{np.percentile(deviations.values, 95):.6f}")

    # Save metrics table
    metrics_df = pd.DataFrame(all_metrics)
    metrics_fname = "stage3_threshold_comparison.csv"
    metrics_df.to_csv(
        os.path.join(os.path.dirname(__file__), "..", metrics_fname),
        index=False,
    )
    print(f"  Saved: {metrics_fname}")

    print("\n" + "=" * 70)
    print("DONE — all simulations complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
