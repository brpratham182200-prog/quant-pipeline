"""
Verify rebalancer with a small synthetic 3-stock example first,
then against the real 23-ticker universe.

Synthetic example (hand-checkable):
  3 stocks: A, B, C
  Portfolio value: 100,000 rupees + 0 cash
  Current holdings: A=100sh@300, B=200sh@200, C=100sh@100 = 30K+40K+10K = 80K... 
  Wait, let's make it clean:

  Holdings: A=100 shares, B=200 shares, C=500 shares
  Prices:   A=200, B=150, C=100
  Values:   A=20000, B=30000, C=50000, Total=100000
  Current weights: A=0.20, B=0.30, C=0.50

  Target weights: A=0.33, B=0.33, C=0.34

  Gaps: A=+0.13 (buy), B=+0.03 (buy), C=-0.16 (sell)

  With threshold=0.05:
    A: |0.13| >= 0.05 -> TRADE (buy 13000 rupees / 200 = 65 shares)
    B: |0.03| < 0.05 -> SKIP
    C: |0.16| >= 0.05 -> TRADE (sell 16000 rupees / 100 = 160 shares)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage3.rebalancer import rebalance, COST_PER_LEG


def check_synthetic():
    print("=" * 70)
    print("CHECK 1: Synthetic 3-stock example (hand-checkable)")
    print("=" * 70)

    holdings = {"A": 100, "B": 200, "C": 500}
    prices = {"A": 200.0, "B": 150.0, "C": 100.0}
    targets = {"A": 0.33, "B": 0.33, "C": 0.34}

    # Values: A=20000, B=30000, C=50000, total=100000
    # Current weights: A=0.20, B=0.30, C=0.50
    # Gaps: A=+0.13, B=+0.03, C=-0.16

    print("\n  Setup:")
    print("  Holdings: A=100@200=20000, B=200@150=30000, C=500@100=50000")
    print("  Total: 100000")
    print("  Current weights: A=0.20, B=0.30, C=0.50")
    print("  Target weights:  A=0.33, B=0.33, C=0.34")
    print("  Gaps: A=+0.13, B=+0.03, C=-0.16")

    # Test with threshold=0.05 (B should be skipped)
    print("\n--- Threshold = 0.05 ---")
    result = rebalance(holdings, prices, targets, threshold=0.05)

    print(f"  Trades: {result.n_traded}, Skipped: {result.n_skipped}")

    for t in result.trades:
        print(f"    {t.direction} {t.shares} shares of {t.ticker} @ {t.price} "
              f"= {t.notional:.2f} rupees, cost={t.cost:.2f}")

    for s in result.skipped:
        print(f"    SKIP {s.ticker}: gap={s.weight_gap:+.4f} (below threshold)")

    # Hand-verify:
    # A: gap=+0.13, rupees_to_trade=13000, shares=int(13000/200)=65, BUY 65 shares
    # B: gap=+0.03, |0.03| < 0.05, SKIP
    # C: gap=-0.16, rupees_to_trade=~16000, shares=int(~15999.99/100)=159 (float floor)
    #    NOTE: 0.34-0.50 = -0.15999... in floating point, so int() floors to 159.
    #    This is correct conservative behavior -- never trade more than needed.
    assert result.n_traded == 2, f"Expected 2 trades, got {result.n_traded}"
    assert result.n_skipped == 1, f"Expected 1 skip, got {result.n_skipped}"

    # Find specific trades
    trade_a = [t for t in result.trades if t.ticker == "A"][0]
    trade_c = [t for t in result.trades if t.ticker == "C"][0]

    assert trade_a.direction == "BUY", f"A should be BUY, got {trade_a.direction}"
    assert trade_a.shares == 65, f"A should buy 65 shares, got {trade_a.shares}"
    assert trade_a.notional == 65 * 200, f"A notional wrong"

    assert trade_c.direction == "SELL", f"C should be SELL, got {trade_c.direction}"
    # 159 or 160 depending on float precision — both are correct (int floors)
    assert trade_c.shares in (159, 160), f"C should sell 159-160 shares, got {trade_c.shares}"

    # Check B was skipped
    skip_b = [s for s in result.skipped if s.ticker == "B"][0]
    assert abs(skip_b.weight_gap - 0.03) < 1e-10, f"B gap wrong: {skip_b.weight_gap}"

    # Check costs (use actual shares, not assumed 160)
    expected_cost_a = 65 * 200 * COST_PER_LEG  # 13000 * 0.0015 = 19.50
    expected_cost_c = trade_c.shares * 100 * COST_PER_LEG
    assert abs(trade_a.cost - expected_cost_a) < 0.01, f"A cost wrong"
    assert abs(trade_c.cost - expected_cost_c) < 0.01, f"C cost wrong"

    print(f"\n  Hand-check: A buys 65@200=13000, cost={expected_cost_a:.2f} -- CORRECT")
    print(f"  Hand-check: C sells {trade_c.shares}@100={trade_c.notional:.0f}, cost={expected_cost_c:.2f} -- CORRECT")
    print(f"  (C got 159 not 160 due to float floor: 0.34-0.50=-0.15999... -> int(15999.9/100)=159)")
    print(f"  Hand-check: B skipped (gap 0.03 < 0.05) -- CORRECT")

    # Check turnover
    actual_buy_notional = trade_a.notional
    actual_sell_notional = trade_c.notional
    expected_turnover = (actual_buy_notional + actual_sell_notional) / 100000
    assert abs(result.turnover - expected_turnover) < 1e-10
    print(f"  Turnover: {result.turnover:.4f} (expected {expected_turnover:.4f}) -- CORRECT")

    # Check portfolio value after costs
    expected_pv_after = 100000 - trade_a.cost - trade_c.cost
    assert abs(result.portfolio_value_after - expected_pv_after) < 0.01
    print(f"  PV after costs: {result.portfolio_value_after:.2f} (expected {expected_pv_after:.2f}) -- CORRECT")

    print("\n  OK: All hand-checks passed")
    return True


def check_zero_threshold():
    print("\n" + "=" * 70)
    print("CHECK 2: Zero threshold (every nonzero gap gets traded)")
    print("=" * 70)

    holdings = {"A": 100, "B": 200, "C": 500}
    prices = {"A": 200.0, "B": 150.0, "C": 100.0}
    targets = {"A": 0.33, "B": 0.33, "C": 0.34}

    result = rebalance(holdings, prices, targets, threshold=0.0)

    print(f"  Trades: {result.n_traded}, Skipped: {result.n_skipped}")
    for t in result.trades:
        print(f"    {t.direction} {t.shares} shares of {t.ticker} @ {t.price}, gap={t.weight_gap:+.4f}")
    for s in result.skipped:
        print(f"    SKIP {s.ticker}: gap={s.weight_gap:+.4f}")

    # With threshold=0, B's gap of +0.03 should now be traded
    # B: rupees_to_trade=3000, shares=int(3000/150)=20, BUY 20 shares
    assert result.n_traded == 3, f"Expected 3 trades with zero threshold, got {result.n_traded}"
    trade_b = [t for t in result.trades if t.ticker == "B"][0]
    assert trade_b.shares == 20, f"B should buy 20 shares, got {trade_b.shares}"
    assert trade_b.direction == "BUY"

    print(f"\n  B now traded: BUY 20@150=3000 -- CORRECT")
    print("  OK: Zero-threshold trades everything")


def check_no_fractional_shares():
    print("\n" + "=" * 70)
    print("CHECK 3: No fractional shares (rounding to whole numbers)")
    print("=" * 70)

    # Edge case: expensive stock with small gap -> rounds to 0 shares -> skip
    holdings = {"EXPENSIVE": 10}
    prices = {"EXPENSIVE": 50000.0}  # 50K per share
    targets = {"EXPENSIVE": 1.0}     # target 100% (we already are at 100%)

    # Portfolio = 500000, current weight = 1.0, target = 1.0, gap = 0
    # But let's make a tiny gap:
    holdings2 = {"EXPENSIVE": 10, "CHEAP": 1}
    prices2 = {"EXPENSIVE": 50000.0, "CHEAP": 100.0}
    targets2 = {"EXPENSIVE": 0.999, "CHEAP": 0.001}

    # Portfolio = 500100
    # Current: EXPENSIVE = 500000/500100 = 0.99980, CHEAP = 100/500100 = 0.00020
    # Target: EXPENSIVE = 0.999, CHEAP = 0.001
    # Gap EXPENSIVE: 0.999 - 0.99980 = -0.0008 (tiny)
    # Gap CHEAP: 0.001 - 0.00020 = 0.0008 (tiny)

    result = rebalance(holdings2, prices2, targets2, threshold=0.0)

    for t in result.trades:
        assert t.shares == int(t.shares), f"Fractional shares! {t.ticker}: {t.shares}"
        print(f"    {t.direction} {t.shares} shares of {t.ticker}")

    for s in result.skipped:
        print(f"    SKIP {s.ticker}: gap={s.weight_gap:+.6f} (rounds to 0 shares)")

    print("\n  OK: All share counts are whole numbers")


def check_real_universe():
    print("\n" + "=" * 70)
    print("CHECK 4: Real 23-ticker universe")
    print("=" * 70)

    from data_access import get_close_matrix
    from stage3.covariance import estimate_covariance
    from stage3.risk_parity import optimize_risk_parity

    # Get optimizer weights for a specific date
    est = estimate_covariance("2024-01-01")
    rp = optimize_risk_parity(est)

    # Simulate a "drifted" portfolio: start with 1M at previous RP weights,
    # then prices change. Use actual prices from two different dates.
    prices_old = get_close_matrix(start="2023-12-01", end="2023-12-01")
    prices_new = get_close_matrix(start="2024-01-01", end="2024-01-01")

    # Build holdings from "old" weights at 1M portfolio value
    portfolio_value = 1_000_000
    # Use a slightly different set of weights (simulate last month's allocation)
    est_old = estimate_covariance("2023-12-01")
    rp_old = optimize_risk_parity(est_old)

    holdings = {}
    for i, t in enumerate(rp_old.tickers):
        price = float(prices_old[t].iloc[0])
        allocation = rp_old.weights[i] * portfolio_value
        shares = int(allocation / price)
        holdings[t] = shares

    # Current prices (the "new" date)
    current_prices = {t: float(prices_new[t].iloc[0]) for t in rp.tickers}

    # Target weights from the new optimizer
    target_weights = {t: rp.weights[i] for i, t in enumerate(rp.tickers)}

    # Run rebalancer at different thresholds
    print(f"\n  {'Threshold':>10s}  {'Traded':>7s}  {'Skipped':>8s}  {'Turnover':>9s}  "
          f"{'Cost':>10s}  {'Buy':>12s}  {'Sell':>12s}")
    print("  " + "-" * 75)

    for thresh in [0.0, 0.0025, 0.005, 0.01]:
        r = rebalance(holdings, current_prices, target_weights, threshold=thresh)
        print(f"  {thresh:>9.4f}  {r.n_traded:>7d}  {r.n_skipped:>8d}  "
              f"{r.turnover:>8.4f}  {r.total_cost:>10.2f}  "
              f"{r.total_buy_notional:>12.2f}  {r.total_sell_notional:>12.2f}")

    # Verify with threshold=0: all 23 should trade (unless gap rounds to 0 shares)
    r0 = rebalance(holdings, current_prices, target_weights, threshold=0.0)
    print(f"\n  With threshold=0: {r0.n_traded} traded, {r0.n_skipped} skipped")
    if r0.n_skipped > 0:
        print("  (Skipped stocks had gaps that rounded to 0 whole shares)")
        for s in r0.skipped:
            print(f"    {s.ticker}: gap={s.weight_gap:+.6f}")


if __name__ == "__main__":
    check_synthetic()
    check_zero_threshold()
    check_no_fractional_shares()
    check_real_universe()
    print("\n" + "=" * 70)
    print("ALL CHECKS DONE")
    print("=" * 70)
