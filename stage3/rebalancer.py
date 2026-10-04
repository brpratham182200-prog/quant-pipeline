"""
rebalancer.py -- Monthly rebalancing logic

Job: compare drifted-actual portfolio weights against fresh optimizer
targets, and decide stock-by-stock what to trade.

Step by step:
  1. Compute current (drifted) weights from holdings + prices
  2. Compute gap per stock: target - current
  3. Skip stocks where |gap| < threshold (configurable)
  4. Convert weight gaps to share counts (whole shares only)
  5. Apply trading costs (15bps per leg, matching Stage 2)
  6. Record turnover and skipped stocks for diagnostics

The threshold is a parameter, not a fixed constant -- we'll test
0%, 0.25%, 0.5%, 1.0% once the simulator exists (Part C).
"""

import numpy as np
from dataclasses import dataclass, field


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
DEFAULT_THRESHOLD = 0.005   # 0.5pp -- will be swept later
COST_PER_LEG = 0.0015       # 15bps per leg, matching Stage 2's execution_handler


# ------------------------------------------------------------------
# Output dataclasses
# ------------------------------------------------------------------
@dataclass
class Trade:
    """One stock's rebalancing trade."""
    ticker: str
    direction: str           # "BUY" or "SELL"
    shares: int              # whole shares (always positive)
    price: float             # price used for the trade
    notional: float          # shares * price (rupees traded)
    cost: float              # notional * COST_PER_LEG
    weight_before: float     # current drifted weight
    weight_target: float     # optimizer's target weight
    weight_gap: float        # target - current


@dataclass
class SkippedStock:
    """A stock that was NOT traded because its gap was below threshold."""
    ticker: str
    weight_before: float
    weight_target: float
    weight_gap: float        # target - current (abs < threshold)


@dataclass
class RebalanceResult:
    """Everything the rebalancer returns -- not just the trade list."""
    trades: list[Trade]
    skipped: list[SkippedStock]
    total_buy_notional: float       # total rupees bought
    total_sell_notional: float      # total rupees sold
    total_cost: float               # total transaction costs paid
    turnover: float                 # sum(abs(notional)) / portfolio_value
    portfolio_value_before: float   # total portfolio value before rebalancing
    portfolio_value_after: float    # after rebalancing (= before - costs)
    threshold_used: float           # the threshold that was applied
    n_traded: int                   # how many stocks were traded
    n_skipped: int                  # how many stocks were skipped


# ------------------------------------------------------------------
# The rebalancer
# ------------------------------------------------------------------
def rebalance(
    holdings: dict[str, int],        # {ticker: shares} -- current share counts
    prices: dict[str, float],        # {ticker: price} -- today's prices
    target_weights: dict[str, float],# {ticker: weight} -- from optimizer
    cash: float = 0.0,               # any uninvested cash
    threshold: float = DEFAULT_THRESHOLD,
) -> RebalanceResult:
    """
    Compare drifted-actual weights against targets and generate trades.

    Args:
        holdings: current share counts per ticker. Can include tickers
            with 0 shares (will be treated as needing to buy from scratch).
        prices: current price per ticker. Must include all tickers in
            target_weights.
        target_weights: optimizer output. Must sum to ~1.0.
        cash: uninvested cash balance.
        threshold: minimum |gap| to trigger a trade (in weight units,
            e.g., 0.005 = 0.5 percentage points).

    Returns:
        RebalanceResult with trades, skipped stocks, costs, and turnover.
    """
    # ----------------------------------------------------------
    # Step 1: Compute current (drifted) weights
    # ----------------------------------------------------------
    tickers = sorted(target_weights.keys())

    # Validate: all tickers in target_weights must have prices
    missing_prices = set(tickers) - set(prices.keys())
    if missing_prices:
        raise ValueError(f"Missing prices for tickers: {missing_prices}")

    # Holdings value per stock
    holdings_value = {}
    for t in tickers:
        shares = holdings.get(t, 0)
        holdings_value[t] = shares * prices[t]

    total_holdings = sum(holdings_value.values())
    portfolio_value = total_holdings + cash

    if portfolio_value <= 0:
        raise ValueError(f"Portfolio value is {portfolio_value} -- cannot rebalance")

    # Current drifted weights
    current_weights = {t: holdings_value[t] / portfolio_value for t in tickers}

    # ----------------------------------------------------------
    # Step 2-3: Compute gaps and apply threshold
    # ----------------------------------------------------------
    trades = []
    skipped = []

    for t in tickers:
        gap = target_weights[t] - current_weights[t]

        if abs(gap) < threshold:
            # Skip: gap too small to be worth trading
            skipped.append(SkippedStock(
                ticker=t,
                weight_before=current_weights[t],
                weight_target=target_weights[t],
                weight_gap=gap,
            ))
            continue

        # ----------------------------------------------------------
        # Step 4: Convert weight gap to share count
        # ----------------------------------------------------------
        rupees_to_trade = gap * portfolio_value
        shares_to_trade = int(abs(rupees_to_trade) / prices[t])  # floor to whole shares

        if shares_to_trade == 0:
            # Rounding ate the entire trade -- treat as skipped
            skipped.append(SkippedStock(
                ticker=t,
                weight_before=current_weights[t],
                weight_target=target_weights[t],
                weight_gap=gap,
            ))
            continue

        direction = "BUY" if gap > 0 else "SELL"
        notional = shares_to_trade * prices[t]

        # ----------------------------------------------------------
        # Step 5: Apply trading cost
        # ----------------------------------------------------------
        cost = notional * COST_PER_LEG

        trades.append(Trade(
            ticker=t,
            direction=direction,
            shares=shares_to_trade,
            price=prices[t],
            notional=notional,
            cost=cost,
            weight_before=current_weights[t],
            weight_target=target_weights[t],
            weight_gap=gap,
        ))

    # ----------------------------------------------------------
    # Step 6-7: Compute totals and turnover
    # ----------------------------------------------------------
    total_buy = sum(t.notional for t in trades if t.direction == "BUY")
    total_sell = sum(t.notional for t in trades if t.direction == "SELL")
    total_cost = sum(t.cost for t in trades)
    total_notional = total_buy + total_sell
    turnover = total_notional / portfolio_value if portfolio_value > 0 else 0.0

    # Portfolio value after costs (trades are buy/sell swaps within the
    # portfolio, so net cash change from trades is approximately zero,
    # but costs are real cash out)
    portfolio_value_after = portfolio_value - total_cost

    # ----------------------------------------------------------
    # Step 8: Return everything
    # ----------------------------------------------------------
    return RebalanceResult(
        trades=trades,
        skipped=skipped,
        total_buy_notional=total_buy,
        total_sell_notional=total_sell,
        total_cost=total_cost,
        turnover=turnover,
        portfolio_value_before=portfolio_value,
        portfolio_value_after=portfolio_value_after,
        threshold_used=threshold,
        n_traded=len(trades),
        n_skipped=len(skipped),
    )
