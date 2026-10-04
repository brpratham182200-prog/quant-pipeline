"""
Stage 2 Backtester — Configuration

Centralizes the assumptions you've explicitly decided on, so they're not
buried inside logic files. Change them here, not in ExecutionHandler.
"""

# --- Fill timing convention ---
# Signal computed on bar t's close -> order filled at bar t+1's open.
# This avoids look-ahead bias (see Q2 of the backtesting theory quiz).
SIGNAL_TO_FILL_LAG_BARS = 1
FILL_PRICE_FIELD = "Open"       # field used for the fill bar
SIGNAL_PRICE_FIELD = "Close"    # field used to generate the signal

# --- Transaction cost model ---
# Flat 30 bps round-trip, blended (taxes + slippage), for Nifty-tier large caps.
# Split evenly across buy and sell legs.
ROUND_TRIP_COST_BPS = 30
PER_LEG_COST_BPS = ROUND_TRIP_COST_BPS / 2  # 15 bps per leg

def per_leg_cost_fraction() -> float:
    """Returns per-leg cost as a fraction of trade notional (e.g. 0.0015 for 15bps)."""
    return PER_LEG_COST_BPS / 10_000


# --- Shared-pool portfolio construction ---
# Both strategies compete for the same capital pool. These are starting
# assumptions, not derived values -- revisit if results look sensitive
# to them.
TOP_K = 5                        # max new entries per day, across BOTH strategies combined
PARTIAL_SCALE_OUT_FRACTION = 0.5 # fraction sold on first BUY->HOLD transition
VOLATILITY_LOOKBACK = 20         # trading days, for inverse-vol position sizing
MAX_DEPLOYMENT_FRACTION = 0.95   # cap on % of equity invested at once; rest is cost/slippage buffer

# --- Conviction-based position replacement ---
# Allows strong new candidates to displace weak existing holdings,
# preventing capital from getting permanently locked into stale winners.
# See portfolio.py Phase 1.5 for the implementation.
MIN_CONVICTION_EDGE = 0.40       # percentile gap (0-1) a challenger needs over the weakest
                                 # defender to justify the replacement + 30bps round-trip cost
MIN_HOLDING_PERIOD = 10          # trading days before a position is eligible for displacement
MAX_REPLACEMENTS_PER_DAY = 2     # cap on conviction-based swaps per bar to limit churn

# Stage 2 must ONLY use data_access.py — never touch raw pickle/CSV directly.
DB_PATH = "db/quant.db"
