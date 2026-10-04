"""
Diagnostic: count Phase 1.5 replacements and trace SUNPHARMA's full trade history.
Re-runs the backtest with the instrumented Portfolio, then reports.
"""
from data_access import get_ticker_info
from data_feed import DataFeed
from strategy import MovingAverageCrossoverStrategy, RSIMeanReversionStrategy
from portfolio import Portfolio
from execution_handler import ExecutionHandler
from engine import BacktestEngine

INITIAL_CASH = 1_000_000
START_DATE = "2010-01-04"
END_DATE = "2026-08-28"
TOP_K = 5
HISTORY_LOOKBACK = 50

tickers = get_ticker_info()["symbol"].tolist()
feed = DataFeed(tickers, START_DATE, END_DATE)
ma = MovingAverageCrossoverStrategy(short_window=20, long_window=50)
rsi = RSIMeanReversionStrategy(period=14)
portfolio = Portfolio(initial_cash=INITIAL_CASH, top_k=TOP_K)
execution = ExecutionHandler()
engine = BacktestEngine(feed, ma, rsi, portfolio, execution, history_lookback=HISTORY_LOOKBACK)

print("Running backtest with instrumentation...")
analyzer = engine.run()

# === Check 1: How many Phase 1.5 replacements fired? ===
print("\n" + "=" * 70)
print("CHECK 1: PHASE 1.5 REPLACEMENT COUNT")
print("=" * 70)
n = len(portfolio.replacement_log)
print("Total conviction-based replacements over 16 years: %d" % n)

if n > 0:
    print("\nAll replacements:")
    for i, r in enumerate(portfolio.replacement_log):
        print("  %3d. defender=%-16s (pct=%.3f, %s, held %d bars) "
              "-> challenger=%-16s (pct=%.3f, %s) edge=%.3f" % (
            i + 1,
            r["defender"], r["defender_pct"], r["defender_strategy"], r["defender_bars_held"],
            r["challenger"], r["challenger_pct"], r["challenger_strategy"],
            r["edge"]))

    # Breakdown by defender/challenger
    defenders_displaced = {}
    challengers_entered = {}
    for r in portfolio.replacement_log:
        defenders_displaced[r["defender"]] = defenders_displaced.get(r["defender"], 0) + 1
        challengers_entered[r["challenger"]] = challengers_entered.get(r["challenger"], 0) + 1

    print("\nDefenders displaced (times each ticker was bumped):")
    for t, c in sorted(defenders_displaced.items(), key=lambda x: -x[1]):
        print("  %-16s: %d" % (t, c))

    print("\nChallengers entered (times each ticker entered via replacement):")
    for t, c in sorted(challengers_entered.items(), key=lambda x: -x[1]):
        print("  %-16s: %d" % (t, c))

# === Check 2: SUNPHARMA trade history ===
print("\n" + "=" * 70)
print("CHECK 2: SUNPHARMA.NS FULL TRADE HISTORY")
print("=" * 70)
sun_trades = [t for t in portfolio.trade_log if t["ticker"] == "SUNPHARMA.NS"]
if not sun_trades:
    print("SUNPHARMA.NS was NEVER traded (zero fills).")
else:
    print("SUNPHARMA.NS fills: %d" % len(sun_trades))
    for i, t in enumerate(sun_trades):
        print("  %3d. %s %4d shares @ %.2f, cost=%.2f, strategy=%s" % (
            i + 1, t["direction"], t["quantity"], t["fill_price"],
            t["cost"], t["strategy_id"]))

# Also check the 3 originally-unheld tickers
print("\n" + "=" * 70)
print("CHECK 3: AXISBANK / HINDUNILVR / LT TRADE HISTORY")
print("=" * 70)
for ticker in ["AXISBANK.NS", "HINDUNILVR.NS", "LT.NS"]:
    trades = [t for t in portfolio.trade_log if t["ticker"] == ticker]
    if not trades:
        print("%s: NEVER traded (zero fills)" % ticker)
    else:
        buys = [t for t in trades if t["direction"] == "BUY"]
        sells = [t for t in trades if t["direction"] == "SELL"]
        print("%s: %d fills (%d buys, %d sells)" % (ticker, len(trades), len(buys), len(sells)))
        for i, t in enumerate(trades):
            print("  %3d. %s %4d shares @ %.2f, cost=%.2f, strategy=%s" % (
                i + 1, t["direction"], t["quantity"], t["fill_price"],
                t["cost"], t["strategy_id"]))

print("\nDONE")
