"""
Ranking

Responsibility: normalize each strategy's raw conviction score into a
percentile rank (0 to 1), so BUY signals from DIFFERENT strategies (MA
crossover's SMA-gap-% vs RSI's oversold-depth) become comparable on a
common scale. This is what makes "pick whichever strategy's signal is
stronger for this ticker" a well-defined operation instead of comparing
apples to oranges.

The actual top-K selection across BOTH strategies happens in
Portfolio.process_signals(), because it needs to know which tickers are
ALREADY held (and by which strategy) before deciding what counts as a
"new entry" candidate -- that's portfolio state, not ranking logic.
"""

from strategy import SignalEvent, Signal


def percentile_scores(signals: list[SignalEvent]) -> dict[str, float]:
    """
    Given one strategy's SignalEvents for a single day, returns
    {ticker: percentile} for BUY signals only, where percentile is in
    [0, 1] -- 1.0 is the single strongest BUY that day, 0.0 is the
    weakest. Tickers with SELL/HOLD signals are excluded from the
    output (they're not entry candidates).

    Percentile rank (not raw min-max scaling) is used deliberately: it's
    robust to outliers -- one extreme score shouldn't compress every
    other score toward 0.
    """
    buys = [s for s in signals if s.signal == Signal.BUY]
    if not buys:
        return {}
    if len(buys) == 1:
        return {buys[0].ticker: 1.0}

    ranked = sorted(buys, key=lambda s: s.score)  # ascending
    n = len(ranked)
    return {s.ticker: i / (n - 1) for i, s in enumerate(ranked)}
