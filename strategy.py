"""
Strategy (base class)

Responsibility: given the current bar and available history for a
ticker, emit a signal. Nothing more.

Owns: your alpha logic (moving averages, momentum, whatever you design).
Does NOT own: position sizing, cash checks, execution, costs. A Strategy
should be testable in complete isolation from Portfolio/ExecutionHandler --
if it needs to know your cash balance to decide on a signal, that logic
is in the wrong class.
"""

from enum import Enum
from dataclasses import dataclass
import pandas as pd


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class SignalEvent:
    ticker: str
    date: pd.Timestamp
    signal: Signal
    score: float = 0.0
    strategy_id: str = ""
    # `strategy_id` identifies which strategy produced this signal (e.g.
    # "MA_CROSSOVER", "RSI_MEANREV"). Needed because both strategies now
    # share one capital pool -- Portfolio uses this to track which
    # strategy "owns" a given open position, so only that strategy's
    # signal governs its exit.
    #
    # `score` is a conviction measure, used ONLY to rank competing BUY
    # signals across tickers on the same day (see ranking.py). It has no
    # meaning for SELL/HOLD signals and is NOT directly comparable across
    # strategies -- normalize via ranking.percentile_scores() first.


class Strategy:
    def generate_signals(self, ticker: str, current_bar: pd.Series,
                          history: pd.DataFrame) -> SignalEvent:
        """
        Must be implemented by subclasses. `current_bar` is bar t (the
        signal bar); `history` is bars up to and including t. The
        resulting signal will be FILLED at bar t+1's open by
        ExecutionHandler -- Strategy itself has no say in fill timing.
        """
        raise NotImplementedError


class MovingAverageCrossoverStrategy(Strategy):
    """
    Trend-following, regime-based (not event-based).

    Signal = BUY whenever fast SMA > slow SMA, SELL whenever fast SMA <
    slow SMA. This is a *persistent* directional bias, re-emitted every
    bar the regime holds -- it is Portfolio's job (in size_order) to
    check current position and no-op if already correctly positioned.
    Keeps this class stateless and independently testable.

    Defaults (20/50) are the standard textbook pair -- easy to justify,
    easy to compare against literature. Tune via constructor args.
    """
    STRATEGY_ID = "MA_CROSSOVER"

    def __init__(self, short_window: int = 20, long_window: int = 50):
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, ticker: str, current_bar: pd.Series,
                          history: pd.DataFrame) -> SignalEvent:
        date = current_bar.name  # assumes history/current_bar indexed by date

        if len(history) < self.long_window:
            # Not enough data yet to compute the slow SMA -- explicit
            # HOLD rather than guessing with a shorter window. This
            # keeps behavior identical across every ticker's warm-up
            # period, which matters when comparing across your 23-ticker
            # universe.
            return SignalEvent(ticker=ticker, date=date, signal=Signal.HOLD)

        closes = history["Close"]
        fast_sma = closes.rolling(self.short_window).mean().iloc[-1]
        slow_sma = closes.rolling(self.long_window).mean().iloc[-1]

        # Normalized gap: how far apart the two averages are, as a % of
        # the slow SMA. This is what makes scores comparable ACROSS
        # tickers with very different price levels (e.g. a stock at
        # ₹3,000 vs one at ₹150) -- an absolute rupee gap would unfairly
        # favor high-priced stocks.
        gap = (fast_sma - slow_sma) / slow_sma

        if fast_sma > slow_sma:
            signal = Signal.BUY
        elif fast_sma < slow_sma:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD  # exact tie -- rare, but must be handled

        return SignalEvent(ticker=ticker, date=date, signal=signal, score=gap,
                            strategy_id=self.STRATEGY_ID)


class RSIMeanReversionStrategy(Strategy):
    """
    Mean reversion via RSI(14), Wilder's smoothing method (the standard
    RSI calculation -- an EMA-style smoothed average of gains/losses,
    not a naive rolling mean).

    Signal = BUY when RSI < oversold_threshold, SELL when RSI >
    overbought_threshold, HOLD in between (maintains whatever position
    is already open -- no forced exit at neutral RSI in this version).

    Long-only assumption baked in: SELL is interpreted downstream as
    "exit to flat," not "open a short." If you want shorting enabled,
    that's a Portfolio-level decision to revisit, not a Strategy one.
    """
    STRATEGY_ID = "RSI_MEANREV"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        if not (0 < oversold < overbought < 100):
            raise ValueError("Require 0 < oversold < overbought < 100")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def _wilder_rsi(self, closes: pd.Series) -> float:
        """
        Standard Wilder RSI. Uses ewm with alpha=1/period, which is
        mathematically equivalent to Wilder's original smoothing method.
        Returns the RSI value for the LAST bar in `closes` only.
        """
        delta = closes.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / self.period, min_periods=self.period, adjust=False).mean()

        last_avg_gain = avg_gain.iloc[-1]
        last_avg_loss = avg_loss.iloc[-1]

        if last_avg_loss == 0:
            return 100.0  # no losses in window -> maximally overbought
        rs = last_avg_gain / last_avg_loss
        return 100 - (100 / (1 + rs))

    def generate_signals(self, ticker: str, current_bar: pd.Series,
                          history: pd.DataFrame) -> SignalEvent:
        date = current_bar.name

        # Wilder smoothing needs `period` diffs to warm up, i.e. period+1 closes.
        if len(history) < self.period + 1:
            return SignalEvent(ticker=ticker, date=date, signal=Signal.HOLD)

        rsi = self._wilder_rsi(history["Close"])

        if rsi < self.oversold:
            signal = Signal.BUY
            # How deep into oversold territory -- RSI=15 outranks RSI=29
            # when both are BUY candidates on the same day.
            score = self.oversold - rsi
        elif rsi > self.overbought:
            signal = Signal.SELL
            score = 0.0  # SELL always passes through ranking unfiltered; score unused
        else:
            signal = Signal.HOLD
            score = 0.0

        return SignalEvent(ticker=ticker, date=date, signal=signal, score=score,
                            strategy_id=self.STRATEGY_ID)
