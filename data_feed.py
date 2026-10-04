"""
DataFeed

Responsibility: stream historical bars one date at a time, wrapping
data_access.py (the ONLY interface to the DB -- raw tables never touched
directly here). Structurally enforces "the strategy can only see today
and earlier": current_bar()/history() only ever expose data up to the
date the iterator has currently yielded.

data_access.py returns lowercase OHLCV columns (open/high/low/close/
volume) plus data_quality. Strategy and ExecutionHandler were written
expecting Title-case ('Close', 'Open') -- normalized here on load, once,
so nothing downstream needs to know about the DB's casing convention.

Owns: nothing about trading logic. Purely an iterator over history.
Does NOT own: signal generation, position sizing, fills.
"""

from typing import Iterator
import pandas as pd
from data_access import get_prices, get_calendar

_COLUMN_RENAME = {
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "volume": "Volume",
}


class DataFeed:
    def __init__(self, tickers: list[str], start_date: str, end_date: str,
                 exclude_quality_issues: bool = False):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.exclude_quality_issues = exclude_quality_issues

        self._data: dict[str, pd.DataFrame] = {}
        self._calendar: pd.DatetimeIndex | None = None
        self._current_date: pd.Timestamp | None = None
        self._position = -1  # index into self._calendar

        self._load()

    def _load(self) -> None:
        """
        Pulls the trading calendar (the canonical intersection-calendar
        dates, per your Stage 1 design decision -- iteration order comes
        from here, NOT from any single ticker's own index) and full
        OHLCV for every ticker, once, up front.
        """
        cal_df = get_calendar(self.start_date, self.end_date)
        if cal_df.empty:
            raise ValueError(
                f"get_calendar returned no rows for {self.start_date}..{self.end_date} "
                f"-- check the date range against what's actually in trading_calendar."
            )
        self._calendar = pd.DatetimeIndex(cal_df["trade_date"])

        for ticker in self.tickers:
            df = get_prices(ticker, self.start_date, self.end_date,
                             exclude_quality_issues=self.exclude_quality_issues)
            if df.empty:
                raise ValueError(
                    f"get_prices returned no rows for {ticker} in "
                    f"{self.start_date}..{self.end_date} -- check the symbol is "
                    f"correct (e.g. 'TCS.NS') and exists in this date range."
                )
            df = df.rename(columns=_COLUMN_RENAME)

            missing_dates = self._calendar.difference(df.index)
            if len(missing_dates) > 0:
                # This should NOT happen given the intersection-calendar
                # design -- if it does, something upstream (DB migration,
                # or a ticker with a different trading history) is
                # inconsistent with the calendar. Fail loudly rather than
                # silently reindex/forward-fill, per your project's
                # "no silent substitution" principle.
                raise ValueError(
                    f"{ticker} is missing {len(missing_dates)} date(s) present in "
                    f"trading_calendar (e.g. {missing_dates[0].date()}) -- this "
                    f"breaks the rectangular-matrix assumption Stage 3 depends on."
                )
            self._data[ticker] = df

    def __iter__(self) -> Iterator[pd.Timestamp]:
        self._position = -1
        return self

    def __next__(self) -> pd.Timestamp:
        self._position += 1
        if self._position >= len(self._calendar):
            raise StopIteration
        self._current_date = self._calendar[self._position]
        return self._current_date

    def current_bar(self, ticker: str) -> pd.Series:
        if self._current_date is None:
            raise RuntimeError("current_bar() called before iteration started -- "
                                "iterate over the DataFeed first (e.g. `for date in feed:`).")
        try:
            row = self._data[ticker].loc[self._current_date]
        except KeyError as e:
            raise KeyError(
                f"No row for {ticker} on {self._current_date.date()} -- "
                f"should be unreachable given the intersection-calendar check in "
                f"_load(); if you see this, that check has a bug."
            ) from e
        row.name = self._current_date  # so callers relying on `current_bar.name` (e.g. Strategy) work
        return row

    def history(self, ticker: str, lookback: int) -> pd.DataFrame:
        if self._current_date is None:
            raise RuntimeError("history() called before iteration started.")
        full = self._data[ticker]
        idx = full.index.get_loc(self._current_date)
        start_idx = max(0, idx - lookback + 1)
        return full.iloc[start_idx: idx + 1]
