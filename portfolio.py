"""
Portfolio

Responsibility: own cash, open positions, and the equity curve for a
SINGLE shared pool used by BOTH strategies (MA crossover and RSI mean
reversion). Resolves competing signals from the two strategies into a
final list of orders.

Ownership rule: once a position is opened by one strategy, only that
strategy's signal governs its exit -- the other strategy's opinion on
that ticker is ignored while the position stays open. This prevents two
different pieces of logic fighting over the same position.

Three-phase signal resolution per bar:
  Phase 1:   Signal-driven exits (SELL -> full exit, BUY->HOLD -> scale-out)
  Phase 1.5: Conviction-based replacement (strong challenger displaces weak
             defender, subject to edge threshold + holding period + daily cap)
  Phase 2:   New entries (remaining top-k candidates sized via inverse-vol)

Owns: position sizing rule (inverse-volatility), cash accounting, exit
rules (partial scale-out then full exit), cross-strategy arbitration,
conviction-based replacement.
Does NOT own: fill price/cost (ExecutionHandler), or the underlying
signal logic itself (Strategy).
"""

from dataclasses import dataclass, field
import pandas as pd
from strategy import SignalEvent, Signal
from ranking import percentile_scores
from config import (PARTIAL_SCALE_OUT_FRACTION, MAX_DEPLOYMENT_FRACTION,
                    MIN_CONVICTION_EDGE, MIN_HOLDING_PERIOD, MAX_REPLACEMENTS_PER_DAY)


@dataclass
class OrderRequest:
    ticker: str
    date: pd.Timestamp
    direction: str          # "BUY" or "SELL"
    quantity: int
    strategy_id: str        # which strategy this order belongs to (ownership tag)


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_price: float = 0.0
    strategy_id: str = ""       # which strategy opened/owns this position
    last_signal: Signal = Signal.HOLD  # tracks BUY->HOLD transition for scale-out logic
    scaled_out: bool = False    # True once the one-time partial scale-out has fired
    bars_held: int = 0          # how many bars this position has been open (for MIN_HOLDING_PERIOD)


class Portfolio:
    def __init__(self, initial_cash: float, top_k: int = 5):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.top_k = top_k
        self.positions: dict[str, Position] = {}
        self.equity_curve: list[dict] = []  # [{date, cash, holdings_value, total_equity}, ...]
        self.replacement_log: list[dict] = []  # Phase 1.5 swap events
        self.trade_log: list[dict] = []  # every fill, for post-hoc analysis

    # ------------------------------------------------------------------
    # Signal resolution + order sizing
    # ------------------------------------------------------------------

    def process_signals(self, ma_signals: list[SignalEvent], rsi_signals: list[SignalEvent],
                         current_prices: dict[str, float],
                         volatilities: dict[str, float]) -> list[OrderRequest]:
        """
        Main entry point, called once per bar with BOTH strategies'
        full signal sets for that day. Returns the list of orders to
        queue for next-bar execution.

        Three phases:
          1.   EXITS -- signal-driven (SELL/HOLD transitions).
          1.5  REPLACEMENT -- conviction-based displacement of weak
               defenders by strong challengers.
          2.   NEW ENTRIES -- remaining top-k candidates, sized via
               inverse-volatility weighting using all available cash
               (including estimated proceeds from Phase 1/1.5 sells).
        """
        ma_by_ticker = {s.ticker: s for s in ma_signals}
        rsi_by_ticker = {s.ticker: s for s in rsi_signals}

        orders: list[OrderRequest] = []

        # Increment bars_held for every open position at the start of
        # each bar -- this counter drives the MIN_HOLDING_PERIOD guard.
        for pos in self.positions.values():
            pos.bars_held += 1

        # === Phase 1: signal-driven exits on currently held positions ===
        for ticker, pos in list(self.positions.items()):
            owner_signal = (ma_by_ticker if pos.strategy_id == "MA_CROSSOVER"
                             else rsi_by_ticker).get(ticker)
            if owner_signal is None:
                continue

            if owner_signal.signal == Signal.SELL:
                orders.append(OrderRequest(ticker=ticker, date=owner_signal.date,
                                            direction="SELL", quantity=pos.quantity,
                                            strategy_id=pos.strategy_id))
                pos.last_signal = Signal.SELL

            elif owner_signal.signal == Signal.HOLD:
                if pos.last_signal == Signal.BUY and not pos.scaled_out:
                    scale_out_qty = int(pos.quantity * PARTIAL_SCALE_OUT_FRACTION)
                    if scale_out_qty > 0:
                        orders.append(OrderRequest(ticker=ticker, date=owner_signal.date,
                                                    direction="SELL", quantity=scale_out_qty,
                                                    strategy_id=pos.strategy_id))
                    pos.scaled_out = True
                pos.last_signal = Signal.HOLD

            else:  # still BUY
                pos.last_signal = Signal.BUY

        # === Phase 1.5: conviction-based replacement ===
        # Tickers that already have a SELL order from Phase 1 are out of
        # play -- don't tangle signal-driven exits with conviction swaps.
        phase1_sell_tickers = {o.ticker for o in orders if o.direction == "SELL"}
        still_held = {t: p for t, p in self.positions.items()
                      if t not in phase1_sell_tickers}

        # Compute percentile rankings across ALL BUY-signaled tickers
        # (held + unheld) so defenders and challengers are on the same scale.
        ma_pct = percentile_scores(ma_signals)
        rsi_pct = percentile_scores(rsi_signals)

        # Merge: one score per ticker, MA wins ties (same rule as before).
        all_scores: dict[str, tuple[float, str]] = {}
        for ticker, pct in ma_pct.items():
            all_scores[ticker] = (pct, "MA_CROSSOVER")
        for ticker, pct in rsi_pct.items():
            if ticker not in all_scores or pct > all_scores[ticker][0]:
                all_scores[ticker] = (pct, "RSI_MEANREV")

        # Separate defenders (held, still BUY, old enough to displace)
        # from challengers (unheld, BUY-signaled).
        defenders = []   # [(ticker, percentile)]
        challengers = []  # [(ticker, percentile, strategy_id)]

        for ticker, (pct, sid) in all_scores.items():
            if ticker in still_held:
                pos = still_held[ticker]
                # Only "beatable" if the OWNING strategy still says BUY
                # and the position has been held long enough.
                owner_signal = (ma_by_ticker if pos.strategy_id == "MA_CROSSOVER"
                                 else rsi_by_ticker).get(ticker)
                if (owner_signal
                        and owner_signal.signal == Signal.BUY
                        and pos.bars_held >= MIN_HOLDING_PERIOD):
                    defenders.append((ticker, pct))
            elif ticker not in phase1_sell_tickers:
                challengers.append((ticker, pct, sid))

        # Weakest defenders first, strongest challengers first.
        defenders.sort(key=lambda x: x[1])          # ascending
        challengers.sort(key=lambda x: x[1], reverse=True)  # descending

        replaced_tickers: set[str] = set()
        challenger_entries: list[tuple[str, str]] = []  # [(ticker, strategy_id)]
        defender_proceeds = 0.0

        d_idx = 0
        for c_ticker, c_pct, c_sid in challengers:
            if len(challenger_entries) >= MAX_REPLACEMENTS_PER_DAY:
                break
            # Advance past already-replaced defenders.
            while d_idx < len(defenders) and defenders[d_idx][0] in replaced_tickers:
                d_idx += 1
            if d_idx >= len(defenders):
                break
            d_ticker, d_pct = defenders[d_idx]
            if c_pct - d_pct >= MIN_CONVICTION_EDGE:
                # Swap: full exit the defender.
                pos = self.positions[d_ticker]
                orders.append(OrderRequest(ticker=d_ticker, date=None,
                                            direction="SELL", quantity=pos.quantity,
                                            strategy_id=pos.strategy_id))
                defender_proceeds += pos.quantity * current_prices.get(d_ticker, pos.avg_price)
                replaced_tickers.add(d_ticker)
                challenger_entries.append((c_ticker, c_sid))
                self.replacement_log.append({
                    "defender": d_ticker, "defender_pct": d_pct,
                    "defender_strategy": pos.strategy_id,
                    "defender_bars_held": pos.bars_held,
                    "challenger": c_ticker, "challenger_pct": c_pct,
                    "challenger_strategy": c_sid,
                    "edge": c_pct - d_pct,
                })
                d_idx += 1
            else:
                # Defenders are sorted ascending -- if the strongest
                # challenger can't beat the weakest defender, no point
                # checking further.
                break

        # === Phase 2: new entries ===
        # Pool = challengers that won replacement + normal top-k picks
        # among remaining unheld, non-selected tickers.
        all_excluded = (set(self.positions.keys())
                        | phase1_sell_tickers
                        | {t for t, _ in challenger_entries})

        phase2_candidates = {t: v for t, v in all_scores.items()
                             if t not in all_excluded}
        phase2_ranked = sorted(phase2_candidates.items(),
                               key=lambda kv: kv[1][0], reverse=True)
        remaining_slots = max(0, self.top_k - len(challenger_entries))
        phase2_selected = phase2_ranked[:remaining_slots]

        # Combine all entries to size together via inverse-vol.
        all_entries = (challenger_entries
                       + [(t, sid) for t, (pct, sid) in phase2_selected])

        if all_entries:
            inv_vols = {t: 1.0 / volatilities[t]
                        for t, _ in all_entries if volatilities.get(t, 0) > 0}
            total_inv_vol = sum(inv_vols.values()) if inv_vols else 0

            total_equity = self.cash + self._holdings_value(current_prices)
            deployable = (total_equity * MAX_DEPLOYMENT_FRACTION
                          - self._holdings_value(current_prices))

            # Add estimated proceeds from Phase 1.5 defender sells --
            # those sells will fill BEFORE these buys on the next bar
            # (sells come first in _pending_orders), so the cash will
            # actually be available when the buys execute.
            deployable += defender_proceeds
            deployable = max(0.0, min(deployable, self.cash + defender_proceeds))

            for ticker, sid in all_entries:
                if ticker not in inv_vols or total_inv_vol == 0:
                    continue
                weight = inv_vols[ticker] / total_inv_vol
                allocation = deployable * weight
                price = current_prices.get(ticker)
                if not price or price <= 0:
                    continue
                quantity = int(allocation // price)
                if quantity > 0:
                    orders.append(OrderRequest(ticker=ticker, date=None,
                                                direction="BUY", quantity=quantity,
                                                strategy_id=sid))

        return orders

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_on_fill(self, ticker: str, direction: str, quantity: int,
                        fill_price: float, cost: float, strategy_id: str = "") -> None:
        """
        Called by the engine after ExecutionHandler reports a fill.
        `cost` already reflects the 30bps friction -- deduct it, don't
        recompute it.
        """
        self.trade_log.append({
            "ticker": ticker, "direction": direction, "quantity": quantity,
            "fill_price": fill_price, "cost": cost, "strategy_id": strategy_id,
        })

        if direction == "BUY":
            self.cash -= (fill_price * quantity + cost)
            pos = self.positions.get(ticker)
            if pos is None:
                self.positions[ticker] = Position(ticker=ticker, quantity=quantity,
                                                   avg_price=fill_price,
                                                   strategy_id=strategy_id,
                                                   last_signal=Signal.BUY)
            else:
                new_qty = pos.quantity + quantity
                pos.avg_price = (pos.avg_price * pos.quantity + fill_price * quantity) / new_qty
                pos.quantity = new_qty

        elif direction == "SELL":
            self.cash += (fill_price * quantity - cost)
            pos = self.positions.get(ticker)
            if pos is None:
                raise ValueError(f"SELL fill for {ticker} with no open position -- bug upstream")
            pos.quantity -= quantity
            if pos.quantity <= 0:
                del self.positions[ticker]
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def _holdings_value(self, current_prices: dict[str, float]) -> float:
        return sum(pos.quantity * current_prices.get(t, pos.avg_price)
                   for t, pos in self.positions.items())

    def mark_to_market(self, date: pd.Timestamp, current_prices: dict[str, float]) -> None:
        holdings_value = self._holdings_value(current_prices)
        self.equity_curve.append({
            "date": date,
            "cash": self.cash,
            "holdings_value": holdings_value,
            "total_equity": self.cash + holdings_value,
        })
