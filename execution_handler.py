"""
ExecutionHandler

Responsibility: turn an OrderRequest into a Fill -- the price you
actually got, and the cost you actually paid. This is the seam where
Stage 4 (order book simulator) will eventually plug in something far
more sophisticated (partial fills, queue position, market impact).
Keep the interface stable now so that swap-in is clean later.

Owns: fill price determination, cost model application.
Does NOT own: whether to trade, or how much (that's Portfolio).
"""

from dataclasses import dataclass
import pandas as pd
from portfolio import OrderRequest
from config import per_leg_cost_fraction, FILL_PRICE_FIELD


@dataclass
class Fill:
    ticker: str
    date: pd.Timestamp
    direction: str
    quantity: int
    fill_price: float
    cost: float           # total cost in currency, not bps


class ExecutionHandler:
    def __init__(self):
        self.per_leg_cost = per_leg_cost_fraction()  # 0.0015 (15 bps) by default

    def execute_order(self, order: OrderRequest, next_bar: pd.Series) -> Fill:
        """
        `next_bar` is bar t+1 -- the engine is responsible for handing
        over the CORRECT bar (today's bar, when the order was queued
        yesterday); this method does not itself reach into DataFeed, to
        avoid any ambiguity about which bar is being used.
        """
        fill_price = next_bar[FILL_PRICE_FIELD]
        if pd.isna(fill_price) or fill_price <= 0:
            raise ValueError(
                f"Invalid fill price for {order.ticker} on "
                f"{next_bar.name}: {fill_price!r} -- refusing to fill "
                f"rather than silently using a bad price."
            )

        notional = fill_price * order.quantity
        cost = notional * self.per_leg_cost

        return Fill(
            ticker=order.ticker,
            date=next_bar.name,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            cost=cost,
        )
