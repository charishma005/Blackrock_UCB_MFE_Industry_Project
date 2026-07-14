"""The weekly analyst-PM meeting, turned into a standing run loop.

"That meeting is not a decoration on top of the design; it is the design."
``Fund.run_meeting`` is one sitting of the meeting: the analysts speak (each
from its own isolated evidence), each PM arbitrates what it heard and expresses
one trade, and the unifying layer sizes the strategies into a single fund book
and feeds capital back down. The layers may run on different clocks in a real
deployment; here one meeting is one rebalance, driven by an ``AsOf`` so the
whole sitting sees only what was knowable at that instant.

``MeetingResult`` is the full, auditable record of one sitting — every view,
every trade, the allocation, and the final netted book — so that when the fund
does well or badly one can ask, separately, whether a view was wrong, the
arbitrage was poor, or the risk was mismanaged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.layered.analysts.base import SingleDriverAnalyst
from src.layered.contracts import DriverView, FundAllocation, StrategyTrade
from src.layered.fund import FundAllocator
from src.layered.pm.base import PortfolioManagerBase
from src.layered.timeline import AsOf


@dataclass
class MeetingResult:
    asof: pd.Timestamp
    views: list[DriverView]                      # analyst layer output
    trades: list[StrategyTrade]                  # PM layer output (pre-allocation)
    allocation: FundAllocation                   # unifying layer output
    sized_trades: list[StrategyTrade]            # PM trades after capital feedback
    book: pd.Series                              # final netted fund book (instrument → weight)

    def views_by_driver(self) -> dict[str, DriverView]:
        return {v.driver: v for v in self.views}


class Fund:
    """A layered fund: a research pool, several PMs, and the unifying layer."""

    def __init__(
        self,
        analysts: list[SingleDriverAnalyst],
        pms: list[PortfolioManagerBase],
        allocator: FundAllocator | None = None,
    ):
        self.analysts = analysts
        self.pms = pms
        self.allocator = allocator or FundAllocator()

    def run_meeting(
        self, world: AsOf, strat_returns: dict[str, pd.Series] | None = None
    ) -> MeetingResult:
        # 1. Analyst layer — each expert forms its view in isolation. The shared
        #    research pool is computed once and handed to every PM; PMs differ in
        #    what they listen to and how they arbitrate, not in the raw research.
        views = [a.form_view(world) for a in self.analysts]

        # 2. PM layer — each PM arbitrates the views it listens to and expresses
        #    one relative-value trade.
        trades = [pm.decide(views, world) for pm in self.pms]

        # 3. Unifying layer — net across strategies, size by conviction/risk/
        #    diversification, and pass capital + constraints back DOWN.
        allocation = self.allocator.allocate(trades, world, strat_returns)

        # 4. Feed the allocation back down: each PM scales its trade by its capital.
        sized = [pm.apply_allocation(t, allocation) for pm, t in zip(self.pms, trades)]

        # 5. The fund book is the netted, sized set of strategy trades.
        book: dict[str, float] = {}
        for t in sized:
            for sym, w in t.legs.items():
                book[sym] = book.get(sym, 0.0) + w
        book_series = pd.Series(book, dtype=float).sort_index()

        return MeetingResult(
            asof=world.asof, views=views, trades=trades, allocation=allocation,
            sized_trades=sized, book=book_series,
        )


def macro_rates_fund(llm_client=None) -> Fund:
    """The one worked instance from the thesis: a single macro-rates strategy fed
    by four single-driver analysts. A single meeting is one instance of the
    pattern; more PMs (equity long/short, credit basis) slot in alongside without
    touching the analyst or fund layers, because the contracts are fixed."""
    from src.layered.analysts.macro_rates import macro_rates_analysts
    from src.layered.pm.macro_rates import MacroRatesPM

    return Fund(
        analysts=macro_rates_analysts(llm_client),
        pms=[MacroRatesPM(llm_client)],
    )
