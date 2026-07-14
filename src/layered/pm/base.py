"""Portfolio-manager base — the two jobs a PM does that analysts cannot.

"A PM does two things the analysts cannot. First, it arbitrates ... Second, and
more distinctively, it expresses." This base class makes that split structural:

    arbitrate(views) -> ArbitratedView   reconcile/weigh/discount the specialists
    express(view, world) -> StrategyTrade map the joint view onto ONE clean RV trade

``decide`` runs the two in sequence. ``arbitrate`` has a sensible generic
implementation (aggregate signed conviction per driver, preserve disagreement,
allow discounting a specialist speaking out of its depth). ``express`` is
abstract: it is "each PM's real edge, and it is where the market's economic
structure lives" — the transmission from driver-space to instrument-space —
so every concrete strategy must supply its own.

Senior to the analysts and answerable for the book, a PM is also where the fund
layer's allocation lands: ``apply_allocation`` scales the PM's trade by the
capital the fund granted it, closing the loop the thesis describes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd

from src.layered.contracts import ArbitratedView, DriverView, FundAllocation, StrategyTrade
from src.layered.timeline import AsOf


class PortfolioManagerBase(ABC):
    """One strategy's portfolio manager."""

    strategy: str = "base"
    listens_to: tuple[str, ...] = ()      # driver names this PM consumes

    def __init__(self, llm_client=None):
        self.llm = llm_client

    # ── Job 1: arbitrate ────────────────────────────────────────────────────
    def _discount(self, view: DriverView) -> float:
        """Multiplier in [0, 1] applied to a view before aggregation.

        Hook for "discount the specialist who is confident but speaking outside
        its competence." Default trusts every view fully; subclasses override.
        """
        return 1.0

    def arbitrate(self, views: list[DriverView]) -> ArbitratedView:
        """Reconcile the specialists into one joint driver view.

        Aggregates signed conviction per driver (mean across any analysts sharing
        a driver, after discounting), and measures *disagreement* — how much the
        analysts pull against each other — which is preserved on the
        ``ArbitratedView`` rather than averaged away, so a concrete PM can use a
        split committee as a signal (e.g. to bet smaller).
        """
        relevant = [v for v in views if not self.listens_to or v.driver in self.listens_to]
        by_driver: dict[str, list[float]] = defaultdict(list)
        asof = max((v.asof for v in relevant), default=pd.Timestamp.min)
        for v in relevant:
            by_driver[v.driver].append(v.signed_conviction * self._discount(v))

        drivers = {d: float(sum(xs) / len(xs)) for d, xs in by_driver.items()}
        disagreement = self._same_driver_disagreement(by_driver)
        return ArbitratedView(asof=asof, drivers=drivers, disagreement=disagreement)

    @staticmethod
    def _same_driver_disagreement(by_driver: dict[str, list[float]]) -> float:
        """Directional split among analysts covering the SAME driver, averaged.

        1 - |Σ s| / Σ|s| per driver: 0 when they agree in sign, → 1 when they
        cancel. With one analyst per driver this is 0; it becomes informative
        once a driver has several analysts. Cross-driver conflict (inflation
        hawkish vs labor dovish) is projected onto a common axis by the concrete
        PM, which understands what the drivers mean.
        """
        vals = []
        for xs in by_driver.values():
            denom = sum(abs(x) for x in xs)
            if denom > 1e-9:
                vals.append(1.0 - abs(sum(xs)) / denom)
        return float(sum(vals) / len(vals)) if vals else 0.0

    # ── Job 2: express ──────────────────────────────────────────────────────
    @abstractmethod
    def express(self, view: ArbitratedView, world: AsOf) -> StrategyTrade:
        """Map the joint driver view onto ONE relative-value trade.

        This is the PM's edge. Must produce a ``StrategyTrade`` whose legs cancel
        the shared/unwanted exposure and isolate only the conviction.
        """

    # ── The meeting: understanding → action ─────────────────────────────────
    def decide(self, views: list[DriverView], world: AsOf) -> StrategyTrade:
        return self.express(self.arbitrate(views), world)

    def apply_allocation(self, trade: StrategyTrade, allocation: FundAllocation) -> StrategyTrade:
        """Scale a trade by the capital the fund granted this strategy."""
        k = allocation.capital.get(self.strategy, 1.0)
        return trade.scaled(k)
