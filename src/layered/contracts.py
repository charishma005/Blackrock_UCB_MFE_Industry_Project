"""The layer interfaces — stable, well-defined contracts between layers.

The thesis is explicit that the commitment "at this stage is not the mechanism
of any interface but the discipline that each layer communicates through a
clean, well-defined contract rather than reaching into another layer's
internals." These pydantic models ARE those contracts. They are intentionally
small and method-agnostic: nothing here presumes how an analyst forms a view,
how a PM arbitrates, or how the fund allocates — those are later, independent
choices, exactly as the thesis intends.

Three contracts, one per interface in the meeting:

    DriverView       analyst  → PM     "here is my expert claim about one driver"
    StrategyTrade    PM       → fund   "here is the one trade + the risk it carries"
    FundAllocation   fund     → PM     "here is your capital and your constraints"

``ArbitratedView`` is not a layer boundary — it is the PM's internal, reconciled
picture of the joint driver view, kept as a typed object so the arbitration step
is observable and the *disagreement* among analysts is preserved rather than
averaged away (one of the open questions the thesis flags).
"""
from __future__ import annotations

from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

# A driver moves up, down, or is going nowhere. This is a statement about the
# DRIVER (inflation, the 2y rate, ...), NOT about any tradeable instrument — the
# separation of driver-space from instrument-space is the analyst/PM boundary.
DriverDirection = Literal["up", "down", "flat"]


class DriverView(BaseModel):
    """An analyst's expert claim about its single driver.

    The thesis specifies exactly what a view carries: "a direction, a degree of
    conviction, a horizon over which the view is expected to hold, and the
    reasoning behind it" — delivered "in a consistent form so that it can be
    compared, combined, and later scored against what actually happened."

    ``level`` records the analyst's current reading of the driver so a view can
    be scored after the fact (did the driver actually move the way it said?),
    which is what lets research quality be attributed separately from trade P&L.
    """

    driver: str                                  # e.g. "inflation", "front_rate"
    asof: pd.Timestamp                           # the moment the view was formed
    direction: DriverDirection
    conviction: float = Field(ge=0.0, le=1.0)    # 0 = no view, 1 = maximal
    horizon_days: int = Field(gt=0)              # how long the view is meant to hold
    reasoning: str = ""
    level: Optional[float] = None                # current measured level, for scoring

    model_config = {"arbitrary_types_allowed": True}

    @property
    def signed_conviction(self) -> float:
        """Direction folded into conviction: +conv up, -conv down, 0 flat."""
        return {"up": 1.0, "down": -1.0, "flat": 0.0}[self.direction] * self.conviction


class ArbitratedView(BaseModel):
    """The PM's reconciled read across the analysts it listens to.

    Not a layer boundary — this is the PM's own working object, exposed as a
    type so arbitration is legible. ``drivers`` is the surviving joint view
    (driver → signed conviction after the PM has discounted, reconciled, and
    weighed the specialists). ``disagreement`` preserves how much the analysts
    conflicted, so it can be used as a signal in its own right rather than being
    silently averaged out.
    """

    asof: pd.Timestamp
    drivers: dict[str, float]                    # driver → signed conviction in [-1, 1]
    disagreement: float = 0.0                    # 0 = unanimous, 1 = maximally split
    notes: str = ""

    model_config = {"arbitrary_types_allowed": True}


class StrategyTrade(BaseModel):
    """A portfolio manager's output: one relative-value trade + its risk.

    "Each PM resolves the research into one unifying relative-value trade." The
    trade is expressed in instrument-space as signed target weights per leg
    (``legs``); a clean RV trade is constructed so the shared/unwanted exposure
    (here, the level of rates) nets out and only the conviction (the curve
    slope) remains — which the ``risk`` block characterizes for the layer above.

    ``conviction`` is the PM's confidence in the *trade* (distinct from any one
    analyst's conviction) and is what the fund layer sizes against.
    """

    strategy: str                                # e.g. "macro_rates"
    asof: pd.Timestamp
    legs: dict[str, float]                       # instrument → signed weight
    conviction: float = Field(ge=0.0, le=1.0)
    rationale: str = ""
    risk: dict = Field(default_factory=dict)     # PM's characterization of the risk

    model_config = {"arbitrary_types_allowed": True}

    @property
    def gross(self) -> float:
        return float(sum(abs(w) for w in self.legs.values()))

    def scaled(self, k: float) -> "StrategyTrade":
        """Return a copy with every leg scaled by ``k`` (capital allocation)."""
        return self.model_copy(update={"legs": {s: w * k for s, w in self.legs.items()}})


class FundAllocation(BaseModel):
    """The unifying layer's decision, fed back DOWN to the PMs.

    "The risk layer passes constraints and adjusted allocations back down,
    closing a loop rather than merely issuing vetoes." One entry per strategy:
    the capital multiplier the fund grants it and any binding constraints. The
    thesis is firm that this layer is a control layer, not a forecasting one —
    hence no views here, only sizing and limits.
    """

    asof: pd.Timestamp
    capital: dict[str, float]                    # strategy → capital multiplier (≥ 0)
    constraints: dict[str, dict] = Field(default_factory=dict)  # strategy → limits
    diagnostics: dict = Field(default_factory=dict)             # netting/vol/breadth

    model_config = {"arbitrary_types_allowed": True}
