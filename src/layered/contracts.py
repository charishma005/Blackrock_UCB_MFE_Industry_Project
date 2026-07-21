"""The layer interfaces — stable, well-defined contracts between layers.

**This file is the merge seam.** This repo owns the upstream — data feeds →
analyst agents → report — and hands off at ``DriverView``. The downstream PM,
ensemble, and risk layers are built by other teammates; they consume
``DriverView`` and implement against ``StrategyTrade`` / ``FundAllocation``. Those
two are kept here, unused upstream, precisely so the people building the layers
above have a typed boundary to write to. Change these shapes only by agreement —
every layer depends on them.

The thesis is explicit that the commitment "at this stage is not the mechanism
of any interface but the discipline that each layer communicates through a
clean, well-defined contract rather than reaching into another layer's
internals." These pydantic models ARE those contracts. They are intentionally
small and method-agnostic: nothing here presumes how an analyst forms a view,
how a PM arbitrates, or how the fund allocates — those are later, independent
choices, exactly as the thesis intends.

Three contracts, one per interface in the meeting:

    DriverView       analyst  → PM     "here is my expert claim about one driver"  ← WE PRODUCE THIS
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

    # ── report-era fields ───────────────────────────────────────────────────
    # An analyst now writes a *report*, and (because PMs are LLMs too) the report
    # is what crosses the layer boundary. The fields above survive as the
    # machine-readable header the graders use, not as PM arithmetic. All optional,
    # so every pre-existing caller is unaffected.
    report: str = ""
    key_evidence: list[str] = Field(default_factory=list)  # feature names leaned on
    falsifier: str = ""                          # what would change this view
    source: str = ""                             # "llm:inflation", "benchmark:persistence"
    degraded: bool = False                       # emitted after a failure — exclude from grading
    # Re-emitted unchanged because no evidence moved since the previous meeting. Not
    # an independent observation: counting carried views as fresh ones is what makes
    # a monthly driver look like it produced 52 opinions a year.
    carried: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @property
    def signed_conviction(self) -> float:
        """Direction folded into conviction: +conv up, -conv down, 0 flat."""
        return {"up": 1.0, "down": -1.0, "flat": 0.0}[self.direction] * self.conviction


class SeriesFeature(BaseModel):
    """One measured quantity, as a short trajectory the analyst can read."""

    name: str
    values: list[float]                          # oldest → newest
    unit: str = ""
    description: str = ""                        # construction only; shown in describe mode


class ScalarFeature(BaseModel):
    """One measured quantity, as of now."""

    name: str
    value: float
    unit: str = ""
    description: str = ""                        # construction only; shown in describe mode


class FeatureSet(BaseModel):
    """Everything measurable an analyst is allowed to see about its driver.

    The input contract for the analyst layer, and deliberately a *measurement*
    object: levels, changes, moving averages, spreads. No score, no direction, no
    signal — every act of judgment belongs to the model that reads this.

    It is also what makes a benchmark comparable. Anything grading against an
    analyst consumes the same ``FeatureSet``, so "did it see more data?" stops
    being a question of discipline and becomes a property of the type.
    """

    driver: str
    asof: pd.Timestamp
    series: list[SeriesFeature] = Field(default_factory=list)
    scalars: list[ScalarFeature] = Field(default_factory=list)
    level_feature: Optional[str] = None          # which feature is the driver's level
    sources_read: list[str] = Field(default_factory=list)   # raw series touched (audit)

    model_config = {"arbitrary_types_allowed": True}

    @property
    def names(self) -> set[str]:
        return {f.name for f in self.series} | {f.name for f in self.scalars}

    @property
    def level(self) -> Optional[float]:
        """The driver's headline measurement — what scoring grades against."""
        if self.level_feature is None:
            return None
        for f in self.series:
            if f.name == self.level_feature:
                return f.values[-1] if f.values else None
        for f in self.scalars:
            if f.name == self.level_feature:
                return f.value
        return None

    def render(self, describe: bool = False) -> str:
        """The measurement block as the analyst sees it — relative time only.

        No absolute dates anywhere: a date is the single token that most helps a
        model recall the period instead of reading the evidence.

        ``describe`` adds each feature's construction note (what it IS, never what
        it implies). Off by default so the un-described arm reproduces exactly.
        """
        lines: list[str] = []
        for f in self.series:
            n = len(f.values)
            unit = f" ({f.unit})" if f.unit else ""
            lines.append(f"{f.name}{unit} — last {n} observations, oldest → newest")
            if describe and f.description:
                lines.append(f"    = {f.description}")
            lines.append("  " + ", ".join(f"{v:.2f}" for v in f.values))
        if self.scalars:
            if lines:
                lines.append("")
            lines.append("Derived measurements")
            if describe and any(f.description for f in self.scalars):
                # per-line block — the aligned column can't carry a description
                for f in self.scalars:
                    unit = f" {f.unit}" if f.unit else ""
                    lines.append(f"  {f.name}  {f.value:+.2f}{unit}")
                    if f.description:
                        lines.append(f"    = {f.description}")
            else:
                width = max(len(f.name) for f in self.scalars)
                for f in self.scalars:
                    unit = f" {f.unit}" if f.unit else ""
                    lines.append(f"  {f.name.ljust(width)}  {f.value:+.2f}{unit}")
        return "\n".join(lines)


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
