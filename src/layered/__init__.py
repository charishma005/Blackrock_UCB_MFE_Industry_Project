"""Layered agent fund — the analyst upstream (data → agent → report).

This package builds the *analyst layer*: a population of isolated single-driver
experts, each reading a partitioned slice of the data through the ``AsOf`` gate and
writing a ``DriverView`` — a claim about ONE driver, in a form that can be compared,
combined, and scored later.

``contracts.py`` holds the stable interfaces between layers, and is the seam the
rest of the fund merges against:

    * analysts emit ``DriverView``      — a claim about ONE driver, scored later
    * a PM emits ``StrategyTrade``      — a relative-value trade + its risk
    * the fund emits ``FundAllocation`` — capital + constraints, fed back down

Only ``DriverView`` is produced here; ``StrategyTrade`` and ``FundAllocation`` are
the downstream (PM / unifying-layer) contracts, kept here so those teammates have a
typed boundary to build against.
"""
from __future__ import annotations

from src.layered.contracts import (
    ArbitratedView,
    DriverDirection,
    DriverView,
    FundAllocation,
    StrategyTrade,
)

__all__ = [
    "ArbitratedView",
    "DriverDirection",
    "DriverView",
    "FundAllocation",
    "StrategyTrade",
]
