"""Layered agent fund — the analyst-PM meeting as a standing architecture.

This package implements the organizing principle set out in the project's
"A Layered Agent Fund" thesis: a fund built as *layers of specialized agents*,
each layer responsible for exactly one kind of work, interacting through the
shape of a weekly analyst-PM meeting.

    analyst layer  →  portfolio-manager layer  →  unifying (risk/allocation) layer
    (isolated          (arbitrate + express          (net exposures across
     single-driver      views as one relative-        strategies, allocate
     expertise)         value trade per strategy)     capital, feed back down)

The three layers are separated by *stable contracts* (``contracts.py``) and
nothing reaches into another layer's internals:

    * analysts emit ``DriverView``      — a claim about ONE driver, scored later
    * a PM emits ``StrategyTrade``      — a relative-value trade + its risk
    * the fund emits ``FundAllocation`` — capital + constraints, fed back down

The design deliberately mirrors, but does not replace, the flat persona
ensemble in ``src/agents`` + ``src/ensemble``. That ensemble blends whole-
investor opinions; this package instead separates *belief formation* (analysts)
from *action selection* (PMs) from *survival* (the fund layer), so each can be
built, scored, and improved on its own — which is the whole claim of the thesis.

Everything here runs offline and deterministically with no API key: each
analyst has a Phase-1 deterministic reading of its driver, and the LLM is an
optional Phase-2 refinement (same pattern as ``src/agents/base.py``).
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
