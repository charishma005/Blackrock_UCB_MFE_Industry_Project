"""The PM layer — one meeting, seven reports, one reconciled view.

The analyst layer produces ``DriverView`` reports, one driver at a time, each on its
own release clock. This layer assembles them into a *meeting* and hands that to an
LLM PM, which reconciles them into an ``ArbitratedView``.

    board.py         the meeting object — each analyst's latest view as of a date
    disagreement.py  how split the panel is, computed (never asked of the model)
    brief.py         N reports rendered into one prompt block
    llm_pm.py        the PM itself — one pod, one meeting, one arbitrated view
    build.py         wiring: pod + llm -> LLMPM

Two properties of this layer are worth stating because they shaped every module.

**Analyst spend is decoupled from PM iteration.** Analysts run once and write JSONL;
the board replays from disk. Iterating on the PM therefore costs one PM run, never
seven analyst runs.

**The drivers do not share a clock.** CPI releases mid-month, the jobs report early,
the market drivers resample to month-end — as of this writing there is no date on
which all seven have a view. So a meeting is an *as-of snap* (each analyst's latest
view known at the meeting date), never a join, and the snap is the layer's
look-ahead choke point in the same way ``AsOf`` is the analyst layer's.
"""
from __future__ import annotations

from src.layered.pm.board import BoardEntry, Meeting, ViewBoard
from src.layered.pm.disagreement import panel_disagreement

__all__ = ["ViewBoard", "Meeting", "BoardEntry", "panel_disagreement"]
