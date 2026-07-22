"""Analyst → PM adapter — the merge seam with the analyst layer.

The analyst layer produces one ``DriverView`` per driver: a direction,
conviction, horizon, reasoning, and current level. The PM pods consume those.
This module is the single point where analyst output is pulled point-in-time for
a rebalance date and handed to the pods, so the coupling lives in exactly one
place instead of being scattered through the engine.

Two analyst sources plug in here, both emitting the same ``DriverView`` contract:

  * **Replay** (wired, $0): the four equity drivers replayed from macro-llm's r7
    vector signals (``src.portfolio.replay_analyst``). No API key — the engine
    gets 14 years of real driver views out of the box.
  * **Live** (opt-in): pass ``live={driver: LLMAnalyst}`` and this feed calls each
    on an ``AsOf`` world. Used once an API key + a paid run are approved; a live
    equity persona is a NEW experiment, not the validated replay signal.

The FRED-macro drivers (inflation, curve_slope, ...) remain unwired here — that
is the upstream analyst team's call — so pods listening only to those still emit
neutral until a live feed is supplied.

The engine calls ``analyst_views_asof(asof, macro_asof)`` (two positional args);
that signature is preserved, so wiring a real feed did not change the call site.
"""
from __future__ import annotations

import pandas as pd

from src.portfolio.replay_analyst import EQUITY_DRIVERS, load_replay_analysts

# Drivers the pods can listen to. Kept here so the placeholder and the real
# wiring agree on the driver name-space (must match personas in src.layered).
KNOWN_DRIVERS: tuple[str, ...] = (
    "inflation", "curve_slope", "term_premium", "financial_conditions",
    "labor_tightness", "inflation_expectations", "balance_sheet",
    "sector_breadth", "vol_regime", "risk_appetite", "positioning",
)

# Lazily-built replay analysts, cached across rebalance dates.
_REPLAY = None


def _replay():
    global _REPLAY
    if _REPLAY is None:
        _REPLAY = load_replay_analysts()
    return _REPLAY


def analyst_views_asof(
    asof: pd.Timestamp,
    macro_asof: dict[str, "pd.Series"] | None = None,
    *,
    live: dict[str, object] | None = None,
) -> dict[str, dict]:
    """Return ``{driver: DriverView-as-dict}`` known as of ``asof``.

    Live analysts (if supplied) take precedence for their drivers; the equity
    replay analysts fill in the rest of ``EQUITY_DRIVERS``. Drivers with no view
    are omitted, so a pod listening only to them keeps emitting neutral.
    """
    views: dict[str, dict] = {}

    if live:
        from src.layered.timeline import AsOf
        world = AsOf(asof=pd.Timestamp(asof), macro=macro_asof or {},
                     prices=pd.DataFrame())
        for driver, analyst in live.items():
            view = analyst.form_view(world)
            views[driver] = view.model_dump(mode="json")

    for driver in EQUITY_DRIVERS:
        if driver in views:
            continue                       # a live analyst already covered it
        analyst = _replay().get(driver)
        if analyst is None:
            continue
        view = analyst.view_asof(pd.Timestamp(asof))
        if view is not None:
            views[driver] = view.model_dump(mode="json")

    return views
