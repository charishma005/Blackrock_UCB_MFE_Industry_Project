"""Analyst → PM adapter — the merge seam with Elias's upstream.

Elias's analyst layer (``src.layered``) produces one ``DriverView`` per driver:
a direction, conviction, horizon, reasoning, and current level. The PM pods
consume those. This module is the single point where the upstream analyst output
is pulled point-in-time for a rebalance date and handed to the pods, so the
coupling to Elias's code lives in exactly one place instead of being scattered
through the engine.

──────────────────────────────────────────────────────────────────────────────
STATUS: PLACEHOLDER. ``analyst_views_asof`` currently returns an empty mapping,
which makes every pod emit neutral — the pipeline runs as pure plumbing with no
dependency on Elias's runner or an API key. To go live, build each analyst via
``src.layered.analysts.build_analyst`` and call ``form_view`` on an ``AsOf`` world
here, returning ``{driver: view_dict}``. The pods already accept that shape.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import pandas as pd

# Drivers the pods can listen to. Kept here so the placeholder and the real
# wiring agree on the driver name-space (must match personas in src.layered).
KNOWN_DRIVERS: tuple[str, ...] = (
    "inflation", "curve_slope", "term_premium", "financial_conditions",
    "labor_tightness", "inflation_expectations", "balance_sheet",
    "sector_breadth", "vol_regime", "risk_appetite", "positioning",
)


def analyst_views_asof(
    asof: pd.Timestamp,
    macro_asof: dict[str, "pd.Series"] | None = None,
) -> dict[str, dict]:
    """Return {driver: {direction, conviction, reasoning, level}} as of ``asof``.

    PLACEHOLDER: returns {} → pods emit neutral. Real implementation reads
    Elias's analysts point-in-time. The signature already takes the
    no-lookahead macro slice the real analysts need, so swapping in the live
    feed does not change the engine's call site.
    """
    _ = (asof, macro_asof)  # placeholder — real feed reads src.layered analysts
    return {}
