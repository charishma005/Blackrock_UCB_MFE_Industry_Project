"""How split the panel is — computed from the board, never asked of the model.

``ArbitratedView.disagreement`` exists so that conflict among the analysts is
preserved as a quantity rather than averaged away. That makes it a property of the
*inputs*, so it is computed here from the views the PM was shown. Asking the model
for it would make an auditable number a matter of opinion, and would let the PM
report a unanimous panel as split.

The one piece of judgment involved is **polarity**, declared per driver in the pod
config. Drivers do not share an orientation: an analyst calling ``curve_slope`` "up"
(steepening) and one calling ``inflation`` "up" are not agreeing about anything until
both are projected onto a common axis. Polarity is that projection. It is declared in
advance and never fitted — deriving it from outcomes would turn this measurement into
a signal, which is the line ``evaluation/panel.py`` draws for the feature layer and
which applies just as much here.
"""
from __future__ import annotations

from typing import Mapping

from src.layered.pm.board import Meeting


def oriented(m: Meeting, polarity: Mapping[str, float]) -> dict[str, float]:
    """Each present analyst's signed conviction, projected onto the pod's axis.

    Uses ``DriverView.signed_conviction`` so the direction→sign map lives in exactly
    one place (the contract) rather than being re-derived per consumer.

    **A driver with no declared polarity is skipped, not defaulted to +1.** A pod that
    reads more analysts than it takes views on (``reads`` wider than ``listens_to``)
    puts drivers on the board whose orientation it never declared; folding those in at
    an assumed +1 would silently make ``disagreement`` a different number depending on
    who the pod happened to be *reading*, which is not what it measures. The axis is
    the pod's own, so only the drivers the pod placed on that axis belong here.
    """
    return {d: float(polarity[d]) * e.view.signed_conviction
            for d, e in m.entries.items() if e.present and d in polarity}


def panel_disagreement(m: Meeting, polarity: Mapping[str, float]) -> float:
    """0 = the panel points one way, 1 = maximally split.

    ``1 - |sum(x)| / sum(|x|)``: the shared direction over the total conviction in the
    room. Conviction-weights itself, since a weak view contributes little to either
    term — two strong analysts in opposition is a real disagreement, two indifferent
    ones is not.

    An all-flat panel returns 0.0 rather than 1.0. Nobody having a view is an absence
    of opinion, not a conflict of opinion, and scoring it as maximal conflict would
    put the quietest meetings at the top of any "where is the panel split?" ranking.
    """
    x = list(oriented(m, polarity).values())
    den = sum(abs(v) for v in x)
    if den == 0.0:
        return 0.0
    return float(1.0 - abs(sum(x)) / den)


def override(arbitrated: Mapping[str, float], m: Meeting) -> float:
    """How far the PM moved the panel, on jointly-present drivers, in [0, 1].

    Mean absolute gap between the PM's conviction on a driver and that driver's own
    analyst's, halved because both live in [-1, 1] and so can differ by 2. Zero means
    the PM restated the panel; large means it overrode it. Reported alongside the IC
    so "the PM helped" can be separated from "the PM changed nothing".
    """
    pairs = [(v, m.entries[d].view.signed_conviction)
             for d, v in arbitrated.items()
             if d in m.entries and m.entries[d].present]
    if not pairs:
        return 0.0
    return float(sum(abs(pm - an) for pm, an in pairs) / (2.0 * len(pairs)))
