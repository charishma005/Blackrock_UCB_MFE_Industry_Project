"""Research-quality scoring — grading analysts on being RIGHT, not on P&L.

The thesis draws a sharp line: an analyst is "judged by whether a view is
correct," a PM by "whether the resulting book makes money." Keeping these apart
is what lets the fund ask, when it does badly, whether "an expert view was
wrong, the arbitrage among views was poor, or the risk was mismanaged — three
very different failures that a monolithic system would blur together."

This module scores the analyst layer on its OWN terms: did each single-driver
view call its driver's next move correctly? A view about inflation is graded
against whether inflation actually rose — never against trade P&L, which belongs
to the PM. Scoring is done view-over-view on the analyst's own reported ``level``
(the metric it committed to), so it is model-consistent and point-in-time: the
view formed at meeting t is graded by the level it reports at meeting t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.layered.contracts import DriverView

_EPS_DEFAULT = 1e-9


def score_driver_views(views: list[DriverView], eps: float = _EPS_DEFAULT) -> dict:
    """Grade one analyst's chronological views against its own realized levels.

    Returns hit rate (directional accuracy on non-flat calls), a conviction-
    weighted information score (signed conviction × realized direction), and the
    sample size. Needs at least two dated views carrying a ``level``.
    """
    dated = sorted([v for v in views if v.level is not None], key=lambda v: v.asof)
    if len(dated) < 2:
        return {"n": 0, "hit_rate": float("nan"), "info_score": float("nan"),
                "avg_conviction": float("nan")}

    move_eps = max(eps, 1e-6)
    hits, scored, info, convs = 0, 0, [], []
    for cur, nxt in zip(dated, dated[1:]):
        realized = float(nxt.level) - float(cur.level)   # what the driver actually did
        moved = abs(realized) > move_eps
        if cur.direction == "flat":
            # a "flat" call is graded only against whether the driver stayed put;
            # no directional bet, so it never enters the info score.
            scored += 1
            hits += int(not moved)
            convs.append(cur.conviction)
            continue
        # A directional call can only be graded over a period where the driver
        # ACTUALLY MOVED. At weekly meetings a monthly driver (CPI, unemployment)
        # reports the same level for weeks; those pairs carry no information and
        # are excluded from the hit rate rather than counted as misses — which
        # would spuriously punish an analyst for the release calendar.
        if not moved:
            continue
        predicted = 1.0 if cur.direction == "up" else -1.0
        realized_dir = float(np.sign(realized))
        scored += 1
        hits += int(predicted == realized_dir)
        info.append(cur.signed_conviction * realized_dir)
        convs.append(cur.conviction)

    return {
        "n": scored,
        "hit_rate": round(hits / scored, 3) if scored else float("nan"),
        "info_score": round(float(np.mean(info)), 3) if info else float("nan"),
        "avg_conviction": round(float(np.mean(convs)), 3) if convs else float("nan"),
    }


class ResearchScorer:
    """Accumulates views per analyst across meetings and grades the layer."""

    def __init__(self):
        self._views: dict[str, list[DriverView]] = {}

    def record(self, views: list[DriverView]) -> None:
        for v in views:
            self._views.setdefault(v.driver, []).append(v)

    def scorecard(self) -> pd.DataFrame:
        """One row per analyst: how right its driver calls have been so far."""
        rows = []
        for driver, vs in self._views.items():
            rows.append({"driver": driver, **score_driver_views(vs)})
        if not rows:
            return pd.DataFrame(columns=["driver", "n", "hit_rate", "info_score", "avg_conviction"]).set_index("driver")
        return pd.DataFrame(rows).set_index("driver")
