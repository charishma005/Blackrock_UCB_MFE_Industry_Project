"""v0 — the relevance-weighted combiner (`src/layered/pm/relevance_pm.py`).

Pins the three things that make it a *walk-forward* relevance weight and not a fitted one:
  * ``equal`` reproduces the mechanical (equal-weight) projection — the sanity anchor;
  * an analyst that has genuinely predicted the traded move gets a larger |weight| than
    one that has not — the combiner actually learns relevance;
  * no look-ahead — the weight at date *t* depends only on data strictly before *t*, so
    early meetings sit in warm-up (zero weight → equal-weight fallback) and a weight is
    unchanged whether or not later dates exist.

Synthetic multi-date board + synthetic macro, so "analyst A predicts the trade, B does
not" is exactly what the test constructs.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.layered.contracts import DriverView
from src.layered.pm.board import ViewBoard
from src.layered.pm.mechanical_pm import MechanicalPM
from src.layered.pm.relevance_pm import RelevancePM

DATES = pd.date_range("2016-01-31", periods=24, freq="ME")


def _view(driver, dt, direction, conviction):
    return DriverView(driver=driver, asof=pd.Timestamp(dt), direction=direction,
                      conviction=conviction, horizon_days=31, level=1.0)


def _board(seqs):
    """seqs: {driver: [direction per DATES]} at conviction 0.5."""
    return ViewBoard({d: [_view(d, dt, dr, 0.5) for dt, dr in zip(DATES, dirs)]
                      for d, dirs in seqs.items()})


def _macro_from(axis_vals):
    """A single-instrument macro dict whose month-end axis == axis_vals over DATES."""
    return {"T10YIE": pd.Series(axis_vals, index=DATES)}


def _cfg(drivers):
    return {"listens_to": {d: {"polarity": 1} for d in drivers}, "reads": list(drivers),
            "trade": {"universe": ["T10YIE"], "sign_convention": "single", "max_legs": 1}}


# A: alternates up/down; the axis is built so its next move == A's oriented view (IC +1).
# B: constant flat (no information).
_A = ["up", "down"] * 12
_B = ["flat"] * 24
# oriented_A at conviction .5 = +.5 / -.5; axis[t+1]-axis[t] = oriented_A[t]
_ORI_A = [0.5 if d == "up" else -0.5 for d in _A]
_AXIS = [2.0]
for o in _ORI_A[:-1]:
    _AXIS.append(_AXIS[-1] + o)


def test_equal_weighting_reproduces_mechanical_projection():
    board = _board({"a": _A, "b": _B})
    pm = RelevancePM("p", _cfg(["a", "b"]), weighting="equal")
    pm.fit(board, DATES, _macro_from(_AXIS))
    mech = MechanicalPM("p", _cfg(["a", "b"]))
    for dt in DATES:
        m = board.at(dt)
        assert pm._rate_projection(m) == pytest.approx(mech._rate_projection(m))


def test_ic_gives_the_predictive_analyst_more_weight():
    board = _board({"a": _A, "b": _B})
    pm = RelevancePM("p", _cfg(["a", "b"]), weighting="ic", min_obs=6)
    pm.fit(board, DATES, _macro_from(_AXIS))
    late = pm._weights[pd.Timestamp(DATES[-1])]           # full history
    assert abs(late["a"]) > 0.3                           # A demonstrably relevant
    assert abs(late["a"]) > abs(late["b"])                # more than the uninformative B


def test_warmup_is_zero_weight_then_falls_back_to_equal():
    board = _board({"a": _A, "b": _B})
    pm = RelevancePM("p", _cfg(["a", "b"]), weighting="ic", min_obs=6)
    pm.fit(board, DATES, _macro_from(_AXIS))
    early = pm._weights[pd.Timestamp(DATES[3])]           # only 3 pairs < min_obs
    assert early == {"a": 0.0, "b": 0.0}
    mech = MechanicalPM("p", _cfg(["a", "b"]))
    m = board.at(DATES[3])
    assert pm._rate_projection(m) == pytest.approx(mech._rate_projection(m))  # equal fallback


def test_no_lookahead_weight_is_unchanged_by_later_dates():
    board = _board({"a": _A, "b": _B})
    macro = _macro_from(_AXIS)
    k = 15
    full = RelevancePM("p", _cfg(["a", "b"]), weighting="ic", min_obs=6).fit(board, DATES, macro)
    trunc = RelevancePM("p", _cfg(["a", "b"]), weighting="ic", min_obs=6).fit(board, DATES[:k], macro)
    for dt in DATES[:k]:                                  # weights before the cut must match
        assert full._weights[pd.Timestamp(dt)] == pytest.approx(trunc._weights[pd.Timestamp(dt)])
