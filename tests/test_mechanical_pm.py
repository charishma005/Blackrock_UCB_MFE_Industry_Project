"""The mechanical PM — the deterministic control the LLM PM's trade is read against.

These tests pin the arithmetic (so the baseline cannot silently change what it means)
and the two honesties: an ``opposed`` slope pod takes no mechanical trade, and an
absent driver is never invented. Synthetic meetings so each rate-axis case can be
constructed directly; then a smoke pass on the real board.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.layered.contracts import DriverView
from src.layered.pm.board import Meeting, ViewBoard
from src.layered.pm.mechanical_pm import MechanicalPM


def _view(driver: str, direction: str, conviction: float) -> DriverView:
    return DriverView(driver=driver, asof=pd.Timestamp("2023-06-30"),
                      direction=direction, conviction=conviction, horizon_days=31,
                      level=1.0)


def _meeting(views: dict[str, DriverView]) -> Meeting:
    """A meeting with every named driver present at age 0 — the board's own shape,
    built directly so a rate-axis case is exactly what the test says it is."""
    b = ViewBoard({d: [v] for d, v in views.items()})
    return b.at("2023-06-30")


def _pm(pod: str, listens_to: dict, trade: dict | None = None) -> MechanicalPM:
    cfg = {"listens_to": listens_to, "reads": list(listens_to)}
    if trade is not None:
        cfg["trade"] = trade
    return MechanicalPM(pod=pod, config=cfg)


# ── the trade sign follows the oriented panel ───────────────────────────────
def test_same_pod_level_trade_takes_both_legs_the_projection_sign():
    """A +1 driver calling 'up' is net upward rate pressure → both legs positive."""
    pm = _pm("lvl", {"inflation": {"polarity": 1}},
             {"universe": ["DGS2", "DGS10"], "sign_convention": "same"})
    av = pm.arbitrate(_meeting({"inflation": _view("inflation", "up", 0.8)}))
    assert av.trade is not None
    assert set(av.trade.legs) == {"DGS2", "DGS10"}
    assert all(w > 0 for w in av.trade.legs.values())          # yields-up bet
    assert av.trade.gross == pytest.approx(1.0)                # unit gross


def test_negative_polarity_flips_the_trade():
    """balance_sheet is -1: an 'up' call (expansion) is DOWNWARD pressure on yields, so
    the level trade goes short even though the analyst said 'up'. This is exactly the
    inversion the answer_space work is about, done in arithmetic."""
    pm = _pm("lvl", {"balance_sheet": {"polarity": -1}},
             {"universe": ["DGS2", "DGS10"], "sign_convention": "same"})
    av = pm.arbitrate(_meeting({"balance_sheet": _view("balance_sheet", "up", 0.6)}))
    assert all(w < 0 for w in av.trade.legs.values())


def test_opposed_pod_takes_no_mechanical_trade():
    """A slope pod needs a front-vs-long split the config does not declare. The baseline
    abstains rather than inventing one — the whole discipline of the module."""
    pm = _pm("crv", {"curve_slope": {"polarity": -1}},
             {"universe": ["DGS2", "DGS10"], "sign_convention": "opposed"})
    av = pm.arbitrate(_meeting({"curve_slope": _view("curve_slope", "up", 0.7)}))
    assert av.trade is None
    assert av.drivers                                          # driver block still forms


def test_single_instrument_pod_takes_one_leg():
    pm = _pm("real", {"inflation": {"polarity": 1}},
             {"universe": ["T10YIE"], "max_legs": 1})
    av = pm.arbitrate(_meeting({"inflation": _view("inflation", "up", 0.5)}))
    assert set(av.trade.legs) == {"T10YIE"}
    assert av.trade.legs["T10YIE"] == pytest.approx(1.0)


def test_max_legs_equal_to_universe_still_trades():
    """The boundary: a 2-instrument same pod with max_legs 2 is fully expressible."""
    pm = _pm("lvl", {"inflation": {"polarity": 1}},
             {"universe": ["DGS2", "DGS10"], "sign_convention": "same", "max_legs": 2})
    av = pm.arbitrate(_meeting({"inflation": _view("inflation", "up", 0.8)}))
    assert av.trade is not None and set(av.trade.legs) == {"DGS2", "DGS10"}


def test_same_pod_abstains_when_max_legs_forbids_all_legs():
    """A same-sign position over more instruments than max_legs permits would require
    choosing WHICH legs to hold — an undeclared decision. The baseline abstains rather
    than pick, mirroring the LLM PM's rejection of an over-legged trade and the opposed
    branch's refusal to fabricate a decomposition."""
    pm = _pm("lvl", {"inflation": {"polarity": 1}},
             {"universe": ["A", "B", "C"], "sign_convention": "same", "max_legs": 2})
    av = pm.arbitrate(_meeting({"inflation": _view("inflation", "up", 0.8)}))
    assert av.trade is None
    assert av.drivers                                          # driver block still forms


def test_flat_panel_is_a_flat_position_not_an_abstention():
    """Net-zero pressure is a decision to hold nothing, scored as a real zero — the same
    distinction the LLM PM's `flat` flag draws. It is a trade with empty legs, not None."""
    pm = _pm("lvl", {"inflation": {"polarity": 1}},
             {"universe": ["DGS2", "DGS10"], "sign_convention": "same"})
    av = pm.arbitrate(_meeting({"inflation": _view("inflation", "flat", 0.0)}))
    assert av.trade is not None and av.trade.legs == {}


# ── the driver block is the consensus blend, and never invents a driver ─────
def test_driver_block_matches_consensus_blend():
    """The per-meeting blend here must equal pm_bench.consensus_blend's arithmetic:
    0.5·own + 0.5·polarity·(oriented panel mean)."""
    pm = _pm("lvl", {"inflation": {"polarity": 1}, "labor_tightness": {"polarity": -1}})
    m = _meeting({"inflation": _view("inflation", "up", 0.8),
                  "labor_tightness": _view("labor_tightness", "down", 0.4)})
    av = pm.arbitrate(m)
    # oriented: inflation +0.8, labor -1*(-0.4)=+0.4 → panel mean +0.6
    # inflation: 0.5*0.8 + 0.5*1*0.6 = 0.70 ; labor: 0.5*(-0.4) + 0.5*(-1)*0.6 = -0.50
    assert av.drivers["inflation"] == pytest.approx(0.70)
    assert av.drivers["labor_tightness"] == pytest.approx(-0.50)


def test_driver_block_equals_pm_bench_consensus_blend():
    """Not just the same formula by hand — the same number as ``pm_bench.consensus_blend``
    itself, the batched control ``pm_bench`` grades as ``ic_mech``. This pins the claim
    that the driver block is one quantity computed in two places, not two that can drift.
    Values stay inside [-1, 1] so the mechanical clip is a no-op and equality is exact."""
    from src.layered.evaluation.pm_bench import consensus_blend

    pol = {"inflation": 1.0, "labor_tightness": -1.0}
    views = {"inflation": _view("inflation", "up", 0.8),
             "labor_tightness": _view("labor_tightness", "down", 0.4)}
    pm = _pm("lvl", {"inflation": {"polarity": 1}, "labor_tightness": {"polarity": -1}})
    block = pm._driver_block(_meeting(views))

    snap = pd.DataFrame({d: [v.signed_conviction] for d, v in views.items()},
                        index=[pd.Timestamp("2023-06-30")])
    mech = consensus_blend(snap, pol)
    assert set(block) == set(views)
    for d in block:
        assert block[d] == pytest.approx(float(mech[d].iloc[0]))


def test_absent_driver_never_filled():
    """A driver the pod listens to but that has no present view stays out of the block —
    never a fabricated 0.0, the rule the whole layer holds."""
    b = ViewBoard({"inflation": [_view("inflation", "up", 0.5)],
                   "term_premium": [DriverView(driver="term_premium",
                                               asof=pd.Timestamp("2024-01-31"),
                                               direction="up", conviction=0.5,
                                               horizon_days=31, level=1.0)]})
    pm = _pm("lvl", {"inflation": {"polarity": 1}, "term_premium": {"polarity": 1}})
    # meeting before term_premium ever reported
    av = pm.arbitrate(b.at("2023-06-30"))
    assert "inflation" in av.drivers
    assert "term_premium" not in av.drivers


# ── smoke on the real board ─────────────────────────────────────────────────
def test_runs_on_the_real_board():
    """End to end on the vendored analyst runs — no key, no spend, deterministic."""
    pytest.importorskip("yaml")
    import os
    if not os.path.exists("reports/ab/inflation_on.jsonl"):
        pytest.skip("board runs not present")
    pm = MechanicalPM.from_pod("duration")
    board = ViewBoard.from_dir("reports/ab", "_on", drivers=pm.reads, **pm.board_kwargs)
    dates = board.meeting_dates(freq=pm.clock_freq)[:12]
    views = [pm.arbitrate(board.at(d)) for d in dates]
    assert any(v.drivers for v in views)
    # every emitted trade names only in-universe instruments
    universe = set(pm.trade_config["universe"])
    for v in views:
        if v.trade is not None:
            assert set(v.trade.legs).issubset(universe)
