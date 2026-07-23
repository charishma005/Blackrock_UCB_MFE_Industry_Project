"""Scoring the PM's trade — the seam where a view becomes a position.

The sign convention is the thing worth defending with tests. It is fixed by the leg
description in the tool schema ("the trade is scored as the weighted sum of yield
changes"), which means a positive weight bets the yield *rises* — the opposite sign
from a price-space bond P&L. Nothing about the numbers looks wrong if that gets
inverted, so it is pinned here by hand-computed arithmetic rather than by round-trip.

No LLM calls; every fixture is synthetic or a written-out run file.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.layered.evaluation.trade_pnl import (forward_yield_change, load_trades,
                                              score_trades, trade_validity, yield_pnl)

TRADE_CONFIG = {"universe": ["DGS2", "DGS10"], "max_legs": 2, "sign_convention": "same"}


def _macro(dgs2: list[float], dgs10: list[float], start: str = "2023-01-31") -> dict:
    idx = pd.date_range(start, periods=len(dgs2), freq="ME")
    return {"DGS2": pd.Series(dgs2, index=idx), "DGS10": pd.Series(dgs10, index=idx)}


def _record(asof: str, legs: dict | None, conviction: float = 0.5,
            raw_legs: list | None = None, degraded: bool = False) -> dict:
    """One run-file line. ``raw_legs`` defaults to whatever survived grounding, so a
    test only has to state it when the two are meant to differ."""
    trade = None if legs is None else {
        "strategy": "duration", "asof": asof, "legs": legs,
        "conviction": conviction, "rationale": "because", "risk": {"tags": ["duration"]},
    }
    if raw_legs is None:
        raw_legs = [] if legs is None else [{"instrument": k, "weight": v}
                                            for k, v in legs.items()]
    raw = {"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.3}]}
    if raw_legs:
        raw["trade"] = {"legs": raw_legs, "conviction": conviction, "rationale": "because"}
    return {"asof": asof, "degraded": degraded, "raw_response": json.dumps(raw),
            "arbitrated": {"drivers": {"inflation": 0.3}, "trade": trade}}


def _run_file(tmp_path, records: list[dict]) -> str:
    p = tmp_path / "run.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return str(p)


# ── the sign convention ─────────────────────────────────────────────────────────
def test_a_long_yield_leg_earns_when_the_yield_rises(tmp_path):
    """+1.0 on DGS10 against a rising yield is a GAIN.

    In price space this same position loses money. If this test ever inverts, every
    P&L number in the notebook silently flips while still looking plausible.
    """
    macro = _macro(dgs2=[1.0, 1.0], dgs10=[3.0, 3.5])
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS10": 1.0})])
    trades = load_trades(path, TRADE_CONFIG)
    pnl = yield_pnl(trades, macro, ["DGS2", "DGS10"])
    assert pnl.loc[pd.Timestamp("2023-01-31")] == pytest.approx(0.5)


def test_two_legs_sum_to_the_hand_computed_pnl(tmp_path):
    macro = _macro(dgs2=[1.00, 1.20], dgs10=[3.00, 2.90])
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS2": -0.4, "DGS10": -0.6})])
    trades = load_trades(path, TRADE_CONFIG)
    pnl = yield_pnl(trades, macro, ["DGS2", "DGS10"])
    # -0.4 * (+0.20) + -0.6 * (-0.10) = -0.08 + 0.06
    assert pnl.iloc[0] == pytest.approx(-0.02)


# ── the outcome grid ────────────────────────────────────────────────────────────
def test_the_forward_level_comes_from_the_data_not_the_sample():
    """A run's LAST meeting is scoreable whenever the data extends past it.

    Shifting within the sample instead would make every truncated run quietly lose its
    final month — the kind of loss that shows up as a smaller ``n`` and nothing else.
    """
    macro = _macro(dgs2=[1.0, 1.0, 1.0], dgs10=[3.0, 3.4, 9.9])
    only_first_two = pd.DatetimeIndex(["2023-01-31", "2023-02-28"])
    fwd = forward_yield_change(macro, ["DGS10"], only_first_two)
    assert fwd["DGS10"].notna().all()
    assert fwd["DGS10"].iloc[1] == pytest.approx(9.9 - 3.4)


def test_the_tail_beyond_the_data_is_dropped_not_zeroed(tmp_path):
    macro = _macro(dgs2=[1.0, 1.0], dgs10=[3.0, 3.2])
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS10": 1.0}),
                                _record("2023-02-28", {"DGS10": 1.0})])
    trades = load_trades(path, TRADE_CONFIG)
    pnl = yield_pnl(trades, macro, ["DGS2", "DGS10"])
    assert list(pnl.index) == [pd.Timestamp("2023-01-31")]


def test_month_end_sampling_does_not_double_count_a_move():
    """Consecutive one-step P&Ls must partition the total move, not overlap it."""
    macro = _macro(dgs2=[1.0] * 4, dgs10=[3.0, 3.1, 3.4, 3.5])
    dates = pd.DatetimeIndex(["2023-01-31", "2023-02-28", "2023-03-31"])
    fwd = forward_yield_change(macro, ["DGS10"], dates)
    assert fwd["DGS10"].sum() == pytest.approx(3.5 - 3.0)


# ── abstention is not a flat position ───────────────────────────────────────────
def test_a_meeting_without_a_trade_contributes_no_observation(tmp_path):
    macro = _macro(dgs2=[1.0, 1.0, 1.0], dgs10=[3.0, 3.5, 4.0])
    path = _run_file(tmp_path, [_record("2023-01-31", None),
                                _record("2023-02-28", {"DGS10": 1.0})])
    trades = load_trades(path, TRADE_CONFIG)
    assert len(trades) == 2 and trades["has_trade"].tolist() == [False, True]
    pnl = yield_pnl(trades, macro, ["DGS2", "DGS10"])
    assert list(pnl.index) == [pd.Timestamp("2023-02-28")]


def test_a_degraded_meeting_is_excluded_entirely(tmp_path):
    path = _run_file(tmp_path, [_record("2023-01-31", None, degraded=True),
                                _record("2023-02-28", {"DGS10": 1.0})])
    assert len(load_trades(path, TRADE_CONFIG)) == 1


# ── the diagnostic half ─────────────────────────────────────────────────────────
def test_sign_violation_fires_on_opposed_legs_for_a_same_sign_pod(tmp_path):
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS2": -0.5, "DGS10": 0.5}),
                                _record("2023-02-28", {"DGS2": 0.5, "DGS10": 0.5})])
    trades = load_trades(path, TRADE_CONFIG)
    assert trades["sign_violation"].tolist() == [True, False]
    assert trade_validity(trades)["sign_violation_rate"] == pytest.approx(0.5)


def test_an_opposed_pod_inverts_the_same_check(tmp_path):
    cfg = {**TRADE_CONFIG, "sign_convention": "opposed"}
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS2": -0.5, "DGS10": 0.5}),
                                _record("2023-02-28", {"DGS2": 0.5, "DGS10": 0.5})])
    assert load_trades(path, cfg)["sign_violation"].tolist() == [False, True]


def test_a_single_leg_trade_can_never_violate_a_sign_convention(tmp_path):
    """Otherwise every ``real`` pod trade — universe of one — reads as a violation."""
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS10": -0.5})])
    assert load_trades(path, TRADE_CONFIG)["sign_violation"].tolist() == [False]


def test_silently_dropped_legs_are_recovered_from_the_raw_response(tmp_path):
    """``_parse_trade`` skips an out-of-universe leg with a bare ``continue``. The raw
    response is the only place that intent survives."""
    path = _run_file(tmp_path, [_record("2023-01-31", {"DGS2": -0.5},
                                        raw_legs=[{"instrument": "DGS2", "weight": -0.5},
                                                  {"instrument": "SPX", "weight": 0.5},
                                                  {"instrument": "DGS10", "weight": 0.0}])])
    t = load_trades(path, TRADE_CONFIG)
    assert t["legs_dropped_universe"].iloc[0] == 1
    assert t["legs_dropped_zero"].iloc[0] == 1
    assert t["n_legs"].iloc[0] == 1


def test_emitted_but_rejected_is_distinguished_from_never_emitted(tmp_path):
    """The saved contract shows ``trade: null`` for both. They need different fixes."""
    path = _run_file(tmp_path, [
        _record("2023-01-31", None),                                    # never emitted
        _record("2023-02-28", None, raw_legs=[{"instrument": "SPX", "weight": 0.5}]),
    ])
    t = load_trades(path, TRADE_CONFIG)
    assert t["emitted"].tolist() == [False, True]
    assert t["has_trade"].tolist() == [False, False]
    v = trade_validity(t)
    assert v["rejected_rate"] == pytest.approx(0.5)
    assert v["emitted_rate"] == pytest.approx(0.5)


def test_a_truncated_raw_response_is_a_finding_not_a_crash(tmp_path):
    rec = _record("2023-01-31", {"DGS10": 0.5})
    rec["raw_response"] = '{"notes": "n", "trade": {"legs": [{"instrum'
    t = load_trades(_run_file(tmp_path, [rec]), TRADE_CONFIG)
    assert t["emitted"].iloc[0] is False or not t["emitted"].iloc[0]
    assert t["has_trade"].iloc[0]  # the grounded trade is still scored


# ── the statistics ──────────────────────────────────────────────────────────────
def test_score_reports_nan_rather_than_inventing_a_t_stat_on_two_points():
    s = score_trades(pd.Series([0.1, -0.1], index=pd.date_range("2023-01-31", periods=2, freq="ME")))
    assert s["n"] == 2 and s["t_stat"] != s["t_stat"]  # NaN


def test_hit_rate_and_sharpe_on_a_known_series():
    idx = pd.date_range("2023-01-31", periods=4, freq="ME")
    s = score_trades(pd.Series([0.1, 0.1, 0.1, -0.1], index=idx))
    assert s["hit_rate"] == pytest.approx(0.75)
    assert s["periods_per_year"] == pytest.approx(365.25 / 30.0, rel=0.1)
    assert s["sharpe_ann"] > 0


def test_conviction_ic_reads_sizing_not_direction():
    """Conviction is unsigned, so this asks: did it size bigger when it was more right?"""
    idx = pd.date_range("2023-01-31", periods=4, freq="ME")
    pnl = pd.Series([-0.2, -0.1, 0.1, 0.2], index=idx)
    good = score_trades(pnl, pd.Series([0.1, 0.2, 0.3, 0.4], index=idx))
    bad = score_trades(pnl, pd.Series([0.4, 0.3, 0.2, 0.1], index=idx))
    assert good["ic_conviction"] == pytest.approx(1.0)
    assert bad["ic_conviction"] == pytest.approx(-1.0)


# ── a deliberate flat is a position, not a silence ──────────────────────────────
def _flat_record(asof: str) -> dict:
    """What `_parse_trade` produces for `{"flat": true, "legs": []}` — a StrategyTrade
    with no legs, which is a decision to carry nothing."""
    raw = {"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.3}],
           "trade": {"flat": True, "legs": [], "conviction": 0.0, "rationale": "sit out"}}
    return {"asof": asof, "degraded": False, "raw_response": json.dumps(raw),
            "arbitrated": {"drivers": {"inflation": 0.3},
                           "trade": {"strategy": "duration", "asof": asof, "legs": {},
                                     "conviction": 0.0, "rationale": "sit out",
                                     "risk": {"tags": []}}}}


def test_a_deliberate_flat_scores_as_a_real_zero(tmp_path):
    """The distinction the `flat` flag exists to preserve: choosing to sit out has an
    outcome. Dropping it would silently condition the record on the PM having acted."""
    macro = _macro(dgs2=[1.0, 1.0], dgs10=[3.0, 3.9])
    t = load_trades(_run_file(tmp_path, [_flat_record("2023-01-31")]), TRADE_CONFIG)
    assert t["has_trade"].iloc[0] and t["flat"].iloc[0] and t["n_legs"].iloc[0] == 0
    pnl = yield_pnl(t, macro, ["DGS2", "DGS10"])
    assert list(pnl.index) == [pd.Timestamp("2023-01-31")]
    assert pnl.iloc[0] == pytest.approx(0.0)


def test_a_flat_is_not_counted_as_an_abstention(tmp_path):
    path = _run_file(tmp_path, [_flat_record("2023-01-31"), _record("2023-02-28", None)])
    v = trade_validity(load_trades(path, TRADE_CONFIG))
    assert v["grounded_rate"] == pytest.approx(0.5)   # the flat counts as a trade
    assert v["flat_rate"] == pytest.approx(1.0)       # ...and it is the only one


def test_a_flat_cannot_violate_a_sign_convention(tmp_path):
    t = load_trades(_run_file(tmp_path, [_flat_record("2023-01-31")]), TRADE_CONFIG)
    assert not t["sign_violation"].iloc[0]
