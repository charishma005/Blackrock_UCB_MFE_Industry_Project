"""CsvReplayAnalyst replays r7 signals point-in-time, with no lookahead."""
from __future__ import annotations

import pandas as pd

from src.portfolio.replay_analyst import (
    POS_FLAT_THRESHOLD,
    CsvReplayAnalyst,
    _direction,
    load_replay_analysts,
)

_REPLAY_CSV = "data/equity_replay/positioning_vector.csv"


def _write_csv(tmp_path, rows):
    p = tmp_path / "d_vector.csv"
    pd.DataFrame(rows, columns=["date", "pos", "conviction", "reasoning"]).to_csv(
        p, index=False)
    return p


def test_direction_mapping_is_strict_at_the_threshold():
    assert _direction(0.5) == "up"
    assert _direction(-0.5) == "down"
    assert _direction(0.0) == "flat"
    # Exactly the threshold is flat (strict inequality).
    assert _direction(POS_FLAT_THRESHOLD) == "flat"
    assert _direction(-POS_FLAT_THRESHOLD) == "flat"
    assert _direction(POS_FLAT_THRESHOLD + 1e-6) == "up"


def test_uses_latest_row_on_or_before_asof(tmp_path):
    csv = _write_csv(tmp_path, [
        ["2020-01-03", 0.5, 0.6, "a"],
        ["2020-01-10", -0.5, 0.7, "b"],
        ["2020-01-17", 0.5, 0.8, "c"],
    ])
    a = CsvReplayAnalyst("d", csv, max_age_days=14)
    v = a.view_asof(pd.Timestamp("2020-01-12"))
    assert v is not None
    assert v.direction == "down" and v.reasoning == "b"
    assert v.source == "replay:d"
    assert v.carried is True                 # row date < asof


def test_exact_date_is_not_carried(tmp_path):
    csv = _write_csv(tmp_path, [["2020-01-03", 0.5, 0.6, "a"]])
    v = CsvReplayAnalyst("d", csv).view_asof(pd.Timestamp("2020-01-03"))
    assert v is not None and v.carried is False


def test_before_first_row_returns_none(tmp_path):
    csv = _write_csv(tmp_path, [["2020-01-03", 0.5, 0.6, "a"]])
    assert CsvReplayAnalyst("d", csv).view_asof(pd.Timestamp("2019-06-01")) is None


def test_stale_row_beyond_max_age_returns_none(tmp_path):
    csv = _write_csv(tmp_path, [["2020-01-03", 0.5, 0.6, "a"]])
    a = CsvReplayAnalyst("d", csv, max_age_days=14)
    assert a.view_asof(pd.Timestamp("2020-03-01")) is None     # >14d stale


def test_truncating_the_future_does_not_change_a_past_view(tmp_path):
    full = _write_csv(tmp_path, [
        ["2020-01-03", 0.5, 0.6, "a"],
        ["2020-01-10", -0.5, 0.7, "b"],
        ["2020-01-17", 0.5, 0.8, "c"],
    ])
    trunc = tmp_path / "t_vector.csv"
    pd.read_csv(full).iloc[:2].to_csv(trunc, index=False)
    asof = pd.Timestamp("2020-01-10")
    v_full = CsvReplayAnalyst("d", full).view_asof(asof)
    v_trunc = CsvReplayAnalyst("d", trunc).view_asof(asof)
    assert v_full.direction == v_trunc.direction
    assert v_full.reasoning == v_trunc.reasoning
    assert v_full.signed_conviction == v_trunc.signed_conviction


def test_vendored_replay_analysts_load_and_span_the_r7_grid():
    analysts = load_replay_analysts()
    assert set(analysts) == {
        "sector_breadth", "vol_regime", "positioning", "risk_appetite"}
    v = analysts["positioning"].view_asof(pd.Timestamp("2015-06-05"))
    assert v is not None and -1.0 <= v.level <= 1.0
    # Pre-grid start (grid begins 2012-01-13) yields no view.
    assert analysts["positioning"].view_asof(pd.Timestamp("2010-01-01")) is None
