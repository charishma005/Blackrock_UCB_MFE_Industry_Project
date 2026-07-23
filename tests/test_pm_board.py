"""The as-of snap — the PM layer's look-ahead gate.

``ViewBoard.at`` is to the PM what ``AsOf`` is to the analyst: the single place a
view can be read, and therefore the single place causality can be broken. These
tests exercise it on synthetic views (so the failure modes can be constructed
directly) and then on the real corpus.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.layered.contracts import DriverView
from src.layered.pm.board import BoardConfigMismatch, ViewBoard, _assert_identical_config


def view(driver: str, asof: str, direction: str = "up", conviction: float = 0.5,
         degraded: bool = False, carried: bool = False) -> DriverView:
    return DriverView(driver=driver, asof=pd.Timestamp(asof), direction=direction,
                      conviction=conviction, horizon_days=31, level=1.0,
                      degraded=degraded, carried=carried)


def test_snap_never_returns_a_future_view():
    """The gate. A view formed after the meeting cannot reach the PM.

    ``expire_after_days`` is set wide so this exercises causality alone; expiry is a
    separate property with its own test, and leaving it at the default here would
    withhold the January view for being old and pass for the wrong reason.
    """
    b = ViewBoard({"inflation": [view("inflation", "2023-01-15"),
                                 view("inflation", "2023-07-15")]},
                  expire_after_days=400)
    entry = b.at("2023-06-30").entries["inflation"]
    assert entry.view.asof == pd.Timestamp("2023-01-15")
    # ...and nowhere on the calendar does the July view leak backwards.
    for ts in pd.date_range("2023-01-15", "2023-07-14", freq="D"):
        got = b.at(ts).entries["inflation"]
        assert got.view is None or got.view.asof <= ts


def test_snap_picks_the_latest_admissible_view():
    b = ViewBoard({"d": [view("d", "2023-01-31"), view("d", "2023-02-28"),
                         view("d", "2023-03-31")]})
    assert b.at("2023-03-15").entries["d"].view.asof == pd.Timestamp("2023-02-28")


def test_age_days_is_the_gap_to_the_meeting():
    b = ViewBoard({"d": [view("d", "2023-06-08")]})
    assert b.at("2023-06-30").entries["d"].age_days == 22


def test_no_view_yet_before_the_first_report():
    b = ViewBoard({"d": [view("d", "2023-06-30")]})
    e = b.at("2023-01-31").entries["d"]
    assert not e.present and e.reason == "no_view_yet"


def test_degraded_view_is_never_snapped():
    """A failed call is an abstention, not an opinion. The board falls back to the
    last real view and the age grows visibly, rather than serving a flat stub."""
    b = ViewBoard({"d": [view("d", "2023-05-31"),
                         view("d", "2023-06-30", degraded=True)]})
    e = b.at("2023-06-30").entries["d"]
    assert e.present and e.view.asof == pd.Timestamp("2023-05-31")
    assert e.age_days == 30


def test_expired_view_is_withheld_and_explained():
    b = ViewBoard({"d": [view("d", "2023-01-31")]}, expire_after_days=95)
    e = b.at("2023-12-31").entries["d"]
    assert not e.present and e.reason == "expired" and e.age_days > 95


def test_stale_is_flagged_but_still_served():
    b = ViewBoard({"d": [view("d", "2023-01-31")]}, stale_after_days=45,
                  expire_after_days=95)
    e = b.at("2023-03-31").entries["d"]
    assert e.present and e.stale


def test_carried_flag_survives_onto_the_entry():
    b = ViewBoard({"d": [view("d", "2023-06-30", carried=True)]})
    assert b.at("2023-06-30").entries["d"].carried


def test_expire_must_exceed_stale():
    with pytest.raises(ValueError):
        ViewBoard({}, stale_after_days=95, expire_after_days=45)


def test_config_identity_assertion_rejects_mixed_legs():
    same = {"start": "2016-01-01", "end": "2025-12-31", "model": "claude-sonnet-5",
            "text_mode": "cue", "text_doc": "statement",
            "describe_features": False, "memory": True}
    mixed = dict(same, model="claude-haiku-4-5")
    with pytest.raises(BoardConfigMismatch):
        _assert_identical_config({"a": {"config": same}, "b": {"config": mixed}})
    _assert_identical_config({"a": {"config": same}, "b": {"config": dict(same)}})


# ── the real corpus ─────────────────────────────────────────────────────────
REPORTS = "reports/ab"
MEETINGS = pd.date_range("2016-01-31", "2025-12-31", freq="ME")


@pytest.fixture(scope="module")
def real_board():
    try:
        return ViewBoard.from_dir(REPORTS, "_on")
    except (FileNotFoundError, BoardConfigMismatch) as e:
        pytest.skip(f"real board unavailable: {e}")


def test_real_board_is_complete(real_board):
    """Every driver present at every meeting. The acceptance gate for the repair."""
    for ts in MEETINGS:
        m = real_board.at(ts)
        assert m.coverage == 1.0, f"{ts.date()}: missing {m.absent}"


def test_view_ages_are_explained_by_the_release_calendar(real_board):
    """No analyst is more than one release behind, except where a release never came.

    31 days is the structural maximum: the slowest clock is monthly, so at month end
    nobody should be further back than that. The single exception in the corpus is
    2025-10-31, where the October 2025 US government shutdown suspended BLS
    publication — the vendored UNRATE series jumps 2025-09-08 → 2025-11-08 with no
    October print, and CPI likewise. The board's correct response to a release that
    never happened is to keep serving the last real view and let its age grow, which
    is what the brief then labels STALE. Asserted rather than tolerated so that a
    *second* such meeting appearing would fail here instead of passing unnoticed.
    """
    stale = {ts: real_board.at(ts).max_age_days for ts in MEETINGS
             if real_board.at(ts).max_age_days > 31}
    assert list(stale) == [pd.Timestamp("2025-10-31")], f"unexplained stale meetings: {stale}"
    # One missed release, not two: still inside the 95-day expiry.
    assert max(stale.values()) < 62


def test_real_board_is_causal(real_board):
    for ts in MEETINGS:
        for d, e in real_board.at(ts).entries.items():
            if e.present:
                assert e.view.asof <= ts, f"{d} at {ts.date()} used a future view"
