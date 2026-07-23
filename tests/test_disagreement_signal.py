"""The disagreement-as-signal evaluator.

The rank-IC core is unit-tested exactly on synthetic series. The end-to-end
``evaluate_run`` is exercised as an integration test against the committed duration run
and board when they are present (they carry vendored, offline data), and skipped
otherwise — the same posture the no-lookahead test takes toward the FRED CSVs.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

from src.layered.evaluation.disagreement_signal import (_rank_ic,
                                                        disagreement_vs_magnitude,
                                                        evaluate_run, summarize)
from src.layered.evaluation.pm_runs import PMRun

_PM_RUN = "reports/pm/duration_on.jsonl"
_BOARD = "reports/ab"


# ── the rank-IC core ─────────────────────────────────────────────────────────────
def test_rank_ic_perfect_monotone():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    s = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    r = _rank_ic(s, s * 2 + 1, "mono")          # strictly increasing transform → IC 1
    assert r["ic"] == pytest.approx(1.0) and r["n"] == 6


def test_rank_ic_perfect_inverse():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    s = pd.Series([1, 2, 3, 4, 5, 6], index=idx, dtype=float)
    r = _rank_ic(s, -s, "inv")
    assert r["ic"] == pytest.approx(-1.0)


def test_rank_ic_degenerate_is_nan():
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    s = pd.Series([1, 2, 3, 4], index=idx, dtype=float)
    const = pd.Series([5, 5, 5, 5], index=idx, dtype=float)
    r = _rank_ic(s, const, "const")
    assert r["ic"] != r["ic"]                    # NaN — no variance in the target


def test_rank_ic_too_few_points():
    idx = pd.date_range("2020-01-31", periods=2, freq="ME")
    s = pd.Series([1.0, 2.0], index=idx)
    assert _rank_ic(s, s, "short")["n"] == 2 and _rank_ic(s, s, "short")["ic"] != _rank_ic(s, s, "short")["ic"]


# ── the no-look-ahead alignment (fully synthetic, offline) ───────────────────────
def test_disagreement_aligns_to_the_future_move_not_the_past():
    """The property the whole module rests on, proved rather than argued: disagreement at
    meeting ``t`` is scored against the ``t → t+1`` move, not the ``t-1 → t`` move.

    A deliberately zig-zag future ``|move|`` sequence [1, 4, 2, 5, 3] with disagreement
    set equal to it makes the forward-aligned rank IC exactly +1. The same data read one
    step in the past would score about -0.6, so ``ic > 0.9`` can only hold if the
    alignment looks forward. ``balance_sheet``'s only raw input is its level feature
    (``WALCL`` → ``fed_assets``), so supplying ``WALCL`` directly fixes the driver level
    to this exact path with no network and no release-dating."""
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    levels = [0.0, 1.0, 5.0, 7.0, 12.0, 15.0]            # forward |moves| = [1, 4, 2, 5, 3]
    macro = {"WALCL": pd.Series(levels, index=dates)}
    disagreement = pd.Series([1.0, 4.0, 2.0, 5.0, 3.0, 9.0], index=dates)
    frame = pd.DataFrame({"balance_sheet": disagreement.to_numpy()}, index=dates)
    empty = pd.Series(dtype=float)
    run = PMRun(path="", pod="balance", model="stub", meta={}, frame=frame,
                disagreement=disagreement, coverage=empty, degraded=empty,
                age=pd.DataFrame(index=dates), notes=empty,
                trades=pd.Series(dtype=object))
    r = disagreement_vs_magnitude(run, steps=1, macro=macro)
    assert r["n"] == 5 and r["ic"] > 0.9


# ── end-to-end on committed data ─────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(_PM_RUN), reason="committed PM run not present")
def test_evaluate_run_end_to_end():
    result = evaluate_run(_PM_RUN, board_dir=_BOARD, board_suffix="_on")
    assert result["n_meetings"] > 0
    assert "ic" in result["magnitude"]
    # the conditioning split is a DataFrame with the two halves when there is enough data
    split = result["accuracy_split"]
    assert isinstance(split, pd.DataFrame)
    if not split.empty:
        assert set(split.index) <= {"low_disagreement", "high_disagreement"}
    # graph question ran (board present) and produced an IC or a recorded error
    assert "graph" in result
    assert isinstance(summarize(result), str) and summarize(result)


@pytest.mark.skipif(not os.path.exists(_PM_RUN), reason="committed PM run not present")
def test_evaluate_run_without_board_omits_graph():
    result = evaluate_run(_PM_RUN)
    assert "graph" not in result
