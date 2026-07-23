"""Two design fixes: the PM's answer space, and the PM's memory of its own position.

Both existed as defects measured on the first duration run (2026-07-22):

  * The pod mandate spoke in rate space while the ``conviction`` field spoke in driver
    space. On 55 of 120 meetings the PM followed the mandate, and ``pm_bench`` graded
    it as driver space — turning a balance_sheet IC of +0.714 into -0.167.
  * The PM never saw its own previous arbitration, so it re-derived a position from
    nothing every month: 45.8% sign flips and mean |Δnet| ≈ mean |net|.

These tests pin the *structure* of both fixes — that the two halves of the prompt agree,
that the grader honours the declaration, and that memory can only ever reach backwards.
No LLM calls.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.layered.contracts import ArbitratedView, DriverView, StrategyTrade
from src.layered.pm.board import ViewBoard
from src.layered.pm.build import build_pm
from src.layered.pm.llm_pm import LLMPM, submit_arbitration_tool

TRADE = {"universe": ["DGS2", "DGS10"], "max_legs": 2, "risk_tags": ["duration"]}


def flat(s: str) -> str:
    """Prompt text is hard-wrapped, so a phrase spanning a line break is not a literal
    substring of it. Collapse whitespace before asserting on wording."""
    return " ".join(s.split())


def _pod(**cfg) -> dict:
    base = {"display_name": "t", "clock_freq": "ME",
            "listens_to": {"inflation": {"polarity": 1},
                           "balance_sheet": {"polarity": -1}}}
    base.update(cfg)
    return base


def _board():
    return ViewBoard({d: [DriverView(driver=d, asof=pd.Timestamp("2023-06-30"),
                                     direction="up", conviction=0.5, horizon_days=31,
                                     level=1.0, report=f"{d} report")]
                      for d in ("inflation", "balance_sheet")})


class FakeLLM:
    """Records what it was shown, so the prompt itself can be asserted on."""

    def __init__(self, payload):
        self.payload, self.seen = payload, []

    def complete(self, system, user, tool):
        self.seen.append({"system": system, "user": user, "tool": tool})
        return self.payload


_PAYLOAD = ('{"notes": "n", "falsifier": "if CPI turns",'
            ' "drivers": [{"driver": "inflation", "conviction": 0.4, "why": "w"}],'
            ' "trade": {"legs": [{"instrument": "DGS10", "weight": -0.6}],'
            ' "conviction": 0.5, "rationale": "r"}}')


# ── the answer space ────────────────────────────────────────────────────────────
def test_the_two_halves_of_the_prompt_agree_in_driver_space():
    """The defect this key closes: the calibration ladder and the field description
    must describe the SAME quantity. Here, the driver's own measurement."""
    pm = LLMPM(pod="t", config=_pod(answer_space="driver"))
    sys_prompt = pm._system_prompt()
    desc = (submit_arbitration_tool(pm.listens_to, answer_space=pm.answer_space)
            ["input_schema"]["properties"]["drivers"]["items"]
            ["properties"]["conviction"]["description"])
    assert "own headline measurement" in sys_prompt.lower() or \
           "driver's own" in sys_prompt.lower()
    assert "headline measurement rises" in desc
    # ...and it says out loud that the rate-axis reading is NOT what is wanted here.
    assert "balance sheet" in sys_prompt.lower()


def test_the_two_halves_of_the_prompt_agree_in_rate_space():
    pm = LLMPM(pod="t", config=_pod(answer_space="rate"))
    sys_prompt = pm._system_prompt()
    desc = (submit_arbitration_tool(pm.listens_to, answer_space=pm.answer_space)
            ["input_schema"]["properties"]["drivers"]["items"]
            ["properties"]["conviction"]["description"])
    assert "pushing" in sys_prompt.lower() and "yields up" in sys_prompt.lower()
    assert "pushing nominal Treasury yields UP" in desc
    assert "headline measurement rises" not in desc


def test_default_is_driver_space_so_an_undeclared_pod_keeps_the_old_contract():
    assert LLMPM(pod="t", config=_pod()).answer_space == "driver"


def test_a_misspelled_answer_space_is_refused_not_defaulted():
    """Silently defaulting would flip the grader's reading of every number in the run
    and leave nothing behind saying why."""
    with pytest.raises(ValueError, match="answer_space"):
        LLMPM(pod="t", config=_pod(answer_space="rates")).answer_space


def test_every_shipped_pod_declares_a_space():
    for pod in ("duration", "curve", "front_end", "real"):
        assert build_pm(pod, None).answer_space in ("driver", "rate")


def test_benchmark_reorients_a_rate_space_run_before_grading(monkeypatch):
    """A rate-space conviction on a -1-polarity driver is the opposite sign from the
    driver-space quantity the levels are built from. Grading it unoriented is exactly
    what produced balance_sheet ic_pm = -0.167 against an analyst's +0.714.

    Built so the PM is a PERFECT driver-space forecaster: its signal is the realised
    move. Under ``driver`` both drivers must score IC +1; under ``rate`` only the
    +1-polarity one may, because re-orientation flips the other back.
    """
    import numpy as np

    import src.layered.evaluation.pm_bench as pb
    from src.layered.evaluation.pm_bench import benchmark

    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    lv = pd.Series(np.arange(40, dtype=float) + (np.arange(40) % 5) * 3.0, index=idx)
    outcome = (lv.shift(-1) - lv)                     # what ICEvaluator will grade against
    pol = {"inflation": 1.0, "balance_sheet": -1.0}

    monkeypatch.setattr(pb, "driver_levels",
                        lambda drivers, dates, macro=None, persona_dir=None:
                        pd.DataFrame({d: lv for d in drivers}))
    monkeypatch.setattr(pb, "analyst_snap",
                        lambda b, d, dr: pd.DataFrame({x: outcome for x in dr}, index=d))

    # A perfect driver-space call on both drivers.
    frame = pd.DataFrame({d: outcome for d in pol}, index=idx).dropna()

    as_driver = benchmark(frame, None, idx, pol, answer_space="driver")
    as_rate = benchmark(frame, None, idx, pol, answer_space="rate")

    assert as_driver.loc["inflation", "ic_pm"] == pytest.approx(1.0)
    assert as_driver.loc["balance_sheet", "ic_pm"] == pytest.approx(1.0)
    # +1 polarity: the two spaces coincide, so nothing moves.
    assert as_rate.loc["inflation", "ic_pm"] == pytest.approx(1.0)
    # -1 polarity: re-orientation flips it, which is the whole point of the key.
    assert as_rate.loc["balance_sheet", "ic_pm"] == pytest.approx(-1.0)


def test_benchmark_refuses_an_unknown_space():
    from src.layered.evaluation.pm_bench import benchmark
    with pytest.raises(ValueError, match="answer_space"):
        benchmark(pd.DataFrame({"d": [1.0]}), None, pd.DatetimeIndex(["2023-01-31"]),
                  {"d": 1.0}, answer_space="yields")


# ── memory ──────────────────────────────────────────────────────────────────────
def test_memory_is_off_by_default_and_changes_neither_prompt():
    """The memory-less arm must reproduce byte-for-byte, or the A/B is not an A/B."""
    off = LLMPM(pod="t", config=_pod(), llm=FakeLLM(_PAYLOAD))
    on = LLMPM(pod="t", config=_pod(), llm=FakeLLM(_PAYLOAD), use_memory=True)
    assert off.memory is None
    assert "previous meeting" not in flat(off._system_prompt()).lower()
    assert "previous meeting" in flat(on._system_prompt()).lower()
    m = _board().at("2023-06-30")
    assert off._user_prompt(m, off.memory) == on._user_prompt(m, None)


def test_the_first_meeting_has_no_memory_block():
    pm = LLMPM(pod="t", config=_pod(), llm=FakeLLM(_PAYLOAD), use_memory=True)
    assert "Your previous meeting" not in pm._user_prompt(_board().at("2023-06-30"),
                                                          pm.memory)


def test_the_second_meeting_is_shown_the_position_it_is_carrying():
    llm = FakeLLM(_PAYLOAD)
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=llm, use_memory=True)
    m = _board().at("2023-06-30")
    pm.arbitrate(m)                                   # forms the memory
    second = pm._user_prompt(m, pm.memory)
    assert "Your previous meeting" in second
    assert "DGS10 -0.60" in second                    # the incumbent position
    assert "inflation +0.40" in second                # the previous call
    assert "if CPI turns" in second                   # the previous falsifier


def test_memory_carries_commitments_not_the_previous_prose():
    """Handing back its own notes invites the model to re-read its own reasoning
    instead of this meeting's reports — the failure the analyst renderer avoids."""
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(_PAYLOAD), use_memory=True)
    m = _board().at("2023-06-30")
    av = pm.arbitrate(m)
    assert av.notes == "n"
    assert "\nn\n" not in pm._render_memory(av)


def test_no_position_is_stated_rather_than_omitted():
    """An empty memory block would read as "no previous meeting", not "I abstained"."""
    payload = '{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.4, "why": "w"}]}'
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(payload), use_memory=True)
    av = pm.arbitrate(_board().at("2023-06-30"))
    assert "carrying no position" in pm._render_memory(av)


def test_a_degraded_meeting_does_not_erase_the_position():
    """A failed call must leave the PM holding what it last actually chose, not an
    empty book it never decided on."""
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(_PAYLOAD), use_memory=True)
    m = _board().at("2023-06-30")
    good = pm.arbitrate(m)
    pm.llm = FakeLLM("not json at all")
    degraded = pm.arbitrate(m)
    assert not degraded.drivers                       # the call did degrade
    assert pm.memory is good                          # ...and the position survived


def test_the_memory_block_carries_no_absolute_date():
    """Same invariant as the brief: a date is the token that most helps a model recall
    the period instead of reading the evidence."""
    import re
    _DATE = re.compile(r"(?<![\d+\-.,])(?:19|20)\d{2}(?![\d.,])")
    av = ArbitratedView(asof=pd.Timestamp("2023-06-30"), drivers={"inflation": 0.4},
                        falsifier="if the 2019 pattern repeats by March 2024",
                        trade=StrategyTrade(strategy="t", asof=pd.Timestamp("2023-06-30"),
                                            legs={"DGS10": -0.6}, conviction=0.5))
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), use_memory=True)
    block = pm._render_memory(av)
    assert not _DATE.search(block)
    assert "March" not in block


def test_the_memory_contract_names_the_over_trading_failure():
    """The instruction is the whole point of the fix: without an incumbent position in
    the prompt, "do not reverse without cause" is not something the model can act on."""
    sys_prompt = flat(LLMPM(pod="t", config=_pod(), use_memory=True)._system_prompt())
    assert "carrying a position" in sys_prompt
    assert "price move is not on its own such a reason" in sys_prompt


# ── the flat position ───────────────────────────────────────────────────────────
_FLAT = ('{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.1, "why": "w"}],'
         ' "trade": {"flat": true, "legs": [], "conviction": 0.0, "rationale": "sit out"}}')


def test_an_explicit_flat_parses_to_a_position_not_to_none():
    """Before `flat` existed, "flatten to neutral" and "I never answered" were the same
    row in the run file. The first memory pilot produced both and they were
    indistinguishable."""
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(_FLAT))
    av = pm.arbitrate(_board().at("2023-06-30"))
    assert av.trade is not None
    assert av.trade.legs == {} and av.trade.gross == 0.0


def test_omitting_the_trade_still_means_no_position_view():
    payload = '{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.1, "why": "w"}]}'
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(payload))
    assert pm.arbitrate(_board().at("2023-06-30")).trade is None


def test_legs_win_over_a_contradictory_flat_flag():
    """A model that names a position and also sets flat is read as holding what it
    named — the concrete answer beats the flag."""
    payload = ('{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.1, "why": "w"}],'
               ' "trade": {"flat": true, "legs": [{"instrument": "DGS10", "weight": -0.4}],'
               ' "conviction": 0.3, "rationale": "r"}}')
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(payload))
    assert pm.arbitrate(_board().at("2023-06-30")).trade.legs == {"DGS10": -0.4}


def test_memory_distinguishes_a_chosen_flat_from_never_having_a_view():
    pm = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(_FLAT), use_memory=True)
    flat_av = pm.arbitrate(_board().at("2023-06-30"))
    assert "chose to be flat" in pm._render_memory(flat_av)

    none_payload = '{"notes": "n", "drivers": [{"driver": "inflation", "conviction": 0.1, "why": "w"}]}'
    pm2 = LLMPM(pod="t", config=_pod(trade=TRADE), llm=FakeLLM(none_payload), use_memory=True)
    assert "took no position view" in pm2._render_memory(pm2.arbitrate(_board().at("2023-06-30")))


def test_the_memory_contract_says_the_trade_is_the_resulting_position():
    """The wording that stops "hold" being expressed as silence."""
    sys_prompt = flat(LLMPM(pod="t", config=_pod(), use_memory=True)._system_prompt())
    assert "position you want to be carrying AFTER this meeting" in sys_prompt
    assert "restate your existing legs to hold" in sys_prompt
